"""
FastAPI Backend for Arabic Dialect Sentiment Analysis Web Application

This module provides a RESTful API for sentiment analysis with:
- Text preprocessing
- Model inference
- Explainability features
- RTL text support
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
import logging
import json
import os
from pathlib import Path
import sys

# Add project root (the arabic-dialect-sentiment folder) to sys.path for imports
# main.py lives in <project>/arabic-dialect-sentiment/backend/main.py
# We need to add '<project>/arabic-dialect-sentiment' so that 'src' package is importable
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data.preprocessor import ArabicTextPreprocessor
from src.utils.config_loader import ConfigLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Arabic Dialect Sentiment Analysis API",
    description="API for analyzing sentiment in Gulf Arabic dialect text",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend communication (dev/prod)
# Configure via ALLOWED_ORIGINS env var (comma-separated). Defaults to '*'
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()] if allowed_origins_env != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configuration
try:
    config = ConfigLoader.load_config("configs/data_config.yaml")
    preprocessor = ArabicTextPreprocessor(config)
    logger.info("Configuration and preprocessor loaded successfully")
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    config = {}
    preprocessor = None

# Pydantic models for request/response
class SentimentRequest(BaseModel):
    text: str = Field(..., description="Arabic text to analyze", min_length=1, max_length=1000)
    dialect: Optional[str] = Field(None, description="Arabic dialect (gulf, egyptian, levantine, msa)")
    include_explanation: bool = Field(False, description="Whether to include model explanation")

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    dialect: str
    explanation: Optional[Dict[str, Any]] = None
    processing_time: float

class BatchSentimentRequest(BaseModel):
    texts: List[str] = Field(..., description="List of Arabic texts to analyze", min_items=1, max_items=100)
    dialect: Optional[str] = Field(None, description="Arabic dialect for all texts")
    include_explanations: bool = Field(False, description="Whether to include explanations")

class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse]
    total_processing_time: float
    average_confidence: float

class HealthResponse(BaseModel):
    status: str
    message: str
    version: str
    timestamp: str

# Global variables for model management
sentiment_model = None
tokenizer = None
model_loaded = False
model_labels = None  # List[str] mapping index->label

def load_sentiment_model():
    """Load the trained sentiment analysis model."""
    global sentiment_model, tokenizer, model_loaded
    
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        # Try model paths in order of preference
        candidate_paths = [
            "models",  # default saved folder
            "models/fine_tuned",
            "models/baselines/transformer_baseline",
        ]

        selected_path = None
        for candidate in candidate_paths:
            if Path(candidate).exists():
                selected_path = candidate
                break

        if selected_path:
            logger.info(f"Loading model from: {selected_path}")
            tokenizer = AutoTokenizer.from_pretrained(selected_path)
            sentiment_model = AutoModelForSequenceClassification.from_pretrained(selected_path)
        else:
            # Use pre-trained model as last resort
            logger.info("Loading pre-trained AraBERT model...")
            model_name = "aubmindlab/bert-base-arabertv2"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=3
            )

        # Move model to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sentiment_model.to(device)

        # Prepare labels depending on number of classes
        num_labels = getattr(sentiment_model.config, "num_labels", 3)
        global model_labels
        if num_labels == 4:
            model_labels = ["NEG", "POS", "NEUTRAL", "OBJ"]
        elif num_labels == 3:
            model_labels = ["negative", "neutral", "positive"]
        else:
            model_labels = [str(i) for i in range(num_labels)]

        model_loaded = True
        logger.info(f"Sentiment model loaded successfully on {device}. Labels: {model_labels}")
        
    except Exception as e:
        logger.error(f"Failed to load sentiment model: {e}")
        model_loaded = False

def predict_sentiment(text: str, include_explanation: bool = False) -> Dict[str, Any]:
    """
    Predict sentiment for a given text.
    
    Args:
        text: Input Arabic text
        include_explanation: Whether to include explanation
        
    Returns:
        Dictionary with prediction results
    """
    import torch
    import time
    
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Preprocess text
        cleaned_text = preprocessor.clean_text(text) if preprocessor else text
        
        # Tokenize
        inputs = tokenizer(
            cleaned_text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        # Ensure tensors are on the same device as model
        device = next(sentiment_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get prediction
        with torch.no_grad():
            outputs = sentiment_model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Map class to sentiment using loaded labels
        sentiment = model_labels[predicted_class] if model_labels and 0 <= predicted_class < len(model_labels) else "unknown"
        
        # Identify dialect
        dialect = preprocessor.identify_dialect(text) if preprocessor else "unknown"
        
        # Prepare response
        result = {
            "text": text,
            "sentiment": sentiment,
            "confidence": confidence,
            "dialect": dialect,
            "processing_time": time.time() - start_time
        }
        
        # Add explanation if requested
        if include_explanation:
            result["explanation"] = generate_explanation(text, inputs, probabilities)
        
        return result
        
    except Exception as e:
        logger.error(f"Error during sentiment prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

def generate_explanation(text: str, inputs: Dict, probabilities):
    """
    Generate explanation for the sentiment prediction.
    
    Args:
        text: Input text
        inputs: Tokenized inputs
        probabilities: Model output probabilities
        
    Returns:
        Explanation dictionary
    """
    try:
        # Get attention weights if available
        attention_weights = None
        if hasattr(sentiment_model, 'get_attention_weights'):
            attention_weights = sentiment_model.get_attention_weights()
        
        # Get top tokens by attention
        top_tokens = []
        if attention_weights is not None:
            # Extract most attended tokens
            import torch as _torch
            attention_scores = attention_weights.mean(dim=1).squeeze()
            k = min(10, attention_scores.numel())
            top_indices = _torch.topk(attention_scores, k=k).indices
            top_tokens = [tokenizer.decode([inputs['input_ids'][0][int(idx)]]) for idx in top_indices]
        
        # Get class probabilities
        class_probs = {model_labels[i] if model_labels and i < len(model_labels) else str(i): float(probabilities[0][i].item()) for i in range(probabilities.shape[-1])}
        
        explanation = {
            "attention_tokens": top_tokens,
            "class_probabilities": class_probs,
            "confidence_threshold": 0.7,
            "explanation_type": "attention_based"
        }
        
        return explanation
        
    except Exception as e:
        logger.warning(f"Failed to generate explanation: {e}")
        return {"error": "Explanation generation failed"}

# API Endpoints

@app.get("/api", response_model=HealthResponse)
async def root():
    """Root endpoint with health information."""
    from datetime import datetime
    
    return HealthResponse(
        status="healthy",
        message="Arabic Dialect Sentiment Analysis API is running",
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    from datetime import datetime
    
    health_status = "healthy" if model_loaded else "degraded"
    message = "API is healthy" if model_loaded else "API is running but model not loaded"
    
    return HealthResponse(
        status=health_status,
        message=message,
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )

@app.post("/api/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """
    Analyze sentiment for a single text.
    
    Args:
        request: Sentiment analysis request
        
    Returns:
        Sentiment analysis result
    """
    try:
        result = predict_sentiment(
            text=request.text,
            include_explanation=request.include_explanation
        )
        
        # Override dialect if provided
        if request.dialect:
            result["dialect"] = request.dialect
        
        return SentimentResponse(**result)
        
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/batch", response_model=BatchSentimentResponse)
async def analyze_sentiment_batch(request: BatchSentimentRequest):
    """
    Analyze sentiment for multiple texts.
    
    Args:
        request: Batch sentiment analysis request
        
    Returns:
        Batch sentiment analysis results
    """
    try:
        import time
        start_time = time.time()
        
        results = []
        for text in request.texts:
            result = predict_sentiment(
                text=text,
                include_explanation=request.include_explanations
            )
            
            # Override dialect if provided
            if request.dialect:
                result["dialect"] = request.dialect
            
            results.append(SentimentResponse(**result))
        
        total_time = time.time() - start_time
        avg_confidence = sum(r.confidence for r in results) / len(results)
        
        return BatchSentimentResponse(
            results=results,
            total_processing_time=total_time,
            average_confidence=avg_confidence
        )
        
    except Exception as e:
        logger.error(f"Batch sentiment analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/preprocess")
async def preprocess_text(text: str = Form(...)):
    """
    Preprocess Arabic text.
    
    Args:
        text: Raw Arabic text
        
    Returns:
        Preprocessed text
    """
    try:
        if not preprocessor:
            raise HTTPException(status_code=503, detail="Preprocessor not available")
        
        cleaned_text = preprocessor.clean_text(text)
        dialect = preprocessor.identify_dialect(text)
        
        return {
            "original_text": text,
            "cleaned_text": cleaned_text,
            "dialect": dialect,
            "text_length": len(cleaned_text)
        }
        
    except Exception as e:
        logger.error(f"Text preprocessing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file for batch processing.
    
    Args:
        file: CSV or text file
        
    Returns:
        Upload confirmation
    """
    try:
        # Check file type
        if not file.filename.endswith(('.csv', '.txt')):
            raise HTTPException(status_code=400, detail="Only CSV and TXT files are supported")
        
        # Read file content
        content = await file.read()
        
        if file.filename.endswith('.csv'):
            import pandas as pd
            from io import StringIO
            df = pd.read_csv(StringIO(content.decode('utf-8')))
            texts = df.iloc[:, 0].tolist()  # Assume first column contains text
        else:
            texts = content.decode('utf-8').split('\n')
            texts = [text.strip() for text in texts if text.strip()]
        
        return {
            "filename": file.filename,
            "texts_count": len(texts),
            "message": "File uploaded successfully"
        }
        
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models/available")
async def get_available_models():
    """Get information about available models."""
    models_info = {
        "sentiment_model": {
            "loaded": model_loaded,
            "type": "transformer",
            "architecture": "BERT-based"
        },
        "preprocessor": {
            "available": preprocessor is not None,
            "type": "Arabic text preprocessor"
        }
    }
    
    return models_info

@app.get("/api/dialects/supported")
async def get_supported_dialects():
    """Get list of supported Arabic dialects."""
    dialects = [
        {
            "code": "gulf",
            "name": "Gulf Arabic",
            "description": "Arabic dialects spoken in the Gulf region"
        },
        {
            "code": "egyptian",
            "name": "Egyptian Arabic",
            "description": "Arabic dialect spoken in Egypt"
        },
        {
            "code": "levantine",
            "name": "Levantine Arabic",
            "description": "Arabic dialects spoken in the Levant"
        },
        {
            "code": "msa",
            "name": "Modern Standard Arabic",
            "description": "Standard Arabic used in formal contexts"
        }
    ]
    
    return {"dialects": dialects}

# Startup and shutdown events

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("Starting Arabic Dialect Sentiment Analysis API...")
    
    # Load the sentiment model
    load_sentiment_model()
    
    logger.info("API startup completed")

# Serve frontend build if available
try:
    build_dir = Path(__file__).parent.parent / "arabic-sentiment-webapp" / "build"
    if build_dir.exists():
        app.mount("/", StaticFiles(directory=str(build_dir), html=True), name="frontend")
        logger.info(f"Mounted frontend build at '/': {build_dir}")
    else:
        logger.warning(f"Frontend build directory not found: {build_dir}")
except Exception as e:
    logger.warning(f"Failed to mount frontend build: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down Arabic Dialect Sentiment Analysis API...")

# Error handlers

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
