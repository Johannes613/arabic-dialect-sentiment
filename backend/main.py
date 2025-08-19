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
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
import logging
import json
import os
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
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

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
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

def load_sentiment_model():
    """Load the trained sentiment analysis model."""
    global sentiment_model, tokenizer, model_loaded
    
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        # Try to load the fine-tuned model first
        model_path = "models/fine_tuned"
        if Path(model_path).exists():
            logger.info("Loading fine-tuned model...")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            # Fall back to baseline model
            baseline_path = "models/baselines/transformer_baseline"
            if Path(baseline_path).exists():
                logger.info("Loading baseline transformer model...")
                tokenizer = AutoTokenizer.from_pretrained(baseline_path)
                sentiment_model = AutoModelForSequenceClassification.from_pretrained(baseline_path)
            else:
                # Use pre-trained model as last resort
                logger.info("Loading pre-trained AraBERT model...")
                model_name = "aubmindlab/bert-base-arabertv2"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, num_labels=3
                )
        
        model_loaded = True
        logger.info("Sentiment model loaded successfully")
        
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
        
        # Get prediction
        with torch.no_grad():
            outputs = sentiment_model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Map class to sentiment
        sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
        sentiment = sentiment_map.get(predicted_class, "unknown")
        
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

def generate_explanation(text: str, inputs: Dict, probabilities: torch.Tensor) -> Dict[str, Any]:
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
            attention_scores = attention_weights.mean(dim=1).squeeze()
            top_indices = torch.topk(attention_scores, k=min(10, len(attention_scores))).indices
            top_tokens = [tokenizer.decode([inputs['input_ids'][0][idx]]) for idx in top_indices]
        
        # Get class probabilities
        class_probs = {
            "negative": probabilities[0][0].item(),
            "neutral": probabilities[0][1].item(),
            "positive": probabilities[0][2].item()
        }
        
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

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health information."""
    from datetime import datetime
    
    return HealthResponse(
        status="healthy",
        message="Arabic Dialect Sentiment Analysis API is running",
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )

@app.get("/health", response_model=HealthResponse)
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

@app.post("/analyze", response_model=SentimentResponse)
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

@app.post("/analyze/batch", response_model=BatchSentimentResponse)
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

@app.post("/preprocess")
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

@app.post("/upload")
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

@app.get("/models/available")
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

@app.get("/dialects/supported")
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
