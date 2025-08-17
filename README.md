<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Arabic Dialect Sentiment Analysis - Enhanced MARBERT Model</title>
</head>
<body style="font-family: Arial, sans-serif; line-height: 1.6; max-width: 1000px; margin: auto; padding: 20px;">

  <h1>Arabic Dialect Sentiment Analysis: Domain-Adapted Transformer for Gulf Arabic</h1>

  <!-- Tech Badges -->
  <div style="margin-bottom: 20px;">
    <img src="https://img.shields.io/badge/python-%233776AB.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python Badge">
    <img src="https://img.shields.io/badge/pytorch-%23EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch Badge">
    <img src="https://img.shields.io/badge/transformers-%23F7931E.svg?style=for-the-badge&logo=huggingface&logoColor=white" alt="Transformers Badge">
    <img src="https://img.shields.io/badge/fastapi-%23000000.svg?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI Badge">
    <img src="https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB" alt="React Badge">
    <img src="https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white" alt="Docker Badge">
    <img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn Badge">
  </div>

  <p><strong>Arabic Dialect Sentiment Analysis</strong> is a comprehensive AI research project focused on developing advanced sentiment analysis capabilities for Gulf Arabic dialects. This project implements a Domain-Adapted Transformer (MARBERT) that achieves outstanding performance on Arabic sentiment classification, specifically addressing the challenges of low-resource language variants.</p>

  <p>
    �� Live Demo:  
    <a href="#" target="_blank">
      Web Application (Coming Soon)
    </a>
  </p>

  <hr/>

  <h2>Project Overview</h2>
  <p>This project addresses the critical challenge of sentiment analysis in Arabic dialects, particularly Gulf Arabic, which has been historically underrepresented in NLP research. The solution combines advanced transformer architecture with domain-adaptive pretraining and sophisticated data augmentation techniques.</p>

  <h3>Key Achievements</h3>
  <ul>
    <li><strong>88% Overall Accuracy</strong> on the ASTD dataset</li>
    <li><strong>86% Macro F1 Score</strong> - exceptional balanced performance</li>
    <li><strong>All classes above 83% F1</strong> - remarkable consistency</li>
    <li><strong>29% improvement</strong> in Macro F1 from baseline models</li>
  </ul>

  <hr/>

  <h2>The Problem</h2>
  <ul>
    <li><strong>Low-Resource Language:</strong> Arabic dialects lack sufficient labeled data for sentiment analysis</li>
    <li><strong>Dialectal Variation:</strong> Gulf Arabic differs significantly from Modern Standard Arabic</li>
    <li><strong>Class Imbalance:</strong> Traditional models perform poorly on minority sentiment classes</li>
    <li><strong>Limited Research:</strong> Few specialized models for Arabic dialect sentiment analysis</li>
    <li><strong>Performance Gap:</strong> Existing models show bias toward majority classes</li>
  </ul>

  <p><strong>Our Enhanced MARBERT Model</strong> addresses these challenges through innovative class balancing, data augmentation, and domain-adaptive training techniques.</p>

  <hr/>

  <h2>Core Features</h2>

  <h3>1. Enhanced MARBERT Architecture</h3>
  <ul>
    <li><strong>Base Model:</strong> UBC-NLP/MARBERT pre-trained on Arabic text</li>
    <li><strong>Classification Head:</strong> 4-class sentiment classification (NEG, POS, NEUTRAL, OBJ)</li>
    <li><strong>Optimized Training:</strong> Custom hyperparameters for Arabic dialect processing</li>
    <li><strong>Device Optimization:</strong> CUDA support with mixed precision training</li>
  </ul>

  <h3>2. Advanced Data Processing</h3>
  <ul>
    <li><strong>Class Balancing:</strong> Undersampling majority classes, oversampling minority classes</li>
    <li><strong>Arabic Text Augmentation:</strong> Synonym replacement, diacritic variations, character-level augmentation</li>
    <li><strong>Text Preprocessing:</strong> URL removal, mention/hashtag cleaning, Arabic character preservation</li>
    <li><strong>Smart Filtering:</strong> Length-based filtering and quality assessment</li>
  </ul>

  <h3>3. Performance Optimization</h3>
  <ul>
    <li><strong>Weighted Loss Functions:</strong> Custom loss computation for class imbalance</li>
    <li><strong>Hyperparameter Tuning:</strong> Optimized learning rates, batch sizes, and training epochs</li>
    <li><strong>Gradient Accumulation:</strong> Effective batch size optimization</li>
    <li><strong>Early Stopping:</strong> Prevents overfitting with validation monitoring</li>
  </ul>

  <hr/>

  <h2>Model Performance & Results</h2>

  <h3>Performance Metrics</h3>
  <table border="1" cellspacing="0" cellpadding="8" style="width: 100%; border-collapse: collapse;">
    <tr style="background-color: #f2f2f2;"><th>Metric</th><th>Original</th><th>Enhanced</th><th>Improvement</th></tr>
    <tr><td><strong>Overall Accuracy</strong></td><td>73%</td><td>88%</td><td>+15%</td></tr>
    <tr><td><strong>Macro F1</strong></td><td>57%</td><td>86%</td><td>+29%</td></tr>
    <tr><td><strong>NEG F1</strong></td><td>57%</td><td>87%</td><td>+30%</td></tr>
    <tr><td><strong>POS F1</strong></td><td>52%</td><td>85%</td><td>+33%</td></tr>
    <tr><td><strong>NEUTRAL F1</strong></td><td>35%</td><td>83%</td><td>+48%</td></tr>
    <tr><td><strong>OBJ F1</strong></td><td>83%</td><td>90%</td><td>+7%</td></tr>
  </table>

  <h3>Class Distribution Analysis</h3>
  <ul>
    <li><strong>Original Dataset:</strong> Highly imbalanced (OBJ: 982, NEG: 251, POS: 117, NEUTRAL: 123)</li>
    <li><strong>Balanced Dataset:</strong> 500 samples per class for fair training</li>
    <li><strong>Augmentation Strategy:</strong> 3x augmentation for minority classes</li>
    <li><strong>Validation Split:</strong> 80/20 train-validation split with stratification</li>
  </ul>

  <hr/>

  <h2>Tech Stack</h2>
  <table border="1" cellspacing="0" cellpadding="8" style="width: 100%; border-collapse: collapse;">
    <tr style="background-color: #f2f2f2;"><th>Component</th><th>Technologies Used</th></tr>
    <tr><td><strong>Core ML Framework</strong></td><td>PyTorch, Transformers, Scikit-learn</td></tr>
    <tr><td><strong>Language Model</strong></td><td>UBC-NLP/MARBERT, AutoModelForSequenceClassification</td></tr>
    <tr><td><strong>Data Processing</strong></td><td>Pandas, NumPy, Arabic text preprocessing</td></tr>
    <tr><td><strong>Training & Evaluation</strong></td><td>Hugging Face Trainer, Custom metrics</td></tr>
    <tr><td><strong>Web Application</strong></td><td>FastAPI (Backend), React (Frontend)</td></tr>
    <tr><td><td>Docker, Docker Compose</td></tr>
    <tr><td><strong>Development</strong></td><td>Google Colab, Jupyter Notebooks</td></tr>
  </table>

  <hr/>

  <h2>Project Structure</h2>

  <h3>Repository Organization</h3>
  <ul>
    <li><strong>src/</strong> - Core Python modules for data processing, model training, and evaluation</li>
    <li><strong>models/</strong> - Trained model files and configurations</li>
    <li><strong>data/</strong> - Dataset files and preprocessing scripts</li>
    <li><strong>webapp/</strong> - FastAPI backend and React frontend</li>
    <li><strong>notebooks/</strong> - Jupyter notebooks for training and experimentation</li>
    <li><strong>scripts/</strong> - Utility scripts for setup and testing</li>
    <li><strong>configs/</strong> - YAML configuration files for different components</li>
  </ul>

  <h3>Key Components</h3>
  <ul>
    <li><strong>ASTD Data Loader:</strong> Specialized loader for Arabic Sentiment Tweets Dataset</li>
    <li><strong>Enhanced Preprocessor:</strong> Arabic-specific text cleaning and augmentation</li>
    <li><strong>MARBERT Fine-tuner:</strong> Custom training pipeline with weighted loss</li>
    <li><strong>Web Interface:</strong> User-friendly sentiment analysis application</li>
  </ul>

  <hr/>

  <h2>Getting Started</h2>

  <h3>Prerequisites</h3>
  <ul>
    <li>Python 3.8+</li>
    <li>PyTorch 1.9+</li>
    <li>Transformers 4.20+</li>
    <li>CUDA-compatible GPU (recommended)</li>
    <li>Google Colab account (for training)</li>
  </ul>

  <h3>Quick Start</h3>

  <ol>
    <li>
      <strong>Clone the repository:</strong>
      <pre><code style="background-color: #f2f2f2; padding: 5px; border-radius: 4px; display: block;">git clone https://github.com/Johannes613/arabic-dialect-sentiment.git
cd arabic-dialect-sentiment</code></pre>
    </li>
    <li>
      <strong>Install dependencies:</strong>
      <pre><code style="background-color: #f2f2f2; padding: 5px; border-radius: 4px; display: block;">pip install -r requirements.txt</code></pre>
    </li>
    <li>
      <strong>Download the trained model:</strong>
      <pre><code style="background-color: #f2f2f2; padding: 5px; border-radius: 4px; display: block;"># Model files are in models/ directory
# Load with: AutoModelForSequenceClassification.from_pretrained("./models")</code></pre>
    </li>
  </ol>

  <h3>Training Your Own Model</h3>

  <ol>
    <li>
      <strong>Upload the training notebook to Google Colab:</strong>
      <pre><code style="background-color: #f2f2f2; padding: 5px; border-radius: 4px; display: block;"># Use notebooks/marbert_finetuning_enhanced.ipynb
# Follow the step-by-step training process</code></pre>
    </li>
    <li>
      <strong>Prepare your dataset:</strong>
      <pre><code style="background-color: #f2f2f2; padding: 5px; border-radius: 4px; display: block;"># Place ASTD dataset in data/raw/
# Run preprocessing scripts for class balancing</code></pre>
    </li>
    <li>
      <strong>Start training:</strong>
      <pre><code style="background-color: #f2f2f2; padding: 5px; border-radius: 4px; display: block;"># Execute training cells in Colab
# Monitor performance metrics</code></pre>
    </li>
  </ol>

  <hr/>

  <h2>API Endpoints</h2>

  <h3>Sentiment Analysis</h3>
  <ul>
    <li><strong>POST /analyze</strong> - Single text sentiment analysis</li>
    <li><strong>POST /analyze/batch</strong> - Batch sentiment analysis</li>
    <li><strong>POST /preprocess</strong> - Arabic text preprocessing</li>
    <li><strong>GET /health</strong> - API health check</li>
  </ul>

  <h3>Model Management</h3>
  <ul>
    <li><strong>GET /model/info</strong> - Model performance metrics</li>
    <li><strong>POST /model/retrain</strong> - Trigger model retraining</li>
  </ul>

  <hr/>

  <h2>Advanced Features</h2>

  <h3>Data Augmentation Techniques</h3>
  <ul>
    <li><strong>Synonym Replacement:</strong> Arabic word synonyms for minority classes</li>
    <li><strong>Diacritic Variations:</strong> Character-level augmentation (ا → أ, إ, آ)</li>
    <li><strong>Smart Augmentation:</strong> Class-specific augmentation strategies</li>
    <li><strong>Quality Control:</strong> Augmentation validation and filtering</li>
  </ul>

  <h3>Training Optimizations</h3>
  <ul>
    <li><strong>Learning Rate Scheduling:</strong> Cosine annealing with warmup</li>
    <li><strong>Gradient Accumulation:</strong> Effective batch size optimization</li>
    <li><strong>Mixed Precision:</strong> FP16 training for faster convergence</li>
    <li><strong>Early Stopping:</strong> Validation-based training termination</li>
  </ul>

  <hr/>

  <h2>Future Enhancements</h2>

  <ul>
    <li><strong>Multi-Dialect Support:</strong> Extend to other Arabic dialects (Egyptian, Levantine)</li>
    <li><strong>Domain Adaptation:</strong> Specialized models for social media, news, reviews</li>
    <li><strong>Real-time Processing:</strong> Streaming sentiment analysis capabilities</li>
    <li><strong>Ensemble Methods:</strong> Combine multiple model architectures</li>
    <li><strong>Active Learning:</strong> Interactive model improvement with user feedback</li>
    <li><strong>Mobile Deployment:</strong> Optimized models for mobile applications</li>
  </ul>

  <hr/>

  <h2>Research Contributions</h2>

  <p>This project contributes to the field of Arabic NLP by:</p>
  <ul>
    <li><strong>Addressing Class Imbalance:</strong> Novel approaches for Arabic sentiment analysis</li>
    <li><strong>Dialectal Adaptation:</strong> Specialized models for Gulf Arabic</li>
    <li><strong>Performance Benchmarking:</strong> New baseline for Arabic sentiment analysis</li>
    <li><strong>Open Source Release:</strong> Complete training pipeline and models</li>
  </ul>

  <hr/>

  <h2>Contributing</h2>

  <p>We welcome contributions to improve Arabic dialect sentiment analysis! Areas for contribution include:</p>
  <ul>
    <li>Additional Arabic dialect support</li>
    <li>Enhanced data augmentation techniques</li>
    <li>Model architecture improvements</li>
    <li>Performance optimization</li>
    <li>Documentation and tutorials</li>
  </ul>

  <hr/>

  <h2>Citation</h2>

  <p>If you use this work in your research, please cite:</p>
  <pre style="background-color: #f2f2f2; padding: 15px; border-radius: 4px; font-size: 12px;">
@misc{arabic_dialect_sentiment_2024,
  title={Enhanced MARBERT for Arabic Dialect Sentiment Analysis},
  author={Your Name},
  year={2024},
  url={https://github.com/Johannes613/arabic-dialect-sentiment}
}
  </pre>

  <hr/>

  <p style="text-align: center; color: #666; font-size: 14px;">
    <strong>Arabic Dialect Sentiment Analysis</strong> - Advancing Arabic NLP Through Domain-Adapted Transformers
  </p>

</body>
</html>