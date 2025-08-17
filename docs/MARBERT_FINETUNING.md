# MARBERT Fine-Tuning for Arabic Sentiment Analysis

This document provides a complete guide for fine-tuning MARBERT on the ASTD dataset for Arabic sentiment analysis.

## Overview

The fine-tuning process involves:
1. **Data Preparation**: Loading and preprocessing the ASTD dataset
2. **Model Setup**: Loading MARBERT model and tokenizer
3. **Training**: Fine-tuning using Hugging Face Trainer
4. **Evaluation**: Assessing performance with Macro-F1 metric
5. **Model Saving**: Storing the fine-tuned model for inference

## Quick Start

### Option 1: Jupyter Notebook (Recommended for Google Colab)

1. **Open the notebook**: `notebooks/marbert_finetuning.ipynb`
2. **Upload to Google Colab** or run locally with Jupyter
3. **Install dependencies**: Run the first cell to install required packages
4. **Execute cells sequentially**: Follow the notebook step-by-step

### Option 2: Standalone Python Script

1. **Install dependencies**:
   ```bash
   pip install -r requirements_finetuning.txt
   ```

2. **Run the script**:
   ```bash
   python scripts/run_marbert_training.py
   ```

### Option 3: Import as Module

```python
from src.models.fine_tune_marbert import MARBERTFineTuner

# Initialize with configuration
config = {
    "model": {"name": "UBC-NLP/MARBERT", "max_length": 128, "num_labels": 4},
    "training": {"learning_rate": 2e-5, "batch_size": 16, "epochs": 3},
    "data": {"data_dir": "data", "output_dir": "models/marbert_sentiment"}
}

fine_tuner = MARBERTFineTuner(config)
fine_tuner.run_fine_tuning()
```

## Dataset Structure

The ASTD dataset should be organized as follows:

```
data/
├── raw/
│   └── Tweets.txt          # Main tweets file with labels
├── processed/
│   ├── splits/             # Processed dataset splits
│   └── preprocessed/       # Preprocessed text data
└── external/               # Additional external data
```

## Configuration

### Model Configuration

```yaml
model:
  name: "UBC-NLP/MARBERT"    # Base model
  max_length: 128            # Maximum sequence length
  num_labels: 4              # Number of sentiment classes
```

### Training Configuration

```yaml
training:
  learning_rate: 2e-5        # Learning rate
  batch_size: 16             # Batch size per device
  epochs: 3                  # Number of training epochs
  warmup_steps: 500          # Warmup steps for learning rate
  weight_decay: 0.01         # Weight decay
  evaluation_strategy: "epoch"  # When to evaluate
  save_strategy: "epoch"     # When to save checkpoints
  metric_for_best_model: "macro_f1"  # Metric for best model selection
```

### Data Configuration

```yaml
data:
  data_dir: "data"           # Data directory
  output_dir: "models/marbert_sentiment"  # Model output directory
  test_size: 0.2             # Test set size
  val_size: 0.2              # Validation set size
  random_state: 42           # Random seed for reproducibility
```

## Training Process

### 1. Data Loading and Preprocessing

- **ASTD Dataset**: Loads tweets and labels from `Tweets.txt`
- **Text Cleaning**: Removes URLs, mentions, hashtags, and special characters
- **Arabic Normalization**: Handles Arabic-specific text normalization
- **Data Splitting**: Creates train/validation/test splits with stratification

### 2. Model Setup

- **Tokenizer**: Loads MARBERT tokenizer
- **Model**: Loads MARBERT with classification head
- **Device**: Automatically detects GPU/CPU availability

### 3. Training

- **Hugging Face Trainer**: Uses optimized training loop
- **Metrics**: Tracks accuracy and Macro-F1
- **Checkpoints**: Saves best model based on validation performance
- **Early Stopping**: Prevents overfitting

### 4. Evaluation

- **Test Set Performance**: Evaluates on held-out test set
- **Confusion Matrix**: Visualizes classification performance
- **Detailed Metrics**: Precision, recall, F1 for each class

## Expected Output

After successful training, you'll have:

```
models/
└── marbert_sentiment/
    ├── config.json          # Model configuration
    ├── pytorch_model.bin    # Model weights
    ├── tokenizer.json       # Tokenizer configuration
    ├── tokenizer_config.json # Tokenizer settings
    └── special_tokens_map.json # Special tokens mapping
```

## Performance Metrics

The model is evaluated using:

- **Macro-F1**: Primary metric for handling class imbalance
- **Accuracy**: Overall classification accuracy
- **Per-class Metrics**: Precision, recall, F1 for each sentiment class

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or sequence length
2. **Slow Training**: Enable mixed precision or use smaller model
3. **Poor Performance**: Check data quality and try different hyperparameters

### Performance Tips

1. **Use GPU**: Training is significantly faster on GPU
2. **Mixed Precision**: Enable for faster training and lower memory usage
3. **Gradient Accumulation**: Use for larger effective batch sizes
4. **Data Parallel**: Scale to multiple GPUs if available

## Customization

### Hyperparameter Tuning

```python
# Example: Grid search over learning rates
learning_rates = [1e-5, 2e-5, 5e-5, 1e-4]
for lr in learning_rates:
    config["training"]["learning_rate"] = lr
    fine_tuner = MARBERTFineTuner(config)
    results = fine_tuner.run_fine_tuning()
    print(f"LR: {lr}, F1: {results['eval_macro_f1']}")
```

### Custom Training Loop

```python
# Use custom training instead of Trainer
fine_tuner.train_with_custom_loop()
```

### Data Augmentation

```python
# Add data augmentation in preprocessing
def augment_text(text):
    # Implement your augmentation logic
    return augmented_text
```

## Integration

### Load Fine-tuned Model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load saved model
model = AutoModelForSequenceClassification.from_pretrained("models/marbert_sentiment")
tokenizer = AutoTokenizer.from_pretrained("models/marbert_sentiment")

# Use for inference
inputs = tokenizer("أنا سعيد جداً", return_tensors="pt")
outputs = model(**inputs)
predictions = torch.softmax(outputs.logits, dim=-1)
```

### Web Application

The fine-tuned model can be integrated into the web application:

```python
# In webapp/backend/main.py
model = AutoModelForSequenceClassification.from_pretrained("models/marbert_sentiment")
```

## Next Steps

1. **Hyperparameter Optimization**: Use Optuna or similar tools
2. **Data Augmentation**: Implement Arabic-specific augmentation
3. **Ensemble Methods**: Combine multiple fine-tuned models
4. **Domain Adaptation**: Continue training on domain-specific data
5. **Production Deployment**: Optimize for inference speed

## References

- [MARBERT Paper](https://arxiv.org/abs/2103.06678)
- [ASTD Dataset](https://github.com/komari6/Arabic-Sentiment-Analysis-Dataset)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Arabic NLP Resources](https://github.com/arabic-nlp/arabic-nlp-resources)
