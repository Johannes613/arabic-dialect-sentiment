#!/usr/bin/env python3
"""
Baseline Model Training Script for Arabic Dialect Sentiment Analysis

This script trains and evaluates baseline models including:
1. Traditional Machine Learning models (Logistic Regression, Random Forest, SVM, Naive Bayes)
2. Transformer-based models (AraBERT, MARBERT, etc.)
3. Comprehensive evaluation and comparison
"""

import os
import sys
import argparse
import logging
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
import pickle
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.config_loader import ConfigLoader
from utils.logging_utils import (
    setup_experiment_logging, log_hyperparameters, log_metrics, 
    log_model_info, log_data_info, setup_tensorboard_logging
)

logger = logging.getLogger(__name__)


class TraditionalMLBaseline:
    """
    Traditional Machine Learning baseline models for Arabic sentiment analysis.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize traditional ML baseline trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.models = {}
        self.results = {}
        
    def prepare_features(self, data: pd.DataFrame, text_column: str = 'cleaned_text') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features using TF-IDF vectorization.
        
        Args:
            data: Input DataFrame
            text_column: Name of the text column
            
        Returns:
            Tuple of (features, labels)
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Extract text and labels
        texts = data[text_column].fillna('').astype(str)
        labels = data['label'].values
        
        # Configure TF-IDF vectorizer
        tfidf_config = self.config.get('traditional_ml', {}).get('feature_extraction', 'tfidf')
        max_features = self.config.get('traditional_ml', {}).get('max_features', 10000)
        ngram_range = tuple(self.config.get('traditional_ml', {}).get('ngram_range', [1, 3]))
        
        # Create and fit TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words=None,  # Arabic stop words would need custom implementation
            lowercase=False,   # Arabic is case-insensitive
            max_df=0.95,
            min_df=2
        )
        
        # Fit and transform
        features = vectorizer.fit_transform(texts)
        
        # Save vectorizer for later use
        self.vectorizer = vectorizer
        
        logger.info(f"Feature extraction completed. Feature matrix shape: {features.shape}")
        return features, labels
    
    def train_models(self, train_features: np.ndarray, train_labels: np.ndarray,
                     val_features: np.ndarray, val_labels: np.ndarray) -> Dict[str, Any]:
        """
        Train traditional ML models.
        
        Args:
            train_features: Training features
            train_labels: Training labels
            val_features: Validation features
            val_labels: Validation labels
            
        Returns:
            Dictionary containing trained models and results
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import classification_report, confusion_matrix
        
        # Get model configurations
        ml_config = self.config.get('traditional_ml', {})
        models_to_train = ml_config.get('models', ['logistic_regression', 'random_forest', 'svm', 'naive_bayes'])
        cv_folds = ml_config.get('cv_folds', 5)
        cv_scoring = ml_config.get('cv_scoring', 'f1_macro')
        
        # Define models
        model_definitions = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': ml_config.get('param_grids', {}).get('logistic_regression', {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2']
                })
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                'params': ml_config.get('param_grids', {}).get('random_forest', {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None]
                })
            },
            'svm': {
                'model': SVC(random_state=42, probability=True),
                'params': ml_config.get('param_grids', {}).get('svm', {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['linear', 'rbf']
                })
            },
            'naive_bayes': {
                'model': MultinomialNB(),
                'params': ml_config.get('param_grids', {}).get('naive_bayes', {
                    'alpha': [0.1, 1.0, 10.0]
                })
            }
        }
        
        results = {}
        
        for model_name in models_to_train:
            if model_name not in model_definitions:
                logger.warning(f"Model {model_name} not defined, skipping...")
                continue
            
            logger.info(f"Training {model_name}...")
            
            try:
                # Get model and parameters
                model_def = model_definitions[model_name]
                model = model_def['model']
                params = model_def['params']
                
                # Perform grid search if parameters are specified
                if params and ml_config.get('grid_search', True):
                    grid_search = GridSearchCV(
                        model, params, cv=cv_folds, scoring=cv_scoring,
                        n_jobs=-1, verbose=1
                    )
                    grid_search.fit(train_features, train_labels)
                    
                    # Get best model
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                    best_score = grid_search.best_score_
                    
                    logger.info(f"Best parameters for {model_name}: {best_params}")
                    logger.info(f"Best CV score: {best_score:.4f}")
                    
                else:
                    # Train without grid search
                    best_model = model
                    best_model.fit(train_features, train_labels)
                    best_params = {}
                    best_score = None
                
                # Evaluate on validation set
                val_predictions = best_model.predict(val_features)
                val_report = classification_report(val_labels, val_predictions, output_dict=True)
                
                # Store results
                results[model_name] = {
                    'model': best_model,
                    'best_params': best_params,
                    'best_cv_score': best_score,
                    'val_accuracy': val_report['accuracy'],
                    'val_f1_macro': val_report['macro avg']['f1-score'],
                    'val_precision_macro': val_report['macro avg']['precision'],
                    'val_recall_macro': val_report['macro avg']['recall'],
                    'val_predictions': val_predictions,
                    'val_report': val_report
                }
                
                # Store model
                self.models[model_name] = best_model
                
                logger.info(f"{model_name} training completed. Validation F1-Macro: {val_report['macro avg']['f1-score']:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        return results
    
    def save_models(self, output_dir: str):
        """
        Save trained models to disk.
        
        Args:
            output_dir: Directory to save models
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            model_path = output_dir / f"{model_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved {model_name} to {model_path}")
        
        # Save vectorizer
        if hasattr(self, 'vectorizer'):
            vectorizer_path = output_dir / "tfidf_vectorizer.pkl"
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            logger.info(f"Saved TF-IDF vectorizer to {vectorizer_path}")


class TransformerBaseline:
    """
    Transformer-based baseline models for Arabic sentiment analysis.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize transformer baseline trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.results = {}
        
    def load_model_and_tokenizer(self):
        """
        Load pre-trained model and tokenizer.
        """
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        model_name = self.config.get('transformer_baseline', {}).get('model_name', 'aubmindlab/bert-base-arabertv2')
        
        logger.info(f"Loading model and tokenizer: {model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model
            num_labels = self.config.get('model', {}).get('num_labels', 3)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                ignore_mismatched_sizes=True
            )
            
            logger.info(f"Model and tokenizer loaded successfully")
            logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
        except Exception as e:
            logger.error(f"Error loading model and tokenizer: {e}")
            raise
    
    def prepare_dataset(self, data: pd.DataFrame, text_column: str = 'cleaned_text') -> 'Dataset':
        """
        Prepare dataset for transformer training.
        
        Args:
            data: Input DataFrame
            text_column: Name of the text column
            
        Returns:
            HuggingFace Dataset object
        """
        from datasets import Dataset
        
        # Prepare texts and labels
        texts = data[text_column].fillna('').astype(str).tolist()
        labels = data['label'].values.tolist()
        
        # Create dataset
        dataset_dict = {
            'text': texts,
            'label': labels
        }
        
        # Add dialect if available
        if 'dialect' in data.columns:
            dataset_dict['dialect'] = data['dialect'].fillna('unknown').tolist()
        
        dataset = Dataset.from_dict(dataset_dict)
        
        logger.info(f"Dataset prepared with {len(dataset)} samples")
        return dataset
    
    def tokenize_function(self, examples):
        """
        Tokenize examples for the model.
        
        Args:
            examples: Batch of examples
            
        Returns:
            Tokenized examples
        """
        max_length = self.config.get('data', {}).get('max_length', 512)
        
        return self.tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
    
    def train_model(self, train_dataset: 'Dataset', val_dataset: 'Dataset') -> Dict[str, Any]:
        """
        Train the transformer model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            
        Returns:
            Training results
        """
        from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        # Tokenize datasets
        train_dataset = train_dataset.map(self.tokenize_function, batched=True)
        val_dataset = val_dataset.map(self.tokenize_function, batched=True)
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Training arguments
        training_config = self.config.get('training', {})
        training_args = TrainingArguments(
            output_dir=self.config.get('output', {}).get('model_save_path', 'models/baselines'),
            num_train_epochs=training_config.get('num_epochs', 5),
            per_device_train_batch_size=training_config.get('batch_size', 16),
            per_device_eval_batch_size=training_config.get('batch_size', 16),
            warmup_steps=training_config.get('warmup_steps', 100),
            weight_decay=training_config.get('weight_decay', 0.01),
            logging_dir=self.config.get('output', {}).get('log_dir', 'logs/baselines'),
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1_macro",
            greater_is_better=True,
            save_total_limit=3,
            report_to="tensorboard" if self.config.get('output', {}).get('use_tensorboard', True) else None
        )
        
        # Define compute_metrics function
        def compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
            acc = accuracy_score(labels, preds)
            return {
                'accuracy': acc,
                'f1_macro': f1,
                'precision_macro': precision,
                'recall_macro': recall
            }
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )
        
        # Train model
        logger.info("Starting transformer model training...")
        train_result = trainer.train()
        
        # Evaluate model
        logger.info("Evaluating transformer model...")
        eval_result = trainer.evaluate()
        
        # Store results
        self.results = {
            'train_loss': train_result.training_loss,
            'eval_metrics': eval_result,
            'trainer': trainer
        }
        
        logger.info(f"Training completed. Final evaluation metrics: {eval_result}")
        
        return self.results
    
    def save_model(self, output_dir: str):
        """
        Save the trained transformer model.
        
        Args:
            output_dir: Directory to save the model
        """
        if self.model is None:
            logger.warning("No model to save")
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        model_path = output_dir / "transformer_baseline"
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        logger.info(f"Transformer model saved to {model_path}")


class BaselineTrainer:
    """
    Main class for training baseline models.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the baseline trainer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = ConfigLoader.load_config(config_path)
        self.setup_logging()
        
        # Initialize baseline trainers
        self.traditional_ml = TraditionalMLBaseline(self.config)
        self.transformer = TransformerBaseline(self.config)
        
        # Create output directories
        self.create_output_directories()
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        setup_experiment_logging(
            experiment_name="baseline_training",
            log_dir=log_config.get('log_dir', 'logs/baselines'),
            config=self.config
        )
    
    def create_output_directories(self):
        """Create necessary output directories."""
        output_config = self.config.get('output', {})
        
        # Create model save directory
        model_save_path = Path(output_config.get('model_save_path', 'models/baselines'))
        model_save_path.mkdir(parents=True, exist_ok=True)
        
        # Create results directory
        results_dir = Path(output_config.get('results_save_path', 'results')).parent
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log directory
        log_dir = Path(output_config.get('log_dir', 'logs/baselines'))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Output directories created")
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load preprocessed data.
        
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        data_config = self.config.get('data', {})
        
        # Load train data
        train_path = data_config.get('train_data_path', 'data/processed/train_data.csv')
        train_data = pd.read_csv(train_path)
        
        # Load validation data
        val_path = data_config.get('val_data_path', 'data/processed/val_data.csv')
        val_data = pd.read_csv(val_path)
        
        # Load test data
        test_path = data_config.get('test_data_path', 'data/processed/test_data.csv')
        test_data = pd.read_csv(test_path)
        
        logger.info(f"Data loaded - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        return train_data, val_data, test_data
    
    def train_traditional_ml(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train traditional ML models.
        
        Args:
            train_data: Training data
            val_data: Validation data
            
        Returns:
            Training results
        """
        if not self.config.get('models', {}).get('traditional_ml', {}).get('enabled', True):
            logger.info("Traditional ML training disabled")
            return {}
        
        logger.info("Starting traditional ML baseline training...")
        
        # Prepare features
        train_features, train_labels = self.traditional_ml.prepare_features(train_data)
        val_features, val_labels = self.traditional_ml.prepare_features(val_data)
        
        # Train models
        results = self.traditional_ml.train_models(train_features, train_labels, val_features, val_labels)
        
        # Save models
        output_dir = self.config.get('output', {}).get('model_save_path', 'models/baselines')
        self.traditional_ml.save_models(output_dir)
        
        return results
    
    def train_transformer(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train transformer model.
        
        Args:
            train_data: Training data
            val_data: Validation data
            
        Returns:
            Training results
        """
        if not self.config.get('models', {}).get('transformer_baseline', {}).get('enabled', True):
            logger.info("Transformer baseline training disabled")
            return {}
        
        logger.info("Starting transformer baseline training...")
        
        # Load model and tokenizer
        self.transformer.load_model_and_tokenizer()
        
        # Prepare datasets
        train_dataset = self.transformer.prepare_dataset(train_data)
        val_dataset = self.transformer.prepare_dataset(val_data)
        
        # Train model
        results = self.transformer.train_model(train_dataset, val_dataset)
        
        # Save model
        output_dir = self.config.get('output', {}).get('model_save_path', 'models/baselines')
        self.transformer.save_model(output_dir)
        
        return results
    
    def save_results(self, traditional_ml_results: Dict, transformer_results: Dict):
        """
        Save all training results.
        
        Args:
            traditional_ml_results: Traditional ML training results
            transformer_results: Transformer training results
        """
        # Combine results
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'traditional_ml': traditional_ml_results,
            'transformer': transformer_results,
            'config': self.config
        }
        
        # Save to JSON
        results_path = self.config.get('output', {}).get('results_save_path', 'results/baseline_results.json')
        results_path = Path(results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Results saved to {results_path}")
        
        # Save to CSV for easy analysis
        csv_results = []
        
        # Traditional ML results
        for model_name, result in traditional_ml_results.items():
            csv_results.append({
                'model_type': 'traditional_ml',
                'model_name': model_name,
                'val_accuracy': result.get('val_accuracy', 0),
                'val_f1_macro': result.get('val_f1_macro', 0),
                'val_precision_macro': result.get('val_precision_macro', 0),
                'val_recall_macro': result.get('val_recall_macro', 0)
            })
        
        # Transformer results
        if transformer_results:
            csv_results.append({
                'model_type': 'transformer',
                'model_name': 'transformer_baseline',
                'val_accuracy': transformer_results.get('eval_metrics', {}).get('eval_accuracy', 0),
                'val_f1_macro': transformer_results.get('eval_metrics', {}).get('eval_f1_macro', 0),
                'val_precision_macro': transformer_results.get('eval_metrics', {}).get('eval_precision_macro', 0),
                'val_recall_macro': transformer_results.get('eval_metrics', {}).get('eval_recall_macro', 0)
            })
        
        # Save CSV
        csv_path = self.config.get('output', {}).get('results_save_path', 'results/baseline_results.csv')
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        results_df = pd.DataFrame(csv_results)
        results_df.to_csv(csv_path, index=False)
        
        logger.info(f"CSV results saved to {csv_path}")
    
    def run_training(self):
        """
        Run the complete baseline training pipeline.
        """
        try:
            logger.info("Starting baseline model training pipeline")
            
            # Load data
            train_data, val_data, test_data = self.load_data()
            
            # Log data information
            log_data_info(
                logger=logger,
                train_samples=len(train_data),
                val_samples=len(val_data),
                test_samples=len(test_data),
                num_classes=len(train_data['label'].unique()),
                class_distribution=train_data['label'].value_counts().to_dict()
            )
            
            # Train traditional ML models
            traditional_ml_results = self.train_traditional_ml(train_data, val_data)
            
            # Train transformer model
            transformer_results = self.train_transformer(train_data, val_data)
            
            # Save results
            self.save_results(traditional_ml_results, transformer_results)
            
            logger.info("Baseline training pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Baseline training pipeline failed: {e}")
            raise


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Arabic Dialect Sentiment Analysis - Baseline Model Training'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/baseline_config.yaml',
        help='Path to configuration file (default: configs/baseline_config.yaml)'
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not Path(args.config).exists():
        print(f"Configuration file not found: {args.config}")
        print("Please ensure the config file exists or specify a different path with --config")
        sys.exit(1)
    
    try:
        # Initialize and run training
        trainer = BaselineTrainer(args.config)
        trainer.run_training()
        
    except Exception as e:
        logger.error(f"Training execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
