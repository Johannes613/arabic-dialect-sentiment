"""
Fine-tuning Script for MARBERT on Arabic Sentiment Analysis

This script implements fine-tuning of the UBC-NLP/MARBERT model on the ASTD dataset
for Arabic sentiment analysis using Hugging Face Transformers.
"""

import os
import sys
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings("ignore")

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    EarlyStoppingCallback
)
from datasets import Dataset as HFDataset

from data.astd_loader import ASTDDataLoader
from utils.config_loader import ConfigLoader
from utils.logging_utils import setup_logging, log_hyperparameters, log_metrics

logger = logging.getLogger(__name__)


class ASTDDataset(Dataset):
    """PyTorch Dataset for ASTD data."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class MARBERTFineTuner:
    """
    Fine-tuner for MARBERT model on Arabic sentiment analysis.
    """
    
    def __init__(self, config_path: str = "configs/fine_tune_config.yaml"):
        """
        Initialize the fine-tuner.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = ConfigLoader.load_config(config_path)
        self.setup_logging()
        self.setup_device()
        self.setup_model_and_tokenizer()
        self.setup_data_loader()
        
        logger.info("MARBERT Fine-tuner initialized successfully")
    
    def setup_logging(self):
        """Setup logging for the fine-tuning process."""
        log_dir = self.config.get('logging', {}).get('log_dir', 'logs')
        experiment_name = self.config.get('logging', {}).get('experiment_name', 'marbert_fine_tune')
        
        setup_logging(
            log_dir=log_dir,
            experiment_name=experiment_name,
            log_level=self.config.get('logging', {}).get('log_level', 'INFO')
        )
        
        # Log configuration
        log_hyperparameters(self.config)
    
    def setup_device(self):
        """Setup device (GPU/CPU) for training."""
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU")
    
    def setup_model_and_tokenizer(self):
        """Setup MARBERT model and tokenizer."""
        model_name = self.config.get('model', {}).get('base_model', 'UBC-NLP/MARBERT')
        num_labels = self.config.get('model', {}).get('num_labels', 4)
        
        logger.info(f"Loading model: {model_name}")
        logger.info(f"Number of labels: {num_labels}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                ignore_mismatched_sizes=True
            )
            
            # Move to device
            self.model.to(self.device)
            
            logger.info(f"Model loaded successfully: {self.model.config.model_type}")
            logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def setup_data_loader(self):
        """Setup ASTD data loader."""
        self.data_loader = ASTDDataLoader()
        
        # Validate dataset
        if not self.data_loader.validate_dataset():
            raise ValueError("Dataset validation failed")
        
        logger.info("ASTD data loader setup completed")
    
    def preprocess_data(self, split_name: str) -> Tuple[List[str], List[int]]:
        """
        Preprocess data for a specific split.
        
        Args:
            split_name: Name of the split to load
            
        Returns:
            Tuple of (texts, labels)
        """
        logger.info(f"Preprocessing data for split: {split_name}")
        
        # Load split data
        split_df = self.data_loader.get_split_data(split_name)
        
        # Extract texts and labels
        texts = split_df['text'].tolist()
        labels = split_df['label_id'].tolist()
        
        logger.info(f"Loaded {len(texts)} samples from {split_name}")
        logger.info(f"Label distribution: {self.data_loader.get_label_distribution(split_df)}")
        
        return texts, labels
    
    def create_datasets(self) -> Tuple[ASTDDataset, ASTDDataset, ASTDDataset]:
        """
        Create train, validation, and test datasets.
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        logger.info("Creating datasets...")
        
        # Load balanced splits
        train_texts, train_labels = self.preprocess_data('4class-balanced-train')
        val_texts, val_labels = self.preprocess_data('4class-balanced-validation')
        test_texts, test_labels = self.preprocess_data('4class-balanced-test')
        
        # Create datasets
        max_length = self.config.get('data', {}).get('tokenization', {}).get('max_length', 128)
        
        train_dataset = ASTDDataset(train_texts, train_labels, self.tokenizer, max_length)
        val_dataset = ASTDDataset(val_texts, val_labels, self.tokenizer, max_length)
        test_dataset = ASTDDataset(test_texts, test_labels, self.tokenizer, max_length)
        
        logger.info(f"Created datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            eval_pred: Evaluation predictions from Trainer
            
        Returns:
            Dictionary of metrics
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        macro_f1 = f1_score(labels, predictions, average='macro')
        weighted_f1 = f1_score(labels, predictions, average='weighted')
        
        # Per-class F1 scores
        per_class_f1 = f1_score(labels, predictions, average=None)
        
        metrics = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'f1_negative': per_class_f1[0],  # NEG
            'f1_positive': per_class_f1[1],  # POS
            'f1_neutral': per_class_f1[2],   # NEUTRAL
            'f1_objective': per_class_f1[3]  # OBJ
        }
        
        return metrics
    
    def train_with_trainer(self, train_dataset: ASTDDataset, val_dataset: ASTDDataset) -> Trainer:
        """
        Train the model using Hugging Face Trainer.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            
        Returns:
            Trained Trainer instance
        """
        logger.info("Starting training with Hugging Face Trainer...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.get('output', {}).get('model_dir', 'models/marbert_fine_tuned'),
            num_train_epochs=self.config.get('training', {}).get('epochs', 3),
            per_device_train_batch_size=self.config.get('training', {}).get('batch_size', 16),
            per_device_eval_batch_size=self.config.get('training', {}).get('batch_size', 16),
            warmup_steps=self.config.get('training', {}).get('warmup_steps', 500),
            weight_decay=self.config.get('training', {}).get('weight_decay', 0.01),
            logging_dir=self.config.get('logging', {}).get('log_dir', 'logs'),
            logging_steps=self.config.get('training', {}).get('logging_steps', 100),
            evaluation_strategy="steps",
            eval_steps=self.config.get('training', {}).get('eval_steps', 500),
            save_steps=self.config.get('training', {}).get('save_steps', 1000),
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            save_total_limit=3,
            dataloader_num_workers=4,
            fp16=torch.cuda.is_available(),
            report_to=["tensorboard"] if self.config.get('logging', {}).get('use_tensorboard', True) else None,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train the model
        logger.info("Training started...")
        trainer.train()
        
        logger.info("Training completed successfully!")
        return trainer
    
    def train_with_custom_loop(self, train_dataset: ASTDDataset, val_dataset: ASTDDataset):
        """
        Train the model using custom training loop.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
        """
        logger.info("Starting training with custom loop...")
        
        # Training parameters
        batch_size = self.config.get('training', {}).get('batch_size', 16)
        epochs = self.config.get('training', {}).get('epochs', 3)
        learning_rate = self.config.get('training', {}).get('learning_rate', 2e-5)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=self.config.get('training', {}).get('warmup_steps', 500),
            num_training_steps=total_steps
        )
        
        # Setup loss function with class weights
        train_df = self.data_loader.get_split_data('4class-balanced-train')
        class_weights = self.data_loader.get_class_weights(train_df)
        class_weights_tensor = torch.tensor([class_weights[i] for i in range(4)], device=self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        
        # Training loop
        best_val_f1 = 0.0
        training_losses = []
        validation_metrics = []
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            # Training phase
            self.model.train()
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    logger.info(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            avg_train_loss = epoch_loss / len(train_loader)
            training_losses.append(avg_train_loss)
            
            # Validation phase
            val_metrics = self.evaluate_model(val_loader)
            validation_metrics.append(val_metrics)
            
            logger.info(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val Macro-F1: {val_metrics['macro_f1']:.4f}")
            
            # Save best model
            if val_metrics['macro_f1'] > best_val_f1:
                best_val_f1 = val_metrics['macro_f1']
                self.save_model("best_model")
                logger.info(f"New best model saved with Macro-F1: {best_val_f1:.4f}")
        
        # Log final metrics
        log_metrics({
            'final_train_loss': training_losses[-1],
            'best_val_macro_f1': best_val_f1,
            'training_losses': training_losses,
            'validation_metrics': validation_metrics
        })
        
        logger.info("Custom training loop completed!")
    
    def evaluate_model(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.
        
        Args:
            data_loader: DataLoader for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        macro_f1 = f1_score(all_labels, all_predictions, average='macro')
        weighted_f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1
        }
    
    def save_model(self, model_name: str = "marbert_fine_tuned"):
        """Save the fine-tuned model."""
        output_dir = Path(self.config.get('output', {}).get('model_dir', 'models'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = output_dir / model_name
        
        # Save model and tokenizer
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        # Save configuration
        config_path = model_path / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
    
    def run_fine_tuning(self):
        """Run the complete fine-tuning pipeline."""
        logger.info("Starting MARBERT fine-tuning pipeline...")
        
        try:
            # Create datasets
            train_dataset, val_dataset, test_dataset = self.create_datasets()
            
            # Choose training method
            use_trainer = self.config.get('training', {}).get('use_trainer', True)
            
            if use_trainer:
                # Train with Hugging Face Trainer
                trainer = self.train_with_trainer(train_dataset, val_dataset)
                
                # Evaluate on test set
                test_results = trainer.evaluate(test_dataset)
                logger.info(f"Test set results: {test_results}")
                
                # Save model
                self.save_model()
                
            else:
                # Train with custom loop
                self.train_with_custom_loop(train_dataset, val_dataset)
                
                # Evaluate on test set
                test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
                test_metrics = self.evaluate_model(test_loader)
                logger.info(f"Test set results: {test_metrics}")
                
                # Save model
                self.save_model()
            
            logger.info("Fine-tuning pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Fine-tuning pipeline failed: {e}")
            raise


def main():
    """Main function to run fine-tuning."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune MARBERT for Arabic sentiment analysis")
    parser.add_argument("--config", type=str, default="configs/fine_tune_config.yaml", 
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    try:
        # Initialize fine-tuner
        fine_tuner = MARBERTFineTuner(args.config)
        
        # Run fine-tuning
        fine_tuner.run_fine_tuning()
        
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
