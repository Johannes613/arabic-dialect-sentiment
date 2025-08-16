#!/usr/bin/env python3
"""
Main Data Preprocessing Script for Arabic Dialect Sentiment Analysis

This script orchestrates the entire data preprocessing pipeline including:
1. Loading and validating raw data
2. Text cleaning and normalization
3. Dataset splitting
4. Tokenization
5. Saving processed datasets
"""

import os
import sys
import argparse
import logging
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import json

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.preprocessor import ArabicTextPreprocessor
from utils.config_loader import ConfigLoader
from utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


class DataPreprocessingPipeline:
    """
    Complete data preprocessing pipeline for Arabic dialect sentiment analysis.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = ConfigLoader.load_config(config_path)
        self.setup_logging()
        self.preprocessor = ArabicTextPreprocessor(self.config)
        
        # Create output directories
        self.create_output_directories()
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        setup_logging(
            level=log_config.get('level', 'INFO'),
            log_dir=log_config.get('log_dir', 'logs')
        )
    
    def create_output_directories(self):
        """Create necessary output directories."""
        data_config = self.config.get('data', {})
        
        # Create processed data directory
        processed_path = Path(data_config.get('processed_data_path', 'data/processed'))
        processed_path.mkdir(parents=True, exist_ok=True)
        
        # Create tokenized data directory
        tokenized_path = Path(data_config.get('tokenized_data_path', 'data/processed/tokenized_data'))
        tokenized_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Output directories created")
    
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw data from the specified path.
        
        Returns:
            Raw data DataFrame
        """
        data_config = self.config.get('data', {})
        raw_data_path = data_config.get('sentiment_dataset_path')
        
        if not raw_data_path or not Path(raw_data_path).exists():
            raise FileNotFoundError(f"Raw data file not found: {raw_data_path}")
        
        logger.info(f"Loading raw data from {raw_data_path}")
        
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1256']
            data = None
            
            for encoding in encodings:
                try:
                    data = pd.read_csv(raw_data_path, encoding=encoding)
                    logger.info(f"Successfully loaded data with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if data is None:
                raise ValueError("Could not load data with any supported encoding")
            
            logger.info(f"Loaded {len(data)} samples with columns: {list(data.columns)}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading raw data: {e}")
            raise
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate the loaded data against requirements.
        
        Args:
            data: Input DataFrame
            
        Returns:
            True if validation passes, False otherwise
        """
        validation_config = self.config.get('validation', {})
        required_columns = validation_config.get('required_columns', ['text', 'label'])
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check minimum samples per class
        min_samples = validation_config.get('min_samples_per_class', 10)
        if 'label' in data.columns:
            class_counts = data['label'].value_counts()
            if (class_counts < min_samples).any():
                logger.warning(f"Some classes have fewer than {min_samples} samples")
        
        # Check for empty text
        if 'text' in data.columns:
            empty_texts = data['text'].isna().sum() + (data['text'] == '').sum()
            if empty_texts > 0:
                logger.warning(f"Found {empty_texts} empty text entries")
        
        logger.info("Data validation completed successfully")
        return True
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the raw data using the ArabicTextPreprocessor.
        
        Args:
            data: Raw data DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Starting data preprocessing")
        
        # Preprocess the dataset
        processed_data = self.preprocessor.preprocess_dataset(
            data=data,
            text_column='text',
            dialect_column='dialect',
            label_column='label'
        )
        
        # Save preprocessing statistics
        if self.config.get('logging', {}).get('save_preprocessing_stats', True):
            stats_path = self.config.get('logging', {}).get('stats_output_path')
            if stats_path:
                self.preprocessor.save_preprocessing_stats(processed_data, stats_path)
        
        logger.info(f"Preprocessing completed. Final dataset size: {len(processed_data)}")
        return processed_data
    
    def split_dataset(self, data: pd.DataFrame) -> tuple:
        """
        Split the dataset into train, validation, and test sets.
        
        Args:
            data: Preprocessed DataFrame
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        splitting_config = self.config.get('splitting', {})
        
        train_ratio = splitting_config.get('train_ratio', 0.7)
        val_ratio = splitting_config.get('val_ratio', 0.15)
        test_ratio = splitting_config.get('test_ratio', 0.15)
        random_seed = splitting_config.get('random_seed', 42)
        stratify = splitting_config.get('stratify', True)
        
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")
        
        logger.info(f"Splitting dataset with ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
        
        # First split: separate test set
        if test_ratio > 0:
            stratify_col = data['label'] if stratify and 'label' in data.columns else None
            train_val_data, test_data = train_test_split(
                data,
                test_size=test_ratio,
                random_state=random_seed,
                stratify=stratify_col
            )
        else:
            train_val_data, test_data = data, pd.DataFrame()
        
        # Second split: separate validation set from remaining data
        if val_ratio > 0:
            val_size = val_ratio / (train_ratio + val_ratio)
            stratify_col = train_val_data['label'] if stratify and 'label' in train_val_data.columns else None
            train_data, val_data = train_test_split(
                train_val_data,
                test_size=val_size,
                random_state=random_seed,
                stratify=stratify_col
            )
        else:
            train_data, val_data = train_val_data, pd.DataFrame()
        
        logger.info(f"Dataset split completed:")
        logger.info(f"  Train: {len(train_data)} samples")
        logger.info(f"  Validation: {len(val_data)} samples")
        logger.info(f"  Test: {len(test_data)} samples")
        
        return train_data, val_data, test_data
    
    def save_split_datasets(self, train_data: pd.DataFrame, 
                           val_data: pd.DataFrame, 
                           test_data: pd.DataFrame):
        """
        Save the split datasets to files.
        
        Args:
            train_data: Training data
            val_data: Validation data
            test_data: Test data
        """
        data_config = self.config.get('data', {})
        
        # Save train data
        if len(train_data) > 0:
            train_path = data_config.get('train_data_path', 'data/processed/train_data.csv')
            train_data.to_csv(train_path, index=False, encoding='utf-8')
            logger.info(f"Training data saved to {train_path}")
        
        # Save validation data
        if len(val_data) > 0:
            val_path = data_config.get('val_data_path', 'data/processed/val_data.csv')
            val_data.to_csv(val_path, index=False, encoding='utf-8')
            logger.info(f"Validation data saved to {val_path}")
        
        # Save test data
        if len(test_data) > 0:
            test_path = data_config.get('test_data_path', 'data/processed/test_data.csv')
            test_data.to_csv(test_path, index=False, encoding='utf-8')
            logger.info(f"Test data saved to {test_path}")
    
    def compute_class_weights(self, train_data: pd.DataFrame) -> dict:
        """
        Compute class weights for imbalanced datasets.
        
        Args:
            train_data: Training data DataFrame
            
        Returns:
            Dictionary mapping class labels to weights
        """
        if 'label' not in train_data.columns:
            logger.warning("No label column found, skipping class weight computation")
            return {}
        
        try:
            # Get unique classes and their counts
            classes = train_data['label'].unique()
            class_counts = train_data['label'].value_counts()
            
            # Compute balanced class weights
            class_weights = compute_class_weight(
                'balanced',
                classes=classes,
                y=train_data['label']
            )
            
            # Create weight dictionary
            weight_dict = dict(zip(classes, class_weights))
            
            logger.info("Class weights computed:")
            for class_label, weight in weight_dict.items():
                count = class_counts[class_label]
                logger.info(f"  {class_label}: weight={weight:.3f}, count={count}")
            
            # Save class weights
            weights_path = Path(self.config.get('data', {}).get('processed_data_path', 'data/processed')) / 'class_weights.json'
            with open(weights_path, 'w', encoding='utf-8') as f:
                json.dump(weight_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Class weights saved to {weights_path}")
            return weight_dict
            
        except Exception as e:
            logger.error(f"Error computing class weights: {e}")
            return {}
    
    def run_pipeline(self):
        """
        Run the complete preprocessing pipeline.
        """
        try:
            logger.info("Starting Arabic Dialect Sentiment Analysis preprocessing pipeline")
            
            # Step 1: Load raw data
            raw_data = self.load_raw_data()
            
            # Step 2: Validate data
            if not self.validate_data(raw_data):
                raise ValueError("Data validation failed")
            
            # Step 3: Preprocess data
            processed_data = self.preprocess_data(raw_data)
            
            # Step 4: Split dataset
            train_data, val_data, test_data = self.split_dataset(processed_data)
            
            # Step 5: Save split datasets
            self.save_split_datasets(train_data, val_data, test_data)
            
            # Step 6: Compute class weights
            self.compute_class_weights(train_data)
            
            logger.info("Preprocessing pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Preprocessing pipeline failed: {e}")
            raise


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Arabic Dialect Sentiment Analysis - Data Preprocessing Pipeline'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/data_config.yaml',
        help='Path to configuration file (default: configs/data_config.yaml)'
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not Path(args.config).exists():
        print(f"Configuration file not found: {args.config}")
        print("Please ensure the config file exists or specify a different path with --config")
        sys.exit(1)
    
    try:
        # Initialize and run pipeline
        pipeline = DataPreprocessingPipeline(args.config)
        pipeline.run_pipeline()
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
