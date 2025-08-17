"""
ASTD Dataset Preprocessing Script

This script preprocesses the ASTD dataset for Arabic sentiment analysis,
integrating with the ASTD loader and applying Arabic-specific text preprocessing.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from data.astd_loader import ASTDDataLoader
from data.preprocessor import ArabicTextPreprocessor
from utils.config_loader import ConfigLoader
from utils.logging_utils import setup_logging, log_hyperparameters, log_metrics

logger = logging.getLogger(__name__)


class ASTDPreprocessor:
    """
    Preprocessor for the ASTD dataset with Arabic-specific text processing.
    """
    
    def __init__(self, config_path: str = "configs/data_config.yaml"):
        """
        Initialize the ASTD preprocessor.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = ConfigLoader.load_config(config_path)
        self.setup_logging()
        self.setup_preprocessor()
        self.setup_data_loader()
        
        logger.info("ASTD Preprocessor initialized successfully")
    
    def setup_logging(self):
        """Setup logging for the preprocessing process."""
        log_dir = self.config.get('logging', {}).get('log_dir', 'logs')
        experiment_name = self.config.get('logging', {}).get('experiment_name', 'astd_preprocessing')
        
        setup_logging(
            log_dir=log_dir,
            experiment_name=experiment_name,
            log_level=self.config.get('logging', {}).get('log_level', 'INFO')
        )
        
        # Log configuration
        log_hyperparameters(self.config)
    
    def setup_preprocessor(self):
        """Setup Arabic text preprocessor."""
        preprocessing_config = self.config.get('preprocessing', {})
        
        self.preprocessor = ArabicTextPreprocessor(
            remove_urls=preprocessing_config.get('remove_urls', True),
            normalize_arabic=preprocessing_config.get('normalize_arabic', True),
            normalize_numbers=preprocessing_config.get('normalize_numbers', True),
            remove_emojis=preprocessing_config.get('remove_emojis', True),
            remove_english=preprocessing_config.get('remove_english', False),
            min_length=preprocessing_config.get('min_length', 10),
            max_length=preprocessing_config.get('max_length', 512)
        )
        
        logger.info("Arabic text preprocessor setup completed")
    
    def setup_data_loader(self):
        """Setup ASTD data loader."""
        self.data_loader = ASTDDataLoader()
        
        # Validate dataset
        if not self.data_loader.validate_dataset():
            raise ValueError("Dataset validation failed")
        
        logger.info("ASTD data loader setup completed")
    
    def preprocess_split(self, split_name: str, save_processed: bool = True) -> pd.DataFrame:
        """
        Preprocess a specific dataset split.
        
        Args:
            split_name: Name of the split to preprocess
            save_processed: Whether to save the processed data
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info(f"Preprocessing split: {split_name}")
        
        # Load split data
        split_df = self.data_loader.get_split_data(split_name)
        
        # Apply preprocessing
        processed_df = self.preprocessor.preprocess_dataset(split_df, text_column='text')
        
        # Add metadata
        processed_df['split_name'] = split_name
        processed_df['original_length'] = split_df['text'].str.len()
        processed_df['processed_length'] = processed_df['text'].str.len()
        processed_df['length_change'] = processed_df['processed_length'] - processed_df['original_length']
        
        # Log preprocessing statistics
        self.log_preprocessing_stats(split_name, split_df, processed_df)
        
        # Save processed data if requested
        if save_processed:
            self.save_processed_split(processed_df, split_name)
        
        return processed_df
    
    def log_preprocessing_stats(self, split_name: str, original_df: pd.DataFrame, processed_df: pd.DataFrame):
        """Log preprocessing statistics for a split."""
        logger.info(f"Preprocessing stats for {split_name}:")
        logger.info(f"  Original samples: {len(original_df)}")
        logger.info(f"  Processed samples: {len(processed_df)}")
        logger.info(f"  Samples removed: {len(original_df) - len(processed_df)}")
        
        if len(processed_df) > 0:
            logger.info(f"  Average original length: {original_df['text'].str.len().mean():.1f}")
            logger.info(f"  Average processed length: {processed_df['text'].str.len().mean():.1f}")
            logger.info(f"  Label distribution: {self.data_loader.get_label_distribution(processed_df)}")
    
    def save_processed_split(self, df: pd.DataFrame, split_name: str):
        """Save processed split data."""
        output_dir = Path(self.config.get('output', {}).get('processed_dir', 'data/processed'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        csv_path = output_dir / f"{split_name}_processed.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"Saved processed {split_name} to {csv_path}")
        
        # Save as pickle for faster loading
        pickle_path = output_dir / f"{split_name}_processed.pkl"
        df.to_pickle(pickle_path)
        logger.info(f"Saved processed {split_name} to {pickle_path}")
    
    def preprocess_all_splits(self) -> Dict[str, pd.DataFrame]:
        """
        Preprocess all available dataset splits.
        
        Returns:
            Dictionary mapping split names to processed DataFrames
        """
        logger.info("Preprocessing all dataset splits...")
        
        processed_splits = {}
        
        # Get all split names
        split_files = list(Path("data").glob("*.txt"))
        split_names = [f.stem for f in split_files if f.name != "Tweets.txt"]
        
        for split_name in split_names:
            try:
                processed_df = self.preprocess_split(split_name, save_processed=True)
                processed_splits[split_name] = processed_df
            except Exception as e:
                logger.error(f"Error preprocessing {split_name}: {e}")
        
        logger.info(f"Preprocessed {len(processed_splits)} splits successfully")
        return processed_splits
    
    def create_balanced_splits(self, processed_splits: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Create balanced splits from processed data.
        
        Args:
            processed_splits: Dictionary of processed splits
            
        Returns:
            Dictionary of balanced splits
        """
        logger.info("Creating balanced splits...")
        
        # Combine all processed data
        all_data = []
        for split_name, df in processed_splits.items():
            df_copy = df.copy()
            df_copy['source_split'] = split_name
            all_data.append(df_copy)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Combined {len(combined_df)} samples from all splits")
        
        # Create balanced splits
        balanced_splits = {}
        label_column = 'label_id'
        
        # Get unique labels
        unique_labels = sorted(combined_df[label_column].unique())
        logger.info(f"Unique labels: {unique_labels}")
        
        # Calculate samples per class for balanced splits
        min_samples_per_class = combined_df[label_column].value_counts().min()
        samples_per_class = min_samples_per_class // 3  # For train/val/test
        
        logger.info(f"Creating balanced splits with {samples_per_class} samples per class per split")
        
        # Create balanced splits
        for split_type in ['train', 'validation', 'test']:
            split_data = []
            
            for label in unique_labels:
                label_data = combined_df[combined_df[label_column] == label]
                
                if split_type == 'train':
                    # Use 70% of data for training
                    split_label_data, _ = train_test_split(
                        label_data, 
                        train_size=0.7, 
                        random_state=42,
                        stratify=label_data[label_column]
                    )
                elif split_type == 'validation':
                    # Use 15% of data for validation
                    _, temp_data = train_test_split(
                        label_data, 
                        train_size=0.7, 
                        random_state=42,
                        stratify=label_data[label_column]
                    )
                    split_label_data, _ = train_test_split(
                        temp_data, 
                        train_size=0.5, 
                        random_state=42,
                        stratify=temp_data[label_column]
                    )
                else:  # test
                    # Use remaining 15% for testing
                    _, temp_data = train_test_split(
                        label_data, 
                        train_size=0.7, 
                        random_state=42,
                        stratify=label_data[label_column]
                    )
                    _, split_label_data = train_test_split(
                        temp_data, 
                        train_size=0.5, 
                        random_state=42,
                        stratify=temp_data[label_column]
                    )
                
                split_data.append(split_label_data)
            
            # Combine all classes for this split
            split_df = pd.concat(split_data, ignore_index=True)
            balanced_splits[f'balanced_{split_type}'] = split_df
            
            logger.info(f"Created balanced_{split_type}: {len(split_df)} samples")
            logger.info(f"  Label distribution: {self.data_loader.get_label_distribution(split_df)}")
        
        return balanced_splits
    
    def save_balanced_splits(self, balanced_splits: Dict[str, pd.DataFrame]):
        """Save balanced splits to files."""
        output_dir = Path(self.config.get('output', {}).get('processed_dir', 'data/processed'))
        splits_dir = output_dir / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)
        
        for split_name, split_df in balanced_splits.items():
            # Save as CSV
            csv_path = splits_dir / f"{split_name}.csv"
            split_df.to_csv(csv_path, index=False, encoding='utf-8')
            
            # Save as pickle
            pickle_path = splits_dir / f"{split_name}.pkl"
            split_df.to_pickle(pickle_path)
            
            logger.info(f"Saved {split_name} to {csv_path} and {pickle_path}")
    
    def calculate_class_weights(self, train_df: pd.DataFrame) -> Dict[int, float]:
        """
        Calculate class weights for handling class imbalance.
        
        Args:
            train_df: Training DataFrame
            
        Returns:
            Dictionary mapping class IDs to weights
        """
        logger.info("Calculating class weights...")
        
        labels = train_df['label_id'].values
        unique_labels = sorted(np.unique(labels))
        
        # Calculate class weights using sklearn
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_labels,
            y=labels
        )
        
        # Create weight dictionary
        weight_dict = {label: weight for label, weight in zip(unique_labels, class_weights)}
        
        logger.info(f"Class weights: {weight_dict}")
        
        # Save class weights
        output_dir = Path(self.config.get('output', {}).get('processed_dir', 'data/processed'))
        weights_file = output_dir / "class_weights.json"
        
        with open(weights_file, 'w') as f:
            json.dump(weight_dict, f, indent=2)
        
        logger.info(f"Saved class weights to {weights_file}")
        
        return weight_dict
    
    def create_vocabulary_stats(self, processed_splits: Dict[str, pd.DataFrame]):
        """Create vocabulary statistics for the processed dataset."""
        logger.info("Creating vocabulary statistics...")
        
        # Combine all texts
        all_texts = []
        for split_df in processed_splits.values():
            all_texts.extend(split_df['text'].tolist())
        
        # Create vocabulary
        from collections import Counter
        word_counts = Counter()
        
        for text in all_texts:
            words = text.split()
            word_counts.update(words)
        
        # Save vocabulary stats
        output_dir = Path(self.config.get('output', {}).get('processed_dir', 'data/processed'))
        vocab_file = output_dir / "vocabulary_stats.json"
        
        vocab_stats = {
            'total_words': sum(word_counts.values()),
            'unique_words': len(word_counts),
            'most_common_words': word_counts.most_common(100),
            'word_frequency_distribution': dict(word_counts.most_common(1000))
        }
        
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(vocab_stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved vocabulary statistics to {vocab_file}")
        logger.info(f"Total words: {vocab_stats['total_words']:,}")
        logger.info(f"Unique words: {vocab_stats['unique_words']:,}")
    
    def run_preprocessing_pipeline(self):
        """Run the complete preprocessing pipeline."""
        logger.info("Starting ASTD preprocessing pipeline...")
        
        try:
            # Preprocess all splits
            processed_splits = self.preprocess_all_splits()
            
            # Create balanced splits
            balanced_splits = self.create_balanced_splits(processed_splits)
            
            # Save balanced splits
            self.save_balanced_splits(balanced_splits)
            
            # Calculate class weights
            train_df = balanced_splits['balanced_train']
            class_weights = self.calculate_class_weights(train_df)
            
            # Create vocabulary statistics
            self.create_vocabulary_stats(processed_splits)
            
            # Log final metrics
            final_metrics = {
                'total_processed_splits': len(processed_splits),
                'total_balanced_splits': len(balanced_splits),
                'total_samples': sum(len(df) for df in processed_splits.values()),
                'balanced_samples': sum(len(df) for df in balanced_splits.values()),
                'class_weights': class_weights
            }
            
            log_metrics(final_metrics)
            
            logger.info("ASTD preprocessing pipeline completed successfully!")
            logger.info(f"Final metrics: {final_metrics}")
            
        except Exception as e:
            logger.error(f"Preprocessing pipeline failed: {e}")
            raise


def main():
    """Main function to run preprocessing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess ASTD dataset for Arabic sentiment analysis")
    parser.add_argument("--config", type=str, default="configs/data_config.yaml", 
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    try:
        # Initialize preprocessor
        preprocessor = ASTDPreprocessor(args.config)
        
        # Run preprocessing pipeline
        preprocessor.run_preprocessing_pipeline()
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
