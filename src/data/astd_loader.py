"""
ASTD (Arabic Sentiment Tweets Dataset) Data Loader

This module provides functionality to load and process the ASTD dataset,
which consists of a main Tweets.txt file and line index files for different splits.
"""

import os
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ASTDDataLoader:
    """
    Data loader for the Arabic Sentiment Tweets Dataset (ASTD).
    
    The dataset structure:
    - Tweets.txt: Main file with tweet text and sentiment labels
    - Line index files: Contain line numbers referencing Tweets.txt
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the ASTD data loader.
        
        Args:
            data_dir: Path to the data directory
        """
        self.data_dir = Path(data_dir)
        self.tweets_file = self.data_dir / "Tweets.txt"
        self.label_mapping = {
            'POS': 1,      # Positive
            'NEG': 0,      # Negative  
            'NEUTRAL': 2,  # Neutral
            'OBJ': 3       # Objective
        }
        self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
        
        # Validate data directory
        if not self.tweets_file.exists():
            raise FileNotFoundError(f"Tweets.txt not found in {data_dir}")
    
    def load_tweets(self) -> pd.DataFrame:
        """
        Load the main Tweets.txt file into a DataFrame.
        
        Returns:
            DataFrame with columns: ['text', 'label', 'label_id', 'file_line_number']
        """
        logger.info("Loading main Tweets.txt file...")
        
        try:
            # Read the TSV file
            df = pd.read_csv(
                self.tweets_file, 
                sep='\t', 
                header=None, 
                names=['text', 'label'],
                encoding='utf-8'
            )
            
            # Add file line numbers (1-based, matching the index files)
            df['file_line_number'] = range(1, len(df) + 1)
            
            # Add label IDs
            df['label_id'] = df['label'].map(self.label_mapping)
            
            # Validate labels
            invalid_labels = df[df['label_id'].isna()]['label'].unique()
            if len(invalid_labels) > 0:
                logger.warning(f"Found invalid labels: {invalid_labels}")
                # Keep track of which rows are valid
                df = df[df['label_id'].notna()].reset_index(drop=True)
            
            logger.info(f"Loaded {len(df)} tweets with {df['label'].nunique()} unique labels")
            logger.info(f"Label distribution:\n{df['label'].value_counts()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading Tweets.txt: {e}")
            raise
    
    def load_split_indices(self, split_name: str) -> List[int]:
        """
        Load line indices for a specific dataset split.
        
        Args:
            split_name: Name of the split (e.g., '4class-balanced-train')
            
        Returns:
            List of line indices (0-based)
        """
        split_file = self.data_dir / f"{split_name}.txt"
        
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        try:
            with open(split_file, 'r', encoding='utf-8') as f:
                indices = [int(line.strip()) - 1 for line in f if line.strip()]  # Convert to 0-based
            
            # Get the actual number of lines in Tweets.txt
            with open(self.tweets_file, 'r', encoding='utf-8') as f:
                max_lines = sum(1 for _ in f)
            
            logger.debug(f"File {split_name}: max_lines={max_lines}, original_indices_count={len(indices)}")
            
            # Validate indices are within bounds
            max_index = max(indices) if indices else 0
            if max_index >= max_lines:
                logger.warning(f"Found indices out of bounds in {split_name}: max={max_index}, file_lines={max_lines}")
                # Filter out invalid indices
                original_count = len(indices)
                indices = [idx for idx in indices if idx < max_lines]
                logger.info(f"Filtered {split_name}: {original_count} -> {len(indices)} valid indices")
                logger.debug(f"Max index after filtering: {max(indices) if indices else 'N/A'}")
            else:
                logger.debug(f"All indices in {split_name} are within bounds: max={max_index}, file_lines={max_lines}")
            
            logger.info(f"Loaded {len(indices)} indices from {split_name}")
            return indices
            
        except Exception as e:
            logger.error(f"Error loading split indices from {split_name}: {e}")
            raise
    
    def get_split_data(self, split_name: str, tweets_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Get the actual tweet data for a specific split.
        
        Args:
            split_name: Name of the split (e.g., '4class-balanced-train')
            tweets_df: Pre-loaded tweets DataFrame (optional)
            
        Returns:
            DataFrame with tweets for the specified split
        """
        if tweets_df is None:
            tweets_df = self.load_tweets()
        
        indices = self.load_split_indices(split_name)
        
        # Convert 0-based indices back to 1-based file line numbers
        file_line_numbers = [idx + 1 for idx in indices]
        
        # Filter tweets by file line numbers
        split_df = tweets_df[tweets_df['file_line_number'].isin(file_line_numbers)].reset_index(drop=True)
        
        logger.info(f"Split {split_name}: {len(split_df)} tweets")
        logger.info(f"Label distribution in split:\n{split_df['label'].value_counts()}")
        
        return split_df
    
    def get_all_splits(self, tweets_df: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
        """
        Load all available dataset splits.
        
        Args:
            tweets_df: Pre-loaded tweets DataFrame (optional)
            
        Returns:
            Dictionary mapping split names to DataFrames
        """
        if tweets_df is None:
            tweets_df = self.load_tweets()
        
        splits = {}
        split_files = list(self.data_dir.glob("*.txt"))
        
        for split_file in split_files:
            if split_file.name != "Tweets.txt":
                split_name = split_file.stem
                try:
                    splits[split_name] = self.get_split_data(split_name, tweets_df)
                except Exception as e:
                    logger.warning(f"Could not load split {split_name}: {e}")
        
        return splits
    
    def get_balanced_splits(self, tweets_df: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
        """
        Get only the balanced dataset splits.
        
        Args:
            tweets_df: Pre-loaded tweets DataFrame (optional)
            
        Returns:
            Dictionary with balanced train/val/test splits
        """
        if tweets_df is None:
            tweets_df = self.load_tweets()
        
        balanced_splits = {}
        split_names = ['4class-balanced-train', '4class-balanced-validation', '4class-balanced-test']
        
        for split_name in split_names:
            try:
                balanced_splits[split_name] = self.get_split_data(split_name, tweets_df)
            except Exception as e:
                logger.warning(f"Could not load balanced split {split_name}: {e}")
        
        return balanced_splits
    
    def get_unbalanced_splits(self, tweets_df: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
        """
        Get only the unbalanced dataset splits.
        
        Args:
            tweets_df: Pre-loaded tweets DataFrame (optional)
            
        Returns:
            Dictionary with unbalanced train/val/test splits
        """
        if tweets_df is None:
            tweets_df = self.load_tweets()
        
        unbalanced_splits = {}
        split_names = ['4class-unbalanced-train', '4class-unbalanced-validation', '4class-unbalanced-test']
        
        for split_name in split_names:
            try:
                unbalanced_splits[split_name] = self.get_split_data(split_name, tweets_df)
            except Exception as e:
                logger.warning(f"Could not load unbalanced split {split_name}: {e}")
        
        return unbalanced_splits
    
    def get_label_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Get the distribution of labels in a DataFrame.
        
        Args:
            df: DataFrame with 'label' column
            
        Returns:
            Dictionary mapping labels to counts
        """
        return df['label'].value_counts().to_dict()
    
    def get_class_weights(self, df: pd.DataFrame) -> Dict[int, float]:
        """
        Calculate class weights for handling class imbalance.
        
        Args:
            df: DataFrame with 'label_id' column
            
        Returns:
            Dictionary mapping class IDs to weights
        """
        label_counts = df['label_id'].value_counts()
        total_samples = len(df)
        
        class_weights = {}
        for label_id in sorted(label_counts.index):
            weight = total_samples / (len(label_counts) * label_counts[label_id])
            class_weights[label_id] = weight
        
        logger.info(f"Calculated class weights: {class_weights}")
        return class_weights
    
    def validate_dataset(self) -> bool:
        """
        Validate the dataset structure and integrity.
        
        Returns:
            True if dataset is valid, False otherwise
        """
        try:
            # Check if main file exists
            if not self.tweets_file.exists():
                logger.error("Tweets.txt not found")
                return False
            
            # Load tweets
            tweets_df = self.load_tweets()
            
            # Check for missing labels
            missing_labels = tweets_df[tweets_df['label_id'].isna()]
            if len(missing_labels) > 0:
                logger.warning(f"Found {len(missing_labels)} tweets with missing labels")
            
            # Get the actual file line count (before filtering)
            with open(self.tweets_file, 'r', encoding='utf-8') as f:
                file_line_count = sum(1 for _ in f)
            
            logger.info(f"File has {file_line_count} lines, loaded {len(tweets_df)} valid tweets")
            
            # Check split files
            split_files = list(self.data_dir.glob("*.txt"))
            split_files = [f for f in split_files if f.name != "Tweets.txt"]
            
            for split_file in split_files:
                try:
                    indices = self.load_split_indices(split_file.stem)
                    # Since load_split_indices already filters invalid indices, just check if we have any
                    if len(indices) == 0:
                        logger.warning(f"Split {split_file.stem} has no valid indices after filtering")
                    elif max(indices) >= file_line_count:
                        logger.error(f"Split {split_file.stem} has indices out of bounds: max={max(indices)}, file_lines={file_line_count}")
                        return False
                    else:
                        logger.debug(f"Split {split_file.stem} validation passed: max_index={max(indices)}, file_lines={file_line_count}")
                except Exception as e:
                    logger.error(f"Error validating split {split_file.stem}: {e}")
                    return False
            
            logger.info("Dataset validation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Dataset validation failed: {e}")
            return False


def main():
    """Example usage of the ASTD data loader."""
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Initialize loader
        loader = ASTDDataLoader()
        
        # Validate dataset
        if not loader.validate_dataset():
            print("Dataset validation failed!")
            return
        
        # Load all tweets
        tweets_df = loader.load_tweets()
        print(f"Loaded {len(tweets_df)} tweets")
        
        # Get balanced splits
        balanced_splits = loader.get_balanced_splits(tweets_df)
        for split_name, split_df in balanced_splits.items():
            print(f"{split_name}: {len(split_df)} tweets")
            print(f"Label distribution: {loader.get_label_distribution(split_df)}")
        
        # Calculate class weights
        train_df = balanced_splits['4class-balanced-train']
        class_weights = loader.get_class_weights(train_df)
        print(f"Class weights: {class_weights}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
