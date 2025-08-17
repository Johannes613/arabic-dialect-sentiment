#!/usr/bin/env python3
"""
Data Organization Script

This script organizes the data folder structure and tests the ASTD data loader.
"""

import os
import shutil
from pathlib import Path
import sys
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.astd_loader import ASTDDataLoader
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def organize_data_folder():
    """Organize the data folder structure."""
    data_dir = Path("data")
    
    logger.info("Organizing data folder structure...")
    
    # Create subdirectories if they don't exist
    subdirs = ["raw", "processed", "external"]
    for subdir in subdirs:
        (data_dir / subdir).mkdir(exist_ok=True)
        logger.info(f"Created directory: {data_dir / subdir}")
    
    # Move main dataset files to raw directory
    raw_dir = data_dir / "raw"
    files_to_move = [
        "Tweets.txt",
        "4class-balanced-train.txt",
        "4class-balanced-validation.txt", 
        "4class-balanced-test.txt",
        "4class-unbalanced-train.txt",
        "4class-unbalanced-validation.txt",
        "4class-unbalanced-test.txt"
    ]
    
    for file_name in files_to_move:
        source = data_dir / file_name
        if source.exists():
            # Create backup in raw directory
            shutil.copy2(source, raw_dir / file_name)
            logger.info(f"Backed up {file_name} to raw/ directory")
    
    # Create processed directory structure
    processed_dir = data_dir / "processed"
    processed_subdirs = ["cleaned", "tokenized", "splits", "features"]
    for subdir in processed_subdirs:
        (processed_dir / subdir).mkdir(exist_ok=True)
        logger.info(f"Created directory: {processed_dir / subdir}")
    
    # Create external directory structure
    external_dir = data_dir / "external"
    external_subdirs = ["pretrained_models", "embeddings", "lexicons"]
    for subdir in external_subdirs:
        (external_dir / subdir).mkdir(exist_ok=True)
        logger.info(f"Created directory: {external_dir / subdir}")
    
    logger.info("Data folder organization completed!")


def test_astd_loader():
    """Test the ASTD data loader functionality."""
    logger.info("Testing ASTD data loader...")
    
    try:
        # Initialize loader
        loader = ASTDDataLoader()
        
        # Validate dataset
        if not loader.validate_dataset():
            logger.error("Dataset validation failed!")
            return False
        
        # Load all tweets
        tweets_df = loader.load_tweets()
        logger.info(f"Successfully loaded {len(tweets_df)} tweets")
        
        # Check if we have existing processed data
        processed_dir = Path("data/processed/splits")
        if processed_dir.exists():
            logger.info("Found existing processed data, loading from there...")
            
            # Load existing processed splits
            balanced_splits = {}
            split_files = list(processed_dir.glob("4class-balanced-*.csv"))
            
            for split_file in split_files:
                split_name = split_file.stem
                split_df = pd.read_csv(split_file, encoding='utf-8')
                balanced_splits[split_name] = split_df
                logger.info(f"Loaded {split_name}: {len(split_df)} tweets")
                label_dist = loader.get_label_distribution(split_df)
                logger.info(f"  Label distribution: {label_dist}")
            
            # Calculate class weights from existing data
            if '4class-balanced-train' in balanced_splits:
                train_df = balanced_splits['4class-balanced-train']
                class_weights = loader.get_class_weights(train_df)
                logger.info(f"Class weights: {class_weights}")
            else:
                logger.warning("No training split found in processed data")
                return False
            
            logger.info("ASTD data loader test completed successfully using existing processed data!")
            return True
        else:
            logger.warning("No processed data found. Please run preprocessing first.")
            return False
        
    except Exception as e:
        logger.error(f"ASTD data loader test failed: {e}")
        return False


def main():
    """Main function to organize data and test loader."""
    logger.info("Starting data organization and testing...")
    
    # Organize data folder
    organize_data_folder()
    
    # Test ASTD loader
    if test_astd_loader():
        logger.info("All tests passed! Data is ready for preprocessing.")
    else:
        logger.error("Some tests failed. Please check the data structure.")
    
    logger.info("Data organization script completed!")


if __name__ == "__main__":
    main()
