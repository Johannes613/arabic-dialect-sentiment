#!/usr/bin/env python3
"""
Standalone script for MARBERT fine-tuning on ASTD dataset
This script can be run directly or imported as a module
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.astd_loader import ASTDDataLoader
from models.fine_tune_marbert import MARBERTFineTuner
from utils.logging_utils import setup_logging, setup_experiment_logging
from utils.config_loader import ConfigLoader

def main():
    """Main function to run MARBERT fine-tuning"""
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config_path = "configs/fine_tune_config.yaml"
    if os.path.exists(config_path):
        config = ConfigLoader().load_config(config_path)
        logger.info("Loaded fine-tuning configuration")
    else:
        # Default configuration
        config = {
            "model": {
                "name": "UBC-NLP/MARBERT",
                "max_length": 128,
                "num_labels": 4
            },
            "training": {
                "learning_rate": 2e-5,
                "batch_size": 16,
                "epochs": 3,
                "warmup_steps": 500,
                "weight_decay": 0.01,
                "evaluation_strategy": "epoch",
                "save_strategy": "epoch",
                "save_total_limit": 2,
                "metric_for_best_model": "macro_f1"
            },
            "data": {
                "data_dir": "data",
                "output_dir": "models/marbert_sentiment",
                "test_size": 0.2,
                "val_size": 0.2,
                "random_state": 42
            }
        }
        logger.info("Using default configuration")
    
    try:
        # Initialize fine-tuner
        logger.info("Initializing MARBERT fine-tuner...")
        fine_tuner = MARBERTFineTuner(config)
        
        # Setup data loader
        logger.info("Setting up data loader...")
        fine_tuner.setup_data_loader()
        
        # Preprocess data
        logger.info("Preprocessing data...")
        fine_tuner.preprocess_data()
        
        # Create datasets
        logger.info("Creating datasets...")
        fine_tuner.create_datasets()
        
        # Setup model and tokenizer
        logger.info("Setting up model and tokenizer...")
        fine_tuner.setup_model_and_tokenizer()
        
        # Run fine-tuning
        logger.info("Starting fine-tuning...")
        fine_tuner.run_fine_tuning()
        
        # Evaluate model
        logger.info("Evaluating model...")
        results = fine_tuner.evaluate_model()
        
        # Save model
        logger.info("Saving model...")
        fine_tuner.save_model()
        
        logger.info("Fine-tuning completed successfully!")
        logger.info(f"Results: {results}")
        
    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}")
        raise

if __name__ == "__main__":
    main()
