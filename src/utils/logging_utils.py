"""
Logging Utilities for Arabic Dialect Sentiment Analysis

This module provides utilities for setting up consistent logging
across the project with proper formatting and file handling.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import os


def setup_logging(level: str = 'INFO', 
                  log_dir: Optional[str] = None,
                  log_file: Optional[str] = None,
                  console_output: bool = True,
                  file_output: bool = True,
                  max_bytes: int = 10 * 1024 * 1024,  # 10MB
                  backup_count: int = 5) -> logging.Logger:
    """
    Setup logging configuration for the project.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
        log_file: Name of the log file
        console_output: Whether to output to console
        file_output: Whether to output to file
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep
        
    Returns:
        Configured logger instance
    """
    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if file_output and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        if not log_file:
            log_file = 'arabic_sentiment.log'
        
        log_path = log_dir / log_file
        
        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels for noisy libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup completed. Level: {level}")
    
    if log_dir:
        logger.info(f"Log directory: {log_dir}")
    
    return logger


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name
        level: Optional logging level override
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    if level:
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(numeric_level)
    
    return logger


def setup_experiment_logging(experiment_name: str,
                            log_dir: str = 'logs',
                            config: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Setup logging for a specific experiment.
    
    Args:
        experiment_name: Name of the experiment
        log_dir: Base log directory
        config: Optional configuration dictionary
        
    Returns:
        Configured logger for the experiment
    """
    # Create experiment-specific log directory
    experiment_log_dir = Path(log_dir) / experiment_name
    experiment_log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(
        level='INFO',
        log_dir=str(experiment_log_dir),
        log_file=f'{experiment_name}.log',
        console_output=True,
        file_output=True
    )
    
    # Log experiment start
    logger.info(f"Starting experiment: {experiment_name}")
    
    # Log configuration if provided
    if config:
        logger.info("Experiment configuration:")
        for key, value in config.items():
            if isinstance(value, dict):
                logger.info(f"  {key}:")
                for sub_key, sub_value in value.items():
                    logger.info(f"    {sub_key}: {sub_value}")
            else:
                logger.info(f"  {key}: {value}")
    
    return logger


def log_hyperparameters(logger: logging.Logger, 
                       hyperparams: Dict[str, Any],
                       prefix: str = "Hyperparameters"):
    """
    Log hyperparameters in a structured format.
    
    Args:
        logger: Logger instance
        hyperparams: Dictionary of hyperparameters
        prefix: Prefix for the log message
    """
    logger.info(f"{prefix}:")
    for key, value in hyperparams.items():
        logger.info(f"  {key}: {value}")


def log_metrics(logger: logging.Logger,
                metrics: Dict[str, float],
                epoch: Optional[int] = None,
                prefix: str = "Metrics"):
    """
    Log metrics in a structured format.
    
    Args:
        logger: Logger instance
        metrics: Dictionary of metrics
        epoch: Optional epoch number
        prefix: Prefix for the log message
    """
    epoch_str = f" (Epoch {epoch})" if epoch is not None else ""
    logger.info(f"{prefix}{epoch_str}:")
    
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, float):
            logger.info(f"  {metric_name}: {metric_value:.4f}")
        else:
            logger.info(f"  {metric_name}: {metric_value}")


def log_training_progress(logger: logging.Logger,
                         epoch: int,
                         total_epochs: int,
                         train_loss: float,
                         val_loss: Optional[float] = None,
                         train_metrics: Optional[Dict[str, float]] = None,
                         val_metrics: Optional[Dict[str, float]] = None):
    """
    Log training progress information.
    
    Args:
        logger: Logger instance
        epoch: Current epoch
        total_epochs: Total number of epochs
        train_loss: Training loss
        val_loss: Validation loss (optional)
        train_metrics: Training metrics (optional)
        val_metrics: Validation metrics (optional)
    """
    logger.info(f"Epoch {epoch}/{total_epochs}")
    logger.info(f"  Training Loss: {train_loss:.4f}")
    
    if val_loss is not None:
        logger.info(f"  Validation Loss: {val_loss:.4f}")
    
    if train_metrics:
        logger.info("  Training Metrics:")
        for metric_name, metric_value in train_metrics.items():
            if isinstance(metric_value, float):
                logger.info(f"    {metric_name}: {metric_value:.4f}")
            else:
                logger.info(f"    {metric_name}: {metric_value}")
    
    if val_metrics:
        logger.info("  Validation Metrics:")
        for metric_name, metric_value in val_metrics.items():
            if isinstance(metric_value, float):
                logger.info(f"    {metric_name}: {metric_value:.4f}")
            else:
                logger.info(f"    {metric_name}: {metric_value}")


def log_model_info(logger: logging.Logger,
                   model_name: str,
                   model_params: int,
                   model_size_mb: Optional[float] = None):
    """
    Log model information.
    
    Args:
        logger: Logger instance
        model_name: Name of the model
        model_params: Number of model parameters
        model_size_mb: Model size in MB (optional)
    """
    logger.info(f"Model: {model_name}")
    logger.info(f"  Parameters: {model_params:,}")
    
    if model_size_mb:
        logger.info(f"  Size: {model_size_mb:.2f} MB")


def log_data_info(logger: logging.Logger,
                  train_samples: int,
                  val_samples: int,
                  test_samples: int,
                  num_classes: int,
                  class_distribution: Optional[Dict[str, int]] = None):
    """
    Log dataset information.
    
    Args:
        logger: Logger instance
        train_samples: Number of training samples
        val_samples: Number of validation samples
        test_samples: Number of test samples
        num_classes: Number of classes
        class_distribution: Class distribution dictionary (optional)
    """
    logger.info("Dataset Information:")
    logger.info(f"  Training samples: {train_samples:,}")
    logger.info(f"  Validation samples: {val_samples:,}")
    logger.info(f"  Test samples: {test_samples:,}")
    logger.info(f"  Total samples: {train_samples + val_samples + test_samples:,}")
    logger.info(f"  Number of classes: {num_classes}")
    
    if class_distribution:
        logger.info("  Class distribution:")
        for class_name, count in class_distribution.items():
            logger.info(f"    {class_name}: {count:,}")


def setup_tensorboard_logging(log_dir: str = 'logs/tensorboard'):
    """
    Setup TensorBoard logging configuration.
    
    Args:
        log_dir: Directory for TensorBoard logs
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.info(f"TensorBoard logging directory: {log_dir}")
    
    return str(log_dir)


def setup_wandb_logging(project_name: str,
                        entity: Optional[str] = None,
                        config: Optional[Dict[str, Any]] = None):
    """
    Setup Weights & Biases logging configuration.
    
    Args:
        project_name: Name of the W&B project
        entity: W&B entity/username (optional)
        config: Configuration dictionary to log (optional)
    """
    try:
        import wandb
        
        # Initialize W&B
        wandb.init(
            project=project_name,
            entity=entity,
            config=config
        )
        
        logger = logging.getLogger(__name__)
        logger.info(f"Weights & Biases logging initialized for project: {project_name}")
        
        return wandb
        
    except ImportError:
        logger = logging.getLogger(__name__)
        logger.warning("Weights & Biases not installed. Skipping W&B logging setup.")
        return None
