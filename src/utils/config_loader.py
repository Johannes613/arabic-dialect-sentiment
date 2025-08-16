"""
Configuration Loader Utility for Arabic Dialect Sentiment Analysis

This module provides utilities for loading, validating, and managing
configuration files in YAML format.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Utility class for loading and managing configuration files.
    """
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If the config file doesn't exist
            yaml.YAMLError: If the YAML file is malformed
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if config is None:
                raise ValueError("Configuration file is empty")
            
            logger.info(f"Configuration loaded from {config_path}")
            return config
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str):
        """
        Save configuration to a YAML file.
        
        Args:
            config: Configuration dictionary to save
            config_path: Path where to save the configuration
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
            
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise
    
    @staticmethod
    def validate_config(config: Dict[str, Any], required_keys: Optional[list] = None) -> bool:
        """
        Validate configuration dictionary.
        
        Args:
            config: Configuration dictionary to validate
            required_keys: List of required top-level keys
            
        Returns:
            True if validation passes, False otherwise
        """
        if not isinstance(config, dict):
            logger.error("Configuration must be a dictionary")
            return False
        
        if required_keys:
            missing_keys = [key for key in required_keys if key not in config]
            if missing_keys:
                logger.error(f"Missing required configuration keys: {missing_keys}")
                return False
        
        logger.info("Configuration validation passed")
        return True
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], 
                     override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries, with override_config taking precedence.
        
        Args:
            base_config: Base configuration
            override_config: Configuration to override with
            
        Returns:
            Merged configuration dictionary
        """
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigLoader.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    @staticmethod
    def get_nested_value(config: Dict[str, Any], key_path: str, 
                        default: Any = None) -> Any:
        """
        Get a nested value from configuration using dot notation.
        
        Args:
            config: Configuration dictionary
            key_path: Dot-separated path to the value (e.g., 'data.max_length')
            default: Default value if key is not found
            
        Returns:
            Value at the specified path or default value
        """
        keys = key_path.split('.')
        current = config
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    @staticmethod
    def set_nested_value(config: Dict[str, Any], key_path: str, value: Any):
        """
        Set a nested value in configuration using dot notation.
        
        Args:
            config: Configuration dictionary to modify
            key_path: Dot-separated path to the value (e.g., 'data.max_length')
            value: Value to set
        """
        keys = key_path.split('.')
        current = config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the value
        current[keys[-1]] = value
    
    @staticmethod
    def load_env_overrides(config: Dict[str, Any], env_prefix: str = "ARABIC_SENTIMENT_") -> Dict[str, Any]:
        """
        Load configuration overrides from environment variables.
        
        Args:
            config: Base configuration dictionary
            env_prefix: Prefix for environment variables
            
        Returns:
            Configuration with environment overrides applied
        """
        overrides = {}
        
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(env_prefix):].lower()
                
                # Convert value to appropriate type
                if value.lower() in ('true', 'false'):
                    overrides[config_key] = value.lower() == 'true'
                elif value.isdigit():
                    overrides[config_key] = int(value)
                elif value.replace('.', '').isdigit():
                    overrides[config_key] = float(value)
                else:
                    overrides[config_key] = value
        
        if overrides:
            logger.info(f"Loaded {len(overrides)} environment variable overrides")
            return ConfigLoader.merge_configs(config, overrides)
        
        return config


def load_config_with_env(config_path: str, env_prefix: str = "ARABIC_SENTIMENT_") -> Dict[str, Any]:
    """
    Convenience function to load configuration with environment variable overrides.
    
    Args:
        config_path: Path to the configuration file
        env_prefix: Prefix for environment variables
        
    Returns:
        Configuration dictionary with environment overrides applied
    """
    config = ConfigLoader.load_config(config_path)
    return ConfigLoader.load_env_overrides(config, env_prefix)
