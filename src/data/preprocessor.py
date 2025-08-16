"""
Arabic Text Preprocessor for Gulf Arabic Dialect Sentiment Analysis

This module provides comprehensive text preprocessing functionality specifically
designed for Gulf Arabic dialects, including text cleaning, normalization,
and dialect-specific handling.
"""

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
import logging
from pathlib import Path
import json

# Arabic-specific imports
try:
    import pyarabic.araby as araby
    import tashkeel
    HAS_ARABIC_LIBS = True
except ImportError:
    HAS_ARABIC_LIBS = False
    logging.warning("Arabic libraries not available. Some features may be limited.")

logger = logging.getLogger(__name__)


class ArabicTextPreprocessor:
    """
    A comprehensive text preprocessor for Arabic dialects, especially Gulf Arabic.
    
    This class handles:
    - Arabic text normalization
    - Dialect-specific text cleaning
    - URL, email, and special character removal
    - Text length filtering
    - Dialect identification
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the preprocessor with configuration.
        
        Args:
            config: Configuration dictionary containing preprocessing parameters
        """
        self.config = config
        self.setup_logging()
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
        
        # Arabic character sets
        self.arabic_chars = set('ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي')
        self.arabic_numbers = set('٠١٢٣٤٥٦٧٨٩')
        self.english_numbers = set('0123456789')
        
    def _compile_patterns(self):
        """Compile regex patterns for efficient text processing."""
        # URL pattern
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
        # Email pattern
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        
        # English text pattern (for mixed Arabic-English text)
        self.english_pattern = re.compile(r'[a-zA-Z]+')
        
        # Emoji pattern
        self.emoji_pattern = re.compile(
            r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]'
        )
        
        # Multiple spaces pattern
        self.multiple_spaces_pattern = re.compile(r'\s+')
        
        # Special characters pattern (keeping Arabic punctuation)
        self.special_chars_pattern = re.compile(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s\w]')
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.config.get('logging', {}).get('level', 'INFO'))
        logging.basicConfig(level=log_level)
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize Arabic text.
        
        Args:
            text: Input Arabic text
            
        Returns:
            Cleaned and normalized text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to string if needed
        text = str(text).strip()
        
        # Apply cleaning steps based on configuration
        if self.config.get('preprocessing', {}).get('remove_urls', True):
            text = self.remove_urls(text)
            
        if self.config.get('preprocessing', {}).get('remove_emails', True):
            text = self.remove_emails(text)
            
        if self.config.get('preprocessing', {}).get('remove_emojis', False):
            text = self.remove_emojis(text)
            
        if self.config.get('preprocessing', {}).get('normalize_arabic', True):
            text = self.normalize_arabic_text(text)
            
        if self.config.get('preprocessing', {}).get('normalize_arabic_numbers', True):
            text = self.normalize_arabic_numbers(text)
            
        # Remove multiple spaces
        text = self.multiple_spaces_pattern.sub(' ', text)
        
        # Final strip
        text = text.strip()
        
        return text
    
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        return self.url_pattern.sub('', text)
    
    def remove_emails(self, text: str) -> str:
        """Remove email addresses from text."""
        return self.email_pattern.sub('', text)
    
    def remove_emojis(self, text: str) -> str:
        """Remove emojis from text."""
        return self.emoji_pattern.sub('', text)
    
    def normalize_arabic_text(self, text: str) -> str:
        """
        Normalize Arabic text using pyarabic library.
        
        Args:
            text: Input Arabic text
            
        Returns:
            Normalized Arabic text
        """
        if not HAS_ARABIC_LIBS:
            logger.warning("pyarabic not available, skipping Arabic normalization")
            return text
        
        try:
            # Normalize Arabic characters
            text = araby.normalize_hamza(text)
            text = araby.normalize_lamalef(text)
            text = araby.normalize_tah(text)
            
            # Remove tashkeel if configured
            if self.config.get('preprocessing', {}).get('remove_tashkeel', False):
                text = araby.strip_tashkeel(text)
            
            # Normalize spaces around Arabic text
            text = araby.normalize_spaces(text)
            
        except Exception as e:
            logger.warning(f"Error during Arabic normalization: {e}")
        
        return text
    
    def normalize_arabic_numbers(self, text: str) -> str:
        """
        Normalize Arabic numbers to English numbers.
        
        Args:
            text: Input text with Arabic numbers
            
        Returns:
            Text with normalized numbers
        """
        if not self.config.get('preprocessing', {}).get('normalize_arabic_numbers', True):
            return text
        
        # Arabic to English number mapping
        arabic_to_english = {
            '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
            '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
        }
        
        for arabic, english in arabic_to_english.items():
            text = text.replace(arabic, english)
        
        return text
    
    def filter_by_length(self, text: str) -> bool:
        """
        Check if text meets length requirements.
        
        Args:
            text: Input text
            
        Returns:
            True if text meets length requirements, False otherwise
        """
        min_length = self.config.get('preprocessing', {}).get('min_length', 3)
        max_length = self.config.get('preprocessing', {}).get('max_length', 512)
        
        text_length = len(text.strip())
        return min_length <= text_length <= max_length
    
    def identify_dialect(self, text: str) -> str:
        """
        Identify the Arabic dialect in the text.
        This is a simplified dialect identification based on common patterns.
        
        Args:
            text: Input Arabic text
            
        Returns:
            Identified dialect (gulf, egyptian, levantine, msa, unknown)
        """
        # Gulf Arabic patterns
        gulf_patterns = [
            r'عشان', r'عندي', r'عندك', r'عنده', r'عندها', r'عندهم',
            r'شلون', r'شخبارك', r'شخبارج', r'شخبارهم',
            r'هذا', r'هذي', r'هذول', r'هذولي',
            r'الي', r'اللي', r'الللي'
        ]
        
        # Egyptian Arabic patterns
        egyptian_patterns = [
            r'ازاي', r'عايز', r'عايزة', r'عايزين',
            r'مش', r'مش عارف', r'مش فاهم',
            r'احنا', r'احنا بنروح'
        ]
        
        # Levantine Arabic patterns
        levantine_patterns = [
            r'شو', r'شو عم', r'شو عم تعمل',
            r'عم', r'عم اتعلم', r'عم ادرس',
            r'بس', r'بس خلص'
        ]
        
        # Count pattern matches
        gulf_count = sum(len(re.findall(pattern, text)) for pattern in gulf_patterns)
        egyptian_count = sum(len(re.findall(pattern, text)) for pattern in egyptian_patterns)
        levantine_count = sum(len(re.findall(pattern, text)) for pattern in levantine_patterns)
        
        # Determine dialect based on pattern frequency
        if gulf_count > egyptian_count and gulf_count > levantine_count:
            return 'gulf'
        elif egyptian_count > gulf_count and egyptian_count > levantine_count:
            return 'egyptian'
        elif levantine_count > gulf_count and levantine_count > egyptian_count:
            return 'levantine'
        else:
            return 'msa'  # Modern Standard Arabic
    
    def preprocess_dataset(self, data: Union[pd.DataFrame, str], 
                          text_column: str = 'text',
                          dialect_column: str = 'dialect',
                          label_column: str = 'label') -> pd.DataFrame:
        """
        Preprocess an entire dataset.
        
        Args:
            data: Input data (DataFrame or path to CSV)
            text_column: Name of the text column
            dialect_column: Name of the dialect column
            label_column: Name of the label column
            
        Returns:
            Preprocessed DataFrame
        """
        # Load data if path is provided
        if isinstance(data, str):
            data = pd.read_csv(data)
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a DataFrame or path to CSV file")
        
        # Validate required columns
        required_columns = self.config.get('validation', {}).get('required_columns', [])
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        logger.info(f"Starting preprocessing of {len(data)} samples")
        
        # Clean text
        data['cleaned_text'] = data[text_column].apply(self.clean_text)
        
        # Filter by length
        initial_length = len(data)
        data = data[data['cleaned_text'].apply(self.filter_by_length)]
        filtered_length = len(data)
        
        logger.info(f"Filtered {initial_length - filtered_length} samples due to length constraints")
        
        # Identify dialect if not present
        if dialect_column not in data.columns:
            data[dialect_column] = data['cleaned_text'].apply(self.identify_dialect)
            logger.info("Added dialect identification column")
        
        # Remove rows with empty cleaned text
        data = data[data['cleaned_text'].str.len() > 0]
        
        # Reset index
        data = data.reset_index(drop=True)
        
        logger.info(f"Preprocessing completed. Final dataset size: {len(data)}")
        
        return data
    
    def save_preprocessing_stats(self, data: pd.DataFrame, 
                               output_path: str) -> Dict:
        """
        Save preprocessing statistics to a JSON file.
        
        Args:
            data: Preprocessed DataFrame
            output_path: Path to save statistics
            
        Returns:
            Dictionary containing preprocessing statistics
        """
        stats = {
            'total_samples': len(data),
            'text_length_stats': {
                'min': data['cleaned_text'].str.len().min(),
                'max': data['cleaned_text'].str.len().max(),
                'mean': data['cleaned_text'].str.len().mean(),
                'median': data['cleaned_text'].str.len().median()
            },
            'dialect_distribution': data.get('dialect', pd.Series()).value_counts().to_dict(),
            'label_distribution': data.get('label', pd.Series()).value_counts().to_dict() if 'label' in data.columns else {},
            'preprocessing_config': self.config
        }
        
        # Save statistics
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Preprocessing statistics saved to {output_path}")
        return stats


def main():
    """Main function for command-line usage."""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description='Arabic Text Preprocessor')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to configuration file')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input CSV file')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output CSV file')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Initialize preprocessor
    preprocessor = ArabicTextPreprocessor(config)
    
    # Preprocess data
    data = preprocessor.preprocess_dataset(args.input)
    
    # Save preprocessed data
    data.to_csv(args.output, index=False, encoding='utf-8')
    logger.info(f"Preprocessed data saved to {args.output}")
    
    # Save statistics if configured
    if config.get('logging', {}).get('save_preprocessing_stats', True):
        stats_path = config.get('logging', {}).get('stats_output_path', 'preprocessing_stats.json')
        preprocessor.save_preprocessing_stats(data, stats_path)


if __name__ == "__main__":
    main()
