#!/usr/bin/env python3
"""
Setup script for Arabic Dialect Sentiment Analysis Project

This script installs the project as a Python package and sets up
all necessary dependencies and configurations.
"""

from setuptools import setup, find_packages
import os
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements_path = this_directory / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    requirements = []

# Development dependencies
dev_requirements = [
    'pytest>=7.4.0',
    'pytest-cov>=4.1.0',
    'black>=23.0.0',
    'flake8>=6.0.0',
    'mypy>=1.0.0',
    'pre-commit>=3.0.0',
]

setup(
    name="arabic-dialect-sentiment",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Domain-Adapted Transformer for Gulf Arabic Dialect Sentiment Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/arabic-dialect-sentiment",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/arabic-dialect-sentiment/issues",
        "Documentation": "https://github.com/yourusername/arabic-dialect-sentiment/docs",
        "Source Code": "https://github.com/yourusername/arabic-dialect-sentiment",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "all": requirements + dev_requirements,
    },
    entry_points={
        "console_scripts": [
            "arabic-sentiment-preprocess=src.data.preprocess:main",
            "arabic-sentiment-baseline=src.models.train_baselines:main",
            "arabic-sentiment-dapt=src.models.dapt_pretraining:main",
            "arabic-sentiment-finetune=src.models.fine_tune:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt", "*.md"],
    },
    zip_safe=False,
    keywords=[
        "arabic",
        "sentiment-analysis",
        "nlp",
        "machine-learning",
        "transformer",
        "bert",
        "dialect",
        "gulf-arabic",
        "arabic-nlp",
        "sentiment-classification",
    ],
)
