#!/usr/bin/env python3
"""
Test script to verify MARBERT fine-tuning setup
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas: {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy: {e}")
        return False
    
    try:
        import sklearn
        print(f"✅ Scikit-learn: {sklearn.__version__}")
    except ImportError as e:
        print(f"❌ Scikit-learn: {e}")
        return False
    
    return True

def test_project_modules():
    """Test if project-specific modules can be imported"""
    print("\nTesting project modules...")
    
    try:
        from data.astd_loader import ASTDDataLoader
        print("✅ ASTDDataLoader imported successfully")
    except ImportError as e:
        print(f"❌ ASTDDataLoader: {e}")
        return False
    
    try:
        from models.fine_tune_marbert import MARBERTFineTuner
        print("✅ MARBERTFineTuner imported successfully")
    except ImportError as e:
        print(f"❌ MARBERTFineTuner: {e}")
        return False
    
    try:
        from utils.config_loader import ConfigLoader
        print("✅ ConfigLoader imported successfully")
    except ImportError as e:
        print(f"❌ ConfigLoader: {e}")
        return False
    
    try:
        from utils.logging_utils import setup_logging
        print("✅ LoggingUtils imported successfully")
    except ImportError as e:
        print(f"❌ LoggingUtils: {e}")
        return False
    
    return True

def test_data_structure():
    """Test if data directory structure exists"""
    print("\nTesting data structure...")
    
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ Data directory does not exist")
        return False
    
    raw_dir = data_dir / "raw"
    if not raw_dir.exists():
        print("❌ Raw data directory does not exist")
        return False
    
    tweets_file = raw_dir / "Tweets.txt"
    if not tweets_file.exists():
        print("❌ Tweets.txt file not found")
        return False
    
    print(f"✅ Data structure verified")
    print(f"   - Data directory: {data_dir}")
    print(f"   - Raw directory: {raw_dir}")
    print(f"   - Tweets file: {tweets_file}")
    
    # Check file size
    file_size = tweets_file.stat().st_size / (1024 * 1024)  # MB
    print(f"   - Tweets file size: {file_size:.2f} MB")
    
    return True

def test_notebook():
    """Test if the notebook was generated"""
    print("\nTesting notebook generation...")
    
    notebook_path = Path("notebooks/marbert_finetuning.ipynb")
    if not notebook_path.exists():
        print("❌ Notebook not found")
        return False
    
    # Check file size
    file_size = notebook_path.stat().st_size / 1024  # KB
    print(f"✅ Notebook generated: {notebook_path}")
    print(f"   - File size: {file_size:.1f} KB")
    
    return True

def test_astd_loader():
    """Test ASTD data loader functionality"""
    print("\nTesting ASTD data loader...")
    
    try:
        from data.astd_loader import ASTDDataLoader
        
        # Initialize loader
        loader = ASTDDataLoader()
        
        # Test validation
        is_valid = loader.validate_dataset()
        if is_valid:
            print("✅ Dataset validation passed")
        else:
            print("❌ Dataset validation failed")
            return False
        
        # Test loading tweets
        tweets_df = loader.load_tweets()
        print(f"✅ Loaded {len(tweets_df)} tweets")
        
        # Test getting splits
        splits = ["4class-balanced-train", "4class-balanced-test", "4class-balanced-validation"]
        for split_name in splits:
            try:
                split_data = loader.get_split_data(split_name, tweets_df)
                print(f"✅ {split_name}: {len(split_data)} samples")
            except Exception as e:
                print(f"❌ Error loading {split_name}: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing ASTD loader: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("MARBERT Fine-tuning Setup Test")
    print("=" * 60)
    
    tests = [
        ("Import Dependencies", test_imports),
        ("Project Modules", test_project_modules),
        ("Data Structure", test_data_structure),
        ("Notebook Generation", test_notebook),
        ("ASTD Data Loader", test_astd_loader),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready for MARBERT fine-tuning.")
        print("\nNext steps:")
        print("1. Open notebooks/marbert_finetuning.ipynb in Google Colab")
        print("2. Or run: python scripts/run_marbert_training.py")
        print("3. Follow the documentation in docs/MARBERT_FINETUNING.md")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\nCommon solutions:")
        print("1. Install missing dependencies: pip install -r requirements_finetuning.txt")
        print("2. Check data directory structure")
        print("3. Verify all source files are present")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
