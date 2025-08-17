# A Domain-Adapted Transformer for Gulf Arabic Dialect Sentiment Analysis

## Project Overview
This project develops a robust sentiment analysis model specifically for the Gulf Arabic dialect, a low-resource language variant that is poorly served by existing AI models. By leveraging Domain-Adaptive Pretraining (DAPT) and fine-tuning on specialized datasets, we demonstrate clear performance gains over general-purpose Arabic language models.

## Problem Statement
While major progress has been made in sentiment analysis for high-resource languages like English, the same is not true for Arabic dialects. The linguistic differences between Modern Standard Arabic (MSA) and Gulf Arabic pose significant challenges. Existing models trained on MSA fail to capture dialect-specific vocabulary, slang, and orthographic variations, resulting in poor performance for social media analysis in the Gulf region.

## Key Objectives
1. **Data Acquisition and Preprocessing**: Collect and preprocess Gulf Arabic text datasets
2. **Establish Strong Baselines**: Implement traditional ML and transformer-based baselines
3. **Implement Domain-Adaptive Pretraining (DAPT)**: Adapt pre-trained models to Gulf Arabic
4. **Model Fine-tuning and Evaluation**: Fine-tune DAPT-adapted models on sentiment data
5. **Develop User Interface**: Create a web application with RTL support
6. **Produce Research-Grade Report**: Document methodology, results, and implications

## Project Structure
```
arabic-dialect-sentiment/
├── data/                           # Data storage and preprocessing
│   ├── raw/                       # Raw datasets
│   ├── processed/                 # Processed datasets
│   └── external/                  # External datasets (Gumar corpus)
├── models/                        # Model checkpoints and artifacts
├── src/                          # Source code
│   ├── data/                     # Data processing modules
│   ├── models/                   # Model architectures and training
│   ├── evaluation/               # Evaluation and metrics
│   └── utils/                    # Utility functions
├── notebooks/                    # Jupyter notebooks for exploration
├── webapp/                       # Web application
│   ├── backend/                  # FastAPI backend
│   └── frontend/                 # React frontend with RTL support
├── configs/                      # Configuration files
├── scripts/                      # Utility scripts
├── tests/                        # Unit tests
├── docs/                         # Documentation and reports
└── requirements.txt              # Python dependencies
```

## Technical Stack
- **Programming Language**: Python 3.8+
- **Machine Learning**: PyTorch, Hugging Face Transformers
- **Data Manipulation**: Pandas, NumPy
- **Model Architecture**: Transformer-based models (AraBERT, MARBERT, Falcon-Arabic)
- **Backend**: FastAPI
- **Frontend**: React with RTL support
- **Containerization**: Docker

## Quick Start
# Data Folder Structure

This folder contains the Arabic Sentiment Tweets Dataset (ASTD) and related data files.

## File Structure

```
data/
├── README.md                    # This file - data documentation
├── Tweets.txt                   # Main dataset with tweet text and sentiment labels
├── 4class-balanced-train.txt    # Balanced training set line indices
├── 4class-balanced-validation.txt # Balanced validation set line indices  
├── 4class-balanced-test.txt     # Balanced test set line indices
├── 4class-unbalanced-train.txt  # Unbalanced training set line indices
├── 4class-unbalanced-validation.txt # Unbalanced validation set line indices
├── 4class-unbalanced-test.txt   # Unbalanced test set line indices
├── external/                     # External datasets (if any)
├── processed/                    # Processed datasets (cleaned, tokenized)
└── raw/                         # Raw data backups
```

## Dataset Description

### ASTD (Arabic Sentiment Tweets Dataset)

The main dataset file `Tweets.txt` contains Arabic tweets with sentiment labels:

- **Format**: Tab-separated values (TSV)
- **Columns**: 
  - Tweet text (Arabic)
  - Sentiment label
- **Labels**:
  - `POS`: Positive sentiment
  - `NEG`: Negative sentiment  
  - `NEUTRAL`: Neutral sentiment
  - `OBJ`: Objective/No sentiment

### Dataset Splits

The dataset provides both balanced and unbalanced splits:

- **Balanced**: Equal representation of all sentiment classes
- **Unbalanced**: Natural distribution of sentiment classes
- **Splits**: Train (70%), Validation (15%), Test (15%)

### Line Index Files

The numbered files (e.g., `4class-balanced-train.txt`) contain line indices that reference specific tweets in the main `Tweets.txt` file. To get a specific tweet:

1. Read the line number from the split file
2. Use that line number to index into `Tweets.txt`
3. Extract the tweet text and sentiment label

## Usage

This dataset is used for:
- Training sentiment analysis models
- Evaluating model performance
- Benchmarking Arabic NLP systems
- Research in Arabic dialect sentiment analysis

## Preprocessing

The data goes through several preprocessing steps:
1. Text cleaning and normalization
2. Label mapping to integers
3. Tokenization using MARBERT
4. Dataset splitting and validation
5. Class weight calculation for imbalanced learning

### Prerequisites
- Python 3.8+
- Git
- Docker (optional)

### Installation
1. Clone the repository:
```bash
git clone <repository-url>
cd arabic-dialect-sentiment
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

### Running the Project

#### Data Preprocessing
```bash
python src/data/preprocess.py --config configs/data_config.yaml
```

#### Training Baselines
```bash
python src/models/train_baselines.py --config configs/baseline_config.yaml
```

#### Domain-Adaptive Pretraining
```bash
python src/models/dapt_pretraining.py --config configs/dapt_config.yaml
```

#### Fine-tuning
```bash
python src/models/fine_tune.py --config configs/fine_tune_config.yaml
```

#### Web Application
```bash
# Backend
cd webapp/backend
uvicorn main:app --reload

# Frontend (in another terminal)
cd webapp/frontend
npm install
npm start
```

## Project Phases

### Phase 1: Project Setup & Baselines (Weeks 1-2)
- [x] Initial project repository with clear structure
- [ ] Data preprocessing scripts and clean dataset
- [ ] Traditional ML baseline model
- [ ] Transformer-based baseline model

### Phase 2: Domain Adaptation & Fine-tuning (Weeks 3-4)
- [ ] DAPT model on Gumar corpus
- [ ] Final fine-tuned sentiment analysis model
- [ ] Comprehensive results logging

### Phase 3: Deployment & Explainability (Week 5)
- [ ] Backend API with inference endpoints
- [ ] Interactive frontend with RTL support
- [ ] Model explainability implementation

### Phase 4: Documentation & Finalization (Week 6)
- [ ] Academic-style report (4-6 pages)
- [ ] Model card with ethical considerations
- [ ] Demo video (1-2 minutes)
- [ ] Clean public GitHub repository

## Contributing
This is a research project. Please refer to the contributing guidelines in `CONTRIBUTING.md`.

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.

## Citation
If you use this work in your research, please cite:
```bibtex
@article{gulf_arabic_sentiment_2024,
  title={A Domain-Adapted Transformer for Gulf Arabic Dialect Sentiment Analysis},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

