# Sarcasm Detection Project

This project implements a sarcasm detection system using BERT (Bidirectional Encoder Representations from Transformers) for text classification.

## Project Overview

The goal of this project is to detect sarcasm in text using machine learning techniques. We use BERT, a state-of-the-art transformer model, to classify text as either sarcastic or non-sarcastic.

## Features

- Text preprocessing and cleaning
- BERT-based model architecture
- Binary classification (sarcastic vs non-sarcastic)
- Comprehensive evaluation metrics

## Technical Stack

- **Framework**: PyTorch
- **Model**: BERT-base-uncased
- **Libraries**:
  - pandas: Data manipulation
  - transformers: BERT implementation
  - scikit-learn: Evaluation metrics
  - torch: Deep learning framework

## Setup

1. Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your data in CSV format with 'comments' and 'contains_slash_s' columns
2. Run the main script:

```bash
python dataset/sarcasm/main.py
```

## Model Architecture

- Base Model: BERT-base-uncased
- Classification Head:
  - Input: 768-dimensional BERT embeddings
  - Output: 2 classes (sarcastic/non-sarcastic)
- Training Parameters:
  - Batch size: 16
  - Learning rate: 1e-5
  - Epochs: 3

## Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score

## Future Improvements

- Try different BERT variants
- Experiment with different hyperparameters
- Add more preprocessing steps
- Implement cross-validation
- Add data augmentation

## License

MIT License
