# SMS Spam Classification with CNN-BiLSTM and Attention

A deep learning-based SMS spam detection system using advanced neural network architectures including CNN-BiLSTM with Attention mechanism. This project achieves high accuracy in classifying SMS messages as spam or legitimate (ham) using state-of-the-art natural language processing techniques.

## Features

- **Advanced Model Architectures**
  - CNN-BiLSTM with Self-Attention (ACB Model)
  - Stacked BiLSTM with Attention
  - Enhanced BiLSTM with regularization

- **Intelligent Text Processing**
  - URL, email, and phone number tokenization
  - CAPS word detection for spam indicators
  - Currency and punctuation normalization
  - Preserves important spam-related features

- **Interactive Web Interface**
  - Real-time SMS classification
  - Batch processing via TXT file upload
  - Confidence score visualization
  - Example messages for testing

- **Model Management**
  - Automatic model selection (improved vs. original)
  - Checkpoint saving during training
  - Model versioning support

## Project Structure

```
CNN-BiLSTM-SMS_Spam_Slassification-Model/
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
├── data/
│   └── raw/
│       ├── spam.csv         # Original SMS spam dataset
│       └── spam_balanced.csv # Balanced dataset for training
├── src/
│   ├── model.py             # Model architectures (CNN-BiLSTM, Attention)
│   ├── preprocessing.py     # Text cleaning and preprocessing
│   ├── train.py            # Training scripts
│   └── utils.py            # Utility functions
└── models/                  # Saved trained models (generated after training)
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd CNN-BiLSTM-SMS_Spam_Slassification-Model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data (required for text preprocessing):
```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

## Usage

### Training the Model

1. Navigate to the source directory:
```bash
cd src
```

2. Train the model:
```bash
python train.py
```

The training script will:
- Load and preprocess the SMS dataset
- Build the CNN-BiLSTM model with attention
- Train with validation split
- Save the best model checkpoints
- Generate performance metrics and visualizations

### Running the Web Application

1. From the project root directory, run:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Use the interface to:
   - Enter SMS messages for classification
   - Upload TXT files for batch processing
   - Try example spam and ham messages
   - View confidence scores and predictions

### Single Message Classification

Enter an SMS message in the text area and click "Classify Message" to get:
- Prediction (Spam or Ham)
- Confidence percentage
- Color-coded results

### Batch Classification

1. Click "Upload .txt file"
2. Select a file with one SMS message per line
3. View results in a table with messages, predictions, and confidence scores

## Model Architecture

### CNN-BiLSTM with Attention (ACB Model)

The improved model combines three powerful components:

1. **Convolutional Neural Network (CNN)**
   - Extracts local features and patterns
   - Multiple filter sizes for different n-grams

2. **Bidirectional LSTM (BiLSTM)**
   - Captures sequential dependencies
   - Processes text in both forward and backward directions

3. **Self-Attention Mechanism**
   - Focuses on important words and phrases
   - Weights different parts of the message based on relevance

### Model Pipeline

```
Input Text
    ↓
Text Preprocessing (URL/Email/Phone tokenization, CAPS detection)
    ↓
Embedding Layer (Word vectors)
    ↓
CNN Layer (Feature extraction)
    ↓
BiLSTM Layer (Sequence processing)
    ↓
Attention Layer (Focus on important features)
    ↓
Dense Layers (Classification)
    ↓
Output (Spam/Ham probability)
```

## Dataset

The project uses SMS spam datasets with balanced classes:
- **spam.csv**: Original dataset
- **spam_balanced.csv**: Balanced version for better training

Dataset features:
- Binary classification (spam vs. ham)
- Real SMS messages
- Multiple spam patterns (promotional, urgent, free offers, etc.)

## Text Preprocessing

Advanced preprocessing pipeline:
- URL replacement with `URL` token
- Email replacement with `EMAIL` token
- Phone number replacement with `PHONE` token
- CAPS word detection (spam indicator)
- Currency amount normalization
- Repeated punctuation handling
- Stopword removal (optional)

## Model Performance

The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC Score

Training includes:
- Train/validation split
- Early stopping
- Model checkpointing
- Learning rate optimization

## Technology Stack

- **Deep Learning**: TensorFlow 2.20, Keras
- **Data Processing**: Pandas, NumPy
- **NLP**: NLTK
- **Web Interface**: Streamlit
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn

## Requirements

```
tensorflow==2.20.0
pandas>=2.2.0
numpy>=2.0.0
nltk>=3.9
streamlit>=1.40.0
scikit-learn>=1.6.0
matplotlib>=3.9.0
seaborn>=0.13.0
```

## Configuration

The model uses the following default parameters:
- Max sequence length: 100 tokens
- Embedding dimension: 128
- LSTM units: 64
- Dropout rate: 0.3-0.5
- Batch size: 32
- Learning rate: Adaptive (Adam optimizer)

These can be adjusted in `src/train.py` and `src/model.py`.

## Examples

### Spam Messages
- "URGENT! you have won 1000$"
- "WINNER!! Click here to claim your FREE prize now!!!"
- "Congratulations! You've been selected for a free iPhone. Call now!"

### Ham Messages
- "Hey! Are we still meeting for lunch at 1pm?"
- "Can you pick up milk on your way home?"
- "Running a bit late, be there in 10 mins"

## Research Background

This implementation is based on research in spam detection using attention-based CNN-BiLSTM architectures, inspired by:
- "Spam review detection using self attention based CNN and bi-directional LSTM"

## Future Improvements

- Multi-language support
- Real-time SMS monitoring
- API endpoints for integration
- Model explainability (LIME/SHAP)
- Mobile application deployment
- Additional datasets for improved generalization

## Troubleshooting

### Model Not Found Error
If you see "No trained model found", ensure you've trained the model first:
```bash
cd src
python train.py
```

### NLTK Data Missing
Download required NLTK data:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

### Memory Issues
If training fails due to memory:
- Reduce batch size in training script
- Reduce max sequence length
- Use a smaller embedding dimension


## Acknowledgments

- SMS Spam Collection Dataset
- TensorFlow and Keras communities
- Research papers on attention mechanisms and spam detection


Built with TensorFlow and Streamlit 
