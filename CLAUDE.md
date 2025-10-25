# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SMS spam detection system using deep learning with CNN-BiLSTM-Attention architecture. The system preserves spam indicators (numbers, currency symbols, exclamation marks, CAPS patterns) during preprocessing, unlike traditional approaches that strip these features.

## Development Commands

### Training Models

```bash
cd src
python train.py
# Interactive prompts:
# - Model: 1=CNN-BiLSTM-Attention (best), 2=Enhanced BiLSTM, 3=Transformer
# - Dataset: 1=original UCI, 2=balanced, 3=large (138k messages)
```

### Running the Application

```bash
# Run Streamlit app (auto-detects available model)
streamlit run app.py

# View TensorBoard training logs
tensorboard --logdir=models/logs
```

### Dataset Management

The training scripts reference multiple dataset options:
- `data/raw/spam.csv` - Original UCI dataset (5,574 messages)
- `data/raw/spam_balanced.csv` - Balanced version
- `data/raw/spam_large.csv` - Large dataset (138k messages)

Training script will prompt if dataset not found. References `download_dataset.py` for obtaining datasets (not currently in repo).

## Architecture Overview

### Preprocessing Pipeline

Implementation in `src/preprocessing.py` (class `ImprovedDataPreprocessor`):

**Key approach** - preserves spam indicators instead of removing them:
- **Keeps spam signals**: numbers, $, !, ?, CAPS patterns
- Replaces URLs/emails/phones with tokens (URL, EMAIL, PHONE) instead of removing
- Adds CAPS_ prefix to all-uppercase words (e.g., "URGENT" → "caps_urgent")
- Normalizes repeated punctuation (!!!! → !!) but keeps them
- Adds spaces around special chars for separate tokenization

Example transformation:
```
Input:  "URGENT! you have won 1000$"
Output: "caps_urgent ! you have won 1000 $"
```

See `src/preprocessing.py:39-89` for the `clean_text()` method.

### Model Architectures

Located in `src/model.py` (class `SpamDetectorModels`):

1. **CNN-BiLSTM-Attention (ACB Model)** - Recommended
   - Multi-scale CNN (2,3,4,5-gram filters) for local feature extraction
   - 2-layer Bidirectional LSTM (256→128 units) for sequential context
   - Custom AttentionLayer (self-attention mechanism)
   - 2 dense layers (256→128) with dropout
   - See `src/model.py:84-167`

2. **Enhanced BiLSTM**
   - Stacked 2-layer BiLSTM without CNN component
   - Faster training, slightly lower accuracy
   - See `src/model.py:170-229`

3. **Transformer-Inspired**
   - MultiHeadAttention + BiLSTM hybrid
   - Most advanced but slower
   - See `src/model.py:232-298`

### Custom AttentionLayer

Implemented in both `src/model.py:22-77` and `app.py:115-165`.

**Critical Implementation Detail**:
The attention layer uses `tf.reduce_sum(uit * self.u, axis=-1)` instead of `K.dot(uit, self.u)` to compute attention scores. The dot product approach causes IndexError with TensorFlow 2.x when handling 3D tensors (batch_size, seq_len, features) × 1D vector (features). This was fixed by using element-wise multiplication followed by sum.

**Model Loading**: When loading saved models with attention, must pass `custom_objects={'AttentionLayer': AttentionLayer}` to `load_model()`.

The layer computes attention scores using:
```python
uit = tanh(W·x + b)
ait = tf.reduce_sum(uit * self.u, axis=-1)  # NOT K.dot(uit, self.u)
ait = softmax(ait)
output = Σ(ait * x)
```

### Training Pipeline

Key features in `src/train.py`:

1. **Class Weight Balancing**:
   - Computes balanced class weights for imbalanced datasets using `sklearn.utils.class_weight`
   - Spam is typically 13% of data, so gets ~6.5x higher weight
   - Applied via `class_weight` parameter in `model.fit()`

2. **Callbacks**:
   - EarlyStopping (monitor='val_loss', patience=5)
   - ModelCheckpoint (saves best model based on val_accuracy)
   - ReduceLROnPlateau (factor=0.5, patience=3, min_lr=1e-7)
   - TensorBoard logging with timestamp

3. **Evaluation**:
   - Uses `src/utils.py` functions:
     - `evaluate_model()` - computes metrics, confusion matrix, classification report
     - `plot_training_history()` - plots accuracy/loss curves
     - `predict_message()` - single message prediction helper

   **Note**: `utils.py:predict_message()` uses legacy cleaning (removes numbers/special chars). This is inconsistent with the preprocessing used during training and may cause prediction errors on spam indicators. The Streamlit app uses the correct cleaning method.

### Application Structure

`app.py` implements smart model loading:

1. **Cascading Model Search** (line 74-82):
   - Tries multiple model paths in priority order:
     1. `spam_classifier_improved_best.h5` (best checkpoint)
     2. `spam_classifier_improved.h5` (final model)
     3. `spam_classifier_best.h5` (legacy best)
     4. `spam_classifier.h5` (legacy final)
   - Handles models with/without custom attention layer

2. **Text Cleaning** (line 160-199):
   - Uses spam-indicator-preserving cleaning (same as preprocessing)
   - Duplicates the `ImprovedDataPreprocessor.clean_text()` logic for consistency

## Key Implementation Patterns

### Code Organization

All files follow the "improved" approach (spam-indicator-preserving preprocessing + advanced models):
- `src/preprocessing.py` - Contains `ImprovedDataPreprocessor` class
- `src/model.py` - Contains `SpamDetectorModels` factory class with multiple architectures
- `src/train.py` - Interactive training script with model/dataset selection
- `src/utils.py` - Shared utilities for evaluation and plotting
- `app.py` - Streamlit application with cascading model search

### Model Persistence

**Training outputs**:
- `models/spam_classifier_improved.h5` - final model
- `models/spam_classifier_improved_best.h5` - best checkpoint
- `models/tokenizer_improved.pickle` - fitted tokenizer
- `models/training_history_improved.png` - plots
- `models/logs/{model}_{timestamp}/` - TensorBoard logs

**App loading priority**:
1. `spam_classifier_improved_best.h5` (best checkpoint)
2. `spam_classifier_improved.h5` (final model)
3. `spam_classifier_best.h5` (original best)
4. `spam_classifier.h5` (original final)

### Vocabulary and Tokenization

Configuration (typically in training scripts):
- `VOCAB_SIZE = 20000` - top N most frequent words
- `MAX_LENGTH = 100` - sequence padding/truncation length
- `EMBEDDING_DIM = 128` - word embedding dimension
- `lower=False` in Tokenizer because cleaning already lowercased

Special tokens:
- `<OOV>` - out-of-vocabulary token
- `CAPS_` prefix - indicates original word was all-caps
- `URL`, `EMAIL`, `PHONE` - placeholders for removed PII

## Common Issues and Solutions

1. **AttentionLayer IndexError**:
   - **Symptom**: `IndexError: pop index out of range` in `AttentionLayer.call()`
   - **Cause**: Using `K.dot(uit, self.u)` for 3D × 1D tensor operation
   - **Solution**: Use `tf.reduce_sum(uit * self.u, axis=-1)` instead
   - Already fixed in `src/model.py:63` and `app.py:152`

2. **Dataset not found**:
   - Training script expects datasets in `data/raw/`:
     - `spam.csv` (UCI dataset - 5,574 messages)
     - `spam_balanced.csv` (balanced version)
     - `spam_large.csv` (138k messages)
   - If missing, script will prompt to run `download_dataset.py` (not in repo)

3. **Model loading errors**:
   - Must pass `custom_objects={'AttentionLayer': AttentionLayer}` when loading models
   - The app handles this automatically in the cascading search
   - Ensure tokenizer pickle version matches model (improved vs legacy)

4. **Inconsistent text cleaning in utils.py**:
   - `utils.py:predict_message()` uses legacy cleaning (line 110-114) that strips spam indicators
   - This differs from `preprocessing.py:clean_text()` used during training
   - For accurate predictions, update `predict_message()` to use `ImprovedDataPreprocessor.clean_text()`
   - Or import and use the cleaning function from `app.py:clean_text_improved()`
