"""
Improved training script for SMS spam detection.

Key improvements:
1. Uses improved preprocessing (keeps spam indicators)
2. CNN-BiLSTM-Attention model architecture
3. Class balancing for imbalanced datasets
4. Better training parameters and callbacks
5. More comprehensive evaluation
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from preprocessing_improved import ImprovedDataPreprocessor
from model_improved import get_model
from utils import plot_training_history, evaluate_model
from datetime import datetime

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def main():
    """Main training function with improved pipeline."""

    print("\n" + "="*80)
    print(" "*25 + "IMPROVED SMS SPAM CLASSIFIER")
    print("="*80)

    # ============================================================================
    # CONFIGURATION
    # ============================================================================

    # Model selection
    print("\nAvailable models:")
    print("1. cnn_bilstm_attention - CNN-BiLSTM with Attention (RECOMMENDED - Best accuracy)")
    print("2. enhanced_bilstm - Enhanced BiLSTM with Attention (Faster training)")
    print("3. transformer - Transformer-inspired model (Most advanced)")

    model_choice = input("\nSelect model (1/2/3) or press Enter for default [1]: ").strip()
    model_map = {
        '1': 'cnn_bilstm_attention',
        '2': 'enhanced_bilstm',
        '3': 'transformer',
        '': 'cnn_bilstm_attention'
    }
    MODEL_TYPE = model_map.get(model_choice, 'cnn_bilstm_attention')

    # Dataset selection
    print("\nAvailable datasets:")
    print("1. data/raw/spam.csv - Original UCI dataset (5,574 messages)")
    print("2. data/raw/spam_balanced.csv - Balanced UCI dataset")

    data_choice = input("\nSelect dataset (1/2/3) or press Enter for default [1]: ").strip()
    data_map = {
        '1': 'data/raw/spam.csv',
        '2': 'data/raw/spam_balanced.csv',
        '3': 'data/raw/spam_large.csv',
        '': 'data/raw/spam.csv'
    }
    DATA_PATH = data_map.get(data_choice, 'data/raw/spam.csv')

    # Check if file exists
    if not os.path.exists(DATA_PATH):
        print(f"\nError: Dataset not found at {DATA_PATH}")
        if 'large' in DATA_PATH:
            print("Run: python download_dataset.py to download the large dataset")
        elif 'balanced' in DATA_PATH:
            print("Run: python download_dataset.py to create balanced dataset")
        sys.exit(1)

    # Model parameters
    VOCAB_SIZE = 20000
    MAX_LENGTH = 100
    EMBEDDING_DIM = 128

    # Training parameters
    EPOCHS = 30
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2

    # Paths
    MODEL_PATH = 'models/spam_classifier_improved.h5'
    TOKENIZER_PATH = 'models/tokenizer_improved.pickle'

    # Create models directory
    os.makedirs('models', exist_ok=True)
    os.makedirs('models/logs', exist_ok=True)

    print("\n" + "="*80)
    print("CONFIGURATION SUMMARY")
    print("="*80)
    print(f"Model type: {MODEL_TYPE}")
    print(f"Dataset: {DATA_PATH}")
    print(f"Vocabulary size: {VOCAB_SIZE}")
    print(f"Max sequence length: {MAX_LENGTH}")
    print(f"Embedding dimension: {EMBEDDING_DIM}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Validation split: {VALIDATION_SPLIT}")
    print("="*80)

    # ============================================================================
    # STEP 1: Load and preprocess data
    # ============================================================================

    print("\n" + "="*80)
    print("STEP 1: LOADING AND PREPROCESSING DATA")
    print("="*80)

    preprocessor = ImprovedDataPreprocessor(
        vocab_size=VOCAB_SIZE,
        max_length=MAX_LENGTH,
        test_size=0.2,
        random_state=42
    )

    X_train, X_test, y_train, y_test = preprocessor.load_and_prepare_data(DATA_PATH)

    # ============================================================================
    # STEP 2: Tokenize and pad sequences
    # ============================================================================

    print("\n" + "="*80)
    print("STEP 2: TOKENIZING AND PADDING SEQUENCES")
    print("="*80)

    X_train_padded, X_test_padded = preprocessor.tokenize_and_pad(X_train, X_test)

    # Save tokenizer
    preprocessor.save_tokenizer(TOKENIZER_PATH)

    # Get class weights for imbalanced data
    class_weights = preprocessor.get_class_weights()

    # ============================================================================
    # STEP 3: Build model
    # ============================================================================

    print("\n" + "="*80)
    print("STEP 3: BUILDING MODEL")
    print("="*80)

    model = get_model(
        model_type=MODEL_TYPE,
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        max_length=MAX_LENGTH
    )

    print("\nModel Summary:")
    model.summary()

    print(f"\nTotal parameters: {model.count_params():,}")

    # ============================================================================
    # STEP 4: Train model
    # ============================================================================

    print("\n" + "="*80)
    print("STEP 4: TRAINING MODEL")
    print("="*80)

    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    callbacks = [
        # Early stopping with more patience for complex models
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),

        # Save best model
        ModelCheckpoint(
            MODEL_PATH.replace('.h5', '_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),

        # Reduce learning rate when learning plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1,
            min_lr=1e-7
        ),

        # TensorBoard logging
        TensorBoard(
            log_dir=f'models/logs/{MODEL_TYPE}_{timestamp}',
            histogram_freq=1
        )
    ]

    print(f"\nStarting training with class weights: {class_weights}")
    print("This may take a while...\n")

    # Train the model
    history = model.fit(
        X_train_padded, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        class_weight=class_weights,  # Handle class imbalance
        verbose=1
    )

    # ============================================================================
    # STEP 5: Save model
    # ============================================================================

    print("\n" + "="*80)
    print("STEP 5: SAVING MODEL")
    print("="*80)

    model.save(MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

    # ============================================================================
    # STEP 6: Evaluate model
    # ============================================================================

    print("\n" + "="*80)
    print("STEP 6: EVALUATING MODEL ON TEST SET")
    print("="*80)

    metrics = evaluate_model(model, X_test_padded, y_test)

    # ============================================================================
    # STEP 7: Plot training history
    # ============================================================================

    print("\n" + "="*80)
    print("STEP 7: PLOTTING TRAINING HISTORY")
    print("="*80)

    plot_training_history(history, save_path='models/training_history_improved.png')

    # ============================================================================
    # STEP 8: Test on example messages
    # ============================================================================

    print("\n" + "="*80)
    print("STEP 8: TESTING ON EXAMPLE MESSAGES")
    print("="*80)

    test_messages = [
        "URGENT! you have won 1000$",
        "Hey, are we still meeting for lunch?",
        "WINNER!! Click here to claim your FREE prize now!!!",
        "Can you pick up milk on your way home?",
        "Congratulations! You've been selected for a free iPhone. Call now!",
        "Running a bit late, be there in 10 mins"
    ]

    from utils import predict_message

    print("\nTesting model predictions:\n")
    for msg in test_messages:
        prediction, probability = predict_message(model, preprocessor.tokenizer, msg, MAX_LENGTH)
        print(f"Message: {msg}")
        print(f"Prediction: {prediction} (confidence: {probability*100 if prediction == 'Spam' else (1-probability)*100:.2f}%)")
        print("-" * 80)

    # ============================================================================
    # COMPLETION
    # ============================================================================

    print("\n" + "="*80)
    print(" "*30 + "TRAINING COMPLETED!")
    print("="*80)
    print(f"\nFinal Test Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Tokenizer saved to: {TOKENIZER_PATH}")
    print(f"\nTo view TensorBoard logs:")
    print(f"  tensorboard --logdir=models/logs")
    print(f"\nTo run the Streamlit app:")
    print(f"  streamlit run app.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
