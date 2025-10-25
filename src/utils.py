import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns


def plot_training_history(history, save_path=None):
    """
    Plot training history (loss and accuracy).

    Args:
        history: Keras training history object
        save_path (str, optional): Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot training & validation accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    # Plot training & validation loss
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")

    plt.show()


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and print metrics.

    Args:
        model: Trained Keras model
        X_test: Test features
        y_test: Test labels

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Make predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrix plot saved to models/confusion_matrix.png")
    plt.show()

    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'predictions': y_pred,
        'probabilities': y_pred_prob
    }


def predict_message(model, tokenizer, message, max_length=100):
    """
    Predict whether a message is spam or ham.

    Args:
        model: Trained Keras model
        tokenizer: Fitted tokenizer
        message (str): Message to classify
        max_length (int): Maximum sequence length

    Returns:
        tuple: (prediction, probability) where prediction is 'Spam' or 'Ham'
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    import re

    # Clean the message
    cleaned_message = message.lower()
    cleaned_message = re.sub(r'http\S+|www\S+|https\S+', '', cleaned_message, flags=re.MULTILINE)
    cleaned_message = re.sub(r'\S+@\S+', '', cleaned_message)
    cleaned_message = re.sub(r'\d+', '', cleaned_message)
    cleaned_message = re.sub(r'\s+', ' ', cleaned_message).strip()

    # Tokenize and pad
    sequence = tokenizer.texts_to_sequences([cleaned_message])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')

    # Predict
    probability = model.predict(padded_sequence, verbose=0)[0][0]
    prediction = 'Spam' if probability > 0.5 else 'Ham'

    return prediction, probability
