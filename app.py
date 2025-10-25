"""
Improved Streamlit app for SMS spam detection.

Works with both original and improved models.
Automatically detects which model is available.
"""

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import os

# Page configuration
st.set_page_config(
    page_title="SMS Spam Detector - Improved",
    page_icon="📱",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
    .spam-result {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
    }
    .spam {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #ef5350;
    }
    .ham {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 2px solid #66bb6a;
    }
    .confidence {
        font-size: 18px;
        color: #555;
        margin-top: 10px;
    }
    .model-info {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_tokenizer():
    """
    Load the trained model and tokenizer.
    Tries improved model first, falls back to original.
    """
    # Try improved model first
    model_paths = [
        ('models/spam_classifier_improved_best.h5', 'models/tokenizer_improved.pickle', 'Improved CNN-BiLSTM-Attention'),
        ('models/spam_classifier_improved.h5', 'models/tokenizer_improved.pickle', 'Improved Model'),
        ('models/spam_classifier_best.h5', 'models/tokenizer.pickle', 'Original BiLSTM'),
        ('models/spam_classifier.h5', 'models/tokenizer.pickle', 'Original Model')
    ]

    for model_path, tokenizer_path, model_name in model_paths:
        if os.path.exists(model_path) and os.path.exists(tokenizer_path):
            try:
                # Load model with custom objects for attention layer
                model = load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer}, compile=False)
                model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )

                # Load tokenizer
                with open(tokenizer_path, 'rb') as f:
                    tokenizer = pickle.load(f)

                return model, tokenizer, model_name
            except Exception as e:
                # If loading fails (e.g., no custom layer needed), try without
                try:
                    model = load_model(model_path)
                    with open(tokenizer_path, 'rb') as f:
                        tokenizer = pickle.load(f)
                    return model, tokenizer, model_name
                except:
                    continue

    st.error("No trained model found. Please train a model first by running:")
    st.code("cd src && python train_improved.py")
    st.code("OR")
    st.code("cd src && python train.py")
    st.stop()


# Custom attention layer for loading improved models
class AttentionLayer(tf.keras.layers.Layer):
    """Self-attention layer."""

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_vector',
            shape=(input_shape[-1],),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        from tensorflow.keras import backend as K
        # Compute attention scores
        # x shape: (batch_size, seq_len, features)
        uit = tf.tanh(K.dot(x, self.W) + self.b)

        # Compute attention weights using einsum for proper dimension handling
        # uit shape: (batch_size, seq_len, features)
        # self.u shape: (features,)
        # ait shape: (batch_size, seq_len)
        ait = tf.reduce_sum(uit * self.u, axis=-1)

        # Apply softmax to get attention weights
        ait = K.exp(ait)
        ait = ait / (K.sum(ait, axis=1, keepdims=True) + K.epsilon())
        ait = K.expand_dims(ait, axis=-1)

        # Apply attention weights
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super(AttentionLayer, self).get_config()


def clean_text_improved(text):
    """
    Improved text cleaning that preserves spam indicators.
    """
    if not isinstance(text, str):
        return ""

    # Replace URLs with token
    text = re.sub(r'http\S+|www\S+|https\S+', ' URL ', text, flags=re.MULTILINE)

    # Replace email with token
    text = re.sub(r'\S+@\S+', ' EMAIL ', text)

    # Replace phone numbers with token
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', ' PHONE ', text)

    # Normalize repeated punctuation
    text = re.sub(r'([!?.]){3,}', r'\1\1', text)

    # Add space around special characters
    text = re.sub(r'([!$£€¥?.])', r' \1 ', text)

    # Handle currency amounts
    text = re.sub(r'([$£€¥])\s+(\d+)', r'\1\2', text)

    # Preserve CAPS words
    words = text.split()
    processed_words = []
    for word in words:
        if len(word) > 2 and word.isupper() and word.isalpha():
            processed_words.append('CAPS_' + word.lower())
        else:
            processed_words.append(word.lower())

    text = ' '.join(processed_words)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def clean_text_original(text):
    """
    Original text cleaning (for backward compatibility).
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def predict_spam(model, tokenizer, message, model_name, max_length=100):
    """
    Predict whether a message is spam or ham.
    """
    # Use improved cleaning if it's an improved model
    if 'Improved' in model_name or 'CNN-BiLSTM' in model_name:
        cleaned_message = clean_text_improved(message)
    else:
        cleaned_message = clean_text_original(message)

    # Tokenize and pad
    sequence = tokenizer.texts_to_sequences([cleaned_message])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')

    # Predict
    probability = model.predict(padded_sequence, verbose=0)[0][0]
    prediction = 'Spam' if probability > 0.5 else 'Ham'

    return prediction, probability


def main():
    """Main Streamlit app."""

    # Initialize session state for message input
    if 'message' not in st.session_state:
        st.session_state.message = ""

    # Title and description
    st.title("📱 SMS Spam Detector - Improved")
    st.markdown("""
    This application uses an **advanced deep learning model** to classify SMS messages as **Spam** or **Ham** (legitimate).
    The improved model uses CNN-BiLSTM architecture with attention mechanism for better accuracy.
    """)

    st.markdown("---")

    # Load model and tokenizer
    with st.spinner("Loading model..."):
        model, tokenizer, model_name = load_model_and_tokenizer()

    # Display model info
    st.markdown(f"""
        <div class="model-info">
            ℹ️ Using model: <strong>{model_name}</strong>
        </div>
    """, unsafe_allow_html=True)

    # Input section
    st.subheader("Enter SMS Message")
    message = st.text_area(
        label="Message",
        value=st.session_state.message,
        placeholder="Type or paste your SMS message here...",
        height=150,
        label_visibility="collapsed"
    )

    # Update session state when user types
    if message != st.session_state.message:
        st.session_state.message = message

    # Classify button
    if st.button("🔍 Classify Message", type="primary", use_container_width=True):
        if message.strip() == "":
            st.warning("Please enter a message to classify.")
        else:
            with st.spinner("Analyzing message..."):
                # Make prediction
                prediction, probability = predict_spam(model, tokenizer, message, model_name)

                # Calculate confidence
                confidence = probability if prediction == 'Spam' else (1 - probability)
                confidence_percent = confidence * 100

                # Display result
                st.markdown("---")
                st.subheader("Classification Result")

                if prediction == 'Spam':
                    st.markdown(f"""
                        <div class="spam-result spam">
                            🚨 SPAM
                        </div>
                        <div class="confidence">
                            Confidence: {confidence_percent:.2f}%
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="spam-result ham">
                            ✅ HAM (Legitimate)
                        </div>
                        <div class="confidence">
                            Confidence: {confidence_percent:.2f}%
                        </div>
                    """, unsafe_allow_html=True)

    # Examples section
    st.markdown("---")
    st.subheader("Example Messages")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Spam Examples:**")
        spam_examples = [
            "URGENT! you have won 1000$",
            "WINNER!! Click here to claim your FREE prize now!!!",
            "Congratulations! You've been selected for a free iPhone. Call now!"
        ]
        for i, example in enumerate(spam_examples, 1):
            if st.button(f"Spam {i}", key=f"spam_{i}", use_container_width=True):
                st.session_state.message = example
                st.rerun()

    with col2:
        st.markdown("**Ham Examples:**")
        ham_examples = [
            "Hey! Are we still meeting for lunch at 1pm?",
            "Can you pick up milk on your way home?",
            "Running a bit late, be there in 10 mins"
        ]
        for i, example in enumerate(ham_examples, 1):
            if st.button(f"Ham {i}", key=f"ham_{i}", use_container_width=True):
                st.session_state.message = example
                st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; font-size: 14px;'>
            <p>Built with TensorFlow and Streamlit | Advanced Deep Learning Architecture</p>
            <p>Model: CNN-BiLSTM with Self-Attention Mechanism</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
