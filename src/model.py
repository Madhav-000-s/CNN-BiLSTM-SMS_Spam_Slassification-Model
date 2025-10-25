"""
Improved model architectures for SMS spam detection.

This module provides advanced deep learning architectures:
1. CNN-BiLSTM with Attention (ACB Model)
2. Stacked BiLSTM with Attention
3. Enhanced BiLSTM with better regularization

Based on research: "Spam review detection using self attention based CNN and bi-directional LSTM"
"""

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D,
    Bidirectional, LSTM, Dense, Dropout, Concatenate,
    MultiHeadAttention, LayerNormalization, Add
)
from tensorflow.keras import backend as K


class AttentionLayer(tf.keras.layers.Layer):
    """
    Self-attention layer for sequence processing.

    This layer computes attention weights and creates a context vector
    by focusing on important parts of the input sequence.
    """

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


class SpamDetectorModels:
    """Factory class for creating different spam detection models."""

    @staticmethod
    def create_cnn_bilstm_attention(vocab_size=20000, embedding_dim=128, max_length=100):
        """
        Create CNN-BiLSTM with Attention model (ACB Model).

        Architecture:
        1. Embedding layer
        2. CNN layer for n-gram feature extraction
        3. Bidirectional LSTM for sequential context
        4. Self-attention mechanism
        5. Dense output layer

        This is the most effective architecture based on research.

        Args:
            vocab_size (int): Size of vocabulary
            embedding_dim (int): Dimension of embedding vectors
            max_length (int): Maximum sequence length

        Returns:
            Model: Compiled Keras model
        """
        print("\n" + "="*70)
        print("Building CNN-BiLSTM-Attention Model (ACB Model)")
        print("="*70)

        # Input layer
        inputs = Input(shape=(max_length,), name='input')

        # Embedding layer
        embedding = Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_length,
            name='embedding'
        )(inputs)

        # CNN layer for local n-gram features
        # Multiple filter sizes to capture different n-grams
        conv_blocks = []
        for filter_size in [2, 3, 4, 5]:
            conv = Conv1D(
                filters=128,
                kernel_size=filter_size,
                activation='relu',
                padding='same',
                name=f'conv_{filter_size}'
            )(embedding)
            conv = MaxPooling1D(pool_size=2, name=f'pool_{filter_size}')(conv)
            conv_blocks.append(conv)

        # Concatenate CNN features
        if len(conv_blocks) > 1:
            cnn_features = Concatenate(name='cnn_concat')(conv_blocks)
        else:
            cnn_features = conv_blocks[0]

        # Bidirectional LSTM for sequential context
        lstm_out = Bidirectional(
            LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
            name='bilstm_1'
        )(cnn_features)

        # Second BiLSTM layer
        lstm_out = Bidirectional(
            LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
            name='bilstm_2'
        )(lstm_out)

        # Self-attention mechanism
        attention_out = AttentionLayer(name='attention')(lstm_out)

        # Dense layers with dropout
        dense = Dense(256, activation='relu', name='dense_1')(attention_out)
        dense = Dropout(0.5, name='dropout_1')(dense)

        dense = Dense(128, activation='relu', name='dense_2')(dense)
        dense = Dropout(0.4, name='dropout_2')(dense)

        # Output layer
        outputs = Dense(1, activation='sigmoid', name='output')(dense)

        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='CNN_BiLSTM_Attention')

        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        return model

    @staticmethod
    def create_enhanced_bilstm(vocab_size=20000, embedding_dim=128, max_length=100):
        """
        Create Enhanced BiLSTM with Attention model.

        Simpler than ACB but still effective.

        Args:
            vocab_size (int): Size of vocabulary
            embedding_dim (int): Dimension of embedding vectors
            max_length (int): Maximum sequence length

        Returns:
            Model: Compiled Keras model
        """
        print("\n" + "="*70)
        print("Building Enhanced BiLSTM-Attention Model")
        print("="*70)

        # Input layer
        inputs = Input(shape=(max_length,), name='input')

        # Embedding layer
        embedding = Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_length,
            name='embedding'
        )(inputs)

        # Stacked Bidirectional LSTM layers
        lstm_out = Bidirectional(
            LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
            name='bilstm_1'
        )(embedding)

        lstm_out = Bidirectional(
            LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
            name='bilstm_2'
        )(lstm_out)

        # Attention mechanism
        attention_out = AttentionLayer(name='attention')(lstm_out)

        # Dense layers
        dense = Dense(128, activation='relu', name='dense_1')(attention_out)
        dense = Dropout(0.5, name='dropout')(dense)

        # Output layer
        outputs = Dense(1, activation='sigmoid', name='output')(dense)

        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs, name='Enhanced_BiLSTM_Attention')

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        return model

    @staticmethod
    def create_transformer_inspired(vocab_size=20000, embedding_dim=128, max_length=100):
        """
        Create a transformer-inspired model with multi-head attention.

        Uses Keras MultiHeadAttention for better performance.

        Args:
            vocab_size (int): Size of vocabulary
            embedding_dim (int): Dimension of embedding vectors
            max_length (int): Maximum sequence length

        Returns:
            Model: Compiled Keras model
        """
        print("\n" + "="*70)
        print("Building Transformer-Inspired Model")
        print("="*70)

        # Input layer
        inputs = Input(shape=(max_length,), name='input')

        # Embedding layer
        x = Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_length,
            name='embedding'
        )(inputs)

        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=8,
            key_dim=embedding_dim,
            dropout=0.1,
            name='multi_head_attention'
        )(x, x)

        # Add & Norm
        x = Add()([x, attention_output])
        x = LayerNormalization(epsilon=1e-6)(x)

        # BiLSTM layer
        x = Bidirectional(
            LSTM(128, return_sequences=False, dropout=0.3),
            name='bilstm'
        )(x)

        # Dense layers
        x = Dense(128, activation='relu', name='dense_1')(x)
        x = Dropout(0.5)(x)

        x = Dense(64, activation='relu', name='dense_2')(x)
        x = Dropout(0.3)(x)

        # Output
        outputs = Dense(1, activation='sigmoid', name='output')(x)

        # Create and compile
        model = Model(inputs=inputs, outputs=outputs, name='Transformer_Inspired')

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        return model


def get_model(model_type='cnn_bilstm_attention', vocab_size=20000, embedding_dim=128, max_length=100):
    """
    Factory function to get the desired model.

    Args:
        model_type (str): Type of model to create
            - 'cnn_bilstm_attention': CNN-BiLSTM-Attention (recommended)
            - 'enhanced_bilstm': Enhanced BiLSTM with Attention
            - 'transformer': Transformer-inspired model
        vocab_size (int): Vocabulary size
        embedding_dim (int): Embedding dimension
        max_length (int): Maximum sequence length

    Returns:
        Model: Compiled Keras model
    """
    models = {
        'cnn_bilstm_attention': SpamDetectorModels.create_cnn_bilstm_attention,
        'enhanced_bilstm': SpamDetectorModels.create_enhanced_bilstm,
        'transformer': SpamDetectorModels.create_transformer_inspired
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")

    return models[model_type](vocab_size, embedding_dim, max_length)
