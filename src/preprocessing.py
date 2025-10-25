"""
Improved preprocessing for SMS spam detection.

Key improvements:
1. KEEPS spam indicators: numbers, dollar signs, exclamation marks
2. Better handling of URLs and special characters
3. Preserves capitalization patterns (spam often uses CAPS)
4. More intelligent text cleaning
"""

import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class ImprovedDataPreprocessor:
    def __init__(self, vocab_size=20000, max_length=100, test_size=0.2, random_state=42):
        """
        Initialize the improved data preprocessor.

        Args:
            vocab_size (int): Maximum number of words to keep in vocabulary
            max_length (int): Maximum length of sequences after padding
            test_size (float): Proportion of dataset to use for testing
            random_state (int): Random seed for reproducibility
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.test_size = test_size
        self.random_state = random_state
        self.tokenizer = None
        self.class_weights = None

    def clean_text(self, text):
        """
        Improved text cleaning that KEEPS spam indicators.

        Args:
            text (str): Raw text to clean

        Returns:
            str: Cleaned text with spam indicators preserved
        """
        if not isinstance(text, str):
            return ""

        # Keep original for case checking
        original = text

        # Replace URLs with a token (don't remove - spam often has URLs)
        text = re.sub(r'http\S+|www\S+|https\S+', ' URL ', text, flags=re.MULTILINE)

        # Replace email with token (spam often has emails)
        text = re.sub(r'\S+@\S+', ' EMAIL ', text)

        # Replace phone numbers with token (but keep the concept)
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', ' PHONE ', text)

        # Normalize repeated punctuation (!!!! → !!) but keep them
        text = re.sub(r'([!?.]){3,}', r'\1\1', text)

        # Add space around special characters to tokenize them separately
        # This helps the model learn that $ ! are spam indicators
        text = re.sub(r'([!$£€¥?.])', r' \1 ', text)

        # Handle currency amounts (e.g., $100, £50) - keep as single token
        text = re.sub(r'([$£€¥])\s+(\d+)', r'\1\2', text)

        # Preserve words with ALL CAPS (spam indicator) by adding a marker
        words = text.split()
        processed_words = []
        for word in words:
            # If word is all caps and length > 2, add CAPS_ prefix
            if len(word) > 2 and word.isupper() and word.isalpha():
                processed_words.append('CAPS_' + word.lower())
            else:
                processed_words.append(word.lower())

        text = ' '.join(processed_words)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def load_and_prepare_data(self, filepath):
        """
        Load and prepare the SMS spam dataset with improved handling.

        Args:
            filepath (str): Path to the CSV file

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print(f"Loading dataset from: {filepath}")

        # Try different encodings
        try:
            df = pd.read_csv(filepath, encoding='latin-1')
        except:
            try:
                df = pd.read_csv(filepath, encoding='utf-8')
            except:
                df = pd.read_csv(filepath)

        # Handle different dataset formats
        if df.shape[1] > 2:
            # UCI format: v1, v2, ...
            df = df.iloc[:, :2]
            df.columns = ['label', 'message']
        else:
            # Already 2 columns
            if df.columns[0] not in ['label', 'Label']:
                df.columns = ['label', 'message']

        # Standardize label column name
        if 'Label' in df.columns:
            df.rename(columns={'Label': 'label'}, inplace=True)

        print(f"Loaded {len(df)} messages")

        # Remove duplicates and null values
        df = df.drop_duplicates()
        df = df.dropna()

        print(f"After removing duplicates: {len(df)} messages")

        # Clean messages with improved preprocessing
        df['cleaned_message'] = df['message'].apply(self.clean_text)

        # Convert labels to binary (ham=0, spam=1)
        # Handle different label formats
        label_mapping = {
            'ham': 0, 'Ham': 0, 'HAM': 0, 'legitimate': 0,
            'spam': 1, 'Spam': 1, 'SPAM': 1, 'smishing': 1
        }
        df['label'] = df['label'].map(lambda x: label_mapping.get(x, x))

        # Ensure labels are integers
        df['label'] = df['label'].astype(int)

        print(f"\nLabel distribution:")
        print(df['label'].value_counts())
        print(f"Spam percentage: {(df['label'].sum() / len(df) * 100):.2f}%")

        # Split features and labels
        X = df['cleaned_message'].values
        y = df['label'].values

        # Calculate class weights for imbalanced data
        self.class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y),
            y=y
        )
        self.class_weights = dict(enumerate(self.class_weights))

        print(f"\nCalculated class weights: {self.class_weights}")

        # Split into train and test sets with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        print(f"\nTrain set: {len(X_train)} messages")
        print(f"Test set: {len(X_test)} messages")

        return X_train, X_test, y_train, y_test

    def tokenize_and_pad(self, X_train, X_test):
        """
        Tokenize and pad sequences.

        Args:
            X_train (array): Training messages
            X_test (array): Testing messages

        Returns:
            tuple: (X_train_padded, X_test_padded)
        """
        # Initialize tokenizer with OOV token
        self.tokenizer = Tokenizer(
            num_words=self.vocab_size,
            oov_token='<OOV>',
            lower=False  # We already lowercased in cleaning
        )

        print("\nTokenizing texts...")
        self.tokenizer.fit_on_texts(X_train)

        print(f"Vocabulary size: {len(self.tokenizer.word_index)}")
        print(f"Using top {self.vocab_size} words")

        # Show some important tokens
        word_freq = sorted(self.tokenizer.word_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"\nTop 20 most frequent tokens:")
        for word, freq in word_freq[:20]:
            print(f"  {word}: {freq}")

        # Convert texts to sequences
        X_train_seq = self.tokenizer.texts_to_sequences(X_train)
        X_test_seq = self.tokenizer.texts_to_sequences(X_test)

        # Pad sequences
        X_train_padded = pad_sequences(X_train_seq, maxlen=self.max_length, padding='post', truncating='post')
        X_test_padded = pad_sequences(X_test_seq, maxlen=self.max_length, padding='post', truncating='post')

        print(f"Sequence shape: {X_train_padded.shape}")

        return X_train_padded, X_test_padded

    def get_class_weights(self):
        """Get computed class weights for imbalanced dataset."""
        if self.class_weights is None:
            raise ValueError("Class weights not computed. Run load_and_prepare_data first.")
        return self.class_weights

    def save_tokenizer(self, filepath):
        """
        Save the tokenizer to a file.

        Args:
            filepath (str): Path to save the tokenizer
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer has not been fitted yet. Call tokenize_and_pad first.")

        with open(filepath, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        print(f"Tokenizer saved to {filepath}")

    @staticmethod
    def load_tokenizer(filepath):
        """
        Load a tokenizer from a file.

        Args:
            filepath (str): Path to the tokenizer file

        Returns:
            Tokenizer: Loaded tokenizer object
        """
        with open(filepath, 'rb') as f:
            tokenizer = pickle.load(f)
        return tokenizer
