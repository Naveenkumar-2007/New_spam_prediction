import os
import re
import numpy as np
import pandas as pd
import joblib
import nltk
from sklearn.preprocessing import LabelEncoder
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import logging

# Download NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

from nltk.corpus import stopwords

logging.basicConfig(level=logging.INFO)

class DataTransformation:
    """
    SMS Text Preprocessing for TensorFlow/Keras Deep Learning
    
    Connection Flow:
    1. Reads: artifacts/train.csv, test.csv (from data_ingestion.py)
    2. Cleans: Text preprocessing (lowercase, remove special chars)
    3. Tokenizes: TensorFlow Tokenizer (converts words to integers)
    4. Sequences: Creates padded sequences for LSTM
    5. Outputs: preprocessing.pkl, train_sequences.pkl, test_sequences.pkl
    6. Next: model_trainer.py uses these for TensorFlow training
    
    Preprocessing Steps:
    - Lowercase conversion
    - Remove URLs, special characters
    - Remove stopwords
    - TensorFlow Tokenization
    - Padding to max_length=100
    """
    
    def __init__(self):
        self.artifacts_dir = "artifacts"
        self.label_encoder = LabelEncoder()
        self.tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
        self.max_length = 100
        
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
        
    def clean_text(self, text):
        """
        Clean SMS/Email text
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Cleaned text
        """
        # Convert to string (handle any non-string values)
        text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        # Remove stopwords (optional - may reduce accuracy for short SMS)
        # Uncomment below if needed:
        # words = text.split()
        # text = ' '.join([w for w in words if w not in self.stop_words])
        
        return text
    
    def preprocess_data(self, df, is_train=True):
        """
        Complete preprocessing pipeline
        
        Args:
            df (pd.DataFrame): Input dataframe
            is_train (bool): Whether this is training data
            
        Returns:
            tuple: (sequences, labels)
        """
        # Clean texts
        texts = df['text'].apply(self.clean_text).tolist()
        labels = df['label'].tolist()
        
        # Tokenize texts
        if is_train:
            self.tokenizer.fit_on_texts(texts)
            logging.info(f"📚 Vocabulary size: {len(self.tokenizer.word_index)}")
            
            # Show most common words
            word_counts = sorted(
                self.tokenizer.word_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            logging.info(f"🔤 Top 10 words: {[w[0] for w in word_counts]}")
        
        # Convert to sequences
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # Pad sequences
        padded_sequences = pad_sequences(
            sequences, 
            maxlen=self.max_length, 
            padding='post', 
            truncating='post'
        )
        
        # Encode labels (spam=1, ham=0)
        if is_train:
            encoded_labels = self.label_encoder.fit_transform(labels)
        else:
            encoded_labels = self.label_encoder.transform(labels)
        
        return padded_sequences, encoded_labels
    
    def initiate_data_transformation(self, train_path, test_path):
        """
        Main transformation pipeline
        
        Args:
            train_path (str): Path to train CSV
            test_path (str): Path to test CSV
            
        Returns:
            tuple: Paths to processed data
        """
        logging.info("=" * 70)
        logging.info("SMS SPAM DETECTION - DATA TRANSFORMATION STARTED")
        logging.info("=" * 70)
        
        try:
            # Load data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info(f"📊 Train shape: {train_df.shape}")
            logging.info(f"📊 Test shape: {test_df.shape}")
            
            # Preprocess train data
            logging.info("\n🔄 Processing training data...")
            X_train, y_train = self.preprocess_data(train_df, is_train=True)
            
            # Preprocess test data
            logging.info("🔄 Processing test data...")
            X_test, y_test = self.preprocess_data(test_df, is_train=False)
            
            logging.info(f"\n✅ Train sequences shape: {X_train.shape}")
            logging.info(f"✅ Test sequences shape: {X_test.shape}")
            
            # Save preprocessing objects
            preprocessing_obj = {
                'label_encoder': self.label_encoder,
                'tokenizer': self.tokenizer,
                'max_length': self.max_length,
                'vocab_size': len(self.tokenizer.word_index) + 1
            }
            
            preprocessing_path = os.path.join(self.artifacts_dir, "preprocessing.pkl")
            joblib.dump(preprocessing_obj, preprocessing_path)
            logging.info(f"💾 Preprocessing objects saved to: {preprocessing_path}")
            
            # Save sequences
            train_seq_path = os.path.join(self.artifacts_dir, "train_sequences.pkl")
            test_seq_path = os.path.join(self.artifacts_dir, "test_sequences.pkl")
            
            joblib.dump({
                'X': X_train,
                'y': y_train
            }, train_seq_path)
            
            joblib.dump({
                'X': X_test,
                'y': y_test
            }, test_seq_path)
            
            logging.info(f"💾 Train sequences saved to: {train_seq_path}")
            logging.info(f"💾 Test sequences saved to: {test_seq_path}")
            
            # Show sample transformations
            logging.info("\n📝 Sample Transformation:")
            sample_text = train_df['text'].iloc[0]
            sample_label = train_df['label'].iloc[0]
            sample_cleaned = self.clean_text(sample_text)
            sample_sequence = X_train[0][:20]  # First 20 tokens
            
            logging.info(f"Original: {sample_text[:80]}...")
            logging.info(f"Cleaned: {sample_cleaned[:80]}...")
            logging.info(f"Label: {sample_label}")
            logging.info(f"Sequence: {sample_sequence}")
            
            logging.info("\n" + "=" * 70)
            logging.info("✅ DATA TRANSFORMATION COMPLETED SUCCESSFULLY")
            logging.info("=" * 70 + "\n")
            
            return train_seq_path, test_seq_path, preprocessing_path
            
        except Exception as e:
            logging.error(f"❌ Error in data transformation: {str(e)}")
            raise e

if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    
    # Run data ingestion first
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()
    
    # Run data transformation
    transformation = DataTransformation()
    transformation.initiate_data_transformation(train_path, test_path)