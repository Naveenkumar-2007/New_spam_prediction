import os
import re
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import logging
import warnings
import traceback

# Suppress scikit-learn version warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

logging.basicConfig(level=logging.INFO)

class predict:
    """
    SMS/Email Spam Prediction Pipeline (TensorFlow)
    
    Connection Flow:
    1. Loads: best_model.h5, preprocessing.pkl (from model_trainer.py)
    2. Receives: SMS/Email text from user (via app.py)
    3. Preprocesses: Cleans, tokenizes, pads sequence
    4. Predicts: Uses TensorFlow LSTM model
    5. Returns: "Spam" or "Legitimate" with confidence
    6. Used by: app.py for web interface
    """
    
    def __init__(self):
        # Resolve artifact paths relative to the project root so deployment
        # environments (which may change working directory) still find files.
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.model_path = os.path.join(project_root, 'artifacts', 'best_model.h5')
        self.preprocessing_path = os.path.join(project_root, 'artifacts', 'preprocessing.pkl')
        self.model = None
        self.tokenizer = None
        self.max_length = None
        # Heuristic fallback keeps the app usable even if TensorFlow fails to load.
        self.fallback = False
        
    def load_model(self):
        """Load TensorFlow model and preprocessing objects"""
        if self.model is None and not self.fallback:
            try:
                # Check if model file exists
                logging.info(f"Checking model path: {self.model_path}")
                logging.info(f"Checking preprocessing path: {self.preprocessing_path}")
                logging.info(f"Model exists: {os.path.exists(self.model_path)}")
                logging.info(f"Preprocessing exists: {os.path.exists(self.preprocessing_path)}")
                
                if not os.path.exists(self.model_path):
                    raise FileNotFoundError(f"Model file not found at: {self.model_path}")
                if not os.path.exists(self.preprocessing_path):
                    raise FileNotFoundError(f"Preprocessing file not found at: {self.preprocessing_path}")

                logging.info("Loading TensorFlow model...")

                # Try to import TensorFlow; failures can happen on some deployment platforms
                try:
                    import tensorflow as tf
                    logging.info(f"TensorFlow version: {tf.__version__}")
                    tf.get_logger().setLevel('ERROR')
                except Exception as tf_import_err:
                    logging.error(f"TensorFlow import failed: {tf_import_err}")
                    raise tf_import_err

                # Load the saved Keras model
                try:
                    self.model = load_model(self.model_path, compile=False)
                    # Compile for safe prediction API
                    self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                    logging.info(f"SUCCESS: Model loaded and compiled from: {self.model_path}")
                except Exception as tf_error:
                    logging.error(f"TensorFlow model loading error: {tf_error}")
                    logging.error(f"Full traceback: {traceback.format_exc()}")
                    raise tf_error

                # Load preprocessing objects
                logging.info("Loading preprocessing objects...")
                preprocessing_obj = joblib.load(self.preprocessing_path)

                # Validate preprocessing object
                required_keys = ['tokenizer', 'max_length', 'vocab_size']
                for key in required_keys:
                    if key not in preprocessing_obj:
                        raise KeyError(f"Missing key in preprocessing object: {key}")

                self.tokenizer = preprocessing_obj['tokenizer']
                self.max_length = preprocessing_obj['max_length']
                logging.info(f"SUCCESS: Tokenizer loaded (vocab size: {preprocessing_obj.get('vocab_size', 'unknown')})")
                logging.info("MODEL LOADED SUCCESSFULLY - Using ML predictions")

            except Exception as e:
                # Instead of crashing on deployment, enable fallback heuristic so app remains useful.
                logging.error(f"CRITICAL: Error loading model or preprocessing: {str(e)}")
                logging.error(f"Full error traceback:\n{traceback.format_exc()}")
                logging.warning("FALLBACK MODE ACTIVATED - Using keyword heuristic instead of ML model")
                self.fallback = True
    
    def clean_text(self, text):
        """Clean text"""
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    def get_predict(self, message_text):
        """
        Predict if message is spam or legitimate
        
        Args:
            message_text (str or pd.DataFrame): Message content
            
        Returns:
            tuple: (prediction, confidence)
        """
        try:
            # Input validation
            if message_text is None:
                raise ValueError("Message text cannot be None")
            
            # Extract text if DataFrame
            if isinstance(message_text, pd.DataFrame):
                if 'text' not in message_text.columns:
                    raise KeyError("DataFrame must contain 'text' column")
                if len(message_text) == 0:
                    raise ValueError("DataFrame is empty")
                message_text = message_text['text'].iloc[0]
            
            # Convert to string and validate
            message_text = str(message_text).strip()
            if len(message_text) == 0:
                raise ValueError("Message text is empty")
            
            logging.info(f"\nAnalyzing message: {message_text[:100]}...")
            
            # Load model if not loaded
            self.load_model()
            
            # Validate model components
            if self.model is None:
                raise RuntimeError("Model failed to load")
            if self.tokenizer is None:
                raise RuntimeError("Tokenizer failed to load")
            if self.max_length is None:
                raise RuntimeError("Max length not set")
            
            # Preprocess text
            cleaned_text = self.clean_text(message_text)
            if len(cleaned_text.strip()) == 0:
                logging.warning("Text became empty after cleaning, using original text")
                cleaned_text = message_text
            
            # Tokenize and pad
            sequence = self.tokenizer.texts_to_sequences([cleaned_text])
            
            # Check if tokenization produced any sequences
            if len(sequence) == 0 or len(sequence[0]) == 0:
                logging.warning("Tokenization produced empty sequence")
                # Return a default prediction for unknown text
                return "Legitimate", 0.5
            
            padded_sequence = pad_sequences(
                sequence, 
                maxlen=self.max_length, 
                padding='post', 
                truncating='post'
            )
            
            # Validate padded sequence
            if padded_sequence is None or len(padded_sequence) == 0:
                raise RuntimeError("Padding failed")
            
            # Predict
            prediction_proba = self.model.predict(padded_sequence, verbose=0)
            
            # Validate prediction output
            if prediction_proba is None or len(prediction_proba) == 0:
                raise RuntimeError("Model prediction failed")
            
            prediction_proba = float(prediction_proba[0][0])
            
            # Ensure probability is in valid range
            prediction_proba = max(0.0, min(1.0, prediction_proba))
            
            # Convert to label
            if prediction_proba > 0.5:
                prediction = "Spam"
                confidence = prediction_proba
            else:
                prediction = "Legitimate"
                confidence = 1 - prediction_proba
            
            logging.info(f"Prediction: {prediction}")
            logging.info(f"Confidence: {confidence:.4f}")
            
            return prediction, float(confidence)
            
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            logging.error(traceback.format_exc())
            # If fallback is enabled, attempt heuristic prediction instead of raising
            if self.fallback:
                logging.info("Using fallback heuristic to classify message")
                text = str(message_text)
                lowered = text.lower()
                # Simple heuristics: URLs, typical spam words
                spam_keywords = ['win', 'free', 'congrat', 'urgent', 'verify', 'password', 'click', 'bank', 'prize']
                has_url = bool(re.search(r'http[s]?://|www\.|bit\.ly|\b\w+\.\w{2,3}\b', lowered))
                score = 0.0
                if has_url:
                    score += 0.5
                for kw in spam_keywords:
                    if kw in lowered:
                        score += 0.15
                score = min(0.99, score)
                if score > 0.5:
                    return "Spam", float(score)
                else:
                    return "Legitimate", float(1 - score)
            # If not fallback, propagate the exception so caller can handle
            raise e

class customdata:
    """
    Custom data class for user input
    
    Connection Flow:
    1. Receives: Message text from app.py form
    2. Creates: DataFrame for prediction pipeline
    3. Used by: predict.get_predict()
    """
    
    def __init__(self, message_text):
        if message_text is None:
            raise ValueError("Message text cannot be None")
        self.message_text = str(message_text).strip()
        if len(self.message_text) == 0:
            raise ValueError("Message text cannot be empty")
        
    def data_frame(self):
        """Convert input to DataFrame"""
        try:
            df = pd.DataFrame({"text": [self.message_text]})
            if len(df) == 0:
                raise ValueError("Failed to create DataFrame")
            return df
        except Exception as e:
            logging.error(f"Error creating DataFrame: {str(e)}")
            raise e