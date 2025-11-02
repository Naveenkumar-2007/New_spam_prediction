import os
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)

class DataIngestion:
    """
    SMS Spam Data Ingestion Component
    
    Connection Flow:
    1. Reads: notebook/spam.csv (Real SMS dataset - 5572 messages)
    2. Cleans: Removes unnecessary columns (Unnamed: 2, 3, 4)
    3. Renames: v1 → label, v2 → text
    4. Splits: 80% train, 20% test (stratified)
    5. Outputs: artifacts/raw.csv, train.csv, test.csv
    6. Next: data_transform.py uses these CSVs
    
    Dataset Format:
    - v1: label (spam/ham)
    - v2: text (SMS content)
    - Unnamed columns: Removed (all NaN)
    """
    
    def __init__(self):
        self.raw_data_path = "C:\\Users\\navee\\Cisco Packet Tracer 8.2.2\\saves\\deep\\spam.csv"
        self.artifacts_dir = "artifacts"
        
    def initiate_data_ingestion(self):
        """
        Main ingestion pipeline for real SMS dataset
        
        Returns:
            tuple: (train_path, test_path)
        """
        logging.info("=" * 70)
        logging.info("SMS SPAM DETECTION - DATA INGESTION STARTED")
        logging.info(f"User: Naveenkumar5151 | Date: 2025-01-11 18:58:34")
        logging.info("=" * 70)
        
        try:
            # Create artifacts directory
            os.makedirs(self.artifacts_dir, exist_ok=True)
            
            # Read real SMS dataset
            logging.info(f"📂 Reading dataset from: {self.raw_data_path}")
            df = pd.read_csv(self.raw_data_path, encoding='latin-1')
            
            logging.info(f"✅ Dataset loaded successfully!")
            logging.info(f"📊 Original shape: {df.shape}")
            logging.info(f"📋 Original columns: {list(df.columns)}")
            
            # Clean dataset - Keep only v1 and v2
            df = df[['v1', 'v2']]
            
            # Rename columns
            df.columns = ['label', 'text']
            
            logging.info(f"🔧 Cleaned columns: {list(df.columns)}")
            
            # Remove duplicates
            original_size = len(df)
            df = df.drop_duplicates()
            duplicates_removed = original_size - len(df)
            logging.info(f"🧹 Removed {duplicates_removed} duplicates")
            
            # Handle missing values
            df = df.dropna()
            logging.info(f"✨ Final dataset shape: {df.shape}")
            
            # Display label distribution
            label_counts = df['label'].value_counts()
            logging.info(f"\n📈 Label Distribution:")
            logging.info(f"   Ham (Legitimate): {label_counts.get('ham', 0)}")
            logging.info(f"   Spam: {label_counts.get('spam', 0)}")
            logging.info(f"   Total: {len(df)} messages")
            
            # Save raw data
            raw_path = os.path.join(self.artifacts_dir, "raw.csv")
            df.to_csv(raw_path, index=False)
            logging.info(f"\n💾 Raw data saved to: {raw_path}")
            
            # Stratified split (80% train, 20% test)
            train_df, test_df = train_test_split(
                df, 
                test_size=0.2, 
                random_state=42,
                stratify=df['label']
            )
            
            # Save train and test sets
            train_path = os.path.join(self.artifacts_dir, "train.csv")
            test_path = os.path.join(self.artifacts_dir, "test.csv")
            
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            
            logging.info(f"📚 Train data: {train_df.shape} → {train_path}")
            logging.info(f"   - Ham: {(train_df['label'] == 'ham').sum()}")
            logging.info(f"   - Spam: {(train_df['label'] == 'spam').sum()}")
            
            logging.info(f"📖 Test data: {test_df.shape} → {test_path}")
            logging.info(f"   - Ham: {(test_df['label'] == 'ham').sum()}")
            logging.info(f"   - Spam: {(test_df['label'] == 'spam').sum()}")
            
            # Show sample messages
            logging.info("\n" + "=" * 70)
            logging.info("📧 Sample SPAM Messages:")
            spam_samples = train_df[train_df['label'] == 'spam']['text'].head(3)
            for idx, msg in enumerate(spam_samples, 1):
                logging.info(f"  {idx}. {msg[:80]}...")
            
            logging.info("\n✉️ Sample HAM (Legitimate) Messages:")
            ham_samples = train_df[train_df['label'] == 'ham']['text'].head(3)
            for idx, msg in enumerate(ham_samples, 1):
                logging.info(f"  {idx}. {msg[:80]}...")
            
            logging.info("\n" + "=" * 70)
            logging.info("✅ DATA INGESTION COMPLETED SUCCESSFULLY")
            logging.info("=" * 70 + "\n")
            
            return train_path, test_path
            
        except FileNotFoundError:
            logging.error(f"❌ Error: File not found - {self.raw_data_path}")
            logging.error("Please ensure 'spam.csv' is in the 'notebook/' directory")
            raise
        except Exception as e:
            logging.error(f"❌ Error in data ingestion: {str(e)}")
            raise e

if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()