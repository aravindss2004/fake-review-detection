"""
Standalone training script for the fake review detection models.
Run this script to train models on your dataset.

Usage:
    python train_model.py --data data/raw/reviews.csv
"""

import sys
import argparse
import logging
from pathlib import Path

# Add backend to path
sys.path.append('backend')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

from preprocessing import TextPreprocessor
from feature_engineering import FeatureExtractor, TFIDFFeatureExtractor, FeatureScaler
from model_trainer import EnsembleModelTrainer
from config import (
    TFIDF_VECTORIZER_PATH, FEATURE_SCALER_PATH,
    TRAIN_DATA_PATH, TEST_DATA_PATH, TEST_SIZE, RANDOM_STATE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(data_path: str) -> pd.DataFrame:
    """Load dataset from CSV file."""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Validate columns
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset must have 'text' and 'label' columns")
    
    # Remove missing values
    df = df.dropna(subset=['text', 'label'])
    
    logger.info(f"Loaded {len(df)} reviews")
    logger.info(f"Label distribution:\n{df['label'].value_counts()}")
    
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess text data."""
    logger.info("Preprocessing text data...")
    
    preprocessor = TextPreprocessor()
    df['cleaned_text'] = preprocessor.preprocess_batch(
        df['text'].tolist(),
        show_progress=True
    )
    
    logger.info("Preprocessing completed!")
    return df


def extract_features(df: pd.DataFrame):
    """Extract all features from the data."""
    logger.info("Extracting features...")
    
    # Extract linguistic features
    feature_extractor = FeatureExtractor()
    linguistic_features_df = feature_extractor.extract_features_batch(
        df['cleaned_text'].tolist(),
        df['text'].tolist()
    )
    
    # Split data
    logger.info("Splitting data into train and test sets...")
    X_text_train, X_text_test, X_ling_train, X_ling_test, y_train, y_test = train_test_split(
        df['cleaned_text'],
        linguistic_features_df,
        df['label'],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df['label']
    )
    
    logger.info(f"Training set: {len(X_text_train)} samples")
    logger.info(f"Test set: {len(X_text_test)} samples")
    
    # TF-IDF vectorization
    logger.info("Creating TF-IDF features...")
    tfidf_extractor = TFIDFFeatureExtractor()
    X_tfidf_train = tfidf_extractor.fit_transform(X_text_train.tolist())
    X_tfidf_test = tfidf_extractor.transform(X_text_test.tolist())
    
    # Scale linguistic features
    logger.info("Scaling linguistic features...")
    scaler = FeatureScaler()
    X_ling_train_scaled = scaler.fit_transform(X_ling_train.values)
    X_ling_test_scaled = scaler.transform(X_ling_test.values)
    
    # Combine all features
    logger.info("Combining features...")
    X_train = np.hstack([X_tfidf_train, X_ling_train_scaled])
    X_test = np.hstack([X_tfidf_test, X_ling_test_scaled])
    
    logger.info(f"Final feature shape: {X_train.shape}")
    
    return (X_train, X_test, y_train.values, y_test.values,
            tfidf_extractor, scaler, X_text_train, X_text_test)


def train_models(X_train, y_train, X_test, y_test):
    """Train ensemble models."""
    logger.info("\n" + "=" * 60)
    logger.info("Starting model training...")
    logger.info("=" * 60)
    
    trainer = EnsembleModelTrainer()
    models = trainer.train(X_train, y_train)
    
    logger.info("\n" + "=" * 60)
    logger.info("Evaluating models...")
    logger.info("=" * 60)
    
    results = trainer.evaluate_all_models(X_test, y_test)
    
    return trainer, results


def save_artifacts(trainer, tfidf_extractor, scaler, df, X_text_train, X_text_test):
    """Save all trained models and artifacts."""
    logger.info("\n" + "=" * 60)
    logger.info("Saving models and artifacts...")
    logger.info("=" * 60)
    
    # Save models
    trainer.save_models()
    
    # Save transformers
    joblib.dump(tfidf_extractor.vectorizer, TFIDF_VECTORIZER_PATH)
    logger.info(f"TF-IDF vectorizer saved to {TFIDF_VECTORIZER_PATH}")
    
    joblib.dump(scaler.scaler, FEATURE_SCALER_PATH)
    logger.info(f"Feature scaler saved to {FEATURE_SCALER_PATH}")
    
    # Save processed data
    train_df = df.loc[X_text_train.index].copy()
    test_df = df.loc[X_text_test.index].copy()
    
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    logger.info(f"Training data saved to {TRAIN_DATA_PATH}")
    
    test_df.to_csv(TEST_DATA_PATH, index=False)
    logger.info(f"Test data saved to {TEST_DATA_PATH}")
    
    logger.info("All artifacts saved successfully!")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train fake review detection models')
    parser.add_argument('--data', type=str, default='data/raw/reviews.csv',
                        help='Path to training data CSV file')
    parser.add_argument('--skip-preprocessing', action='store_true',
                        help='Skip preprocessing if data is already clean')
    
    args = parser.parse_args()
    
    try:
        # Load data
        df = load_data(args.data)
        
        # Preprocess
        if not args.skip_preprocessing:
            df = preprocess_data(df)
        else:
            logger.info("Skipping preprocessing...")
            df['cleaned_text'] = df['text']
        
        # Extract features
        (X_train, X_test, y_train, y_test,
         tfidf_extractor, scaler, X_text_train, X_text_test) = extract_features(df)
        
        # Train models
        trainer, results = train_models(X_train, y_train, X_test, y_test)
        
        # Save artifacts
        save_artifacts(trainer, tfidf_extractor, scaler, df, X_text_train, X_text_test)
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE!")
        logger.info("=" * 60)
        logger.info("\nModel Performance Summary:")
        
        for model_name, metrics in results.items():
            logger.info(f"\n{model_name.upper()}:")
            for metric_name, value in metrics.items():
                logger.info(f"  {metric_name}: {value:.4f}")
        
        logger.info("\n" + "=" * 60)
        logger.info("Next Steps:")
        logger.info("1. Start the backend: cd backend && python app.py")
        logger.info("2. Start the frontend: cd frontend && npm start")
        logger.info("3. Open http://localhost:3000 in your browser")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
