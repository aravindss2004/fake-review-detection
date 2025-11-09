"""
Prediction module for inference on new reviews.
"""
import numpy as np
import pandas as pd
import joblib
from typing import List, Dict, Any, Union
import logging
from pathlib import Path

from config import (
    TFIDF_VECTORIZER_PATH, FEATURE_SCALER_PATH,
    LIGHTGBM_MODEL_PATH, CATBOOST_MODEL_PATH, XGBOOST_MODEL_PATH,
    ENSEMBLE_MODEL_PATH, FAKE_THRESHOLD
)
from preprocessing import TextPreprocessor
from feature_engineering import FeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FakeReviewPredictor:
    """
    Predict whether reviews are fake or genuine using trained ensemble model.
    """
    
    def __init__(self, use_ensemble: bool = True):
        """
        Initialize the predictor.
        
        Args:
            use_ensemble (bool): Whether to use ensemble model or individual models
        """
        self.use_ensemble = use_ensemble
        self.preprocessor = TextPreprocessor()
        self.feature_extractor = FeatureExtractor()
        
        self.tfidf_vectorizer = None
        self.feature_scaler = None
        self.lightgbm_model = None
        self.catboost_model = None
        self.xgboost_model = None
        self.ensemble_model = None
        
        self.models_loaded = False
    
    def load_models(self):
        """Load all trained models and transformers."""
        logger.info("Loading models and transformers...")
        
        try:
            # Load TF-IDF vectorizer
            if TFIDF_VECTORIZER_PATH.exists():
                self.tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
                logger.info("TF-IDF vectorizer loaded")
            else:
                raise FileNotFoundError(f"TF-IDF vectorizer not found at {TFIDF_VECTORIZER_PATH}")
            
            # Load feature scaler
            if FEATURE_SCALER_PATH.exists():
                self.feature_scaler = joblib.load(FEATURE_SCALER_PATH)
                logger.info("Feature scaler loaded")
            else:
                raise FileNotFoundError(f"Feature scaler not found at {FEATURE_SCALER_PATH}")
            
            # Load models
            if self.use_ensemble:
                if ENSEMBLE_MODEL_PATH.exists():
                    self.ensemble_model = joblib.load(ENSEMBLE_MODEL_PATH)
                    logger.info("Ensemble model loaded")
                else:
                    raise FileNotFoundError(f"Ensemble model not found at {ENSEMBLE_MODEL_PATH}")
            else:
                # Load individual models
                if LIGHTGBM_MODEL_PATH.exists():
                    self.lightgbm_model = joblib.load(LIGHTGBM_MODEL_PATH)
                    logger.info("LightGBM model loaded")
                
                if CATBOOST_MODEL_PATH.exists():
                    self.catboost_model = joblib.load(CATBOOST_MODEL_PATH)
                    logger.info("CatBoost model loaded")
                
                if XGBOOST_MODEL_PATH.exists():
                    self.xgboost_model = joblib.load(XGBOOST_MODEL_PATH)
                    logger.info("XGBoost model loaded")
            
            self.models_loaded = True
            logger.info("All models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def preprocess_and_extract_features(self, texts: List[str]) -> np.ndarray:
        """
        Preprocess texts and extract all features.
        
        Args:
            texts (List[str]): List of raw review texts
            
        Returns:
            np.ndarray: Combined feature matrix
        """
        if not self.models_loaded:
            raise ValueError("Models must be loaded before prediction")
        
        # Store original texts
        original_texts = texts.copy()
        
        # Preprocess texts
        preprocessed_texts = self.preprocessor.preprocess_batch(texts)
        
        # Extract linguistic features
        linguistic_features_df = self.feature_extractor.extract_features_batch(
            preprocessed_texts, original_texts
        )
        linguistic_features = linguistic_features_df.values
        
        # Scale linguistic features
        linguistic_features_scaled = self.feature_scaler.transform(linguistic_features)
        
        # Extract TF-IDF features
        tfidf_features = self.tfidf_vectorizer.transform(preprocessed_texts).toarray()
        
        # Combine all features
        combined_features = np.hstack([tfidf_features, linguistic_features_scaled])
        
        return combined_features
    
    def predict_single(self, text: str) -> Dict[str, Any]:
        """
        Predict for a single review.
        
        Args:
            text (str): Review text
            
        Returns:
            Dict: Prediction result with label, probability, and confidence
        """
        return self.predict_batch([text])[0]
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict for multiple reviews.
        
        Args:
            texts (List[str]): List of review texts
            
        Returns:
            List[Dict]: List of prediction results
        """
        if not self.models_loaded:
            self.load_models()
        
        # Extract features
        features = self.preprocess_and_extract_features(texts)
        
        # Get predictions
        if self.use_ensemble:
            predictions = self.ensemble_model.predict(features)
            probabilities = self.ensemble_model.predict_proba(features)
        else:
            # Average predictions from individual models
            lgb_pred = self.lightgbm_model.predict_proba(features)
            cat_pred = self.catboost_model.predict_proba(features)
            xgb_pred = self.xgboost_model.predict_proba(features)
            
            probabilities = (lgb_pred + cat_pred + xgb_pred) / 3
            predictions = (probabilities[:, 1] >= FAKE_THRESHOLD).astype(int)
        
        # Format results
        results = []
        for i, text in enumerate(texts):
            result = {
                'text': text,
                'prediction': 'Fake' if predictions[i] == 1 else 'Genuine',
                'label': int(predictions[i]),
                'confidence': float(probabilities[i][predictions[i]]),
                'fake_probability': float(probabilities[i][1]),
                'genuine_probability': float(probabilities[i][0])
            }
            results.append(result)
        
        return results
    
    def predict_from_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Predict for reviews in a DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame with review texts
            text_column (str): Name of the text column
            
        Returns:
            pd.DataFrame: DataFrame with predictions
        """
        texts = df[text_column].tolist()
        predictions = self.predict_batch(texts)
        
        # Add predictions to dataframe
        df = df.copy()
        df['prediction'] = [p['prediction'] for p in predictions]
        df['label'] = [p['label'] for p in predictions]
        df['confidence'] = [p['confidence'] for p in predictions]
        df['fake_probability'] = [p['fake_probability'] for p in predictions]
        df['genuine_probability'] = [p['genuine_probability'] for p in predictions]
        
        return df
    
    def get_prediction_summary(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary statistics for predictions.
        
        Args:
            predictions (List[Dict]): List of prediction results
            
        Returns:
            Dict: Summary statistics
        """
        total = len(predictions)
        fake_count = sum(1 for p in predictions if p['label'] == 1)
        genuine_count = total - fake_count
        
        avg_confidence = np.mean([p['confidence'] for p in predictions])
        avg_fake_prob = np.mean([p['fake_probability'] for p in predictions])
        
        return {
            'total_reviews': total,
            'fake_reviews': fake_count,
            'genuine_reviews': genuine_count,
            'fake_percentage': (fake_count / total * 100) if total > 0 else 0,
            'genuine_percentage': (genuine_count / total * 100) if total > 0 else 0,
            'average_confidence': float(avg_confidence),
            'average_fake_probability': float(avg_fake_prob)
        }


class ScrapePredictor:
    """
    Placeholder for future scraping functionality.
    This feature is disabled for ToS compliance and security.
    """
    
    @staticmethod
    def scrape_and_predict(product_url: str) -> Dict[str, str]:
        """
        Placeholder function for scraping reviews from product URLs.
        
        Args:
            product_url (str): Product URL
            
        Returns:
            Dict: Message indicating feature is disabled
        """
        return {
            'status': 'disabled',
            'message': 'Scraping feature is disabled in this version for security and ToS compliance.',
            'alternative': 'Please manually input reviews or upload a CSV file.'
        }


if __name__ == "__main__":
    logger.info("Predictor Module - Ready for inference!")
