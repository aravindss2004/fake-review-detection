"""
Feature engineering module for extracting linguistic features from reviews.
"""
import string
import numpy as np
import pandas as pd
from typing import Dict, List
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import logging

from config import MAX_FEATURES_TFIDF, NGRAM_RANGE, MIN_DF, MAX_DF, NUMERIC_FEATURES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extract linguistic and statistical features from review text.
    """
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.feature_names = NUMERIC_FEATURES
    
    def extract_length_features(self, text: str) -> Dict[str, float]:
        """
        Extract length-based features.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, float]: Length features
        """
        if not isinstance(text, str) or len(text) == 0:
            return {
                'char_count': 0,
                'word_count': 0,
                'avg_word_length': 0
            }
        
        words = text.split()
        char_count = len(text)
        word_count = len(words)
        avg_word_length = char_count / word_count if word_count > 0 else 0
        
        return {
            'char_count': char_count,
            'word_count': word_count,
            'avg_word_length': avg_word_length
        }
    
    def extract_punctuation_features(self, text: str) -> Dict[str, float]:
        """
        Extract punctuation-based features.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, float]: Punctuation features
        """
        if not isinstance(text, str):
            return {
                'punctuation_count': 0,
                'exclamation_count': 0,
                'question_count': 0
            }
        
        punctuation_count = sum(1 for char in text if char in string.punctuation)
        exclamation_count = text.count('!')
        question_count = text.count('?')
        
        return {
            'punctuation_count': punctuation_count,
            'exclamation_count': exclamation_count,
            'question_count': question_count
        }
    
    def extract_capital_features(self, text: str) -> Dict[str, float]:
        """
        Extract capital letter features.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, float]: Capital letter features
        """
        if not isinstance(text, str) or len(text) == 0:
            return {'capital_ratio': 0}
        
        capital_count = sum(1 for char in text if char.isupper())
        capital_ratio = capital_count / len(text)
        
        return {'capital_ratio': capital_ratio}
    
    def extract_sentiment_features(self, text: str) -> Dict[str, float]:
        """
        Extract sentiment features using TextBlob.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, float]: Sentiment features
        """
        try:
            blob = TextBlob(text)
            return {
                'sentiment_polarity': blob.sentiment.polarity,
                'sentiment_subjectivity': blob.sentiment.subjectivity
            }
        except:
            return {
                'sentiment_polarity': 0.0,
                'sentiment_subjectivity': 0.0
            }
    
    def extract_all_features(self, text: str, original_text: str = None) -> Dict[str, float]:
        """
        Extract all linguistic features.
        
        Args:
            text (str): Preprocessed text
            original_text (str): Original text (for punctuation/capital features)
            
        Returns:
            Dict[str, float]: All extracted features
        """
        if original_text is None:
            original_text = text
        
        features = {}
        
        # Length features from preprocessed text
        features.update(self.extract_length_features(text))
        
        # Punctuation features from original text
        features.update(self.extract_punctuation_features(original_text))
        
        # Capital features from original text
        features.update(self.extract_capital_features(original_text))
        
        # Sentiment features from original text
        features.update(self.extract_sentiment_features(original_text))
        
        return features
    
    def extract_features_batch(self, texts: List[str], 
                               original_texts: List[str] = None) -> pd.DataFrame:
        """
        Extract features for multiple texts.
        
        Args:
            texts (List[str]): List of preprocessed texts
            original_texts (List[str]): List of original texts
            
        Returns:
            pd.DataFrame: DataFrame with extracted features
        """
        if original_texts is None:
            original_texts = texts
        
        features_list = []
        for text, original in zip(texts, original_texts):
            features = self.extract_all_features(text, original)
            features_list.append(features)
        
        return pd.DataFrame(features_list)


class TFIDFFeatureExtractor:
    """
    TF-IDF vectorization for text features.
    """
    
    def __init__(self, max_features: int = MAX_FEATURES_TFIDF,
                 ngram_range: tuple = NGRAM_RANGE,
                 min_df: int = MIN_DF,
                 max_df: float = MAX_DF):
        """
        Initialize TF-IDF vectorizer.
        
        Args:
            max_features (int): Maximum number of features
            ngram_range (tuple): Range of n-grams
            min_df (int): Minimum document frequency
            max_df (float): Maximum document frequency
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}',
            stop_words='english'
        )
        self.is_fitted = False
    
    def fit(self, texts: List[str]):
        """
        Fit the TF-IDF vectorizer.
        
        Args:
            texts (List[str]): Training texts
        """
        logger.info("Fitting TF-IDF vectorizer...")
        self.vectorizer.fit(texts)
        self.is_fitted = True
        logger.info(f"TF-IDF vectorizer fitted with {len(self.vectorizer.vocabulary_)} features")
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to TF-IDF features.
        
        Args:
            texts (List[str]): Input texts
            
        Returns:
            np.ndarray: TF-IDF feature matrix
        """
        if not self.is_fitted:
            raise ValueError("TF-IDF vectorizer must be fitted before transform")
        
        return self.vectorizer.transform(texts).toarray()
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit and transform texts.
        
        Args:
            texts (List[str]): Training texts
            
        Returns:
            np.ndarray: TF-IDF feature matrix
        """
        self.fit(texts)
        return self.transform(texts)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names from vectorizer."""
        if not self.is_fitted:
            return []
        return self.vectorizer.get_feature_names_out().tolist()


class FeatureScaler:
    """
    Scale numeric features.
    """
    
    def __init__(self):
        """Initialize scaler."""
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, features: np.ndarray):
        """
        Fit the scaler.
        
        Args:
            features (np.ndarray): Training features
        """
        self.scaler.fit(features)
        self.is_fitted = True
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features.
        
        Args:
            features (np.ndarray): Input features
            
        Returns:
            np.ndarray: Scaled features
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        return self.scaler.transform(features)
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """
        Fit and transform features.
        
        Args:
            features (np.ndarray): Training features
            
        Returns:
            np.ndarray: Scaled features
        """
        self.fit(features)
        return self.transform(features)


if __name__ == "__main__":
    # Test feature extraction
    extractor = FeatureExtractor()
    
    test_texts = [
        "This product is AMAZING!!! Best purchase ever!",
        "Terrible quality. Complete waste of money.",
        "It's okay, nothing special. Works as expected."
    ]
    
    print("Testing Feature Extractor\n" + "=" * 50)
    for text in test_texts:
        features = extractor.extract_all_features(text)
        print(f"\nText: {text}")
        print("Features:")
        for key, value in features.items():
            print(f"  {key}: {value:.3f}")
