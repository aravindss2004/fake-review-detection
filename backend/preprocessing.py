"""
Text preprocessing module for fake review detection.
Handles cleaning, tokenization, lemmatization, and stopword removal.
"""
import re
import string
import spacy
from textblob import TextBlob
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging

from config import (
    SPACY_MODEL, URL_PATTERN, EMAIL_PATTERN, PHONE_PATTERN
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Comprehensive text preprocessing pipeline for review data.
    """
    
    def __init__(self):
        """Initialize the preprocessor with spaCy model."""
        try:
            self.nlp = spacy.load(SPACY_MODEL, disable=['parser', 'ner'])
            logger.info(f"Loaded spaCy model: {SPACY_MODEL}")
        except OSError:
            logger.error(f"spaCy model '{SPACY_MODEL}' not found. Please run: python -m spacy download {SPACY_MODEL}")
            raise
        
        # Define stopwords
        self.stopwords = self.nlp.Defaults.stop_words
        
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing URLs, emails, phone numbers, and extra whitespace.
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(URL_PATTERN, ' ', text)
        
        # Remove emails
        text = re.sub(EMAIL_PATTERN, ' ', text)
        
        # Remove phone numbers
        text = re.sub(PHONE_PATTERN, ' ', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove numbers (optional - can keep if relevant)
        # text = re.sub(r'\d+', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def remove_punctuation(self, text: str) -> str:
        """
        Remove punctuation from text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text without punctuation
        """
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def tokenize_and_lemmatize(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """
        Tokenize and lemmatize text using spaCy.
        
        Args:
            text (str): Input text
            remove_stopwords (bool): Whether to remove stopwords
            
        Returns:
            List[str]: List of lemmatized tokens
        """
        doc = self.nlp(text)
        
        tokens = []
        for token in doc:
            # Skip if token is stopword (if remove_stopwords is True)
            if remove_stopwords and token.text in self.stopwords:
                continue
            
            # Skip if token is only whitespace
            if token.is_space:
                continue
            
            # Add lemmatized token
            if token.lemma_ and len(token.lemma_) > 1:
                tokens.append(token.lemma_)
        
        return tokens
    
    def preprocess(self, text: str, remove_stopwords: bool = True, 
                   remove_punct: bool = True) -> str:
        """
        Complete preprocessing pipeline.
        
        Args:
            text (str): Raw text
            remove_stopwords (bool): Whether to remove stopwords
            remove_punct (bool): Whether to remove punctuation
            
        Returns:
            str: Preprocessed text
        """
        # Clean text
        text = self.clean_text(text)
        
        # Remove punctuation if specified
        if remove_punct:
            text = self.remove_punctuation(text)
        
        # Tokenize and lemmatize
        tokens = self.tokenize_and_lemmatize(text, remove_stopwords)
        
        # Join tokens back into string
        return ' '.join(tokens)
    
    def preprocess_batch(self, texts: List[str], show_progress: bool = False) -> List[str]:
        """
        Preprocess multiple texts efficiently.
        
        Args:
            texts (List[str]): List of raw texts
            show_progress (bool): Whether to show progress bar
            
        Returns:
            List[str]: List of preprocessed texts
        """
        if show_progress:
            try:
                from tqdm import tqdm
                texts = tqdm(texts, desc="Preprocessing")
            except ImportError:
                pass
        
        return [self.preprocess(text) for text in texts]
    
    def get_sentiment(self, text: str) -> Dict[str, float]:
        """
        Extract sentiment features using TextBlob.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, float]: Sentiment polarity and subjectivity
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


def preprocess_dataframe(df: pd.DataFrame, text_column: str = 'text',
                         preprocessor: TextPreprocessor = None) -> pd.DataFrame:
    """
    Preprocess text data in a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        text_column (str): Name of the text column
        preprocessor (TextPreprocessor): Preprocessor instance
        
    Returns:
        pd.DataFrame: DataFrame with preprocessed text
    """
    if preprocessor is None:
        preprocessor = TextPreprocessor()
    
    logger.info(f"Preprocessing {len(df)} reviews...")
    
    # Create a copy
    df = df.copy()
    
    # Preprocess text
    df['cleaned_text'] = preprocessor.preprocess_batch(
        df[text_column].tolist(),
        show_progress=True
    )
    
    logger.info("Preprocessing completed!")
    return df


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = TextPreprocessor()
    
    test_texts = [
        "This product is AMAZING!!! Best purchase ever! Visit http://example.com for more info.",
        "Terrible quality. Complete waste of money. DO NOT BUY!!!",
        "It's okay, nothing special. Works as expected."
    ]
    
    print("Testing Text Preprocessor\n" + "=" * 50)
    for i, text in enumerate(test_texts, 1):
        print(f"\nOriginal {i}: {text}")
        cleaned = preprocessor.clean_text(text)
        print(f"Cleaned {i}: {cleaned}")
        processed = preprocessor.preprocess(text)
        print(f"Processed {i}: {processed}")
        sentiment = preprocessor.get_sentiment(text)
        print(f"Sentiment {i}: {sentiment}")
