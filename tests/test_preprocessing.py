"""
Unit tests for preprocessing module.
"""
import sys
sys.path.append('../backend')

import unittest
from preprocessing import TextPreprocessor


class TestPreprocessing(unittest.TestCase):
    """Test cases for text preprocessing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = TextPreprocessor()
    
    def test_clean_text_removes_urls(self):
        """Test that URLs are removed from text."""
        text = "Check out http://example.com for more info"
        cleaned = self.preprocessor.clean_text(text)
        self.assertNotIn("http", cleaned)
    
    def test_clean_text_removes_emails(self):
        """Test that email addresses are removed."""
        text = "Contact me at test@example.com"
        cleaned = self.preprocessor.clean_text(text)
        self.assertNotIn("@", cleaned)
    
    def test_clean_text_lowercases(self):
        """Test that text is converted to lowercase."""
        text = "THIS IS UPPERCASE"
        cleaned = self.preprocessor.clean_text(text)
        self.assertEqual(cleaned, "this is uppercase")
    
    def test_remove_punctuation(self):
        """Test punctuation removal."""
        text = "Hello, world! How are you?"
        cleaned = self.preprocessor.remove_punctuation(text)
        self.assertNotIn(",", cleaned)
        self.assertNotIn("!", cleaned)
        self.assertNotIn("?", cleaned)
    
    def test_tokenize_and_lemmatize(self):
        """Test tokenization and lemmatization."""
        text = "running ran runs"
        tokens = self.preprocessor.tokenize_and_lemmatize(text)
        # All should be lemmatized to 'run'
        self.assertTrue(all(token == 'run' for token in tokens))
    
    def test_preprocess_pipeline(self):
        """Test complete preprocessing pipeline."""
        text = "This is a GREAT product! Visit http://example.com"
        processed = self.preprocessor.preprocess(text)
        
        # Should be lowercase
        self.assertEqual(processed, processed.lower())
        
        # Should not contain URLs
        self.assertNotIn("http", processed)
        
        # Should not be empty
        self.assertGreater(len(processed), 0)
    
    def test_get_sentiment(self):
        """Test sentiment extraction."""
        positive_text = "This is amazing and wonderful!"
        negative_text = "This is terrible and awful!"
        
        pos_sentiment = self.preprocessor.get_sentiment(positive_text)
        neg_sentiment = self.preprocessor.get_sentiment(negative_text)
        
        self.assertGreater(pos_sentiment['sentiment_polarity'], 0)
        self.assertLess(neg_sentiment['sentiment_polarity'], 0)


if __name__ == '__main__':
    unittest.main()
