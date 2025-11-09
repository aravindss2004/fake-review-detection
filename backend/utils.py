"""
Utility functions for the fake review detection system.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Union, List, Dict, Any
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(file_path: Union[str, Path], 
                 text_column: str = 'text',
                 label_column: str = 'label') -> pd.DataFrame:
    """
    Load dataset from CSV file.
    
    Args:
        file_path: Path to CSV file
        text_column: Name of the text column
        label_column: Name of the label column
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Dataset loaded: {len(df)} samples")
        
        # Check required columns
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in dataset")
        
        if label_column in df.columns:
            # Check label distribution
            label_counts = df[label_column].value_counts()
            logger.info(f"Label distribution:\n{label_counts}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise


def save_dataset(df: pd.DataFrame, file_path: Union[str, Path]):
    """
    Save dataset to CSV file.
    
    Args:
        df: DataFrame to save
        file_path: Path to save CSV file
    """
    try:
        df.to_csv(file_path, index=False)
        logger.info(f"Dataset saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving dataset: {str(e)}")
        raise


def validate_input_text(text: str) -> bool:
    """
    Validate input text.
    
    Args:
        text: Input text
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(text, str):
        return False
    
    if len(text.strip()) == 0:
        return False
    
    if len(text) > 10000:  # Max 10k characters
        return False
    
    return True


def validate_csv_file(file_path: Union[str, Path], required_columns: List[str]) -> bool:
    """
    Validate CSV file format.
    
    Args:
        file_path: Path to CSV file
        required_columns: List of required column names
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        df = pd.read_csv(file_path, nrows=1)
        
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Required column '{col}' not found")
                return False
        
        return True
    
    except Exception as e:
        logger.error(f"Error validating CSV: {str(e)}")
        return False


def format_predictions_for_export(predictions: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Format predictions for export to CSV.
    
    Args:
        predictions: List of prediction results
        
    Returns:
        pd.DataFrame: Formatted predictions
    """
    df = pd.DataFrame(predictions)
    
    # Select and order columns
    columns = ['text', 'prediction', 'confidence', 'fake_probability', 'genuine_probability']
    df = df[columns]
    
    # Round probabilities
    df['confidence'] = df['confidence'].round(4)
    df['fake_probability'] = df['fake_probability'].round(4)
    df['genuine_probability'] = df['genuine_probability'].round(4)
    
    return df


def calculate_statistics(df: pd.DataFrame, label_column: str = 'label') -> Dict[str, Any]:
    """
    Calculate dataset statistics.
    
    Args:
        df: DataFrame
        label_column: Name of the label column
        
    Returns:
        Dict: Statistics
    """
    stats = {
        'total_samples': len(df),
        'label_distribution': df[label_column].value_counts().to_dict() if label_column in df.columns else {},
        'missing_values': df.isnull().sum().to_dict(),
        'columns': list(df.columns)
    }
    
    return stats


def create_response(success: bool, message: str, data: Any = None, 
                   error: str = None) -> Dict[str, Any]:
    """
    Create standardized API response.
    
    Args:
        success: Whether request was successful
        message: Response message
        data: Response data
        error: Error message if any
        
    Returns:
        Dict: Formatted response
    """
    response = {
        'success': success,
        'message': message
    }
    
    if data is not None:
        response['data'] = data
    
    if error is not None:
        response['error'] = error
    
    return response


def sanitize_text(text: str) -> str:
    """
    Sanitize text input.
    
    Args:
        text: Input text
        
    Returns:
        str: Sanitized text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Truncate if too long
    if len(text) > 10000:
        text = text[:10000]
    
    return text.strip()


def batch_data(data: List[Any], batch_size: int = 32) -> List[List[Any]]:
    """
    Split data into batches.
    
    Args:
        data: List of data items
        batch_size: Size of each batch
        
    Returns:
        List[List]: List of batches
    """
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]


def get_model_info() -> Dict[str, Any]:
    """
    Get information about loaded models.
    
    Returns:
        Dict: Model information
    """
    from config import (
        LIGHTGBM_MODEL_PATH, CATBOOST_MODEL_PATH, 
        XGBOOST_MODEL_PATH, ENSEMBLE_MODEL_PATH,
        TFIDF_VECTORIZER_PATH, FEATURE_SCALER_PATH
    )
    
    model_info = {
        'models': {
            'lightgbm': LIGHTGBM_MODEL_PATH.exists(),
            'catboost': CATBOOST_MODEL_PATH.exists(),
            'xgboost': XGBOOST_MODEL_PATH.exists(),
            'ensemble': ENSEMBLE_MODEL_PATH.exists()
        },
        'transformers': {
            'tfidf_vectorizer': TFIDF_VECTORIZER_PATH.exists(),
            'feature_scaler': FEATURE_SCALER_PATH.exists()
        }
    }
    
    return model_info


if __name__ == "__main__":
    logger.info("Utility functions loaded successfully!")
