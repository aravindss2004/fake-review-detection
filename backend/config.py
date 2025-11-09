"""
Configuration file for the fake review detection system.
"""
import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = BASE_DIR / 'models'
LOGS_DIR = BASE_DIR / 'logs'

# Ensure directories exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model paths
TFIDF_VECTORIZER_PATH = MODELS_DIR / 'tfidf_vectorizer.joblib'
FEATURE_SCALER_PATH = MODELS_DIR / 'feature_scaler.joblib'
LIGHTGBM_MODEL_PATH = MODELS_DIR / 'lightgbm_model.joblib'
CATBOOST_MODEL_PATH = MODELS_DIR / 'catboost_model.joblib'
XGBOOST_MODEL_PATH = MODELS_DIR / 'xgboost_model.joblib'
ENSEMBLE_MODEL_PATH = MODELS_DIR / 'ensemble_model.joblib'

# Data paths
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / 'train.csv'
TEST_DATA_PATH = PROCESSED_DATA_DIR / 'test.csv'

# Preprocessing settings
SPACY_MODEL = 'en_core_web_sm'
MAX_FEATURES_TFIDF = 5000
NGRAM_RANGE = (1, 2)  # Unigrams and bigrams
MIN_DF = 2
MAX_DF = 0.95

# Model hyperparameters
LIGHTGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': -1,
    'min_child_samples': 20,
    'n_estimators': 200,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

CATBOOST_PARAMS = {
    'iterations': 200,
    'learning_rate': 0.05,
    'depth': 6,
    'l2_leaf_reg': 3,
    'border_count': 128,
    'random_seed': 42,
    'verbose': False,
    'task_type': 'CPU',
    'thread_count': -1
}

XGBOOST_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'gamma': 0,
    'reg_alpha': 0,
    'reg_lambda': 1,
    'random_state': 42,
    'n_jobs': -1
}

# Training settings
TEST_SIZE = 0.2
RANDOM_STATE = 42
CROSS_VALIDATION_FOLDS = 5

# API settings
API_HOST = '0.0.0.0'
API_PORT = 5000
DEBUG = True
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max file size

# Logging settings
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = LOGS_DIR / 'app.log'

# Feature names
NUMERIC_FEATURES = [
    'char_count',
    'word_count',
    'avg_word_length',
    'punctuation_count',
    'exclamation_count',
    'question_count',
    'capital_ratio',
    'sentiment_polarity',
    'sentiment_subjectivity'
]

# Text cleaning patterns
URL_PATTERN = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
PHONE_PATTERN = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'

# Threshold for fake detection
FAKE_THRESHOLD = 0.5
