# Trained Models Directory

This directory contains trained machine learning models.

## Model Files

After training, the following files will be generated:

- `lightgbm_model.joblib` - LightGBM classifier (~900 KB)
- `catboost_model.joblib` - CatBoost classifier (~450 KB)
- `xgboost_model.joblib` - XGBoost classifier (~600 KB)
- `ensemble_model.joblib` - Voting ensemble (~4 MB)
- `tfidf_vectorizer.joblib` - TF-IDF transformer (~180 KB)
- `feature_scaler.joblib` - Feature scaler (~1 KB)

**Total size: ~6 MB**

## Why Models Are Not Included

The trained model files are **not included** in this repository because:

1. **Size**: Models are large (6+ MB) and not suitable for Git
2. **Reproducibility**: Users should train on their own datasets
3. **Customization**: Models should be trained on your specific data
4. **Best Practice**: Trained models are build artifacts, not source code

## How to Train Models

### Prerequisites

1. Ensure you have the dataset in `data/raw/` (see `data/raw/README.md`)
2. Install all requirements: `pip install -r requirements.txt`

### Training Steps

```bash
# 1. Prepare the data (if not already done)
python prepare_data.py

# 2. Train the models
python train_model.py
```

### Training Time

- **Preprocessing**: ~10-15 minutes for 50,000 reviews
- **Model Training**: ~5-10 minutes
- **Total**: ~15-25 minutes

### Expected Accuracy

After training on 50,000 reviews, you should achieve:

- **LightGBM**: ~85-86% accuracy
- **CatBoost**: ~82-83% accuracy  
- **XGBoost**: ~83-84% accuracy
- **Ensemble**: ~84-85% accuracy

## Quick Start

If you want to test the system without training:

1. Download pre-trained models from the releases page (if available)
2. Or train your own using the steps above
3. Then start the backend: `cd backend && python app.py`

## Model Details

### LightGBM (Best Performance)
- Type: Gradient Boosting Decision Tree
- Features: 5,009 (TF-IDF + Linguistic)
- Hyperparameters: Optimized for speed and accuracy

### CatBoost
- Type: Gradient Boosting with categorical features support
- Features: Same as above
- Advantages: Handles categorical data well

### XGBoost  
- Type: Extreme Gradient Boosting
- Features: Same as above
- Advantages: Robust and widely used

### Ensemble (Voting Classifier)
- Combines predictions from all three models
- Uses soft voting (probability averaging)
- Often more robust than individual models

## Troubleshooting

**"Models not found" error:**
- Make sure you've run `python train_model.py`
- Check that all 6 `.joblib` files are in this directory

**Low accuracy:**
- Ensure you have enough training data (10,000+ reviews minimum)
- Check that your dataset is balanced (similar number of fake and genuine)
- Verify data quality and preprocessing

**Training taking too long:**
- Reduce dataset size in `prepare_data.py`
- Consider using a smaller vocabulary size in feature engineering

## Need Help?

See the main [README.md](../README.md) or [INSTALLATION.md](../INSTALLATION.md) for more details.
