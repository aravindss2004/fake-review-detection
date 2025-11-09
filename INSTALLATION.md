# Installation Guide

Complete step-by-step installation instructions for the Fake Review Detection System.

## System Requirements

- **Operating System:** Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python:** 3.8 or higher
- **Node.js:** 16.0 or higher
- **RAM:** 8GB minimum (16GB recommended for training)
- **Disk Space:** 5GB free space

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/fake-review-detection.git
cd fake-review-detection
```

---

## Step 2: Backend Setup

### 2.1 Create Virtual Environment

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2.2 Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- Flask (web framework)
- scikit-learn (ML utilities)
- LightGBM, CatBoost, XGBoost (ensemble models)
- spaCy (NLP preprocessing)
- TextBlob (sentiment analysis)
- pandas, numpy (data manipulation)
- And more...

### 2.3 Download spaCy Language Model

```bash
python -m spacy download en_core_web_sm
```

### 2.4 Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 2.5 Verify Backend Installation

```bash
cd backend
python -c "import flask, sklearn, lightgbm, catboost, xgboost, spacy; print('✓ All packages installed successfully!')"
```

---

## Step 3: Frontend Setup

### 3.1 Navigate to Frontend Directory

```bash
cd ../frontend
```

### 3.2 Install Node Dependencies

```bash
npm install
```

This will install:
- React (UI framework)
- Tailwind CSS (styling)
- Chart.js (visualizations)
- React Router (navigation)
- Axios (API calls)

### 3.3 Create Environment File

```bash
# Copy example env file
cp .env.example .env
```

Edit `.env` and set:
```
REACT_APP_API_URL=http://localhost:5000
```

---

## Step 4: Prepare Dataset

### Option A: Download Recommended Dataset

1. Visit [Kaggle - Amazon Reviews](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews)
2. Download the dataset
3. Extract and save as `data/raw/reviews.csv`

The CSV should have columns:
- `text` - Review text
- `label` - 0 (genuine) or 1 (fake)

### Option B: Use Sample Data

Create a sample dataset for testing:

```bash
cd ..
python -c "
import pandas as pd
data = {
    'text': [
        'Great product, highly recommend!',
        'AMAZING!!! BEST EVER!!! BUY NOW!!!',
        'Good quality, works as expected.',
        'I got this free. Five stars!',
        'Decent product, nothing special.',
        'TERRIBLE! WORST PURCHASE EVER!!!',
        'Nice item, arrived on time.',
        'OMG THIS IS THE BEST THING EVER!!!'
    ] * 125,  # Repeat to get 1000 samples
    'label': [0, 1, 0, 1, 0, 1, 0, 1] * 125
}
pd.DataFrame(data).to_csv('data/raw/reviews.csv', index=False)
print('✓ Sample dataset created!')
"
```

---

## Step 5: Train Models

### 5.1 Using Jupyter Notebook (Recommended)

```bash
# Install Jupyter if not already installed
pip install jupyter

# Start Jupyter
jupyter notebook notebooks/training_evaluation.ipynb
```

Follow the notebook cells to:
1. Load and explore data
2. Preprocess text
3. Extract features
4. Train ensemble models
5. Evaluate performance
6. Save models

### 5.2 Using Python Script (Alternative)

Create `train.py` in the root directory:

```python
import sys
sys.path.append('backend')

from preprocessing import TextPreprocessor
from feature_engineering import FeatureExtractor, TFIDFFeatureExtractor, FeatureScaler
from model_trainer import EnsembleModelTrainer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
from config import *

# Load data
df = pd.read_csv('data/raw/reviews.csv')
print(f"Loaded {len(df)} reviews")

# Preprocess
preprocessor = TextPreprocessor()
df['cleaned'] = preprocessor.preprocess_batch(df['text'].tolist(), True)

# Extract features
extractor = FeatureExtractor()
ling_feats = extractor.extract_features_batch(df['cleaned'].tolist(), df['text'].tolist())

# Split
X_text_tr, X_text_te, X_ling_tr, X_ling_te, y_tr, y_te = train_test_split(
    df['cleaned'], ling_feats, df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# TF-IDF
tfidf = TFIDFFeatureExtractor()
X_tfidf_tr = tfidf.fit_transform(X_text_tr.tolist())
X_tfidf_te = tfidf.transform(X_text_te.tolist())

# Scale
scaler = FeatureScaler()
X_ling_tr_sc = scaler.fit_transform(X_ling_tr.values)
X_ling_te_sc = scaler.transform(X_ling_te.values)

# Combine
X_train = np.hstack([X_tfidf_tr, X_ling_tr_sc])
X_test = np.hstack([X_tfidf_te, X_ling_te_sc])

# Train
trainer = EnsembleModelTrainer()
models = trainer.train(X_train, y_tr.values)
results = trainer.evaluate_all_models(X_test, y_te.values)

# Save
trainer.save_models()
joblib.dump(tfidf.vectorizer, TFIDF_VECTORIZER_PATH)
joblib.dump(scaler.scaler, FEATURE_SCALER_PATH)

print("✓ Training complete! Models saved.")
```

Run:
```bash
python train.py
```

---

## Step 6: Verify Installation

### 6.1 Check Model Files

```bash
ls models/
```

Should show:
- `ensemble_model.joblib`
- `lightgbm_model.joblib`
- `catboost_model.joblib`
- `xgboost_model.joblib`
- `tfidf_vectorizer.joblib`
- `feature_scaler.joblib`

### 6.2 Test Backend API

```bash
cd backend
python app.py
```

You should see:
```
Starting Fake Review Detection API Server
Host: 0.0.0.0
Port: 5000
```

Open browser and visit: `http://localhost:5000/health`

### 6.3 Test Frontend

In a new terminal:
```bash
cd frontend
npm start
```

Browser should open automatically at `http://localhost:3000`

---

## Step 7: Run the Complete System

### Terminal 1 - Backend:
```bash
cd backend
python app.py
```

### Terminal 2 - Frontend:
```bash
cd frontend
npm start
```

Now you can:
1. Navigate to `http://localhost:3000`
2. Enter review text or upload CSV
3. Click "Analyze Reviews"
4. View predictions and statistics

---

## Troubleshooting

### Issue: spaCy model not found

**Solution:**
```bash
python -m spacy download en_core_web_sm
```

### Issue: Module not found errors

**Solution:**
```bash
pip install -r requirements.txt --force-reinstall
```

### Issue: Port already in use

**Solution:**
```bash
# For backend (edit backend/config.py)
API_PORT = 5001

# For frontend
PORT=3001 npm start
```

### Issue: CORS errors in browser

**Solution:** Ensure backend is running and CORS is enabled in `backend/app.py`

### Issue: Models not loading

**Solution:**
```bash
# Retrain models
cd notebooks
jupyter notebook training_evaluation.ipynb
# Run all cells
```

### Issue: npm install fails

**Solution:**
```bash
# Clear cache and retry
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

---

## Quick Start Script

Create `start.sh` (Linux/macOS) or `start.bat` (Windows):

**start.sh:**
```bash
#!/bin/bash
echo "Starting Fake Review Detection System..."

# Start backend
cd backend
source ../venv/bin/activate
python app.py &
BACKEND_PID=$!

# Start frontend
cd ../frontend
npm start &
FRONTEND_PID=$!

echo "System started!"
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo "Press Ctrl+C to stop"

wait
```

**start.bat (Windows):**
```batch
@echo off
echo Starting Fake Review Detection System...

start cmd /k "cd backend && venv\Scripts\activate && python app.py"
start cmd /k "cd frontend && npm start"

echo System started!
```

Make executable and run:
```bash
chmod +x start.sh
./start.sh
```

---

## Next Steps

1. **Customize Models:** Adjust hyperparameters in `backend/config.py`
2. **Add More Data:** Improve accuracy by training on larger datasets
3. **Deploy:** See `DEPLOYMENT.md` for cloud deployment options
4. **Contribute:** Submit pull requests on GitHub

---

## Support

- **Documentation:** See `README.md`
- **Datasets:** See `DATASETS.md`
- **Deployment:** See `DEPLOYMENT.md`
- **Issues:** https://github.com/yourusername/fake-review-detection/issues

---

## Verify Complete Installation

Run this verification script:

```bash
python -c "
import sys
sys.path.append('backend')

print('Checking installation...')
checks = []

try:
    import flask
    checks.append(('Flask', True))
except: checks.append(('Flask', False))

try:
    import sklearn
    checks.append(('scikit-learn', True))
except: checks.append(('scikit-learn', False))

try:
    import lightgbm
    checks.append(('LightGBM', True))
except: checks.append(('LightGBM', False))

try:
    import catboost
    checks.append(('CatBoost', True))
except: checks.append(('CatBoost', False))

try:
    import xgboost
    checks.append(('XGBoost', True))
except: checks.append(('XGBoost', False))

try:
    import spacy
    nlp = spacy.load('en_core_web_sm')
    checks.append(('spaCy + model', True))
except: checks.append(('spaCy + model', False))

try:
    import textblob
    checks.append(('TextBlob', True))
except: checks.append(('TextBlob', False))

print('\nInstallation Status:')
for name, status in checks:
    status_str = '✓' if status else '✗'
    print(f'{status_str} {name}')

all_ok = all(status for _, status in checks)
if all_ok:
    print('\n✓ All components installed successfully!')
else:
    print('\n✗ Some components missing. Please reinstall.')
"
```

✅ **Installation Complete!**
