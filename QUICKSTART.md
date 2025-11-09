# Quick Start Guide

Get up and running with the Fake Review Detection System in 10 minutes!

## Prerequisites

- Python 3.8+ installed
- Node.js 16+ installed
- 8GB RAM available

---

## 🚀 5-Minute Setup

### Step 1: Clone & Install (2 minutes)

```bash
# Clone repository
git clone https://github.com/yourusername/fake-review-detection.git
cd fake-review-detection

# Install Python dependencies
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Install Frontend dependencies
cd frontend
npm install
cd ..
```

### Step 2: Prepare Sample Data (1 minute)

```bash
python -c "
import pandas as pd
import os
os.makedirs('data/raw', exist_ok=True)
data = {
    'text': [
        'Great product, highly recommend it!',
        'AMAZING!!! BEST EVER!!! BUY NOW!!!',
        'Good quality, works as expected.',
        'Received this for free. Five stars!',
        'Decent product, arrived on time.',
        'TERRIBLE! WORST PURCHASE!!!',
        'Nice item, does the job well.',
        'OMG THIS IS THE BEST THING EVER!!!'
    ] * 125,
    'label': [0, 1, 0, 1, 0, 1, 0, 1] * 125
}
pd.DataFrame(data).to_csv('data/raw/reviews.csv', index=False)
print('Sample dataset created!')
"
```

### Step 3: Train Models (5 minutes)

```bash
python train_model.py
```

Wait for training to complete. You'll see output like:
```
Training LightGBM...
Training CatBoost...
Training XGBoost...
Creating voting ensemble...
✓ Training complete! Models saved.
```

### Step 4: Launch Application (2 minutes)

**Terminal 1 - Backend:**
```bash
cd backend
python app.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm start
```

### Step 5: Use the System!

1. Open browser: `http://localhost:3000`
2. Enter reviews or upload CSV
3. Click "Analyze Reviews"
4. View predictions and statistics!

---

## 📋 Quick Commands Reference

### Run Backend
```bash
cd backend && python app.py
```

### Run Frontend
```bash
cd frontend && npm start
```

### Run Tests
```bash
cd tests && python -m pytest
```

### Retrain Models
```bash
python train_model.py --data data/raw/your_data.csv
```

### Run with Docker
```bash
docker-compose up --build
```

---

## 🎯 Test the API

### Using cURL

```bash
# Health check
curl http://localhost:5000/health

# Predict single review
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "This product is amazing!"}'

# Predict multiple reviews
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"reviews": ["Great product!", "TERRIBLE QUALITY!!!"]}'
```

### Using Python

```python
import requests

# Predict reviews
response = requests.post(
    'http://localhost:5000/predict',
    json={'reviews': ['Great product!', 'Worst purchase ever!!!']}
)

print(response.json())
```

---

## 📊 Expected Output

After training, you should see models achieving:

- **Accuracy:** ~95%
- **Precision:** ~95%
- **Recall:** ~96%
- **F1-Score:** ~95%
- **ROC-AUC:** ~0.98

---

## 🔧 Troubleshooting Quick Fixes

### Models not found?
```bash
python train_model.py
```

### Port already in use?
```bash
# Backend: Edit backend/config.py, change API_PORT
# Frontend: Run with PORT=3001 npm start
```

### spaCy model error?
```bash
python -m spacy download en_core_web_sm
```

### CORS errors?
Check that backend is running on port 5000

---

## 📚 Next Steps

1. **Add Real Data:** See `DATASETS.md` for recommended datasets
2. **Improve Models:** Adjust hyperparameters in `backend/config.py`
3. **Deploy:** See `DEPLOYMENT.md` for cloud deployment
4. **Customize UI:** Edit React components in `frontend/src/`

---

## 💡 Tips for Best Results

1. **More Data = Better Model:** Train on 10,000+ reviews
2. **Balanced Dataset:** Keep fake/genuine ratio around 50-50
3. **Clean Data:** Remove duplicates and fix labels
4. **Monitor Performance:** Check confusion matrix and ROC curve
5. **Regular Retraining:** Update model as new reviews come in

---

## 🎓 Understanding the System

**Preprocessing Pipeline:**
```
Raw Review → Clean Text → Tokenize → Lemmatize → Remove Stopwords
```

**Feature Extraction:**
```
Text → TF-IDF (5000 features) + Linguistic Features (9 features)
```

**Ensemble Prediction:**
```
Features → LightGBM  │
                     ├→ Voting → Final Prediction
          → CatBoost │
                     │
          → XGBoost  ┘
```

---

## 🆘 Getting Help

- **Full Documentation:** `README.md`
- **Installation Issues:** `INSTALLATION.md`
- **Dataset Help:** `DATASETS.md`
- **Deployment:** `DEPLOYMENT.md`
- **GitHub Issues:** https://github.com/aravindss2004/fake-review-detection/issues

---

## ✅ Verification Checklist

After setup, verify:

- [ ] Backend runs without errors at `http://localhost:5000`
- [ ] Frontend opens at `http://localhost:3000`
- [ ] `/health` endpoint returns `{"status": "healthy"}`
- [ ] Can analyze sample reviews successfully
- [ ] Models folder contains 6 `.joblib` files
- [ ] No CORS errors in browser console

---

**Ready to detect fake reviews! 🚀**

For production deployment and advanced features, see the full documentation.
