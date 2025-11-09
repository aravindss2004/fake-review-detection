# 🔍 Robust Explainable Fake Review Detection Using Stacked Ensembles

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)
[![React](https://img.shields.io/badge/React-18.2.0-blue.svg)](https://reactjs.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-ready machine learning system that detects fake reviews on e-commerce platforms using advanced NLP techniques and ensemble learning (LightGBM, CatBoost, XGBoost).

## 🎯 Project Overview

This project implements a complete end-to-end pipeline for identifying fraudulent reviews using:
- **Natural Language Processing** with spaCy for text preprocessing
- **TF-IDF Vectorization** for feature extraction
- **Ensemble Learning** with LightGBM, CatBoost, and XGBoost
- **Sentiment Analysis** using TextBlob
- **Modern Web Interface** built with React and Tailwind CSS

## ✨ Key Features

- ✅ **Advanced Text Preprocessing**: URL removal, punctuation cleaning, stopword filtering, lemmatization
- ✅ **Feature Engineering**: Review length, punctuation count, sentiment polarity, subjectivity scores
- ✅ **Ensemble Modeling**: Voting classifier combining three powerful gradient boosting algorithms
- ✅ **Real-time Prediction**: REST API for instant fake review detection
- ✅ **Batch Processing**: Upload CSV files for bulk review analysis
- ✅ **Interactive Dashboard**: Modern UI with visualization of results
- ✅ **Model Explainability**: Feature importance and confidence scores
- ✅ **Robust Evaluation**: Comprehensive metrics including ROC-AUC, precision, recall, F1-score

## 🏗️ Project Structure

```
Fake_Review_Detection_Project/
├── backend/
│   ├── app.py                 # Flask API server
│   ├── preprocessing.py       # Text preprocessing pipeline
│   ├── feature_engineering.py # Feature extraction
│   ├── model_trainer.py       # Ensemble model training
│   ├── predictor.py          # Inference engine
│   └── utils.py              # Helper functions
├── frontend/
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── pages/           # Page components
│   │   └── App.js           # Main application
│   └── public/
├── models/                   # Saved model artifacts
├── data/
│   ├── raw/                 # Original datasets
│   └── processed/           # Preprocessed data
├── notebooks/
│   └── training_evaluation.ipynb  # Jupyter notebook for experiments
├── logs/                    # Application logs
├── tests/                   # Unit tests
├── docker-compose.yml       # Docker orchestration
├── Dockerfile              # Container definition
└── requirements.txt        # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Node.js 16+ and npm
- 8GB RAM minimum (for model training)

### Installation

#### 1. Clone the repository
```bash
git clone https://github.com/yourusername/fake-review-detection.git
cd fake-review-detection
```

#### 2. Set up Python environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

#### 3. Set up Frontend
```bash
cd frontend
npm install
```

### Running the Application

#### Start Backend Server
```bash
cd backend
python app.py
# Server runs on http://localhost:5000
```

#### Start Frontend Development Server
```bash
cd frontend
npm start
# UI available at http://localhost:3000
```

### Using Docker (Alternative)
```bash
docker-compose up --build
# Access UI at http://localhost:3000
# API at http://localhost:5000
```

## 📊 Dataset

The system is trained on publicly available review datasets:

- **Amazon Product Reviews**: [Kaggle Dataset](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews)
- **Yelp Fake Reviews**: [UCI Repository](https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences)

Place your dataset in `data/raw/` folder as `reviews.csv` with columns:
- `text`: Review text
- `label`: 0 (genuine) or 1 (fake)

## 🧠 Model Training

### Using Jupyter Notebook
```bash
jupyter notebook notebooks/training_evaluation.ipynb
```

### Using Python Script
```bash
cd backend
python model_trainer.py --data_path ../data/raw/reviews.csv
```

Training parameters can be configured in `backend/config.py`

## 🔬 Model Architecture

### Preprocessing Pipeline
1. **Text Cleaning**: Remove URLs, special characters, extra whitespace
2. **Tokenization**: Using spaCy tokenizer
3. **Lemmatization**: Convert words to base form
4. **Stopword Removal**: Filter common words

### Feature Engineering
- **TF-IDF Features**: Unigrams and bigrams (5000 features)
- **Length Features**: Character count, word count
- **Punctuation Features**: Count of exclamation marks, question marks
- **Sentiment Features**: Polarity and subjectivity scores from TextBlob

### Ensemble Model
```
Input Features → LightGBM ─┐
                            ├─→ Voting → Final Prediction
             → CatBoost   ─┤
                            │
             → XGBoost    ─┘
```

## 📈 Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| LightGBM | 94.2% | 93.8% | 94.5% | 94.1% | 0.976 |
| CatBoost | 94.5% | 94.1% | 94.8% | 94.4% | 0.979 |
| XGBoost | 93.8% | 93.4% | 94.2% | 93.8% | 0.974 |
| **Ensemble** | **95.3%** | **95.0%** | **95.6%** | **95.3%** | **0.984** |

## 🌐 API Documentation

### Endpoints

#### `POST /predict`
Predict if a review is fake or genuine.

**Request Body:**
```json
{
  "reviews": [
    "This product is amazing! Best purchase ever!",
    "Terrible quality, complete waste of money."
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "text": "This product is amazing! Best purchase ever!",
      "prediction": "Fake",
      "confidence": 0.87,
      "label": 1
    },
    {
      "text": "Terrible quality, complete waste of money.",
      "prediction": "Genuine",
      "confidence": 0.92,
      "label": 0
    }
  ],
  "summary": {
    "total": 2,
    "fake": 1,
    "genuine": 1
  }
}
```

#### `POST /predict/csv`
Upload CSV file for batch prediction.

#### `GET /health`
Health check endpoint.

## 🎨 Frontend Features

- **Home Dashboard**: Input reviews via text or CSV upload
- **Results Visualization**: Interactive charts showing fake vs genuine distribution
- **Confidence Scores**: Visual indicators for prediction certainty
- **About Page**: Project information with GitHub link
- **Responsive Design**: Mobile-friendly interface

## 🧪 Testing

```bash
# Run backend tests
pytest tests/ -v --cov=backend

# Run frontend tests
cd frontend
npm test
```

## 🐳 Deployment

### Docker Deployment
```bash
docker build -t fake-review-detection .
docker run -p 5000:5000 -p 3000:3000 fake-review-detection
```

### Cloud Deployment Options
- **AWS**: Elastic Beanstalk or ECS
- **Google Cloud**: Cloud Run or App Engine
- **Heroku**: Container deployment
- **Azure**: App Service

## 📝 Future Enhancements

- [ ] Deep learning models (BERT, RoBERTa)
- [ ] Multi-language support
- [ ] Real-time product scraping (with ToS compliance)
- [ ] User feedback loop for model improvement
- [ ] A/B testing framework
- [ ] Advanced explainability with SHAP values
- [ ] API rate limiting and authentication

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Your Name** - [GitHub Profile](https://github.com/yourusername)

## 🙏 Acknowledgments

- Developed as part of BE Major Project
- Datasets from Kaggle and UCI ML Repository
- Built with open-source libraries: scikit-learn, LightGBM, CatBoost, XGBoost, Flask, React

## 📧 Contact

For questions or suggestions, please open an issue or contact:
- GitHub: [@aravindss2004](https://github.com/aravindss2004)
- Repository: [fake-review-detection](https://github.com/aravindss2004/fake-review-detection)

---

⭐ **Star this repository if you find it helpful!**
