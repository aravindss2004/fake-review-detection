# API Documentation

Complete API reference for the Fake Review Detection System.

## Base URL

```
http://localhost:5000
```

For production, replace with your deployed URL.

---

## Authentication

Currently, the API does not require authentication. For production, implement:
- API keys
- OAuth2
- JWT tokens

---

## Endpoints

### 1. Health Check

Check if the API is running and models are loaded.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "models_loaded": true,
  "transformers_loaded": true,
  "details": {
    "models": {
      "lightgbm": true,
      "catboost": true,
      "xgboost": true,
      "ensemble": true
    },
    "transformers": {
      "tfidf_vectorizer": true,
      "feature_scaler": true
    }
  }
}
```

**Status Codes:**
- `200 OK`: System is healthy
- `500 Internal Server Error`: System issues

---

### 2. Predict Single/Multiple Reviews

Analyze review(s) for authenticity.

**Endpoint:** `POST /predict`

**Request Body (Single Review):**
```json
{
  "review": "This product is amazing! Best purchase ever!"
}
```

**Request Body (Multiple Reviews):**
```json
{
  "reviews": [
    "Great product, highly recommend!",
    "TERRIBLE! WORST PURCHASE!!!",
    "Good quality, works as expected."
  ]
}
```

**Response:**
```json
{
  "success": true,
  "message": "Predictions generated successfully",
  "data": {
    "predictions": [
      {
        "text": "Great product, highly recommend!",
        "prediction": "Genuine",
        "label": 0,
        "confidence": 0.92,
        "fake_probability": 0.08,
        "genuine_probability": 0.92
      },
      {
        "text": "TERRIBLE! WORST PURCHASE!!!",
        "prediction": "Fake",
        "label": 1,
        "confidence": 0.87,
        "fake_probability": 0.87,
        "genuine_probability": 0.13
      }
    ],
    "summary": {
      "total_reviews": 2,
      "fake_reviews": 1,
      "genuine_reviews": 1,
      "fake_percentage": 50.0,
      "genuine_percentage": 50.0,
      "average_confidence": 0.895,
      "average_fake_probability": 0.475
    }
  }
}
```

**Status Codes:**
- `200 OK`: Successful prediction
- `400 Bad Request`: Invalid input
- `500 Internal Server Error`: Prediction failed

**Validation Rules:**
- Text cannot be empty
- Maximum text length: 10,000 characters
- Maximum batch size: Unlimited (but consider memory)

---

### 3. Predict from CSV

Upload CSV file for batch prediction.

**Endpoint:** `POST /predict/csv`

**Request:**
- Content-Type: `multipart/form-data`
- Field name: `file`
- File type: `.csv`

**CSV Format:**
```csv
text
"Great product!"
"Terrible quality!"
"Works as expected."
```

**Alternative column names accepted:**
- `text`
- `review`
- `review_text`
- `content`

**Response:**
```json
{
  "success": true,
  "message": "CSV predictions generated successfully",
  "data": {
    "total_reviews": 100,
    "summary": {
      "total_reviews": 100,
      "fake_reviews": 35,
      "genuine_reviews": 65,
      "fake_percentage": 35.0,
      "genuine_percentage": 65.0,
      "average_confidence": 0.91
    },
    "output_file": "predictions_20240115_103000.csv",
    "preview": [
      {
        "text": "Great product!",
        "prediction": "Genuine",
        "confidence": 0.94,
        "fake_probability": 0.06,
        "genuine_probability": 0.94
      }
    ]
  }
}
```

**Status Codes:**
- `200 OK`: Successful processing
- `400 Bad Request`: Invalid file or format
- `413 Payload Too Large`: File exceeds 16MB
- `500 Internal Server Error`: Processing failed

---

### 4. Model Information

Get information about loaded models.

**Endpoint:** `GET /model/info`

**Response:**
```json
{
  "success": true,
  "message": "Model information retrieved",
  "data": {
    "models": {
      "lightgbm": true,
      "catboost": true,
      "xgboost": true,
      "ensemble": true
    },
    "transformers": {
      "tfidf_vectorizer": true,
      "feature_scaler": true
    }
  }
}
```

---

### 5. Scraping Endpoint (Disabled)

**Endpoint:** `POST /scrape`

**Response:**
```json
{
  "success": false,
  "message": "Scraping feature is disabled",
  "data": {
    "status": "disabled",
    "message": "Scraping feature is disabled in this version for security and ToS compliance.",
    "alternative": "Please manually input reviews or upload a CSV file."
  }
}
```

**Status Code:** `501 Not Implemented`

---

## Error Responses

All error responses follow this format:

```json
{
  "success": false,
  "message": "Error description",
  "error": "Detailed error message"
}
```

### Common Error Codes

- `400 Bad Request`: Invalid input data
- `404 Not Found`: Endpoint doesn't exist
- `413 Payload Too Large`: File/request too large
- `500 Internal Server Error`: Server error
- `501 Not Implemented`: Feature not available

---

## Rate Limiting

Currently no rate limiting. For production:

```
Rate Limit: 100 requests/minute per IP
Header: X-RateLimit-Remaining: 95
```

---

## Examples

### Python

```python
import requests

# Single review
response = requests.post(
    'http://localhost:5000/predict',
    json={'review': 'This product is amazing!'}
)
print(response.json())

# Multiple reviews
response = requests.post(
    'http://localhost:5000/predict',
    json={
        'reviews': [
            'Great quality!',
            'Terrible product!!!'
        ]
    }
)
print(response.json())

# CSV upload
with open('reviews.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/predict/csv',
        files={'file': f}
    )
print(response.json())
```

### cURL

```bash
# Single review
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "Amazing product!"}'

# Multiple reviews
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"reviews": ["Great!", "Terrible!!!"]}'

# CSV upload
curl -X POST http://localhost:5000/predict/csv \
  -F "file=@reviews.csv"

# Health check
curl http://localhost:5000/health
```

### JavaScript (Axios)

```javascript
import axios from 'axios';

const API_URL = 'http://localhost:5000';

// Single review
const predictSingle = async (review) => {
  const response = await axios.post(`${API_URL}/predict`, {
    review: review
  });
  return response.data;
};

// Multiple reviews
const predictMultiple = async (reviews) => {
  const response = await axios.post(`${API_URL}/predict`, {
    reviews: reviews
  });
  return response.data;
};

// CSV upload
const predictCSV = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await axios.post(`${API_URL}/predict/csv`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  return response.data;
};

// Usage
predictSingle('Great product!').then(console.log);
```

---

## Response Time

Average response times:
- Single review: ~100-200ms
- Batch (10 reviews): ~500-800ms
- CSV (100 reviews): ~2-5 seconds

*Times may vary based on server specifications.*

---

## CORS

CORS is enabled for all origins in development. For production, configure specific origins in `backend/app.py`:

```python
from flask_cors import CORS

CORS(app, origins=['https://your-frontend-domain.com'])
```

---

## Pagination

For large CSV uploads, consider implementing pagination:

```json
{
  "page": 1,
  "per_page": 100,
  "total_pages": 5,
  "total_reviews": 500,
  "predictions": [...]
}
```

*Currently not implemented. Process entire file at once.*

---

## Webhooks

For async processing (future feature):

```json
{
  "webhook_url": "https://your-app.com/webhook",
  "csv_file": "..."
}
```

*Not yet implemented.*

---

## SDK Support

Official SDKs planned for:
- Python
- JavaScript/Node.js
- Java
- Ruby

*Coming soon!*

---

## Versioning

Current version: `v1.0.0`

API versioning will be introduced in future releases:
```
http://localhost:5000/v1/predict
http://localhost:5000/v2/predict
```

---

## Support

For API issues or questions:
- GitHub Issues: https://github.com/aravindss2004/fake-review-detection/issues
- Documentation: See README.md

---

## Changelog

### v1.0.0 (Current)
- Initial API release
- Basic prediction endpoints
- CSV upload support
- Health check endpoint
- Model info endpoint

---

**Last Updated:** January 2024
