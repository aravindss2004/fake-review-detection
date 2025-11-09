# Deployment Guide

This guide covers deploying the Fake Review Detection System to various platforms.

## Table of Contents
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
  - [Heroku](#heroku)
  - [AWS](#aws)
  - [Google Cloud](#google-cloud)
  - [Azure](#azure)
- [Production Checklist](#production-checklist)

---

## Local Development

### Backend
```bash
# Navigate to backend
cd backend

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Run Flask server
python app.py

# Server runs on http://localhost:5000
```

### Frontend
```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start development server
npm start

# UI available at http://localhost:3000
```

---

## Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Build and start services
docker-compose up --build

# Run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Using Dockerfile Only

```bash
# Build image
docker build -t fake-review-detection .

# Run container
docker run -p 5000:5000 -v $(pwd)/models:/app/models fake-review-detection
```

---

## Cloud Deployment

### Heroku

#### 1. Create Heroku App
```bash
# Install Heroku CLI
# Windows: Download from https://devcenter.heroku.com/articles/heroku-cli
# macOS: brew tap heroku/brew && brew install heroku

# Login
heroku login

# Create app
heroku create fake-review-detector

# Set buildpacks
heroku buildpacks:add heroku/python
heroku buildpacks:add heroku/nodejs
```

#### 2. Create Procfile
```procfile
web: python backend/app.py
```

#### 3. Deploy
```bash
# Commit changes
git add .
git commit -m "Prepare for Heroku deployment"

# Push to Heroku
git push heroku main

# Open app
heroku open
```

---

### AWS (Elastic Beanstalk)

#### 1. Install EB CLI
```bash
pip install awsebcli
```

#### 2. Initialize EB Application
```bash
eb init -p python-3.10 fake-review-detection --region us-east-1
```

#### 3. Create Environment and Deploy
```bash
eb create production-env
eb deploy
eb open
```

#### 4. Configure Environment
```bash
# Set environment variables
eb setenv FLASK_ENV=production

# Scale instances
eb scale 2
```

---

### Google Cloud Platform (App Engine)

#### 1. Create app.yaml
```yaml
runtime: python310

instance_class: F2

env_variables:
  FLASK_ENV: 'production'

handlers:
- url: /static
  static_dir: frontend/build/static

- url: /.*
  script: auto
```

#### 2. Deploy
```bash
# Install gcloud SDK
# Follow: https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login

# Create project
gcloud projects create fake-review-detection

# Set project
gcloud config set project fake-review-detection

# Deploy
gcloud app deploy

# Open app
gcloud app browse
```

---

### Microsoft Azure (App Service)

#### 1. Install Azure CLI
```bash
# Windows: Download from https://aka.ms/installazurecliwindows
# macOS: brew install azure-cli
```

#### 2. Login and Create Resources
```bash
# Login
az login

# Create resource group
az group create --name FakeReviewRG --location eastus

# Create App Service plan
az appservice plan create --name FakeReviewPlan --resource-group FakeReviewRG --sku B1 --is-linux

# Create web app
az webapp create --resource-group FakeReviewRG --plan FakeReviewPlan --name fake-review-detector --runtime "PYTHON|3.10"
```

#### 3. Deploy
```bash
# Deploy from local Git
az webapp deployment source config-local-git --name fake-review-detector --resource-group FakeReviewRG

# Get deployment URL
az webapp deployment list-publishing-credentials --name fake-review-detector --resource-group FakeReviewRG --query scmUri --output tsv

# Add remote and push
git remote add azure <deployment-url>
git push azure main
```

---

## Production Checklist

### Security
- [ ] Set `DEBUG=False` in production
- [ ] Use environment variables for secrets
- [ ] Enable HTTPS/SSL
- [ ] Implement rate limiting
- [ ] Add authentication if needed
- [ ] Sanitize all user inputs
- [ ] Set up CORS properly

### Performance
- [ ] Enable caching (Redis)
- [ ] Use production WSGI server (Gunicorn)
- [ ] Optimize model loading
- [ ] Implement request queuing
- [ ] Set up CDN for static files
- [ ] Enable gzip compression

### Monitoring
- [ ] Set up error tracking (Sentry)
- [ ] Configure logging
- [ ] Monitor API performance
- [ ] Set up health checks
- [ ] Track model accuracy over time
- [ ] Set up alerts

### Scaling
- [ ] Use load balancer
- [ ] Implement horizontal scaling
- [ ] Cache model predictions
- [ ] Consider serverless functions
- [ ] Use message queues for async tasks

---

## Environment Variables

Create `.env` file:

```bash
# Backend
FLASK_ENV=production
FLASK_DEBUG=0
SECRET_KEY=your-secret-key-here

# API
API_HOST=0.0.0.0
API_PORT=5000
MAX_CONTENT_LENGTH=16777216

# Models
MODEL_PATH=/app/models

# Frontend
REACT_APP_API_URL=https://your-api-url.com
```

---

## Using Gunicorn (Production Server)

Install Gunicorn:
```bash
pip install gunicorn
```

Update requirements.txt:
```
gunicorn==21.2.0
```

Run with Gunicorn:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 backend.app:app
```

Update Procfile for Heroku:
```
web: gunicorn -w 4 -b 0.0.0.0:$PORT backend.app:app
```

---

## Nginx Configuration (Optional)

For serving with Nginx as reverse proxy:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /static {
        alias /path/to/frontend/build/static;
    }
}
```

---

## Continuous Deployment

### GitHub Actions Example

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Production

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Deploy to Heroku
      uses: akhileshns/heroku-deploy@v3.12.12
      with:
        heroku_api_key: ${{secrets.HEROKU_API_KEY}}
        heroku_app_name: "fake-review-detector"
        heroku_email: "your-email@example.com"
```

---

## Troubleshooting

### Common Issues

**Model files not found:**
```bash
# Ensure models are uploaded
ls models/
# Should show: ensemble_model.joblib, tfidf_vectorizer.joblib, etc.
```

**Port already in use:**
```bash
# Change port in config.py or use different port
export API_PORT=5001
```

**Memory issues:**
```bash
# Increase container memory
docker run -m 2g -p 5000:5000 fake-review-detection
```

---

## Support

For deployment issues:
- GitHub Issues: https://github.com/aravindss2004/fake-review-detection/issues
- Documentation: Check README.md

---

**Production URL Examples:**
- Heroku: `https://fake-review-detector.herokuapp.com`
- AWS: `http://fake-review-env.eba-xxx.us-east-1.elasticbeanstalk.com`
- GCP: `https://fake-review-detection.uc.r.appspot.com`
