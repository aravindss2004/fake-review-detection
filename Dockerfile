# Multi-stage Dockerfile for Fake Review Detection System

# Stage 1: Build Frontend
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --only=production
COPY frontend/ ./
RUN npm run build

# Stage 2: Python Backend
FROM python:3.10-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Copy backend code
COPY backend/ ./backend/
COPY models/ ./models/
COPY data/ ./data/
COPY logs/ ./logs/

# Copy frontend build
COPY --from=frontend-builder /app/frontend/build ./frontend/build

# Expose ports
EXPOSE 5000

# Environment variables
ENV FLASK_APP=backend/app.py
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "backend/app.py"]
