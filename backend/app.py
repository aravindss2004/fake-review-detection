"""
Flask API server for fake review detection.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import traceback
from werkzeug.utils import secure_filename
import os
from datetime import datetime

from config import (
    API_HOST, API_PORT, DEBUG, MAX_CONTENT_LENGTH,
    LOG_FILE, LOG_FORMAT, LOG_LEVEL
)
from predictor import FakeReviewPredictor, ScrapePredictor
from utils import (
    validate_input_text, validate_csv_file, format_predictions_for_export,
    create_response, sanitize_text, get_model_info
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
CORS(app)

# Initialize predictor
predictor = None

# Upload folder
UPLOAD_FOLDER = Path(__file__).parent.parent / 'data' / 'uploads'
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv'}


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def initialize_predictor():
    """Initialize the predictor with loaded models."""
    global predictor
    try:
        if predictor is None:
            logger.info("Initializing predictor...")
            predictor = FakeReviewPredictor(use_ensemble=True)
            predictor.load_models()
            logger.info("Predictor initialized successfully!")
        return True
    except Exception as e:
        logger.error(f"Error initializing predictor: {str(e)}")
        logger.error(traceback.format_exc())
        return False


@app.route('/', methods=['GET'])
def home():
    """Home endpoint."""
    return jsonify({
        'message': 'Fake Review Detection API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            '/health': 'Health check',
            '/predict': 'Predict single or multiple reviews',
            '/predict/csv': 'Predict from CSV file',
            '/model/info': 'Get model information',
            '/scrape': 'Scrape product reviews (disabled)'
        }
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    model_info = get_model_info()
    
    all_models_loaded = all(model_info['models'].values())
    all_transformers_loaded = all(model_info['transformers'].values())
    
    status = 'healthy' if (all_models_loaded and all_transformers_loaded) else 'unhealthy'
    
    return jsonify({
        'status': status,
        'timestamp': datetime.now().isoformat(),
        'models_loaded': all_models_loaded,
        'transformers_loaded': all_transformers_loaded,
        'details': model_info
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict if reviews are fake or genuine.
    
    Request body:
    {
        "reviews": ["review text 1", "review text 2", ...]
    }
    or
    {
        "review": "single review text"
    }
    """
    try:
        # Initialize predictor if not already done
        if not initialize_predictor():
            return jsonify(create_response(
                False, 
                "Model not loaded. Please train the model first.",
                error="Models not found"
            )), 500
        
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify(create_response(
                False,
                "No data provided",
                error="Request body is empty"
            )), 400
        
        # Handle single review
        if 'review' in data:
            review_text = sanitize_text(data['review'])
            
            if not validate_input_text(review_text):
                return jsonify(create_response(
                    False,
                    "Invalid review text",
                    error="Text is empty or too long (max 10000 characters)"
                )), 400
            
            reviews = [review_text]
        
        # Handle multiple reviews
        elif 'reviews' in data:
            reviews = [sanitize_text(r) for r in data['reviews']]
            
            # Validate all reviews
            invalid_reviews = [i for i, r in enumerate(reviews) if not validate_input_text(r)]
            if invalid_reviews:
                return jsonify(create_response(
                    False,
                    f"Invalid reviews at indices: {invalid_reviews}",
                    error="Some reviews are empty or too long"
                )), 400
        
        else:
            return jsonify(create_response(
                False,
                "Missing 'review' or 'reviews' field",
                error="Invalid request format"
            )), 400
        
        # Make predictions
        logger.info(f"Predicting {len(reviews)} reviews...")
        predictions = predictor.predict_batch(reviews)
        summary = predictor.get_prediction_summary(predictions)
        
        logger.info(f"Predictions completed: {summary}")
        
        return jsonify(create_response(
            True,
            "Predictions generated successfully",
            data={
                'predictions': predictions,
                'summary': summary
            }
        ))
    
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify(create_response(
            False,
            "Error generating predictions",
            error=str(e)
        )), 500


@app.route('/predict/csv', methods=['POST'])
def predict_csv():
    """
    Predict from uploaded CSV file.
    
    File must contain a column named 'text' with review texts.
    """
    try:
        # Initialize predictor if not already done
        if not initialize_predictor():
            return jsonify(create_response(
                False,
                "Model not loaded. Please train the model first.",
                error="Models not found"
            )), 500
        
        # Check if file is present
        if 'file' not in request.files:
            return jsonify(create_response(
                False,
                "No file provided",
                error="File field missing in request"
            )), 400
        
        file = request.files['file']
        
        # Check if filename is empty
        if file.filename == '':
            return jsonify(create_response(
                False,
                "No file selected",
                error="Empty filename"
            )), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify(create_response(
                False,
                "Invalid file type",
                error="Only CSV files are allowed"
            )), 400
        
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"File uploaded: {filepath}")
        
        # Read CSV
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            return jsonify(create_response(
                False,
                "Error reading CSV file",
                error=str(e)
            )), 400
        
        # Check for text column
        text_column = None
        for col in ['text', 'review', 'review_text', 'content']:
            if col in df.columns:
                text_column = col
                break
        
        if text_column is None:
            return jsonify(create_response(
                False,
                "CSV must contain a 'text' column with review texts",
                error=f"Available columns: {list(df.columns)}"
            )), 400
        
        logger.info(f"Processing {len(df)} reviews from CSV...")
        
        # Make predictions
        df_with_predictions = predictor.predict_from_dataframe(df, text_column=text_column)
        
        # Get summary
        predictions = df_with_predictions.to_dict('records')
        summary = predictor.get_prediction_summary([
            {'label': row['label'], 'confidence': row['confidence'], 
             'fake_probability': row['fake_probability']}
            for row in predictions
        ])
        
        # Save results
        output_filename = f"predictions_{timestamp}.csv"
        output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        df_with_predictions.to_csv(output_filepath, index=False)
        
        logger.info(f"Predictions saved to {output_filepath}")
        
        return jsonify(create_response(
            True,
            "CSV predictions generated successfully",
            data={
                'total_reviews': len(df),
                'summary': summary,
                'output_file': output_filename,
                'predictions': predictions  # Return ALL predictions
            }
        ))
    
    except Exception as e:
        logger.error(f"Error in predict_csv endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify(create_response(
            False,
            "Error processing CSV file",
            error=str(e)
        )), 500


@app.route('/model/info', methods=['GET'])
def model_info():
    """Get information about loaded models."""
    try:
        info = get_model_info()
        
        return jsonify(create_response(
            True,
            "Model information retrieved",
            data=info
        ))
    
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify(create_response(
            False,
            "Error retrieving model information",
            error=str(e)
        )), 500


@app.route('/scrape', methods=['POST'])
def scrape_reviews():
    """
    Placeholder endpoint for scraping reviews.
    This feature is disabled for ToS compliance.
    """
    return jsonify(create_response(
        False,
        "Scraping feature is disabled",
        data=ScrapePredictor.scrape_and_predict("")
    )), 501


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    return jsonify(create_response(
        False,
        "File too large",
        error=f"Maximum file size is {MAX_CONTENT_LENGTH / (1024 * 1024):.0f} MB"
    )), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify(create_response(
        False,
        "Endpoint not found",
        error="The requested endpoint does not exist"
    )), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify(create_response(
        False,
        "Internal server error",
        error="An unexpected error occurred"
    )), 500


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Starting Fake Review Detection API Server")
    logger.info("=" * 60)
    logger.info(f"Host: {API_HOST}")
    logger.info(f"Port: {API_PORT}")
    logger.info(f"Debug: {DEBUG}")
    logger.info("=" * 60)
    
    # Try to initialize predictor on startup
    initialize_predictor()
    
    # Run the app
    app.run(host=API_HOST, port=API_PORT, debug=DEBUG)
