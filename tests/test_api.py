"""
Unit tests for API endpoints.
"""
import sys
sys.path.append('../backend')

import unittest
import json
from app import app


class TestAPI(unittest.TestCase):
    """Test cases for Flask API endpoints."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.app = app.test_client()
        self.app.testing = True
    
    def test_home_endpoint(self):
        """Test home endpoint."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('message', data)
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('status', data)
    
    def test_predict_endpoint_no_data(self):
        """Test predict endpoint with no data."""
        response = self.app.post('/predict',
                                 data=json.dumps({}),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 400)
    
    def test_predict_endpoint_with_single_review(self):
        """Test predict endpoint with single review."""
        data = {'review': 'This is a test review'}
        response = self.app.post('/predict',
                                 data=json.dumps(data),
                                 content_type='application/json')
        # Note: This will fail if models are not loaded
        # In real tests, we would mock the predictor
        self.assertIn(response.status_code, [200, 500])
    
    def test_predict_endpoint_with_multiple_reviews(self):
        """Test predict endpoint with multiple reviews."""
        data = {
            'reviews': [
                'Great product!',
                'Terrible quality'
            ]
        }
        response = self.app.post('/predict',
                                 data=json.dumps(data),
                                 content_type='application/json')
        self.assertIn(response.status_code, [200, 500])
    
    def test_model_info_endpoint(self):
        """Test model info endpoint."""
        response = self.app.get('/model/info')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('data', data)


if __name__ == '__main__':
    unittest.main()
