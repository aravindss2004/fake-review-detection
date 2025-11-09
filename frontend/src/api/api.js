import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const predictReviews = async (reviews) => {
  try {
    const response = await api.post('/predict', { reviews });
    return response.data;
  } catch (error) {
    throw error.response?.data || { message: 'Failed to predict reviews' };
  }
};

export const predictCSV = async (file) => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await api.post('/predict/csv', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    throw error.response?.data || { message: 'Failed to process CSV file' };
  }
};

export const checkHealth = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    throw error.response?.data || { message: 'Health check failed' };
  }
};

export const getModelInfo = async () => {
  try {
    const response = await api.get('/model/info');
    return response.data;
  } catch (error) {
    throw error.response?.data || { message: 'Failed to get model info' };
  }
};

export default api;
