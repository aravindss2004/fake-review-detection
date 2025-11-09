import React, { useState } from 'react';
import { DocumentTextIcon, CloudArrowUpIcon } from '@heroicons/react/24/outline';
import { predictReviews, predictCSV } from '../api/api';
import ResultsTable from '../components/ResultsTable';
import StatsChart from '../components/StatsChart';

const Home = () => {
  const [inputMode, setInputMode] = useState('text');
  const [reviewText, setReviewText] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [predictions, setPredictions] = useState(null);
  const [summary, setSummary] = useState(null);

  const handleTextSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const reviews = reviewText.split('\n').filter(r => r.trim() !== '');
      
      if (reviews.length === 0) {
        setError('Please enter at least one review');
        setLoading(false);
        return;
      }

      const result = await predictReviews(reviews);
      
      if (result.success) {
        setPredictions(result.data.predictions);
        setSummary(result.data.summary);
      } else {
        setError(result.message || 'Failed to analyze reviews');
      }
    } catch (err) {
      setError(err.message || 'An error occurred while analyzing reviews');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      if (!selectedFile) {
        setError('Please select a CSV file');
        setLoading(false);
        return;
      }

      const result = await predictCSV(selectedFile);
      
      if (result.success) {
        setPredictions(result.data.predictions);
        setSummary(result.data.summary);
      } else {
        setError(result.message || 'Failed to process CSV file');
      }
    } catch (err) {
      setError(err.message || 'An error occurred while processing the file');
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file && file.type === 'text/csv') {
      setSelectedFile(file);
      setError('');
    } else {
      setError('Please select a valid CSV file');
      setSelectedFile(null);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl md:text-5xl font-extrabold text-gray-900 mb-4">
            Fake Review Detection System
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Powered by Machine Learning • Ensemble Models • Real-time Analysis
          </p>
        </div>

        {/* Input Section */}
        <div className="card max-w-4xl mx-auto">
          {/* Mode Selector */}
          <div className="flex justify-center space-x-4 mb-8">
            <button
              onClick={() => setInputMode('text')}
              className={`flex items-center space-x-2 px-6 py-3 rounded-lg font-medium transition-all ${
                inputMode === 'text'
                  ? 'bg-blue-600 text-white shadow-lg'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              <DocumentTextIcon className="h-5 w-5" />
              <span>Text Input</span>
            </button>
            <button
              onClick={() => setInputMode('file')}
              className={`flex items-center space-x-2 px-6 py-3 rounded-lg font-medium transition-all ${
                inputMode === 'file'
                  ? 'bg-blue-600 text-white shadow-lg'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              <CloudArrowUpIcon className="h-5 w-5" />
              <span>CSV Upload</span>
            </button>
          </div>

          {/* Text Input Form */}
          {inputMode === 'text' && (
            <form onSubmit={handleTextSubmit}>
              <div className="mb-6">
                <label htmlFor="reviews" className="block text-sm font-medium text-gray-700 mb-2">
                  Enter Reviews (one per line)
                </label>
                <textarea
                  id="reviews"
                  rows="8"
                  className="input-field font-mono text-sm"
                  placeholder="This product is amazing! Best purchase ever!&#10;Quality is decent, arrived on time.&#10;Terrible product, complete waste of money..."
                  value={reviewText}
                  onChange={(e) => setReviewText(e.target.value)}
                  required
                />
              </div>
              
              <button
                type="submit"
                disabled={loading}
                className="btn-primary w-full disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? 'Analyzing...' : 'Analyze Reviews'}
              </button>
            </form>
          )}

          {/* File Upload Form */}
          {inputMode === 'file' && (
            <form onSubmit={handleFileUpload}>
              <div className="mb-6">
                <label htmlFor="csv-file" className="block text-sm font-medium text-gray-700 mb-2">
                  Upload CSV File (must contain 'text' column)
                </label>
                <div className="mt-2 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-lg hover:border-blue-400 transition-colors">
                  <div className="space-y-2 text-center">
                    <CloudArrowUpIcon className="mx-auto h-12 w-12 text-gray-400" />
                    <div className="flex text-sm text-gray-600">
                      <label
                        htmlFor="csv-file"
                        className="relative cursor-pointer bg-white rounded-md font-medium text-blue-600 hover:text-blue-500 focus-within:outline-none"
                      >
                        <span>Upload a file</span>
                        <input
                          id="csv-file"
                          type="file"
                          accept=".csv"
                          className="sr-only"
                          onChange={handleFileChange}
                        />
                      </label>
                      <p className="pl-1">or drag and drop</p>
                    </div>
                    <p className="text-xs text-gray-500">CSV file up to 16MB</p>
                    {selectedFile && (
                      <p className="text-sm text-green-600 font-medium">
                        Selected: {selectedFile.name}
                      </p>
                    )}
                  </div>
                </div>
              </div>
              
              <button
                type="submit"
                disabled={loading || !selectedFile}
                className="btn-primary w-full disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? 'Processing...' : 'Analyze CSV'}
              </button>
            </form>
          )}

          {/* Error Message */}
          {error && (
            <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-sm text-red-700">{error}</p>
            </div>
          )}
        </div>

        {/* Results */}
        {summary && <StatsChart summary={summary} />}
        {predictions && <ResultsTable predictions={predictions} />}
      </div>
    </div>
  );
};

export default Home;
