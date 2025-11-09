import React, { useState } from 'react';
import { CheckCircleIcon, XCircleIcon, ChevronLeftIcon, ChevronRightIcon, ArrowDownTrayIcon } from '@heroicons/react/24/solid';

const ResultsTable = ({ predictions }) => {
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(50);
  
  if (!predictions || predictions.length === 0) {
    return null;
  }

  // Pagination calculations
  const totalPages = Math.ceil(predictions.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const currentPredictions = predictions.slice(startIndex, endIndex);

  const goToPage = (page) => {
    setCurrentPage(Math.max(1, Math.min(page, totalPages)));
  };

  // Download results as CSV
  const downloadCSV = () => {
    const headers = ['Review Text', 'Prediction', 'Confidence', 'Fake Probability', 'Genuine Probability'];
    const csvContent = [
      headers.join(','),
      ...predictions.map(pred => [
        `"${pred.text.replace(/"/g, '""')}"`,
        pred.prediction,
        pred.confidence.toFixed(4),
        pred.fake_probability.toFixed(4),
        pred.genuine_probability.toFixed(4)
      ].join(','))
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `fake_review_analysis_${new Date().toISOString().slice(0, 10)}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="card mt-8">
      <div className="flex justify-between items-center mb-6 flex-wrap gap-4">
        <h2 className="text-2xl font-bold text-gray-900">
          Analysis Results
          <span className="text-lg font-normal text-gray-600 ml-2">
            ({predictions.length} total reviews)
          </span>
        </h2>
        
        <div className="flex items-center space-x-4">
          {predictions.length > 10 && (
            <div className="flex items-center space-x-2">
              <label className="text-sm text-gray-600">Show:</label>
              <select
                value={itemsPerPage}
                onChange={(e) => {
                  setItemsPerPage(Number(e.target.value));
                  setCurrentPage(1);
                }}
                className="border border-gray-300 rounded px-3 py-1 text-sm"
              >
                <option value={10}>10</option>
                <option value={25}>25</option>
                <option value={50}>50</option>
                <option value={100}>100</option>
                <option value={predictions.length}>All</option>
              </select>
              <span className="text-sm text-gray-600">per page</span>
            </div>
          )}
          
          <button
            onClick={downloadCSV}
            className="flex items-center space-x-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors shadow-sm"
          >
            <ArrowDownTrayIcon className="h-5 w-5" />
            <span>Download CSV</span>
          </button>
        </div>
      </div>
      
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                #
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Review Text
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Prediction
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Confidence
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Fake Probability
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {currentPredictions.map((pred, index) => (
              <tr key={startIndex + index} className="hover:bg-gray-50 transition-colors">
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {startIndex + index + 1}
                </td>
                <td className="px-6 py-4 text-sm text-gray-900 max-w-md">
                  <div className="line-clamp-2">{pred.text}</div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex items-center space-x-2">
                    {pred.prediction === 'Fake' ? (
                      <>
                        <XCircleIcon className="h-5 w-5 text-red-500" />
                        <span className="text-sm font-medium text-red-700 bg-red-100 px-3 py-1 rounded-full">
                          Fake
                        </span>
                      </>
                    ) : (
                      <>
                        <CheckCircleIcon className="h-5 w-5 text-green-500" />
                        <span className="text-sm font-medium text-green-700 bg-green-100 px-3 py-1 rounded-full">
                          Genuine
                        </span>
                      </>
                    )}
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex items-center">
                    <div className="w-16 bg-gray-200 rounded-full h-2 mr-2">
                      <div
                        className={`h-2 rounded-full ${
                          pred.confidence > 0.8
                            ? 'bg-green-500'
                            : pred.confidence > 0.6
                            ? 'bg-yellow-500'
                            : 'bg-red-500'
                        }`}
                        style={{ width: `${pred.confidence * 100}%` }}
                      ></div>
                    </div>
                    <span className="text-sm text-gray-700">
                      {(pred.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">
                  {(pred.fake_probability * 100).toFixed(1)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="mt-6 flex items-center justify-between border-t border-gray-200 pt-4">
          <div className="flex-1 flex justify-between sm:hidden">
            <button
              onClick={() => goToPage(currentPage - 1)}
              disabled={currentPage === 1}
              className="relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Previous
            </button>
            <button
              onClick={() => goToPage(currentPage + 1)}
              disabled={currentPage === totalPages}
              className="ml-3 relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Next
            </button>
          </div>
          <div className="hidden sm:flex-1 sm:flex sm:items-center sm:justify-between">
            <div>
              <p className="text-sm text-gray-700">
                Showing <span className="font-medium">{startIndex + 1}</span> to{' '}
                <span className="font-medium">{Math.min(endIndex, predictions.length)}</span> of{' '}
                <span className="font-medium">{predictions.length}</span> results
              </p>
            </div>
            <div>
              <nav className="relative z-0 inline-flex rounded-md shadow-sm -space-x-px" aria-label="Pagination">
                <button
                  onClick={() => goToPage(currentPage - 1)}
                  disabled={currentPage === 1}
                  className="relative inline-flex items-center px-2 py-2 rounded-l-md border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <ChevronLeftIcon className="h-5 w-5" />
                </button>
                
                {/* Page numbers */}
                {[...Array(totalPages)].map((_, i) => {
                  const page = i + 1;
                  if (
                    page === 1 ||
                    page === totalPages ||
                    (page >= currentPage - 2 && page <= currentPage + 2)
                  ) {
                    return (
                      <button
                        key={page}
                        onClick={() => goToPage(page)}
                        className={`relative inline-flex items-center px-4 py-2 border text-sm font-medium ${
                          currentPage === page
                            ? 'z-10 bg-blue-50 border-blue-500 text-blue-600'
                            : 'bg-white border-gray-300 text-gray-500 hover:bg-gray-50'
                        }`}
                      >
                        {page}
                      </button>
                    );
                  } else if (page === currentPage - 3 || page === currentPage + 3) {
                    return <span key={page} className="relative inline-flex items-center px-4 py-2 border border-gray-300 bg-white text-sm font-medium text-gray-700">...</span>;
                  }
                  return null;
                })}
                
                <button
                  onClick={() => goToPage(currentPage + 1)}
                  disabled={currentPage === totalPages}
                  className="relative inline-flex items-center px-2 py-2 rounded-r-md border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <ChevronRightIcon className="h-5 w-5" />
                </button>
              </nav>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ResultsTable;
