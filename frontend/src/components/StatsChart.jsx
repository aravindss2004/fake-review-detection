import React from 'react';
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement } from 'chart.js';
import { Pie, Bar } from 'react-chartjs-2';

ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement);

const StatsChart = ({ summary }) => {
  if (!summary) {
    return null;
  }

  const pieData = {
    labels: ['Genuine', 'Fake'],
    datasets: [
      {
        data: [summary.genuine_reviews, summary.fake_reviews],
        backgroundColor: ['#10b981', '#ef4444'],
        borderColor: ['#059669', '#dc2626'],
        borderWidth: 2,
      },
    ],
  };

  const barData = {
    labels: ['Genuine', 'Fake'],
    datasets: [
      {
        label: 'Count',
        data: [summary.genuine_reviews, summary.fake_reviews],
        backgroundColor: ['rgba(16, 185, 129, 0.7)', 'rgba(239, 68, 68, 0.7)'],
        borderColor: ['#10b981', '#ef4444'],
        borderWidth: 2,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom',
      },
    },
  };

  return (
    <div className="card mt-8">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Summary Statistics</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
          <p className="text-sm text-blue-600 font-medium">Total Reviews</p>
          <p className="text-3xl font-bold text-blue-900 mt-2">
            {summary.total_reviews}
          </p>
        </div>
        
        <div className="bg-green-50 rounded-lg p-4 border border-green-200">
          <p className="text-sm text-green-600 font-medium">Genuine</p>
          <p className="text-3xl font-bold text-green-900 mt-2">
            {summary.genuine_reviews}
          </p>
          <p className="text-xs text-green-600 mt-1">
            {summary.genuine_percentage.toFixed(1)}%
          </p>
        </div>
        
        <div className="bg-red-50 rounded-lg p-4 border border-red-200">
          <p className="text-sm text-red-600 font-medium">Fake</p>
          <p className="text-3xl font-bold text-red-900 mt-2">
            {summary.fake_reviews}
          </p>
          <p className="text-xs text-red-600 mt-1">
            {summary.fake_percentage.toFixed(1)}%
          </p>
        </div>
        
        <div className="bg-purple-50 rounded-lg p-4 border border-purple-200">
          <p className="text-sm text-purple-600 font-medium">Avg Confidence</p>
          <p className="text-3xl font-bold text-purple-900 mt-2">
            {(summary.average_confidence * 100).toFixed(1)}%
          </p>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div>
          <h3 className="text-lg font-semibold text-gray-900 mb-4 text-center">
            Distribution (Pie Chart)
          </h3>
          <div className="h-64">
            <Pie data={pieData} options={options} />
          </div>
        </div>
        
        <div>
          <h3 className="text-lg font-semibold text-gray-900 mb-4 text-center">
            Distribution (Bar Chart)
          </h3>
          <div className="h-64">
            <Bar data={barData} options={options} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default StatsChart;
