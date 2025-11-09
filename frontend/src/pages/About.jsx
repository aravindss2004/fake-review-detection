import React from 'react';
import { AcademicCapIcon, BeakerIcon, ChartBarIcon, CodeBracketIcon } from '@heroicons/react/24/outline';

const About = () => {
  const features = [
    {
      icon: BeakerIcon,
      title: 'Advanced NLP',
      description: 'Utilizes spaCy for text preprocessing, lemmatization, and linguistic feature extraction',
    },
    {
      icon: ChartBarIcon,
      title: 'Ensemble Learning',
      description: 'Combines LightGBM, CatBoost, and XGBoost for superior prediction accuracy',
    },
    {
      icon: CodeBracketIcon,
      title: 'Production Ready',
      description: 'Full-stack solution with Flask backend and React frontend for real-time predictions',
    },
    {
      icon: AcademicCapIcon,
      title: 'Research Grade',
      description: 'Built for BE Major Project with comprehensive evaluation and documentation',
    },
  ];

  const technologies = {
    backend: ['Python', 'Flask', 'scikit-learn', 'LightGBM', 'CatBoost', 'XGBoost', 'spaCy', 'TextBlob'],
    frontend: ['React', 'Tailwind CSS', 'Chart.js', 'Axios'],
    tools: ['Jupyter', 'NumPy', 'Pandas', 'Matplotlib', 'Seaborn'],
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Hero Section */}
        <div className="text-center mb-16">
          <h1 className="text-5xl font-extrabold text-gray-900 mb-6">
            About This Project
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
            A comprehensive machine learning system designed to detect fake reviews on e-commerce platforms
            using state-of-the-art NLP techniques and ensemble learning methods.
          </p>
        </div>

        {/* Project Overview */}
        <div className="card mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-6">Project Overview</h2>
          <div className="prose max-w-none text-gray-700 space-y-4">
            <p className="text-lg leading-relaxed">
              This project implements an end-to-end solution for identifying fraudulent reviews using advanced
              machine learning techniques. The system processes review text through multiple stages of preprocessing,
              feature extraction, and ensemble prediction to achieve high accuracy in distinguishing fake reviews
              from genuine ones.
            </p>
            <p className="text-lg leading-relaxed">
              Developed as part of a BE Major Project, this system demonstrates the practical application of
              Natural Language Processing (NLP) and ensemble learning in solving real-world problems affecting
              e-commerce platforms and consumer trust.
            </p>
          </div>
        </div>

        {/* Key Features */}
        <div className="mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-8 text-center">Key Features</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <div key={index} className="card hover:shadow-xl transition-shadow">
                  <div className="flex items-start space-x-4">
                    <div className="flex-shrink-0">
                      <div className="flex items-center justify-center h-12 w-12 rounded-lg bg-blue-100">
                        <Icon className="h-6 w-6 text-blue-600" />
                      </div>
                    </div>
                    <div>
                      <h3 className="text-xl font-semibold text-gray-900 mb-2">
                        {feature.title}
                      </h3>
                      <p className="text-gray-600 leading-relaxed">
                        {feature.description}
                      </p>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Technologies Used */}
        <div className="card mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-8 text-center">Technologies Used</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div>
              <h3 className="text-xl font-semibold text-blue-600 mb-4">Backend</h3>
              <div className="flex flex-wrap gap-2">
                {technologies.backend.map((tech, index) => (
                  <span
                    key={index}
                    className="px-3 py-1 bg-blue-50 text-blue-700 rounded-full text-sm font-medium border border-blue-200"
                  >
                    {tech}
                  </span>
                ))}
              </div>
            </div>
            <div>
              <h3 className="text-xl font-semibold text-green-600 mb-4">Frontend</h3>
              <div className="flex flex-wrap gap-2">
                {technologies.frontend.map((tech, index) => (
                  <span
                    key={index}
                    className="px-3 py-1 bg-green-50 text-green-700 rounded-full text-sm font-medium border border-green-200"
                  >
                    {tech}
                  </span>
                ))}
              </div>
            </div>
            <div>
              <h3 className="text-xl font-semibold text-purple-600 mb-4">Tools</h3>
              <div className="flex flex-wrap gap-2">
                {technologies.tools.map((tech, index) => (
                  <span
                    key={index}
                    className="px-3 py-1 bg-purple-50 text-purple-700 rounded-full text-sm font-medium border border-purple-200"
                  >
                    {tech}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Model Architecture */}
        <div className="card mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-6">Model Architecture</h2>
          <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-8 border-2 border-blue-200">
            <div className="text-center space-y-4">
              <div className="bg-white rounded-lg p-4 shadow-md inline-block">
                <p className="font-mono text-sm text-gray-700">Input Review Text</p>
              </div>
              <div className="text-2xl text-gray-400">↓</div>
              <div className="bg-white rounded-lg p-4 shadow-md inline-block">
                <p className="font-mono text-sm text-gray-700">Preprocessing (spaCy)</p>
              </div>
              <div className="text-2xl text-gray-400">↓</div>
              <div className="flex justify-center space-x-4">
                <div className="bg-white rounded-lg p-4 shadow-md">
                  <p className="font-mono text-sm text-gray-700">TF-IDF Features</p>
                </div>
                <div className="bg-white rounded-lg p-4 shadow-md">
                  <p className="font-mono text-sm text-gray-700">Linguistic Features</p>
                </div>
              </div>
              <div className="text-2xl text-gray-400">↓</div>
              <div className="flex justify-center space-x-4">
                <div className="bg-blue-100 rounded-lg p-4 shadow-md border-2 border-blue-300">
                  <p className="font-mono text-sm text-blue-700 font-semibold">LightGBM</p>
                </div>
                <div className="bg-green-100 rounded-lg p-4 shadow-md border-2 border-green-300">
                  <p className="font-mono text-sm text-green-700 font-semibold">CatBoost</p>
                </div>
                <div className="bg-purple-100 rounded-lg p-4 shadow-md border-2 border-purple-300">
                  <p className="font-mono text-sm text-purple-700 font-semibold">XGBoost</p>
                </div>
              </div>
              <div className="text-2xl text-gray-400">↓</div>
              <div className="bg-gradient-to-r from-yellow-100 to-orange-100 rounded-lg p-4 shadow-lg border-2 border-yellow-400 inline-block">
                <p className="font-mono text-sm text-gray-800 font-bold">Voting Ensemble</p>
              </div>
              <div className="text-2xl text-gray-400">↓</div>
              <div className="bg-white rounded-lg p-4 shadow-md inline-block">
                <p className="font-mono text-sm text-gray-700">Fake / Genuine Prediction</p>
              </div>
            </div>
          </div>
        </div>

        {/* Performance Metrics */}
        <div className="card mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-6 text-center">Expected Performance</h2>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Model</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Accuracy</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Precision</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Recall</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">F1-Score</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">ROC-AUC</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                <tr>
                  <td className="px-6 py-4 text-sm font-medium text-gray-900">LightGBM</td>
                  <td className="px-6 py-4 text-sm text-gray-700">~94%</td>
                  <td className="px-6 py-4 text-sm text-gray-700">~94%</td>
                  <td className="px-6 py-4 text-sm text-gray-700">~95%</td>
                  <td className="px-6 py-4 text-sm text-gray-700">~94%</td>
                  <td className="px-6 py-4 text-sm text-gray-700">~0.98</td>
                </tr>
                <tr>
                  <td className="px-6 py-4 text-sm font-medium text-gray-900">CatBoost</td>
                  <td className="px-6 py-4 text-sm text-gray-700">~95%</td>
                  <td className="px-6 py-4 text-sm text-gray-700">~94%</td>
                  <td className="px-6 py-4 text-sm text-gray-700">~95%</td>
                  <td className="px-6 py-4 text-sm text-gray-700">~94%</td>
                  <td className="px-6 py-4 text-sm text-gray-700">~0.98</td>
                </tr>
                <tr>
                  <td className="px-6 py-4 text-sm font-medium text-gray-900">XGBoost</td>
                  <td className="px-6 py-4 text-sm text-gray-700">~94%</td>
                  <td className="px-6 py-4 text-sm text-gray-700">~93%</td>
                  <td className="px-6 py-4 text-sm text-gray-700">~94%</td>
                  <td className="px-6 py-4 text-sm text-gray-700">~94%</td>
                  <td className="px-6 py-4 text-sm text-gray-700">~0.97</td>
                </tr>
                <tr className="bg-green-50">
                  <td className="px-6 py-4 text-sm font-bold text-green-900">Ensemble</td>
                  <td className="px-6 py-4 text-sm font-semibold text-green-700">~95%</td>
                  <td className="px-6 py-4 text-sm font-semibold text-green-700">~95%</td>
                  <td className="px-6 py-4 text-sm font-semibold text-green-700">~96%</td>
                  <td className="px-6 py-4 text-sm font-semibold text-green-700">~95%</td>
                  <td className="px-6 py-4 text-sm font-semibold text-green-700">~0.98</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* GitHub Link */}
        <div className="card bg-gradient-to-r from-gray-900 to-blue-900 text-white text-center">
          <h2 className="text-3xl font-bold mb-4">View on GitHub</h2>
          <p className="text-lg mb-6 opacity-90">
            Explore the complete source code, documentation, and contribute to the project
          </p>
          <a
            href="https://github.com/aravindss2004/fake-review-detection"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center space-x-2 bg-white text-gray-900 px-8 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-colors shadow-lg"
          >
            <svg className="h-6 w-6" fill="currentColor" viewBox="0 0 24 24">
              <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
            </svg>
            <span>GitHub Repository</span>
          </a>
        </div>

        {/* Project Info */}
        <div className="mt-12 text-center text-gray-600">
          
          <p className="text-sm mt-2">
            BE Major Project • Machine Learning & NLP
          </p>
        </div>
      </div>
    </div>
  );
};

export default About;
