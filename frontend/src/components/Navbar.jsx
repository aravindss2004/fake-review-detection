import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { ShieldCheckIcon } from '@heroicons/react/24/solid';

const Navbar = () => {
  const location = useLocation();
  
  const isActive = (path) => {
    return location.pathname === path;
  };
  
  return (
    <nav className="bg-white shadow-lg border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <Link to="/" className="flex items-center space-x-3">
              <ShieldCheckIcon className="h-8 w-8 text-blue-600" />
              <span className="text-xl font-bold text-gray-900">
                Fake Review Detection
              </span>
            </Link>
          </div>
          
          <div className="flex items-center space-x-4">
            <Link
              to="/"
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                isActive('/')
                  ? 'bg-blue-100 text-blue-700'
                  : 'text-gray-700 hover:bg-gray-100'
              }`}
            >
              Home
            </Link>
            <Link
              to="/about"
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                isActive('/about')
                  ? 'bg-blue-100 text-blue-700'
                  : 'text-gray-700 hover:bg-gray-100'
              }`}
            >
              About
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
