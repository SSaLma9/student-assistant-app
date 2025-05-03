import React, { useState } from 'react';
import { toast } from 'react-toastify';
import axios from 'axios';
import { LockClosedIcon } from '@heroicons/react/24/solid';

const Login = ({ onLogin }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [isRegister, setIsRegister] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const endpoint = isRegister ? '/register' : '/login';
      const response = await axios.post(`${process.env.REACT_APP_API_BASE_URL}${endpoint}`, {
        username,
        password
      });
      onLogin(username, response.data.token);
      toast.success(isRegister ? 'Registered and logged in successfully!' : 'Logged in successfully!');
      setUsername('');
      setPassword('');
    } catch (err) {
      console.error('Login/Register error:', err);
      toast.error(err.response?.data?.error || err.response?.data?.detail || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-indigo-100 to-blue-200 animate-fade-in">
      <div className="bg-white p-6 sm:p-8 rounded-xl shadow-2xl w-full max-w-md sm:max-w-lg transform transition-all hover:scale-105">
        <div className="flex justify-center mb-6">
          <LockClosedIcon className="h-12 w-12 text-indigo-600" />
        </div>
        <h2 className="text-2xl font-bold text-center text-gray-800 mb-6">
          {isRegister ? 'Create Account' : 'Welcome Back'}
        </h2>
        <form onSubmit={handleSubmit} className="space-y-4 sm:space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700">Username</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="mt-1 w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500"
              required
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="mt-1 w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500"
              required
            />
          </div>
          <button
            type="submit"
            disabled={loading}
            className={`w-full bg-indigo-600 text-white py-2 px-4 rounded-lg hover:bg-indigo-700 transition duration-300 flex items-center justify-center ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            {loading ? (
              <svg className="animate-spin h-5 w-5 mr-2 text-white" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
              </svg>
            ) : null}
            {isRegister ? 'Register' : 'Login'}
          </button>
        </form>
        <p className="mt-4 text-center text-sm text-gray-600">
          {isRegister ? 'Already have an account?' : "Don't have an account?"}
          <button
            onClick={() => setIsRegister(!isRegister)}
            className="ml-1 text-indigo-600 hover:text-indigo-800 font-medium"
          >
            {isRegister ? 'Login' : 'Register'}
          </button>
        </p>
      </div>
    </div>
  );
};

export default Login;
