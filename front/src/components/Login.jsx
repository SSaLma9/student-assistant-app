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
      const response = await axios.post(`${process.env.REACT_APP_API_URL}${endpoint}`, {
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
    <div className="container">
      <div className="text-center">
        <LockClosedIcon className="icon" />
      </div>
      <h2 className="text-xl text-center mt-10">
        {isRegister ? 'Create Account' : 'Welcome Back'}
      </h2>
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label className="form-label">Username</label>
          <input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
          />
        </div>
        <div className="form-group">
          <label className="form-label">Password</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </div>
        <button type="submit" disabled={loading}>
          {loading ? (
            <svg className="spinner" viewBox="0 0 24 24">
              <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" style={{ opacity: 0.25 }} />
              <path fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" style={{ opacity: 0.75 }} />
            </svg>
          ) : null}
          {isRegister ? 'Register' : 'Login'}
        </button>
      </form>
      <p className="text-sm text-center mt-10">
        {isRegister ? 'Already have an account?' : "Don't have an account?"}
        <button
          onClick={() => setIsRegister(!isRegister)}
          style={{ background: 'none', border: 'none', color: '#007bff', cursor: 'pointer' }}
        >
          {isRegister ? 'Login' : 'Register'}
        </button>
      </p>
    </div>
  );
};

export default Login;
