import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { toast } from 'react-toastify';
import { ArrowLeftIcon, DocumentArrowUpIcon } from '@heroicons/react/24/solid';

const LectureManager = ({ selectedCourse, setView, setSelectedLecture, token }) => {
  const [lectures, setLectures] = useState([]);
  const [lectureName, setLectureName] = useState('');
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [memoryWarning, setMemoryWarning] = useState(false);

  // Check server resources
  useEffect(() => {
    const checkResources = async () => {
      try {
        const response = await axios.get(
          `${process.env.REACT_APP_API_BASE_URL}/resources`,
          { headers: { Authorization: `Bearer ${token}` } }
        );
        if (response.data.memory.percent > 80 || response.data.disk.percent > 85) {
          setMemoryWarning(true);
          toast.warn('Server resources are high. Consider uploading a smaller file.');
        }
      } catch (err) {
        console.error('Failed to check resources:', err);
      }
    };
    checkResources();
  }, [token]);

  // Fetch lectures
  useEffect(() => {
    if (!selectedCourse) return;
    
    const fetchLectures = async () => {
      setLoading(true);
      try {
        const response = await axios.get(
          `${process.env.REACT_APP_API_BASE_URL}/lectures/${selectedCourse}`,
          {
            headers: { Authorization: `Bearer ${token}` },
            timeout: 10000
          }
        );
        setLectures(response.data.lectures || []);
      } catch (err) {
        handleApiError(err, 'Failed to fetch lectures');
      } finally {
        setLoading(false);
      }
    };
    
    fetchLectures();
  }, [selectedCourse, token]);

  // Handle API errors
  const handleApiError = (err, defaultMessage) => {
    console.error('API Error:', err.response || err);
    
    const errorMessage = err.response?.data?.error || 
      (err.code === 'ECONNABORTED' ? 'Request timed out' : 
      err.message.includes('Network Error') ? 'Network connection failed' : 
      defaultMessage);
      
    toast.error(errorMessage);
    setError(errorMessage);
    
    if (err.response?.status === 507) {
      setMemoryWarning(true);
    }
  };

  // Validate file
  const validateFile = (file) => {
    if (!file) {
      return 'Please select a file';
    }
    
    if (file.type !== 'application/pdf') {
      return 'Only PDF files are allowed';
    }
    
    if (file.size > 5 * 1024 * 1024) { // 5MB
      return 'File size exceeds 5MB limit';
    }
    
    return null;
  };

  // Handle file upload
  const handleUpload = async (e) => {
    e.preventDefault();
    setError('');
    setMemoryWarning(false);
    
    if (!lectureName.trim()) {
      setError('Lecture name is required');
      toast.error('Lecture name is required');
      return;
    }
    
    if (!/^[a-zA-Z0-9_-]+$/.test(lectureName)) {
      setError('Only letters, numbers, underscores and hyphens allowed');
      toast.error('Invalid lecture name format');
      return;
    }

    const fileError = validateFile(file);
    if (fileError) {
      setError(fileError);
      toast.error(fileError);
      return;
    }
    
    setLoading(true);
    
    try {
      const formData = new FormData();
      formData.append('lecture_name', lectureName.trim());
      formData.append('course_name', selectedCourse);
      formData.append('file', file);

      // Log FormData
      for (let [key, value] of formData.entries()) {
        console.log(`${key}:`, value);
      }

      const source = axios.CancelToken.source();
      const timeout = setTimeout(() => {
        source.cancel('Upload timed out after 2 minutes');
      }, 120000);

      await axios.post(
        `${process.env.REACT_APP_API_BASE_URL}/lectures`,
        formData,
        {
          headers: {
            Authorization: `Bearer ${token}`,
            'Content-Type': 'multipart/form-data'
          },
          cancelToken: source.token,
          onUploadProgress: (progressEvent) => {
            const percentCompleted = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            );
            console.log(`Upload progress: ${percentCompleted}%`);
          }
        }
      );

      clearTimeout(timeout);
      
      const response = await axios.get(
        `${process.env.REACT_APP_API_BASE_URL}/lectures/${selectedCourse}`,
        {
          headers: { Authorization: `Bearer ${token}` },
          timeout: 10000
        }
      );
      
      setLectures(response.data.lectures || []);
      setLectureName('');
      setFile(null);
      
      toast.success('Lecture uploaded successfully!');
    } catch (err) {
      if (axios.isCancel(err)) {
        setError('Upload timed out');
        toast.error('Upload took too long. Try a smaller file.');
      } else {
        handleApiError(err, 'Failed to upload lecture');
      }
    } finally {
      setLoading(false);
    }
  };

  // Handle lecture selection
  const handleSelectLecture = (lecture) => {
    setSelectedLecture(lecture);
    setView('study');
  };

  return (
    <div className="bg-white p-6 rounded-xl shadow-lg animate-fade-in max-w-3xl mx-auto">
      <div className="flex items-center mb-6">
        <button
          onClick={() => setView('courses')}
          className="text-indigo-600 hover:text-indigo-800 mr-4"
          aria-label="Back to courses"
        >
          <ArrowLeftIcon className="h-6 w-6" />
        </button>
        <h2 className="text-2xl font-bold text-gray-800">
          Lectures for {selectedCourse}
        </h2>
      </div>

      {memoryWarning && (
        <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 mb-6">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg
                className="h-5 w-5 text-yellow-400"
                viewBox="0 0 20 20"
                fill="currentColor"
              >
                <path
                  fillRule="evenodd"
                  d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                  clipRule="evenodd"
                />
              </svg>
            </div>
            <div className="ml-3">
              <p className="text-sm text-yellow-700">
                The server is low on resources. Try uploading a smaller file or 
                contact support to upgrade your plan.
              </p>
            </div>
          </div>
        </div>
      )}

      {error && (
        <div className="bg-red-50 border-l-4 border-red-400 p-4 mb-6">
          <p className="text-sm text-red-700">{error}</p>
        </div>
      )}

      <form onSubmit={handleUpload} className="space-y-6 mb-8">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Lecture Name
          </label>
          <input
            type="text"
            value={lectureName}
            onChange={(e) => setLectureName(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500"
            placeholder="e.g., Introduction to Algorithms"
            required
            pattern="[a-zA-Z0-9_-]+"
            title="Only letters, numbers, underscores, and hyphens allowed"
          />
          <p className="mt-1 text-xs text-gray-500">
            Only letters, numbers, underscores (_), and hyphens (-) allowed
          </p>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            PDF File
          </label>
          <div className="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-lg">
            <div className="space-y-1 text-center">
              <DocumentArrowUpIcon className="mx-auto h-12 w-12 text-gray-400" />
              <div className="flex text-sm text-gray-600">
                <label
                  htmlFor="file-upload"
                  className="relative cursor-pointer bg-white rounded-md font-medium text-indigo-600 hover:text-indigo-500 focus-within:outline-none"
                >
                  <span>Upload a file</span>
                  <input
                    id="file-upload"
                    name="file-upload"
                    type="file"
                    className="sr-only"
                    accept="application/pdf"
                    onChange={(e) => setFile(e.target.files?.[0] || null)}
                  />
                </label>
                <p className="pl-1">or drag and drop</p>
              </div>
              <p className="text-xs text-gray-500">
                PDF up to 5MB
              </p>
              {file && (
                <p className="text-sm text-gray-900 mt-2">
                  Selected: {file.name} ({(file.size / 1024 / 1024).toFixed(2)}MB)
                </p>
              )}
            </div>
          </div>
        </div>

        <button
          type="submit"
          disabled={loading}
          className={`w-full bg-indigo-600 text-white py-3 px-4 rounded-lg hover:bg-indigo-700 transition duration-300 flex items-center justify-center ${
            loading ? 'opacity-70 cursor-not-allowed' : ''
          }`}
        >
          {loading ? (
            <>
              <svg
                className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                ></circle>
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 3 7.938l3-2.647z"
                ></path>
              </svg>
              Processing...
            </>
          ) : (
            'Upload Lecture'
          )}
        </button>
      </form>

      <div>
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          Available Lectures
        </h3>
        {loading && lectures.length === 0 ? (
          <div className="flex justify-center items-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
          </div>
        ) : lectures.length === 0 ? (
          <div className="bg-gray-50 p-6 rounded-lg text-center">
            <p className="text-gray-600">
              No lectures available. Upload your first lecture above!
            </p>
          </div>
        ) : (
          <ul className="space-y-3">
            {lectures.map((lecture) => (
              <li
                key={lecture.lecture_name || lecture}
                className="bg-gray-50 p-4 rounded-lg hover:bg-indigo-50 cursor-pointer transition flex justify-between items-center"
                onClick={() => handleSelectLecture(lecture)}
              >
                <span className="font-medium text-gray-800">
                  {lecture.lecture_name || lecture}
                </span>
                <svg
                  className="h-5 w-5 text-indigo-600"
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 20 20"
                  fill="currentColor"
                >
                  <path
                    fillRule="evenodd"
                    d="M7.293 14district.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z"
                    clipRule="evenodd"
                  />
                </svg>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
};

export default LectureManager;
