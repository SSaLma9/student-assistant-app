import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { toast } from 'react-toastify';
import { ArrowLeftIcon } from '@heroicons/react/24/solid';

const LectureManager = ({ selectedCourse, setView, setSelectedLecture, token }) => {
  const [lectures, setLectures] = useState([]);
  const [lectureName, setLectureName] = useState('');
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!selectedCourse) return;
    const fetchLectures = async () => {
      setLoading(true);
      try {
        const response = await axios.get(`${process.env.REACT_APP_API_URL}/lectures/${selectedCourse}`, {
          headers: { Authorization: `Bearer ${token}` }
        });
        setLectures(response.data.lectures || []);
      } catch (err) {
        toast.error(err.response?.data?.error || 'Failed to fetch lectures');
      } finally {
        setLoading(false);
      }
    };
    fetchLectures();
  }, [selectedCourse, token]);

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file || !lectureName || !selectedCourse) {
      setError('Please provide lecture name, course, and PDF file');
      toast.error('Please provide lecture name, course, and PDF file');
      return;
    }
    setLoading(true);
    setError('');
    const formData = new FormData();
    formData.append('lecture_name', lectureName);
    formData.append('course_name', selectedCourse);
    formData.append('file', file);

    try {
      await axios.post(`${process.env.REACT_APP_API_URL}/lectures`, formData, {
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'multipart/form-data'
        }
      });
      setLectures([...lectures, lectureName]);
      setLectureName('');
      setFile(null);
      toast.success('Lecture uploaded successfully!');
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to upload lecture');
      toast.error(err.response?.data?.error || 'Failed to upload lecture');
    } finally {
      setLoading(false);
    }
  };

  const handleSelectLecture = (lecture) => {
    setSelectedLecture(lecture);
    setView('study');
  };

  return (
    <div className="bg-white p-4 sm:p-6 rounded-xl shadow-lg animate-fade-in">
      <div className="flex items-center mb-6">
        <button
          onClick={() => setView('courses')}
          className="text-indigo-600 hover:text-indigo-800 mr-4"
        >
          <ArrowLeftIcon className="h-6 w-6" />
        </button>
        <h2 className="text-xl sm:text-2xl font-bold text-gray-800">Manage Lectures for {selectedCourse}</h2>
      </div>
      {error && <p className="text-red-500 mb-4 text-sm sm:text-base">{error}</p>}
      <div className="space-y-6 sm:space-y-0 sm:flex sm:space-x-6">
        <form onSubmit={handleUpload} className="space-y-4 mb-6 sm:mb-0 sm:w-1/2">
          <div>
            <label className="block text-sm sm:text-base font-medium text-gray-700">Lecture Name</label>
            <input
              type="text"
              value={lectureName}
              onChange={(e) => setLectureName(e.target.value)}
              className="mt-1 w-full p-3 sm:p-4 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 text-sm sm:text-base"
              required
            />
          </div>
          <div>
            <label className="block text-sm sm:text-base font-medium text-gray-700">Upload Lecture PDF</label>
            <input
              type="file"
              accept="application/pdf"
              onChange={(e) => setFile(e.target.files[0])}
              className="mt-1 w-full p-3 sm:p-4 border border-gray-300 rounded-lg text-sm sm:text-base"
              required
            />
          </div>
          <button
            type="submit"
            disabled={loading}
            className="w-full bg-indigo-600 text-white p-3 sm:p-4 rounded-lg hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 active:bg-indigo-800 disabled:bg-indigo-300 text-sm sm:text-base font-medium"
          >
            {loading ? (
              <svg className="animate-spin h-5 w-5 mr-2 text-white" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
              </svg>
            ) : null}
            Upload Lecture
          </button>
        </form>
        <div className="sm:w-1/2">
          <h3 className="text-lg sm:text-xl font-semibold text-gray-800 mb-4">Available Lectures</h3>
          {loading ? (
            <p className="text-gray-600 text-sm sm:text-base">Loading lectures...</p>
          ) : lectures.length === 0 ? (
            <p className="text-gray-600 text-sm sm:text-base">No lectures available. Upload one above!</p>
          ) : (
            <ul className="space-y-2">
              {lectures.map((lecture) => (
                <li
                  key={lecture}
                  className="bg-indigo-50 p-3 sm:p-4 rounded-lg hover:bg-indigo-100 cursor-pointer transition text-sm sm:text-base"
                  onClick={() => handleSelectLecture(lecture)}
                >
                  {lecture}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
};

export default LectureManager;
