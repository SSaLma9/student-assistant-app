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
    <div className="card">
      <div className="header">
        <button
          onClick={() => setView('courses')}
          className="back-button"
        >
          <ArrowLeftIcon />
        </button>
        <h2 className="header-title">Manage Lectures for {selectedCourse}</h2>
      </div>
      {error && <p className="error-text">{error}</p>}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
        <form onSubmit={handleUpload}>
          <div className="form-group">
            <label className="form-label">Lecture Name</label>
            <input
              type="text"
              value={lectureName}
              onChange={(e) => setLectureName(e.target.value)}
              required
            />
          </div>
          <div className="form-group">
            <label className="form-label">Upload Lecture PDF</label>
            <input
              type="file"
              accept="application/pdf"
              onChange={(e) => setFile(e.target.files[0])}
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
            Upload Lecture
          </button>
        </form>
        <div>
          <h3 className="content-title">Available Lectures</h3>
          {loading ? (
            <p>Loading lectures...</p>
          ) : lectures.length === 0 ? (
            <p>No lectures available. Upload one above!</p>
          ) : (
            <ul>
              {lectures.map((lecture) => (
                <li
                  key={lecture}
                  className="list-item"
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
