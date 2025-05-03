import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { toast } from 'react-toastify';
import { ArrowLeftIcon } from '@heroicons/react/24/solid';

const CourseManager = ({ setView, setSelectedCourse, token }) => {
  const [courses, setCourses] = useState([]);
  const [courseName, setCourseName] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchCourses = async () => {
      setLoading(true);
      try {
        const response = await axios.get(`${process.env.REACT_APP_API_URL}/courses`, {
          headers: { Authorization: `Bearer ${token}` }
        });
        setCourses(response.data.courses || []);
      } catch (err) {
        toast.error(err.response?.data?.error || 'Failed to fetch courses');
      } finally {
        setLoading(false);
      }
    };
    fetchCourses();
  }, [token]);

  const handleCreateCourse = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    try {
      await axios.post(`${process.env.REACT_APP_API_URL}/courses`,
        { course_name: courseName },
        { headers: { Authorization: `Bearer ${token}` } }
      );
      setCourses([...courses, courseName]);
      setCourseName('');
      toast.success('Course created successfully!');
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to create course');
      toast.error(err.response?.data?.error || 'Failed to create course');
    } finally {
      setLoading(false);
    }
  };

  const handleSelectCourse = (course) => {
    setSelectedCourse(course);
    setView('lectures');
  };

  return (
    <div className="card">
      <div className="header">
        <button
          onClick={() => setView('dashboard')}
          className="back-button"
        >
          <ArrowLeftIcon />
        </button>
        <h2 className="header-title">Manage Courses</h2>
      </div>
      {error && <p className="error-text">{error}</p>}
      <form onSubmit={handleCreateCourse}>
        <div className="form-group">
          <label className="form-label">New Course Name</label>
          <input
            type="text"
            value={courseName}
            onChange={(e) => setCourseName(e.target.value)}
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
          Add Course
        </button>
      </form>
      <div className="mt-10">
        <h3 className="content-title">Your Courses</h3>
        {loading ? (
          <p>Loading courses...</p>
        ) : courses.length === 0 ? (
          <p>No courses available. Add one above!</p>
        ) : (
          <ul>
            {courses.map((course) => (
              <li
                key={course}
                className="list-item"
                onClick={() => handleSelectCourse(course)}
              >
                {course}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
};

export default CourseManager;
