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
    <div className="bg-white p-4 sm:p-6 rounded-xl shadow-lg animate-fade-in">
      <div className="flex items-center mb-6">
        <button
          onClick={() => setView('dashboard')}
          className="text-indigo-600 hover:text-indigo-800 mr-4"
        >
          <ArrowLeftIcon className="h-6 w-6" />
        </button>
        <h2 className="text-xl sm:text-2xl font-bold text-gray-800">Manage Courses</h2>
      </div>
      {error && <p className="text-red-500 mb-4 text-sm sm:text-base">{error}</p>}
      <form onSubmit={handleCreateCourse} className="space-y-4 mb-6">
        <div>
          <label className="block text-sm sm:text-base font-medium text-gray-700">New Course Name</label>
          <input
            type="text"
            value={courseName}
            onChange={(e) => setCourseName(e.target.value)}
            className="mt-1 w-full p-3 sm:p-4 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 text-sm sm:text-base"
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
          Add Course
        </button>
      </form>
      <div>
        <h3 className="text-lg sm:text-xl font-semibold text-gray-800 mb-4">Your Courses</h3>
        {loading ? (
          <p className="text-gray-600 text-sm sm:text-base">Loading courses...</p>
        ) : courses.length === 0 ? (
          <p className="text-gray-600 text-sm sm:text-base">No courses available. Add one above!</p>
        ) : (
          <ul className="space-y-2">
            {courses.map((course) => (
              <li
                key={course}
                className="bg-indigo-50 p-3 sm:p-4 rounded-lg hover:bg-indigo-100 cursor-pointer transition text-sm sm:text-base"
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
