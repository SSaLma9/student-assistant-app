import React, { useState } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import Login from './components/Login';
import Dashboard from './components/Dashboard';
import CourseManager from './components/CourseManager';
import LectureManager from './components/LectureManager';
import StudyAssistant from './components/StudyAssistant';
import ExamMode from './components/ExamMode';

const App = () => {
  const [view, setView] = useState('login');
  const [username, setUsername] = useState('');
  const [token, setToken] = useState('');
  const [selectedCourse, setSelectedCourse] = useState('');
  const [selectedLecture, setSelectedLecture] = useState('');

  const handleLogin = (username, token) => {
    setUsername(username);
    setToken(token);
    localStorage.setItem('token', token);
    setView('dashboard');
    toast.success(`Welcome, ${username}!`);
  };

  const handleLogout = () => {
    setUsername('');
    setToken('');
    localStorage.removeItem('token');
    setSelectedCourse('');
    setSelectedLecture('');
    setView('login');
    toast.info('Logged out successfully');
  };

  const renderView = () => {
    switch (view) {
      case 'login':
        return <Login onLogin={handleLogin} />;
      case 'dashboard':
        return <Dashboard onLogout={handleLogout} setView={setView} />;
      case 'courses':
        return <CourseManager setView={setView} setSelectedCourse={setSelectedCourse} token={token} />;
      case 'lectures':
        return <LectureManager selectedCourse={selectedCourse} setView={setView} setSelectedLecture={setSelectedLecture} token={token} />;
      case 'study':
        return <StudyAssistant selectedLecture={selectedLecture} setView={setView} token={token} />;
      case 'exam':
        return <ExamMode selectedLecture={selectedLecture} setView={setView} token={token} />;
      default:
        return <Login onLogin={handleLogin} />;
    }
  };

  return (
    <div className="min-h-screen">
      {view !== 'login' && (
        <nav className="bg-indigo-600 text-white p-4 shadow-lg">
          <div className="max-w-7xl mx-auto flex flex-col sm:flex-row justify-between items-center">
            <h1 className="text-xl font-bold mb-2 sm:mb-0">Student Assistant</h1>
            <div className="flex flex-row flex-wrap items-center gap-2 sm:gap-4">
              <span className="text-sm">Welcome, {username}</span>
              <button onClick={() => setView('dashboard')} className="hover:bg-indigo-700 px-3 py-2 rounded transition">
                Dashboard
              </button>
              <button onClick={() => setView('courses')} className="hover:bg-indigo-700 px-3 py-2 rounded transition">
                Courses
              </button>
              <button onClick={() => setView('lectures')} className="hover:bg-indigo-700 px-3 py-2 rounded transition">
                Lectures
              </button>
              <button onClick={() => setView('study')} className="hover:bg-indigo-700 px-3 py-2 rounded transition">
                Study
              </button>
              <button onClick={() => setView('exam')} className="hover:bg-indigo-700 px-3 py-2 rounded transition">
                Exam
              </button>
              <button onClick={handleLogout} className="hover:bg-red-600 px-3 py-2 rounded transition">
                Logout
              </button>
            </div>
          </div>
        </nav>
      )}
      <main className="max-w-7xl mx-auto p-4 sm:p-6 lg:p-8">
        {renderView()}
      </main>
      <ToastContainer position="top-right" autoClose={3000} hideProgressBar={false} />
    </div>
  );
};

export default App;
