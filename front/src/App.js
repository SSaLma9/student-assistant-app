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
    <div className="app-container">
      {view !== 'login' && (
        <nav className="nav-bar">
          <div className="nav-content">
            <h1 className="nav-title">Student Assistant</h1>
            <div className="nav-items">
              <span className="nav-username">Welcome, {username}</span>
              <button onClick={() => setView('dashboard')} className="nav-button">
                Dashboard
              </button>
              <button onClick={() => setView('courses')} className="nav-button">
                Courses
              </button>
              <button onClick={() => setView('lectures')} className="nav-button">
                Lectures
              </button>
              <button onClick={() => setView('study')} className="nav-button">
                Study
              </button>
              <button onClick={() => setView('exam')} className="nav-button">
                Exam
              </button>
              <button onClick={handleLogout} className="nav-button logout-button">
                Logout
              </button>
            </div>
          </div>
        </nav>
      )}
      <main className="main-content">
        {renderView()}
      </main>
      <ToastContainer position="top-right" autoClose={3000} hideProgressBar={false} />
    </div>
  );
};

export default App;
