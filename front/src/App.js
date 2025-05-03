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
  const [isNavOpen, setIsNavOpen] = useState(false);

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
        <>
          {/* Hamburger Menu for Mobile */}
          <button
            className="hamburger-menu"
            onClick={() => setIsNavOpen(!isNavOpen)}
          >
            <svg className="menu-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16m-7 6h7"></path>
            </svg>
          </button>

          {/* Navigation Bar */}
          <nav className={`nav-bar ${isNavOpen ? 'nav-open' : 'nav-closed'}`}>
            <div className="nav-content">
              <h1 className="nav-title">Student Assistant</h1>
              <div className="nav-items">
                <span className="nav-username">Welcome, {username}</span>
                <button onClick={() => { setView('dashboard'); setIsNavOpen(false); }} className="nav-button">
                  Dashboard
                </button>
                <button onClick={() => { setView('courses'); setIsNavOpen(false); }} className="nav-button">
                  Courses
                </button>
                <button onClick={() => { setView('lectures'); setIsNavOpen(false); }} className="nav-button">
                  Lectures
                </button>
                <button onClick={() => { setView('study'); setIsNavOpen(false); }} className="nav-button">
                  Study
                </button>
                <button onClick={() => { setView('exam'); setIsNavOpen(false); }} className="nav-button">
                  Exam
                </button>
                <button onClick={() => { handleLogout(); setIsNavOpen(false); }} className="nav-button logout-button">
                  Logout
                </button>
              </div>
            </div>
          </nav>

          {/* Overlay for Mobile Nav */}
          {isNavOpen && (
            <div className="nav-overlay" onClick={() => setIsNavOpen(false)}></div>
          )}
        </>
      )}
      <main className="main-content">
        {renderView()}
      </main>
      <ToastContainer position="top-right" autoClose={3000} hideProgressBar={false} />
    </div>
  );
};

export default App;
