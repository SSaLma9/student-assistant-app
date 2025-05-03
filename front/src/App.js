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
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-blue-50 to-indigo-100 p-2 sm:p-4">
      {view !== 'login' && (
        <>
          {/* Hamburger Menu for Mobile */}
          <button
            className="md:hidden p-2 text-white focus:outline-none z-50"
            onClick={() => setIsNavOpen(!isNavOpen)}
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16m-7 6h7"></path>
            </svg>
          </button>

          {/* Navigation Bar */}
          <nav
            className={`bg-indigo-600 text-white shadow-lg transition-all duration-300 ${
              isNavOpen ? 'max-h-screen' : 'max-h-0 md:max-h-screen'
            } md:max-h-screen overflow-hidden`}
          >
            <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center p-2 sm:p-4">
              <h1 className="text-lg sm:text-xl font-bold">Student Assistant</h1>
              <div className="flex flex-col md:flex-row items-center space-y-2 md:space-y-0 md:space-x-4 mt-2 md:mt-0">
                <span className="text-sm sm:text-base">Welcome, {username}</span>
                <button onClick={() => { setView('dashboard'); setIsNavOpen(false); }} className="hover:bg-indigo-700 px-2 sm:px-3 py-1 sm:py-2 rounded transition text-sm sm:text-base">
                  Dashboard
                </button>
                <button onClick={() => { setView('courses'); setIsNavOpen(false); }} className="hover:bg-indigo-700 px-2 sm:px-3 py-1 sm:py-2 rounded transition text-sm sm:text-base">
                  Courses
                </button>
                <button onClick={() => { setView('lectures'); setIsNavOpen(false); }} className="hover:bg-indigo-700 px-2 sm:px-3 py-1 sm:py-2 rounded transition text-sm sm:text-base">
                  Lectures
                </button>
                <button onClick={() => { setView('study'); setIsNavOpen(false); }} className="hover:bg-indigo-700 px-2 sm:px-3 py-1 sm:py-2 rounded transition text-sm sm:text-base">
                  Study
                </button>
                <button onClick={() => { setView('exam'); setIsNavOpen(false); }} className="hover:bg-indigo-700 px-2 sm:px-3 py-1 sm:py-2 rounded transition text-sm sm:text-base">
                  Exam
                </button>
                <button onClick={() => { handleLogout(); setIsNavOpen(false); }} className="hover:bg-red-600 px-2 sm:px-3 py-1 sm:py-2 rounded transition text-sm sm:text-base">
                  Logout
                </button>
              </div>
            </div>
          </nav>

          {/* Overlay for Mobile Nav */}
          {isNavOpen && (
            <div
              className="fixed inset-0 bg-black opacity-50 md:hidden"
              onClick={() => setIsNavOpen(false)}
            ></div>
          )}
        </>
      )}
      <main className="max-w-7xl mx-auto p-2 sm:p-4 lg:p-6 flex-1">
        {renderView()}
      </main>
      <ToastContainer position="top-right" autoClose={3000} hideProgressBar={false} />
    </div>
  );
};

export default App;
