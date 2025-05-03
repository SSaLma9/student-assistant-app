import React, { useState } from 'react';
import { BookOpenIcon, AcademicCapIcon, DocumentTextIcon, ClipboardDocumentCheckIcon } from '@heroicons/react/24/solid';

const Dashboard = ({ onLogout, setView }) => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const buttons = [
    { label: 'Manage Courses', view: 'courses', icon: BookOpenIcon },
    { label: 'Manage Lectures', view: 'lectures', icon: DocumentTextIcon },
    { label: 'Study Assistant', view: 'study', icon: AcademicCapIcon },
    { label: 'Exam Mode', view: 'exam', icon: ClipboardDocumentCheckIcon },
  ];

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  return (
    <div className="min-h-screen bg-gray-100 flex">
      {/* Hamburger Menu for Mobile */}
      <button
        className="md:hidden p-4 text-gray-700 focus:outline-none z-50"
        onClick={toggleSidebar}
      >
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16m-7 6h7"></path>
        </svg>
      </button>

      {/* Sidebar */}
      <div
        className={`fixed inset-y-0 left-0 w-64 bg-white shadow-lg transform ${
          isSidebarOpen ? 'translate-x-0' : '-translate-x-full'
        } md:relative md:translate-x-0 transition-transform duration-300 ease-in-out z-40`}
      >
        <div className="p-6">
          <h2 className="text-xl sm:text-2xl font-bold mb-6 text-gray-800">Dashboard</h2>
          <nav className="space-y-2">
            {buttons.map(({ label, view, icon: Icon }) => (
              <button
                key={label}
                onClick={() => { setView(view); setIsSidebarOpen(false); }}
                className="w-full text-left px-4 py-2 text-gray-700 hover:bg-indigo-100 rounded-md text-sm sm:text-base flex items-center"
              >
                <Icon className="h-5 w-5 mr-2" />
                {label}
              </button>
            ))}
            <button
              onClick={onLogout}
              className="w-full text-left px-4 py-2 text-gray-700 hover:bg-red-100 rounded-md text-sm sm:text-base mt-4"
            >
              Logout
            </button>
          </nav>
        </div>
      </div>

      {/* Overlay for Mobile Sidebar */}
      {isSidebarOpen && (
        <div
          className="fixed inset-0 bg-black opacity-50 md:hidden"
          onClick={toggleSidebar}
        ></div>
      )}

      {/* Main Content */}
      <div className="flex-1 p-4 sm:p-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {buttons.map(({ label, view, icon: Icon }) => (
            <button
              key={label}
              onClick={() => setView(view)}
              className="bg-indigo-600 text-white p-4 sm:p-6 rounded-lg hover:bg-indigo-700 transition duration-300 flex flex-col items-center justify-center transform hover:scale-105 text-sm sm:text-base"
            >
              <Icon className="h-6 w-6 sm:h-8 sm:w-8 mb-2" />
              <span className="font-semibold">{label}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
