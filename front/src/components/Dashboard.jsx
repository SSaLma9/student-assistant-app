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
    <div className="dashboard-container">
      <button
        className="hamburger-menu"
        onClick={toggleSidebar}
      >
        <svg className="menu-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16m-7 6h7"></path>
        </svg>
      </button>

      <div className={`dashboard-sidebar ${isSidebarOpen ? 'sidebar-open' : 'sidebar-closed'}`}>
        <h2 className="header-title">Dashboard</h2>
        {buttons.map(({ label, view }) => (
          <button
            key={label}
            onClick={() => { setView(view); setIsSidebarOpen(false); }}
            className="nav-button"
          >
            {label}
          </button>
        ))}
        <button
          onClick={() => { onLogout(); setIsSidebarOpen(false); }}
          className="nav-button logout-button"
        >
          Logout
        </button>
      </div>

      {isSidebarOpen && (
        <div className="nav-overlay" onClick={toggleSidebar}></div>
      )}

      <div className="dashboard-content">
        <div className="grid">
          {buttons.map(({ label, view, icon: Icon }) => (
            <div
              key={label}
              onClick={() => setView(view)}
              className="grid-item"
            >
              <Icon className="grid-item-icon" />
              <span className="grid-item-text">{label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
