import React from 'react';
import { BookOpenIcon, AcademicCapIcon, DocumentTextIcon, ClipboardDocumentCheckIcon } from '@heroicons/react/24/solid';

const Dashboard = ({ onLogout, setView }) => {
  const buttons = [
    { label: 'Manage Courses', view: 'courses', icon: BookOpenIcon },
    { label: 'Manage Lectures', view: 'lectures', icon: DocumentTextIcon },
    { label: 'Study Assistant', view: 'study', icon: AcademicCapIcon },
    { label: 'Exam Mode', view: 'exam', icon: ClipboardDocumentCheckIcon },
  ];

  return (
    <div className="bg-white p-6 sm:p-8 rounded-xl shadow-lg animate-fade-in">
      <h2 className="text-3xl font-bold text-gray-800 mb-6 sm:mb-8 text-center">Dashboard</h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6">
        {buttons.map(({ label, view, icon: Icon }) => (
          <button
            key={label}
            onClick={() => setView(view)}
            className="bg-indigo-600 text-white p-4 sm:p-6 rounded-lg hover:bg-indigo-700 transition duration-300 flex flex-col items-center justify-center transform hover:scale-105"
          >
            <Icon className="h-6 sm:h-8 w-6 sm:w-8 mb-2" />
            <span className="text-base sm:text-lg font-semibold">{label}</span>
          </button>
        ))}
      </div>
    </div>
  );
};

export default Dashboard;
