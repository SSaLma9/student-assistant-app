import React, { useState } from 'react';
import axios from 'axios';
import { toast } from 'react-toastify';
import { ArrowLeftIcon } from '@heroicons/react/24/solid';

const StudyAssistant = ({ selectedLecture, setView, token }) => {
  const [task, setTask] = useState('Summarize');
  const [customQuestion, setCustomQuestion] = useState('');
  const [content, setContent] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const tasks = ['Summarize', 'Explain', 'Examples', 'Custom Question'];

  const handleGenerate = async () => {
    if (!selectedLecture) {
      setError('Please select a lecture first');
      toast.error('Please select a lecture first');
      return;
    }
    if (task === 'Custom Question' && !customQuestion) {
      setError('Please enter a custom question');
      toast.error('Please enter a custom question');
      return;
    }
    setLoading(true);
    setError('');
    try {
        const response = await axios.post(`${process.env.REACT_APP_API_BASE_URL}/study`,
        { task, lecture_name: selectedLecture, question: customQuestion },
        { headers: { Authorization: `Bearer ${token}` } }
      );
      setContent(response.data.content);
      toast.success('Content generated successfully!');
      if (task === 'Custom Question') {
        setCustomQuestion('');
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to generate content');
      toast.error(err.response?.data?.error || 'Failed to generate content');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white p-6 sm:p-8 rounded-xl shadow-lg animate-fade-in">
      <div className="flex items-center mb-4 sm:mb-6">
        <button
          onClick={() => setView('lectures')}
          className="text-indigo-600 hover:text-indigo-800 mr-2 sm:mr-4"
        >
          <ArrowLeftIcon className="h-5 sm:h-6 w-5 sm:w-6" />
        </button>
        <h2 className="text-xl sm:text-2xl font-bold text-gray-800">Study Assistant for {selectedLecture}</h2>
      </div>
      {error && <p className="text-red-500 mb-2 sm:mb-4">{error}</p>}
      <div className="space-y-4 sm:space-y-6">
        <div>
          <label className="block text-sm font-medium text-gray-700">Select Study Task</label>
          <select
            value={task}
            onChange={(e) => setTask(e.target.value)}
            className="mt-1 w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500"
          >
            {tasks.map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </div>
        {task === 'Custom Question' && (
          <div>
            <label className="block text-sm font-medium text-gray-700">Your Question</label>
            <textarea
              value={customQuestion}
              onChange={(e) => setCustomQuestion(e.target.value)}
              className="mt-1 w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500"
              placeholder="Type your question here..."
              rows="4"
            />
          </div>
        )}
        <button
          onClick={handleGenerate}
          disabled={loading || !selectedLecture}
          className={`w-full bg-indigo-600 text-white py-2 px-4 rounded-lg hover:bg-indigo-700 transition duration-300 flex items-center justify-center ${loading || !selectedLecture ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          {loading ? (
            <svg className="animate-spin h-4 sm:h-5 w-4 sm:w-5 mr-1 sm:mr-2 text-white" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
            </svg>
          ) : null}
          Generate Content
        </button>
        {content && (
          <div className="bg-gray-100 p-4 sm:p-6 rounded-lg">
            <h3 className="text-lg font-semibold text-gray-800 mb-2 sm:mb-4">Generated Content</h3>
            <p className="text-gray-700 whitespace-pre-wrap">{content}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default StudyAssistant;
