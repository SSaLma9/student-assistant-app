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
      const response = await axios.post(`${process.env.REACT_APP_API_URL}/study`,
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
    <div className="card">
      <div className="header">
        <button
          onClick={() => setView('lectures')}
          className="back-button"
        >
          <ArrowLeftIcon />
        </button>
        <h2 className="header-title">Study Assistant for {selectedLecture}</h2>
      </div>
      {error && <p className="error-text">{error}</p>}
      <div>
        <div className="form-group">
          <label className="form-label">Select Study Task</label>
          <select
            value={task}
            onChange={(e) => setTask(e.target.value)}
          >
            {tasks.map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </div>
        {task === 'Custom Question' && (
          <div className="form-group">
            <label className="form-label">Your Question</label>
            <textarea
              value={customQuestion}
              onChange={(e) => setCustomQuestion(e.target.value)}
              placeholder="Type your question here..."
              rows="4"
            />
          </div>
        )}
        <button onClick={handleGenerate} disabled={loading || !selectedLecture}>
          {loading ? (
            <svg className="spinner" viewBox="0 0 24 24">
              <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" style={{ opacity: 0.25 }} />
              <path fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" style={{ opacity: 0.75 }} />
            </svg>
          ) : null}
          Generate Content
        </button>
        {content && (
          <div className="content-box mt-10">
            <h3 className="content-title">Generated Content</h3>
            <p className="content-text">{content}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default StudyAssistant;
