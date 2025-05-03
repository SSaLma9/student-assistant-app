import React, { useState } from 'react';
import axios from 'axios';
import { toast } from 'react-toastify';
import { ArrowLeftIcon } from '@heroicons/react/24/solid';

const ExamMode = ({ selectedLecture, setView, token }) => {
  const [examType, setExamType] = useState('MCQs');
  const [difficulty, setDifficulty] = useState('Easy');
  const [questions, setQuestions] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [answers, setAnswers] = useState({});
  const [feedback, setFeedback] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const examTypes = ['MCQs', 'Essay Questions'];
  const difficulties = ['Easy', 'Medium', 'Hard'];

  const handleGenerateExam = async () => {
    if (!selectedLecture) {
      setError('Please select a lecture first');
      toast.error('Please select a lecture first');
      return;
    }
    setLoading(true);
    setError('');
    try {
      const response = await axios.post(`${process.env.REACT_APP_API_URL}/exam`, 
        { lecture_name: selectedLecture, exam_type: examType, difficulty },
        { headers: { Authorization: `Bearer ${token}` } }
      );
      setQuestions(response.data.questions || []);
      setCurrentIndex(0);
      setAnswers({});
      setFeedback('');
      toast.success('Exam generated successfully!');
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to generate exam');
      toast.error(err.response?.data?.error || 'Failed to generate exam');
    } finally {
      setLoading(false);
    }
  };

  const handleAnswerSubmit = async (questionId, answer) => {
    if (!answer) {
      toast.error('Please provide an answer');
      return;
    }
    setLoading(true);
    try {
      const response = await axios.post(`${process.env.REACT_APP_API_URL}/exam/grade`,
        { question_id: questionId, answer },
        { headers: { Authorization: `Bearer ${token}` } }
      );
      setFeedback(response.data.feedback);
      toast.success('Answer submitted successfully!');
    } catch (err) {
      toast.error(err.response?.data?.error || 'Failed to grade answer');
    } finally {
      setLoading(false);
    }
  };

  const currentQuestion = questions[currentIndex];

  return (
    <div className="card">
      <div className="header">
        <button
          onClick={() => setView('lectures')}
          className="back-button"
        >
          <ArrowLeftIcon />
        </button>
        <h2 className="header-title">Exam Mode for {selectedLecture}</h2>
      </div>
      {error && <p className="error-text">{error}</p>}
      <div className="grid">
        <div className="form-group">
          <label className="form-label">Exam Type</label>
          <select
            value={examType}
            onChange={(e) => setExamType(e.target.value)}
          >
            {examTypes.map((type) => (
              <option key={type} value={type}>{type}</option>
            ))}
          </select>
        </div>
        <div className="form-group">
          <label className="form-label">Difficulty</label>
          <select
            value={difficulty}
            onChange={(e) => setDifficulty(e.target.value)}
          >
            {difficulties.map((diff) => (
              <option key={diff} value={diff}>{diff}</option>
            ))}
          </select>
        </div>
        <button onClick={handleGenerateExam} disabled={loading || !selectedLecture}>
          {loading ? (
            <svg className="spinner" viewBox="0 0 24 24">
              <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" style={{ opacity: 0.25 }} />
              <path fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" style={{ opacity: 0.75 }} />
            </svg>
          ) : null}
          Generate Exam
        </button>
      </div>
      {questions.length > 0 && currentQuestion && (
        <div>
          <h3 className="content-title mt-10">
            Question {currentIndex + 1} of {questions.length}
          </h3>
          <div className="content-box">
            <p className="text-lg">{currentQuestion.question}</p>
            {currentQuestion.type === 'mcq' && currentQuestion.options && (
              <div style={{ marginTop: '10px' }}>
                {currentQuestion.options.map((option, idx) => (
                  <label key={idx} style={{ display: 'flex', alignItems: 'center', margin: '5px 0' }}>
                    <input
                      type="radio"
                      name={`question-${currentQuestion.id}`}
                      value={option}
                      checked={answers[currentQuestion.id] === option}
                      onChange={(e) => setAnswers({ ...answers, [currentQuestion.id]: e.target.value })}
                      style={{ marginRight: '5px' }}
                    />
                    <span>{option}</span>
                  </label>
                ))}
              </div>
            )}
            {currentQuestion.type === 'essay' && (
              <textarea
                value={answers[currentQuestion.id] || ''}
                onChange={(e) => setAnswers({ ...answers, [currentQuestion.id]: e.target.value })}
                placeholder="Write your essay answer here..."
                rows="6"
              />
            )}
          </div>
          <button
            onClick={() => handleAnswerSubmit(currentQuestion.id, answers[currentQuestion.id] || '')}
            disabled={loading}
          >
            {loading ? (
              <svg className="spinner" viewBox="0 0 24 24">
                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" style={{ opacity: 0.25 }} />
                <path fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" style={{ opacity: 0.75 }} />
              </svg>
            ) : null}
            Submit Answer
          </button>
          {feedback && (
            <div className="content-box mt-10">
              <h3 className="content-title">Feedback</h3>
              <p className="content-text">{feedback}</p>
            </div>
          )}
          <div className="button-group mt-10">
            <button
              onClick={() => setCurrentIndex(currentIndex - 1)}
              disabled={currentIndex === 0}
              style={{ backgroundColor: currentIndex === 0 ? '#a3a3a3' : '#4b5563' }}
            >
              Previous
            </button>
            <button
              onClick={() => setCurrentIndex(currentIndex + 1)}
              disabled={currentIndex === questions.length - 1}
              style={{ backgroundColor: currentIndex === questions.length - 1 ? '#a3a3a3' : '#4b5563' }}
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default ExamMode;
