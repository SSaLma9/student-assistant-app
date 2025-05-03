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
    <div className="bg-white p-4 sm:p-6 rounded-xl shadow-lg animate-fade-in">
      <div className="flex items-center mb-6">
        <button
          onClick={() => setView('lectures')}
          className="text-indigo-600 hover:text-indigo-800 mr-4"
        >
          <ArrowLeftIcon className="h-6 w-6" />
        </button>
        <h2 className="text-xl sm:text-2xl font-bold text-gray-800">Exam Mode for {selectedLecture}</h2>
      </div>
      {error && <p className="text-red-500 mb-4 text-sm sm:text-base">{error}</p>}
      <div className="space-y-6 mb-6 sm:space-y-0 sm:grid sm:grid-cols-2 sm:gap-4">
        <div>
          <label className="block text-sm sm:text-base font-medium text-gray-700">Exam Type</label>
          <select
            value={examType}
            onChange={(e) => setExamType(e.target.value)}
            className="mt-1 w-full p-3 sm:p-4 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500 text-sm sm:text-base"
          >
            {examTypes.map((type) => (
              <option key={type} value={type}>{type}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-sm sm:text-base font-medium text-gray-700">Difficulty</label>
          <select
            value={difficulty}
            onChange={(e) => setDifficulty(e.target.value)}
            className="mt-1 w-full p-3 sm:p-4 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500 text-sm sm:text-base"
          >
            {difficulties.map((diff) => (
              <option key={diff} value={diff}>{diff}</option>
            ))}
          </select>
        </div>
        <button
          onClick={handleGenerateExam}
          disabled={loading || !selectedLecture}
          className="w-full bg-indigo-600 text-white p-3 sm:p-4 rounded-lg hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 active:bg-indigo-800 disabled:bg-indigo-300 text-sm sm:text-base font-medium"
        >
          {loading ? (
            <svg className="animate-spin h-5 w-5 mr-2 text-white" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
            </svg>
          ) : null}
          Generate Exam
        </button>
      </div>
      {questions.length > 0 && currentQuestion && (
        <div className="space-y-6">
          <h3 className="text-xl sm:text-2xl font-semibold text-gray-800">
            Question {currentIndex + 1} of {questions.length}
          </h3>
          <div className="bg-blue-50 p-4 sm:p-6 rounded-lg shadow overflow-y-auto max-h-[50vh] sm:max-h-[60vh]">
            <p className="text-lg sm:text-xl font-medium text-gray-800 mb-4">{currentQuestion.question}</p>
            {currentQuestion.type === 'mcq' && currentQuestion.options && (
              <div className="space-y-2">
                {currentQuestion.options.map((option, idx) => (
                  <label key={idx} className="flex items-center space-x-2">
                    <input
                      type="radio"
                      name={`question-${currentQuestion.id}`}
                      value={option}
                      checked={answers[currentQuestion.id] === option}
                      onChange={(e) => setAnswers({ ...answers, [currentQuestion.id]: e.target.value })}
                      className="h-4 w-4 sm:h-5 sm:w-5 text-indigo-600"
                    />
                    <span className="text-sm sm:text-base">{option}</span>
                  </label>
                ))}
              </div>
            )}
            {currentQuestion.type === 'essay' && (
              <textarea
                value={answers[currentQuestion.id] || ''}
                onChange={(e) => setAnswers({ ...answers, [currentQuestion.id]: e.target.value })}
                className="w-full p-3 sm:p-4 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500 text-sm sm:text-base"
                placeholder="Write your essay answer here..."
                rows="6"
              />
            )}
          </div>
          <button
            onClick={() => handleAnswerSubmit(currentQuestion.id, answers[currentQuestion.id] || '')}
            disabled={loading}
            className="w-full bg-indigo-600 text-white p-3 sm:p-4 rounded-lg hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 active:bg-indigo-800 disabled:bg-indigo-300 text-sm sm:text-base font-medium"
          >
            {loading ? (
              <svg className="animate-spin h-5 w-5 mr-2 text-white" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
              </svg>
            ) : null}
            Submit Answer
          </button>
          {feedback && (
            <div className="bg-green-50 p-4 sm:p-6 rounded-lg overflow-y-auto max-h-[40vh] sm:max-h-[50vh]">
              <h3 className="text-lg sm:text-xl font-semibold text-gray-800 mb-2">Feedback</h3>
              <p className="text-gray-700 whitespace-pre-wrap text-sm sm:text-base">{feedback}</p>
            </div>
          )}
          <div className="flex flex-col sm:flex-row justify-between space-y-4 sm:space-y-0 sm:space-x-4">
            <button
              onClick={() => setCurrentIndex(currentIndex - 1)}
              disabled={currentIndex === 0}
              className={`w-full sm:w-auto bg-gray-600 text-white p-3 sm:p-4 rounded-lg ${currentIndex === 0 ? 'opacity-50 cursor-not-allowed' : 'hover:bg-gray-700'}`}
            >
              Previous
            </button>
            <button
              onClick={() => setCurrentIndex(currentIndex + 1)}
              disabled={currentIndex === questions.length - 1}
              className={`w-full sm:w-auto bg-gray-600 text-white p-3 sm:p-4 rounded-lg ${currentIndex === questions.length - 1 ? 'opacity-50 cursor-not-allowed' : 'hover:bg-gray-700'}`}
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
