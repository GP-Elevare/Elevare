import React, { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import './App.css';
import './Questions.css';

function Questions() {
  const [questions, setQuestions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [currentAnswer, setCurrentAnswer] = useState('');

  const navigate = useNavigate();

  useEffect(() => {
    const fetchQuestions = async () => {
      try {
        const response = await fetch('http://localhost:5000/questions');
        if (!response.ok) throw new Error('Unable to load questions');
        let data = await response.json();
        let questionArray = data.questions || data.result || data;
        if (!Array.isArray(questionArray)) questionArray = [];
        questionArray = questionArray.slice(0, 5);
        setQuestions(questionArray);
      } catch (err) {
        console.error(err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchQuestions();
  }, []);

  const handleNext = async () => {
    const updatedQuestions = [...questions];
    updatedQuestions[currentIndex].student_answer = currentAnswer;
    setQuestions(updatedQuestions);
    setCurrentAnswer('');

    if (currentIndex === questions.length - 1) {
      try {
        const response = await fetch('http://localhost:5000/save-answers', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ updatedData: updatedQuestions }),
        });
        if (response.ok) {
          navigate('/questions-feedback');
        } else {
          alert('Failed to save answers.');
        }
      } catch (err) {
        console.error('Error saving answers', err);
        alert('Error connecting to server.');
      }
    } else {
      setCurrentIndex(currentIndex + 1);
    }
  };

  return (
    <div className="page-wrapper">
      <nav className="nav">
        <Link to="/" className="nav-logo nav-link-a">Elevare</Link>
        <div className="nav-links">
          <Link to="/" className="nav-link nav-link-a">Home</Link>
        </div>
      </nav>

      <div className="feedback-container">
        <div className="page-header">
          <h2>Q&amp;A session</h2>
          <p>Answer each question based on your slide content</p>
        </div>

        {error && <div className="error-message">{error}</div>}

        {loading ? (
          <p className="loading-text">Loading your questions...</p>
        ) : questions.length > 0 ? (
          <>
            {/* Progress dots */}
            <div className="q-progress-bar">
              {questions.map((_, i) => (
                <div
                  key={i}
                  className={`q-progress-dot ${i < currentIndex ? 'done' : i === currentIndex ? 'active' : ''}`}
                />
              ))}
            </div>

            {/* Question */}
            <div className="q-box">
              <div className="q-counter">Question {currentIndex + 1} of {questions.length}</div>
              <div className="q-text">{questions[currentIndex].question}</div>
            </div>

            {/* Answer */}
            <textarea
              className="answer-area"
              placeholder="Type your answer here..."
              value={currentAnswer}
              onChange={(e) => setCurrentAnswer(e.target.value)}
            />

            <div className="q-footer-row">
              <span className="q-hint">Take your time — no time limit</span>
              <button className="btn-primary" onClick={handleNext}>
                {currentIndex === questions.length - 1 ? 'Submit answers' : 'Next question'}
              </button>
            </div>
          </>
        ) : (
          <p className="loading-text">No questions yet. Please upload a PPTX first.</p>
        )}
      </div>

      <footer className="site-footer">
        <span className="footer-brand">Elevare</span>
        <div className="footer-links">
          <span className="footer-link">Privacy</span>
          <span className="footer-link">Terms</span>
        </div>
      </footer>
    </div>
  );
}

export default Questions;
