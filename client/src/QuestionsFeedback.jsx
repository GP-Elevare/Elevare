import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import './App.css';
import './QuestionsFeedback.css';

function QuestionsFeedback() {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchResults = async () => {
      try {
        const response = await fetch('http://localhost:5000/qa-results');
        if (!response.ok) throw new Error('Unable to load graded results');
        const data = await response.json();
        setResults(data.qa_results || []);
      } catch (err) {
        console.error(err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchResults();
  }, []);

  const getLabelClass = (label) => {
    if (!label) return '';
    const l = label.toLowerCase();
    if (l.includes('incorrect')) return 'label-incorrect';
    if (l.includes('partial'))   return 'label-partial';
    if (l.includes('correct'))   return 'label-correct';
    return '';
  };

  const getPillClass = (label) => {
    if (!label) return '';
    const l = label.toLowerCase();
    if (l.includes('incorrect')) return 'incorrect';
    if (l.includes('partial'))   return 'partial';
    if (l.includes('correct'))   return 'correct';
    return '';
  };

  const countByType = (type) =>
    results.filter((r) => r.label && r.label.toLowerCase().includes(type)).length;

  return (
    <div className="page-wrapper">
      <nav className="nav">
        <Link to="/" className="nav-logo nav-link-a">Elevare</Link>
        <div className="nav-links">
          <Link to="/" className="nav-link nav-link-a">Home</Link>
          <Link to="/" className="nav-link-a">
            <button className="nav-cta">New session</button>
          </Link>
        </div>
      </nav>

      <div className="feedback-container">
        <div className="page-header">
          <h2>Q&amp;A results</h2>
          <p>Here's how you did on the practice questions</p>
        </div>

        {error && <div className="error-message">{error}</div>}

        {loading ? (
          <p className="loading-text">Loading your grades...</p>
        ) : results.length > 0 ? (
          <>
            {/* Summary pills */}
            <div className="result-summary">
              <span className="result-pill correct">{countByType('correct')} correct</span>
              <span className="result-pill partial">{countByType('partial')} partial</span>
              <span className="result-pill incorrect">{countByType('incorrect')} incorrect</span>
            </div>

            {results.map((item, index) => (
              <div key={index} className="qa-result-card">
                <div className="qa-result-header">
                  <span className="qa-result-num">Question {index + 1}</span>
                  <span className={`status-badge ${getLabelClass(item.label)}`}>
                    {item.label ? item.label.toUpperCase() : 'UNGRADED'}
                  </span>
                </div>
                <div className="qa-body">
                  <p><strong>Question:</strong> {item.question}</p>
                  <p>
                    <strong>Your answer:</strong>{' '}
                    {item.student_answer
                      ? item.student_answer
                      : <em style={{ color: 'var(--text-faint)' }}>No answer provided</em>
                    }
                  </p>
                  <div className="qa-feedback-block">
                    <div className="qa-feedback-label">Feedback</div>
                    {item.feedback}
                  </div>
                </div>
              </div>
            ))}

            <div className="action-row" style={{ marginTop: 24 }}>
              <Link to="/" className="nav-link-a">
                <button className="btn-primary">Back to home</button>
              </Link>
              <Link to="/" className="nav-link-a">
                <button className="btn-secondary">New session</button>
              </Link>
            </div>
          </>
        ) : (
          <p className="loading-text">No results found. Please complete the questions first.</p>
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

export default QuestionsFeedback;
