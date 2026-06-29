import React, { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import "./QuestionsFeedback.css";

function QuestionsFeedback() {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

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

  // Helper function to color-code labels
  const getLabelClass = (label) => {
    if (!label) return "label-default";
    const lowerLabel = label.toLowerCase();
    if (lowerLabel.includes("incorrect")) return "label-incorrect";
    if (lowerLabel.includes("partially")) return "label-partial";
    if (lowerLabel.includes("correct")) return "label-correct";
    return "label-default";
  };

  return (
    <div className="page-wrapper">
      {/* Navigation Bar */}
      <header className="top-bar">
        <div className="logo-container">
          <div className="logo-icon">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="22"></line></svg>
          </div>
          <div className="title">Elev<span className="highlight">are</span></div>
        </div>
        
        <div className="header-center">
          Feedback & Results
        </div>

        <button className="sign-in-btn">Sign In</button>
      </header>

      <main className="feedback-page-container">
        <div className="feedback-header-section">
          <h2 className="main-heading">Q&A Grading Feedback</h2>
          <p className="subtitle">Review your answers and see how you can improve your delivery.</p>
        </div>
        
        {error && <div className="error-message">{error}</div>}

        {loading ? (
          <p className="loading-text">Analyzing and loading your grades...</p>
        ) : results.length > 0 ? (
          <div className="results-list">
            {results.map((item, index) => (
              <div key={index} className="result-card">
                <div className="card-header">
                  <h3>Question {index + 1}</h3>
                  <span className={`status-badge ${getLabelClass(item.label)}`}>
                    {item.label ? item.label.toUpperCase() : "UNGRADED"}
                  </span>
                </div>
                
                <div className="card-body">
                  <div className="qa-block">
                    <div className="block-label">Question</div>
                    <div className="block-text bold">{item.question}</div>
                  </div>
                  
                  <div className="qa-block">
                    <div className="block-label">Your Answer</div>
                    <div className="block-text">{item.student_answer || <em className="no-answer">No answer provided</em>}</div>
                  </div>
                  
                  <div className="feedback-block">
                    <div className="block-label ai-label">
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"></path></svg>
                      AI Feedback
                    </div>
                    <div className="block-text">{item.feedback}</div>
                  </div>
                </div>
              </div>
            ))}
            
            <div className="bottom-actions">
              <button className="primary-btn" onClick={() => navigate('/')}>
                Practice Again
              </button>
            </div>
          </div>
        ) : (
          <div className="empty-state">
            <p className="loading-text">No results found. Please complete a Q&A session first.</p>
            <button className="primary-btn" onClick={() => navigate('/')}>Go to Home</button>
          </div>
        )}
      </main>
    </div>
  );
}

export default QuestionsFeedback;