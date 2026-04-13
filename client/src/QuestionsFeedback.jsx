import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import "./Feedback.css";
import "./Home.css";
import user from "./assets/user-stroke-rounded.png";

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

  // Helper function to color-code labels
  const getLabelClass = (label) => {
    if (!label) return "";
    const lowerLabel = label.toLowerCase();
    if (lowerLabel.includes("incorrect")) return "label-incorrect";
    if (lowerLabel.includes("partially")) return "label-partial";
    if (lowerLabel.includes("correct")) return "label-correct";
    return "";
  };

  return (
    <div>
      {/* Navigation Bar */}
      <div className="top-bar">
        <div id="title">Elevare</div>
        <div className="side-top-bar">
          <Link to="/" className="nav-link">
            <div>Home</div>
          </Link>
          <div>Start</div>
          <div>Demo</div>
          <div><img src={user} alt="User" id="user-icon" /></div>
        </div>
      </div>

      <div className="feedback-container" style={{ paddingBottom: '60px' }}>
        <h2 id="big">Q&A Grading Feedback</h2>
        
        {error && <div className="error-message">{error}</div>}

        {loading ? (
          <p className="loading-text">Loading your grades...</p>
        ) : results.length > 0 ? (
          <div className="feedback-content">
            {results.map((item, index) => (
              <div key={index} className="qa-card">
                <div className="qa-header">
                  <h3>Question {index + 1}</h3>
                  <span className={`status-badge ${getLabelClass(item.label)}`}>
                    {item.label ? item.label.toUpperCase() : "UNGRADED"}
                  </span>
                </div>
                
                <div className="qa-body">
                  <p className="qa-text"><strong>Question:</strong> {item.question}</p>
                  <p className="qa-text"><strong>Your Answer:</strong> {item.student_answer || <em>No answer provided</em>}</p>
                  
                  <div className="feedback-text-box" style={{ marginTop: '15px' }}>
                    <strong>Feedback:</strong> <br/>
                    {item.feedback}
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="loading-text">No results found. Please complete the questions first.</p>
        )}
      </div>

      <div className="bottom-bar">
        <div id="title">Elevare</div>
      </div>
    </div>
  );
}

export default QuestionsFeedback;