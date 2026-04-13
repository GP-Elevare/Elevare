import React, { useEffect, useState, useRef } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import './App.css';
import './Feedback.css';

function Feedback() {
  const [feedback, setFeedback] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);

  const navigate = useNavigate();
  const fileInputRef = useRef(null);

  useEffect(() => {
    const fetchFeedback = async () => {
      try {
        const response = await fetch('http://localhost:5000/feedback');
        if (!response.ok) throw new Error('Unable to load feedback data');
        const data = await response.json();
        setFeedback(data);
      } catch (err) {
        console.error(err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchFeedback();
  }, []);

  const handleGetQuestionsClick = () => {
    fileInputRef.current.click();
  };

  const handleFileChange = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    setUploading(true);
    const formData = new FormData();
    formData.append('powerpoint', file);
    try {
      const response = await fetch('http://localhost:5000/upload-ppt', {
        method: 'POST',
        body: formData,
      });
      if (response.ok) {
        const data = await response.json();
        navigate('/questions', { state: { questions: data } });
      } else {
        alert('PPT upload failed.');
      }
    } catch (err) {
      console.error(err);
      alert('Error connecting to server.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="page-wrapper">
      <input
        type="file"
        ref={fileInputRef}
        style={{ display: 'none' }}
        accept=".pptx,.ppt"
        onChange={handleFileChange}
      />

      <nav className="nav">
        <Link to="/" className="nav-logo nav-link-a">Elevare</Link>
        <div className="nav-links">
          <Link to="/" className="nav-link nav-link-a">Home</Link>
          <button className="nav-cta" onClick={handleGetQuestionsClick} disabled={uploading}>
            {uploading ? 'Processing...' : 'Practice Q&A'}
          </button>
        </div>
      </nav>

      <div className="feedback-container">
        <div className="page-header">
          <h2>Your analysis</h2>
          <p>Based on your most recent session</p>
        </div>

        {error && <div className="error-message">{error}</div>}

        {loading ? (
          <p className="loading-text">Analyzing your performance...</p>
        ) : (
          <>
            {/* Score row — shown if we have structured scores */}
            <div className="score-row">
              <div className="score-card">
                <div className="score-num">—</div>
                <div className="score-lbl">Overall</div>
              </div>
              <div className="score-card">
                <div className="score-num">—</div>
                <div className="score-lbl">Clarity</div>
              </div>
              <div className="score-card">
                <div className="score-num">—</div>
                <div className="score-lbl">Pacing</div>
              </div>
            </div>

            <div className="feedback-block">
              <div className="feedback-block-label">Feedback</div>
              <div className="feedback-text-box">
                {feedback?.stage3_feedback?.full_formatted_text || 'No feedback text available.'}
              </div>
            </div>

            <div className="action-row">
              <button className="btn-primary" onClick={handleGetQuestionsClick} disabled={uploading}>
                {uploading ? 'Processing PPT...' : 'Practice Q&A from slides'}
              </button>
              <Link to="/" className="nav-link-a">
                <button className="btn-secondary">Analyze another video</button>
              </Link>
            </div>
          </>
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

export default Feedback;
