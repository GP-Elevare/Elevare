import React, { useEffect, useState, useRef } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import "./Feedback.css";

function Feedback() {
  const [feedback, setFeedback] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);

  const navigate = useNavigate();
  const fileInputRef = useRef(null);

  // Fetch initial feedback on load
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

  // Trigger hidden file input
  const handleGetQuestionsClick = () => {
    fileInputRef.current.click();
  };

  // Handle PPT upload and navigation
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
        // Redirect to questions page with the data
        navigate('/questions', { state: { questions: data } });
      } else {
        alert("PPT upload failed.");
      }
    } catch (err) {
      console.error(err);
      alert("Error connecting to server.");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="page-wrapper">
      {/* Hidden File Input for PPT */}
      <input 
        type="file" 
        ref={fileInputRef} 
        style={{ display: 'none' }} 
        accept=".pptx, .ppt, .pdf"
        onChange={handleFileChange} 
      />

      {/* Navigation Bar */}
      <header className="top-bar">
        <div className="logo-container">
          <div className="logo-icon">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="22"></line></svg>
          </div>
          <div className="title">Elev<span className="highlight">are</span></div>
        </div>
        
        <div className="header-center">
          Video Analysis
        </div>

        <button className="sign-in-btn">Sign In</button>
      </header>

      <main className="feedback-page-container">
        <div className="feedback-header-section">
          <h2 className="main-heading">Analysis Results</h2>
          <p className="subtitle">Review your presentation's emotion and delivery feedback.</p>
        </div>
        
        {error && <div className="error-message">{error}</div>}
        
        {loading ? (
          <p className="loading-text">Analyzing your performance...</p>
        ) : (
          <div className="feedback-content">
            <div className="result-card">
              <div className="card-header">
                <h3>AI Feedback</h3>
              </div>
              <div className="card-body">
                <div className="feedback-text-box">
                  {feedback?.full_formatted_text || 'No feedback text available.'}
                </div>
              </div>
            </div>

            <div className="bottom-actions">
              <button 
                className="secondary-btn" 
                onClick={() => navigate('/')}
              >
                Back to Home
              </button>
              <button
                className="primary-btn" 
                onClick={handleGetQuestionsClick} 
                disabled={uploading}
              >
                {uploading ? 'Processing Slides...' : 'Upload Slides for Q&A'}
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default Feedback;