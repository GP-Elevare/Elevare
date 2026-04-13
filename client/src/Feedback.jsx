import React, { useEffect, useState, useRef } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import "./Feedback.css";
import "./Home.css";
import user from "./assets/user-stroke-rounded.png";

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
    <div>
      {/* Hidden File Input for PPT */}
      <input 
        type="file" 
        ref={fileInputRef} 
        style={{ display: 'none' }} 
        accept=".pptx, .ppt"
        onChange={handleFileChange} 
      />

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

      <div className="feedback-container">
        <h2 id="big">Analysis Results</h2>
        
        {error && <div className="error-message">{error}</div>}

        {loading ? (
          <p className="loading-text">Analyzing your performance...</p>
        ) : (
          <div className="feedback-content">
            <section className="feedback-section">
              <div className="feedback-text-box">
                {feedback?.stage3_feedback?.full_formatted_text || 'No feedback text available.'}
              </div>
            </section>

            {/* PPT Upload Button */}
            <div className="action-area">
              <button 
                id="learn" 
                onClick={handleGetQuestionsClick} 
                disabled={uploading}
              >
                {uploading ? 'Processing PPT...' : 'get questions'}
              </button>
            </div>
          </div>
        )}
      </div>

      <div className="bottom-bar">
        <div id="title">Elevare</div>
      </div>
    </div>
  );
}

export default Feedback;