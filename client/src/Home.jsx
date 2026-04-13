import { useNavigate, Link } from 'react-router-dom';
import { useState, useRef } from 'react';
import './App.css';
import './Home.css';

function Home() {
  const navigate = useNavigate();
  const fileInputRef = useRef(null);
  const [loading, setLoading] = useState(false);

  const handleButtonClick = () => {
    fileInputRef.current.click();
  };

  const handleFileChange = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append('video', file);
    formData.append('fps', 30);
    formData.append('intervalSec', 1);
    try {
      const response = await fetch('http://localhost:5000/process-video-ai', {
        method: 'POST',
        body: formData,
      });
      if (response.ok) {
        const data = await response.json();
        navigate('/feedback', { state: { videoData: data } });
      } else {
        alert('Upload failed.');
      }
    } catch (err) {
      console.error(err);
      alert('Error connecting to server.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-wrapper">
      <input
        type="file"
        ref={fileInputRef}
        style={{ display: 'none' }}
        accept="video/*"
        onChange={handleFileChange}
      />

      {/* Nav */}
      <nav className="nav">
        <span className="nav-logo">Elevare</span>
        <div className="nav-links">
          <Link to="/" className="nav-link nav-link-a">Home</Link>
          <button className="nav-cta" onClick={handleButtonClick} disabled={loading}>
            {loading ? 'Processing...' : 'Get started'}
          </button>
        </div>
      </nav>

      {/* Hero */}
      <section className="hero">
        <div className="hero-eyebrow">AI-powered coaching</div>
        <h1>Speak with <span>confidence</span>,<br />every time</h1>
        <p className="hero-sub">
          Upload a video of your presentation and get instant AI feedback on
          delivery, clarity, body language, and more.
        </p>
        <div className="hero-actions">
          <button className="btn-primary" onClick={handleButtonClick} disabled={loading}>
            {loading ? 'Processing...' : 'Analyze my speaking'}
          </button>
        </div>
      </section>

      {/* Features */}
      <section className="features-section">
        <div className="features-grid">
          <div className="feat-card">
            <div className="feat-icon">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#534AB7" strokeWidth="2">
                <circle cx="12" cy="12" r="10"/>
                <path d="M12 8v4l3 3"/>
              </svg>
            </div>
            <h3>Instant analysis</h3>
            <p>Get detailed feedback within seconds of uploading your video.</p>
          </div>
          <div className="feat-card">
            <div className="feat-icon">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#534AB7" strokeWidth="2">
                <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>
              </svg>
            </div>
            <h3>Smart Q&amp;A</h3>
            <p>Test your knowledge with AI-generated questions from your slides.</p>
          </div>
          <div className="feat-card">
            <div className="feat-icon">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#534AB7" strokeWidth="2">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
              </svg>
            </div>
            <h3>Track progress</h3>
            <p>See how your skills improve session over session.</p>
          </div>
        </div>
      </section>

      {/* How it works */}
      <section className="steps-section">
        <div className="section-label">How it works</div>
        <h2 className="section-title">Three steps to better speaking</h2>
        <div className="steps-grid">
          <div className="step-card">
            <span className="step-num">Step 1</span>
            <h3>Record or upload</h3>
            <p>Film yourself giving a presentation or upload an existing video.</p>
          </div>
          <div className="step-card">
            <span className="step-num">Step 2</span>
            <h3>Get AI feedback</h3>
            <p>Our AI analyzes your pacing, filler words, posture, and delivery.</p>
          </div>
          <div className="step-card">
            <span className="step-num">Step 3</span>
            <h3>Practice & improve</h3>
            <p>Answer slide-based questions and track your growth over time.</p>
          </div>
        </div>
      </section>

      {/* CTA banner */}
      <div className="cta-banner">
        <h2>Ready to level up?</h2>
        <p>Join thousands of speakers who use Elevare to sharpen their skills.</p>
        <button className="btn-white" onClick={handleButtonClick} disabled={loading}>
          {loading ? 'Processing...' : 'Upload your first video'}
        </button>
      </div>

      {/* Footer */}
      <footer className="site-footer">
        <span className="footer-brand">Elevare</span>
        <div className="footer-links">
          <span className="footer-link">Privacy</span>
          <span className="footer-link">Terms</span>
          <span className="footer-link">Contact</span>
        </div>
      </footer>
    </div>
  );
}

export default Home;
