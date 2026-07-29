import React from 'react';
import { Link } from 'react-router-dom';
import "./Signup.css"; // Can share the layout styling with signup

function Signin() {
  return (
    <div className="page-wrapper">
      <header className="top-bar">
        <Link to="/" className="logo-link">
          <div className="logo-container">
            <div className="logo-icon">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="22"></line></svg>
            </div>
            <div className="title">Elev<span className="highlight">are</span></div>
          </div>
        </Link>
      </header>

      <main className="signup-page-container">
        <div className="signup-card">
          
          {/* Left Branding Panel */}
          <div className="signup-brand-panel">
            <div className="brand-icon-large">
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="22"></line></svg>
            </div>
            <h2 className="brand-title">Elev<span className="highlight">are</span></h2>
            <p className="brand-subtitle">Your AI-powered public speaking coach</p>
          </div>

          {/* Right Form Panel */}
          <div className="signup-form-panel">
            <div className="form-header">
              <h1>Welcome back</h1>
              <p>Log in to access your dashboard and past sessions.</p>
            </div>

            <form className="signup-form" onSubmit={(e) => e.preventDefault()}>
              <div className="input-group">
                <label>Email address</label>
                <input type="email" placeholder="you@example.com" />
              </div>

              <div className="input-group">
                <label>Password</label>
                <input type="password" placeholder="Enter your password" />
              </div>

              <button type="submit" className="submit-btn">Sign In</button>
            </form>

            <div className="divider">
              <span>or</span>
            </div>

            <div className="login-prompt">
              Don't have an account? <Link to="/signup" className="login-link">Sign up</Link>
            </div>
          </div>

        </div>
      </main>
    </div>
  );
}

export default Signin;