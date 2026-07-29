import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from './AuthContext';
import "./Signup.css"; // Can share the layout styling with signup

function Signin() {
  const navigate = useNavigate();
  const { login } = useAuth();

  const [formData, setFormData] = useState({
    email: "",
    password: "",
  });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      const res = await fetch("http://localhost:5000/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || "Login failed");
      }

      // Save token + user info via AuthContext so the whole app knows we're logged in
      login(data.token, data.user);

      navigate("/"); // change this to your actual landing route
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

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

            <form className="signup-form" onSubmit={handleSubmit}>
              <div className="input-group">
                <label>Email address</label>
                <input
                  type="email"
                  name="email"
                  placeholder="you@example.com"
                  value={formData.email}
                  onChange={handleChange}
                  required
                />
              </div>

              <div className="input-group">
                <label>Password</label>
                <input
                  type="password"
                  name="password"
                  placeholder="Enter your password"
                  value={formData.password}
                  onChange={handleChange}
                  required
                />
              </div>

              {error && <p className="form-error">{error}</p>}

              <button type="submit" className="submit-btn" disabled={loading}>
                {loading ? "Signing in..." : "Sign In"}
              </button>
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