import React from 'react';

const LandingPage = ({ setView }) => (
  <div className="card">
    <h1>Elevare: Public speaking coach</h1>
    <p>How would you like to provide your video?</p>
    <div className="button-group">
      <button onClick={() => setView('upload')}>📁 Upload Video</button>
      <button onClick={() => setView('record')}>🎥 Record Live</button>
    </div>
  </div>
);

export default LandingPage;