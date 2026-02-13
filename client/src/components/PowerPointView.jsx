import React, { useState } from 'react';

const PowerPointView = ({ onBack }) => {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState('');

  const handleUpload = async () => {
    if (!file) return alert("Select a PowerPoint file!");

    const formData = new FormData();
    formData.append('powerpoint', file);

    try {
      const response = await fetch('http://localhost:5000/upload-ppt', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setStatus(data.message);
    } catch (err) {
      setStatus("Upload failed.");
    }
  };

  return (
    <div className="card">
      <button className="back-btn" onClick={onBack}>← Back</button>
      <h2>PowerPoint Upload</h2>
      <p>Upload your presentation slides here.</p>
      
      <input 
        type="file" 
        accept=".pptx, .ppt" 
        onChange={(e) => setFile(e.target.files[0])} 
      />
      
      <button onClick={handleUpload} style={{ marginTop: '20px' }}>
        Upload Slides
      </button>

      {status && <div className="success-msg">{status}</div>}
    </div>
  );
};

export default PowerPointView;