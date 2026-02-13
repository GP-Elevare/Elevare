import React, { useState } from 'react';

const UploadView = ({ onBack }) => {
  const [file, setFile] = useState(null);
  const [fps, setFps] = useState(30);
  const [loading, setLoading] = useState(false);

  const handleUpload = async () => {
    if (!file) return alert("Select a file!");
    setLoading(true);

    const formData = new FormData();
    formData.append('video', file);
    formData.append('fps', fps);

    try {
      const response = await fetch('http://localhost:5000/process-video-ai', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      window.location.href = `http://localhost:5000${data.downloadUrl}`;
    } catch (err) {
      alert("Error uploading file.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <button className="back-btn" onClick={onBack}>← Back</button>
      <h2>Upload Video</h2>
      <input type="file" accept="video/*" onChange={(e) => setFile(e.target.files[0])} />
      
      <div className="input-group" style={{marginTop: '20px'}}>
        <label>Target FPS:</label>
        <input type="number" value={fps} onChange={(e) => setFps(e.target.value)} />
      </div>

      <button onClick={handleUpload} disabled={loading}>
        {loading ? 'Processing...' : 'Upload & Convert'}
      </button>
    </div>
  );
};

export default UploadView;