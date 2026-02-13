import React, { useState, useRef } from 'react';

const RecordView = ({ onBack }) => {
  const [recording, setRecording] = useState(false);
  const [videoBlob, setVideoBlob] = useState(null);
  const [fps, setFps] = useState(30);
  const [loading, setLoading] = useState(false);
  
  const mediaRecorderRef = useRef(null);
  const videoRef = useRef(null);

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
    videoRef.current.srcObject = stream;
    
    mediaRecorderRef.current = new MediaRecorder(stream);
    const chunks = [];

    mediaRecorderRef.current.ondataavailable = (e) => chunks.push(e.data);
    
    mediaRecorderRef.current.onstop = () => {
      const blob = new Blob(chunks, { type: 'video/mp4' });
      setVideoBlob(blob);
      stream.getTracks().forEach(track => track.stop()); 
    };

    mediaRecorderRef.current.start();
    setRecording(true);
  };

  const stopRecording = () => {
    mediaRecorderRef.current.stop();
    setRecording(false);
  };

  const handleProcess = async () => {
    if (!videoBlob) return;
    setLoading(true);

    const formData = new FormData();
    formData.append('video', videoBlob, 'webcam-record.mp4');
    formData.append('fps', fps);

    try {
      const response = await fetch('http://localhost:5000/process-video', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (data.downloadUrl) {
        window.location.href = `http://localhost:5000${data.downloadUrl}`;
      }
    } catch (err) {
      console.error(err);
      alert("Error processing recording.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <button className="back-btn" onClick={onBack}>← Back</button>
      <h2>Record Live</h2>
      
      {!videoBlob ? (
        <video ref={videoRef} autoPlay muted className="preview-window" />
      ) : (
        <div className="success-msg">Recording Captured!</div>
      )}
      
      <div className="button-group">
        {!recording ? (
          <button onClick={startRecording}>Start Recording</button>
        ) : (
          <button className="stop-btn" onClick={stopRecording}>Stop Recording</button>
        )}
      </div>

      {videoBlob && (
        <div className="process-section">
          <div className="input-group">
            <label>Target FPS:</label>
            <input 
              type="number" 
              value={fps} 
              onChange={(e) => setFps(e.target.value)} 
            />
          </div>
          <button onClick={handleProcess} disabled={loading}>
            {loading ? 'Processing...' : 'Convert & Download'}
          </button>
        </div>
      )}
    </div>
  );
};

export default RecordView;