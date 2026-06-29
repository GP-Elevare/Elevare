import { useNavigate } from 'react-router-dom';
import { useState, useRef } from 'react';
import "./Home.css";

function Home() {
  const navigate = useNavigate();
  const fileInputRef = useRef(null);
  const [loading, setLoading] = useState(false);
  
  // State to track if the user is uploading a video or slides
  const [uploadMode, setUploadMode] = useState('video'); // 'video' or 'slides'

  const handleButtonClick = () => {
    fileInputRef.current.click();
  };

  const handleFileChange = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setLoading(true);

    const formData = new FormData();
    
    // Append parameters based on the selected mode
    if (uploadMode === 'video') {
      formData.append('video', file);
      formData.append('fps', 30);
      formData.append("intervalSec", 1);
    } else {
      formData.append('powerpoint', file);
    }

    // Set the endpoint dynamically based on the current mode
    const endpoint = uploadMode === 'video' 
      ? 'http://localhost:5000/process-video-ai' 
      : 'http://localhost:5000/upload-ppt';

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        
        // Navigate based on the current upload mode
        if (uploadMode === 'video') {
          navigate('/feedback', { state: { data: data, mode: uploadMode } });
        } else {
          navigate('/questions', { state: { questions: data } });
        }
        
      } else {
        alert("Upload failed.");
      }
    } catch (err) {
      console.error(err);
      alert("Error connecting to server.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <input
        type="file"
        ref={fileInputRef}
        style={{ display: 'none' }}
        /* Dynamically accept file types based on mode */
        accept={uploadMode === 'video' ? "video/*" : ".pdf,.pptx,.ppt"}
        onChange={handleFileChange}
      />

      <header className="top-bar">
        <div className="logo-container">
          <div className="logo-icon">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="22"></line></svg>
          </div>
          <div className="title">Elev<span className="highlight">are</span></div>
        </div>
        <button className="sign-in-btn">Sign In</button>
      </header>

      <main className="main-content">
        <div className="hero">
          <div className="badge">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"></path></svg>
            AI-powered speaking coach
          </div>
          <h1 className="main-heading">Practice makes you <span className="highlight">confident</span></h1>
          <p className="subtitle">Upload your video or slides and get personalized feedback to help you<br />speak with clarity and calm.</p>
        </div>

        <div className="options-container">
          <div 
            className={`option-card ${uploadMode === 'video' ? 'active' : ''}`}
            onClick={() => setUploadMode('video')}
          >
            <div className="card-icon-wrapper">
              <div className={`card-icon ${uploadMode === 'video' ? '' : 'light'}`}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="23 7 16 12 23 17 23 7"></polygon><rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect></svg>
              </div>
            </div>
            <h3>Video analysis</h3>
            <p>Upload a recording and get<br />emotion and delivery feedback</p>
          </div>
          <div 
            className={`option-card ${uploadMode === 'slides' ? 'active' : ''}`}
            onClick={() => setUploadMode('slides')}
          >
            <div className="card-icon-wrapper">
              <div className={`card-icon ${uploadMode === 'slides' ? '' : 'light'}`}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect><line x1="8" y1="21" x2="16" y2="21"></line><line x1="12" y1="17" x2="12" y2="21"></line></svg>
              </div>
            </div>
            <h3>Slides practice</h3>
            <p>Upload your PPT and answer<br />AI-generated questions</p>
          </div>
        </div>

        <div className="upload-box" onClick={handleButtonClick}>
          <div className="upload-icon">
            {uploadMode === 'video' ? (
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="23 7 16 12 23 17 23 7"></polygon><rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect></svg>
            ) : (
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect><line x1="8" y1="21" x2="16" y2="21"></line><line x1="12" y1="17" x2="12" y2="21"></line></svg>
            )}
          </div>
          
          {/* Dynamic text rendering based on state */}
          <h3>
            {uploadMode === 'video' 
              ? 'Drop your video here' 
              : 'Drop your presentation here'}
          </h3>
          <p>
            {uploadMode === 'video' 
              ? <>Record yourself presenting, then upload it for in-depth<br />emotion and delivery analysis.</>
              : <>Upload your presentation deck and prepare yourself with<br />AI-generated questions.</>}
          </p>

          <button 
            className="browse-btn" 
            disabled={loading} 
            onClick={(e) => { 
              e.stopPropagation(); 
              handleButtonClick(); 
            }}
          >
            {loading ? 'Processing...' : 'Browse files'}
          </button>
        </div>
      </main>
    </div>
  );
}

export default Home;