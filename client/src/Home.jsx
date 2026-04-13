import { useNavigate } from 'react-router-dom';
import { useState, useRef } from 'react';
import "./Home.css"
import user from "./assets/user-stroke-rounded.png"
import guy from "./assets/Business_C_B069_070_1004048570.jpg"

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
    formData.append("intervalSec", 1);

    try {
      const response = await fetch('http://localhost:5000/process-video-ai', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();

        navigate('/feedback', { state: { videoData: data } }); 
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
    <div>
      <input 
        type="file" 
        ref={fileInputRef} 
        style={{ display: 'none' }} 
        accept="video/*"
        onChange={handleFileChange} 
      />

      <div className="top-bar">
        <div id="title">Elevare</div>
        <div className="side-top-bar">
          <div>Home</div>
          <div>Start</div>
          <div>Demo</div>
          <div><img src={user} alt="User" id="user-icon" /></div>
        </div>
      </div>

      <div className="section">
        <img src={guy} alt="Guy" id="guy" />
        <div className="CTA">
          <div id="big">Master public speaking with AI</div>
          <div>Want to learn how to improve your public speaking skills?</div>
          <button id="learn">learn more</button>
        </div>
      </div>

      <div className="section section-no">
        <div className="content">
          <div id="big">What are you waiting for?</div>
          <div>Star recording yourself now!</div>
          
          <button 
            id="learn" 
            onClick={handleButtonClick} 
            disabled={loading}
          >
            {loading ? 'Processing...' : 'upload'}
          </button>
        </div>
      </div>

      <div className="bottom-bar">
        <div id="title">Elevare</div>
      </div>
    </div>
  );
}

export default Home;