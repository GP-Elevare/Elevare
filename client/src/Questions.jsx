import React, { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import "./Questions.css";

function Questions() {
  const [questions, setQuestions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [currentIndex, setCurrentIndex] = useState(0);

  // State for the text inside the box right now
  const [currentAnswer, setCurrentAnswer] = useState("");

  const navigate = useNavigate();

  useEffect(() => {
    const fetchQuestions = async () => {
      try {
        const response = await fetch('http://localhost:5000/questions');
        if (!response.ok) throw new Error('Unable to load questions');
        
        let data = await response.json();
        
        // Handle the data whether it's wrapped in an object or just a raw array
        let questionArray = data.questions || data.result || data; 
        if (!Array.isArray(questionArray)) questionArray = [];

        // LIMIT TO 5 ENTRIES
        questionArray = questionArray.slice(0, 5);
        
        setQuestions(questionArray);
      } catch (err) {
        console.error(err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchQuestions();
  }, []);

  const handleNext = async () => {
    // 1. Copy the questions array and update the student_answer for the current question
    const updatedQuestions = [...questions];
    updatedQuestions[currentIndex].student_answer = currentAnswer;
    setQuestions(updatedQuestions);

    // 2. Clear the textarea for the next question
    setCurrentAnswer("");

    // 3. If it is the last question, send the updated array to the server
    if (currentIndex === questions.length - 1) {
      try {
        const response = await fetch('http://localhost:5000/save-answers', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ updatedData: updatedQuestions }), 
        });

        if (response.ok) {
          alert("All answers saved and graded successfully!");
          navigate('/questions-feedback');
        } else {
          alert("Failed to save answers.");
        }
      } catch (err) {
        console.error("Error saving answers", err);
        alert("Error connecting to server.");
      }
    } else {
      // Move to the next question
      setCurrentIndex(currentIndex + 1);
    }
  };

  // Calculate word count
  const wordCount = currentAnswer.trim() === "" ? 0 : currentAnswer.trim().split(/\s+/).length;

  // Calculate header progress bar width
  const progressPercent = questions.length > 0 ? ((currentIndex + 1) / questions.length) * 100 : 0;

  return (
    <div className="page-wrapper">
      <header className="top-bar">
        <div className="logo-container">
          <div className="logo-icon">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="22"></line></svg>
          </div>
          <div className="title">Elev<span className="highlight">are</span></div>
        </div>
        
        <div className="header-center">
          Q&A Session
        </div>

        <button className="sign-in-btn">Sign In</button>
        
        {/* Dynamic top progress line */}
        <div className="header-progress-line" style={{ width: `${progressPercent}%` }}></div>
      </header>

      <main className="qa-container">
        {error && <div className="error-message">{error}</div>}

        {loading ? (
          <p className="loading-text">Loading your questions...</p>
        ) : questions.length > 0 ? (
          <>
            <div className="question-header">
              <div className="question-meta">
                Question <span className="bold-num">{currentIndex + 1}</span> of <span className="bold-num">{questions.length}</span>
              </div>
              <div className="progress-dots">
                {questions.map((_, index) => (
                  <span 
                    key={index} 
                    className={`dot ${index === currentIndex ? 'active' : ''} ${index < currentIndex ? 'completed' : ''}`}
                  ></span>
                ))}
              </div>
            </div>

            <div className="question-card">
              {questions[currentIndex].question}
            </div>

            <div className="answer-section">
              <div className="answer-label">YOUR ANSWER</div>
              <textarea 
                className="dark-textarea"
                placeholder="Type your answer here..."
                value={currentAnswer} 
                onChange={(e) => setCurrentAnswer(e.target.value)} 
              ></textarea>
              
              <div className="textarea-footer">
                <span className="word-count">{wordCount} words</span>
                <button 
                  className={`next-button ${currentAnswer.trim().length > 0 ? 'active' : ''}`} 
                  onClick={handleNext}
                >
                  {currentIndex === questions.length - 1 ? "Submit All" : "Next"}
                </button>
              </div>
            </div>

            <div className="relax-message">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path></svg>
              Take your time — there's no rush
            </div>
          </>
        ) : (
          <p className="loading-text">No questions generated yet. Please upload a presentation first.</p>
        )}
      </main>
    </div>
  );
}

export default Questions;