import React, { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import "./Questions.css";
import "./Home.css";
import user from "./assets/user-stroke-rounded.png";

function Questions() {
  const [questions, setQuestions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [currentIndex, setCurrentIndex] = useState(0);

  // New states for tracking answers
  const [currentAnswer, setCurrentAnswer] = useState("");
  const [allAnswers, setAllAnswers] = useState({});

  const navigate = useNavigate();

  useEffect(() => {
    const fetchQuestions = async () => {
      try {
        const response = await fetch('http://localhost:5000/questions');
        if (!response.ok) throw new Error('Unable to load questions');
        
        const data = await response.json();
        
        // Adjusting based on your index.js JSON structure
        const questionArray = data.questions?.questions || data.questions || [];
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
    // 1. Save the current answer to our state object
    const updatedAnswers = {
      ...allAnswers,
      [questions[currentIndex]]: currentAnswer
    };
    setAllAnswers(updatedAnswers);

    // 2. Clear the textarea for the next question
    setCurrentAnswer("");

    // 3. Check if we are on the very last question
    if (currentIndex === questions.length - 1) {
      try {
        // Send all collected answers to the backend
        const response = await fetch('http://localhost:5000/save-answers', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ answers: updatedAnswers }),
        });

        if (response.ok) {
          alert("All answers saved successfully!");
          navigate('/'); // Send the user back to the home page
        } else {
          alert("Failed to save answers.");
        }
      } catch (err) {
        console.error("Error saving answers", err);
        alert("Error connecting to server.");
      }
    } else {
      // If not the last question, just move to the next one
      setCurrentIndex(currentIndex + 1);
    }
  };

  return (
    <div>
      {/* Navigation Bar */}
      <div className="top-bar">
        <div id="title">Elevare</div>
        <div className="side-top-bar">
          <Link to="/" className="nav-link"><div>Home</div></Link>
          <div>Start</div>
          <div>Demo</div>
          <div><img src={user} alt="User" id="user-icon" /></div>
        </div>
      </div>

      <div className="feedback-container">
        <h2 id="big">Q&A Session</h2>

        {error && <div className="error-message">{error}</div>}

        {loading ? (
          <p className="loading-text">Loading your questions...</p>
        ) : questions.length > 0 ? (
          <>
            {/* Question Display */}
            <div className="question">
              <div id="question-label">
                Question {currentIndex + 1} of {questions.length}:
              </div>
              <div className="question-text">
                {questions[currentIndex]}
              </div>
            </div>

            {/* Answer Input */}
            <div className="answer">
              <div id="answer-label">Your Answer:</div>
              <textarea 
                id="answer-input" 
                placeholder="Type your answer here..."
                rows="4"
                value={currentAnswer} // Tie the textarea to React state
                onChange={(e) => setCurrentAnswer(e.target.value)} // Update state on type
              ></textarea>
              
              <div className="button-group">
                {/* Note: Removed the "disabled" attribute so the final click triggers the POST request */}
                <button id="learn" onClick={handleNext}>
                  {currentIndex === questions.length - 1 ? "Submit All" : "Next Question"}
                </button>
              </div>
            </div>
          </>
        ) : (
          <p className="loading-text">No questions generated yet. Please upload a PPTX first.</p>
        )}
      </div>

      {/* Footer */}
      <div className="bottom-bar">
        <div id="title">Elevare</div>
      </div>
    </div>
  );
}

export default Questions;