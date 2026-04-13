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
          body: JSON.stringify({ updatedData: updatedQuestions }), // Sending the array of objects
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

  return (
    <div>
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
            <div className="question">
              <div id="question-label">
                Question {currentIndex + 1} of {questions.length}:
              </div>
              {/* Note the .question here, because it's now an object */}
              <div className="question-text">
                {questions[currentIndex].question}
              </div>
            </div>

            <div className="answer">
              <div id="answer-label">Your Answer:</div>
              <textarea 
                id="answer-input" 
                placeholder="Type your answer here..."
                rows="4"
                value={currentAnswer} 
                onChange={(e) => setCurrentAnswer(e.target.value)} 
              ></textarea>
              
              <div className="button-group">
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

      <div className="bottom-bar">
        <div id="title">Elevare</div>
      </div>
    </div>
  );
}

export default Questions;