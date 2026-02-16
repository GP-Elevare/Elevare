import React, { useEffect, useState } from 'react';

const DataViewer = ({ onBack }) => {
  const [feedback, setFeedback] = useState(null);
  const [questions, setQuestions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [fbRes, qRes] = await Promise.all([
          fetch('http://localhost:5000/feedback'),
          fetch('http://localhost:5000/questions'),
        ]);

        if (!fbRes.ok) throw new Error('Unable to load feedback');
        if (!qRes.ok) throw new Error('Unable to load questions');

        const fbJson = await fbRes.json();
        const qJson = await qRes.json();
        setFeedback(fbJson);
        setQuestions(qJson);
      } catch (err) {
        console.error(err);
        setError(err.message);
      }
    };

    fetchData();
  }, []);

  return (
    <div className="card">
      <button className="back-btn" onClick={onBack}>← Back</button>
      <h2>Feedback & Questions</h2>
      {error && <div className="error">{error}</div>}

      <section>
        <h3>Feedback</h3>
        {feedback ? (
          <div className="formatted-text">
            {feedback.stage3_feedback?.full_formatted_text || 'No text available'}
          </div>
        ) : (
          <p>Loading feedback…</p>
        )}
      </section>

      <section style={{ marginTop: '2rem' }}>
        <h3>Questions</h3>
        {questions ? (
          questions.questions?.questions ? (
            <ul className="question-list">
              {questions.questions.questions.map((q, idx) => (
                <li key={idx}>{q}</li>
              ))}
            </ul>
          ) : (
            <p>No questions found</p>
          )
        ) : (
          <p>Loading questions…</p>
        )}
      </section>
    </div>
  );
};

export default DataViewer;
