import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import './index.css'
import Home from './Home'
import Feedback from './Feedback'
import Questions from './Questions'
import QuestionsFeedback from './QuestionsFeedback';
createRoot(document.getElementById('root')).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/feedback" element={<Feedback />} />
        <Route path="/questions" element={<Questions />} />
        <Route path="/questions-feedback" element={<QuestionsFeedback />} />
      </Routes>
    </BrowserRouter>
  </StrictMode>,
)
