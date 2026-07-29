import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import './index.css'
import Home from './Home'
import Feedback from './Feedback'
import Questions from './Questions'
import QuestionsFeedback from './QuestionsFeedback';
import Signup from './Signup';
import Signin from './Signin';
import { AuthProvider } from './AuthContext';

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/feedback" element={<Feedback />} />
          <Route path="/questions" element={<Questions />} />
          <Route path="/questions-feedback" element={<QuestionsFeedback />} />
          <Route path="/signup" element={<Signup/>} />
          <Route path="/signin" element={<Signin/>} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  </StrictMode>,
)