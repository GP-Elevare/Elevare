import React, { useState } from 'react';
import LandingPage from './components/LandingPage';
import UploadView from './components/UploadView';
import RecordView from './components/RecordView';
import PowerPointView from './components/PowerPointView';
import './App.css';

function App() {
  const [view, setView] = useState('landing');

  return (
    <div className="container">
      {view === 'landing' && <LandingPage setView={setView} />}
      {view === 'upload' && <UploadView onBack={() => setView('landing')} />}
      {view === 'record' && <RecordView onBack={() => setView('landing')} />}
      {view === 'ppt' && <PowerPointView onBack={() => setView('landing')} />}
    </div>
  );
}

export default App;