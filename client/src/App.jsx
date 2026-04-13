import React, { useState } from 'react';
import './App.css';

function App() {
  const [view, setView] = useState('landing');

  return (
    <div className="container">
      {view === 'landing' && <LandingPage setView={setView} />}
      {view === 'upload' && <UploadView onBack={() => setView('landing')} />}
      {view === 'record' && <RecordView onBack={() => setView('landing')} />}
      {view === 'ppt' && <PowerPointView onBack={() => setView('landing')} />}
      {view === 'viewer' && <DataViewer onBack={() => setView('landing')} />}
    </div>
  );
}

export default App;
