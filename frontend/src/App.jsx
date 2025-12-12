import React, { useState } from 'react'
import AnalysisDashboard from './components/AnalysisDashboard'
import ConfigUpload from './components/ConfigUpload'
import PDFReportViewer from './components/PDFReportViewer'
import './App.css'

function App() {
  const [currentView, setCurrentView] = useState('dashboard')
  const [analysisResults, setAnalysisResults] = useState(null)

  return (
    <div className="App">
      <header className="app-header">
        <div className="header-content">
          <h1>🛡️ LLM Security Framework</h1>
          <p>M.Tech Project - Advanced Ensemble Detection with CVSS 4.0 & MITRE ATT&CK</p>
        </div>
        <nav className="nav-tabs">
          <button 
            className={`nav-tab ${currentView === 'dashboard' ? 'active' : ''}`}
            onClick={() => setCurrentView('dashboard')}
          >
            🏠 Security Analysis
          </button>
          <button 
            className={`nav-tab ${currentView === 'config' ? 'active' : ''}`}
            onClick={() => setCurrentView('config')}
          >
            ⚙️ Configuration Manager
          </button>
          <button 
            className={`nav-tab ${currentView === 'reports' ? 'active' : ''}`}
            onClick={() => setCurrentView('reports')}
            disabled={!analysisResults}
          >
            📊 Reports
          </button>
        </nav>
      </header>

      <main className="app-main">
        {currentView === 'dashboard' && (
          <AnalysisDashboard 
            onAnalysisComplete={setAnalysisResults}
            analysisResults={analysisResults}
          />
        )}
        {currentView === 'config' && <ConfigUpload />}
        {currentView === 'reports' && analysisResults && (
          <PDFReportViewer analysisResults={analysisResults} />
        )}
      </main>

      <footer className="app-footer">
        <p>© 2025 LLM Security Framework - M.Tech Project | 
           Advanced Ensemble Detection + CVSS 4.0 + MITRE ATT&CK Integration</p>
      </footer>
    </div>
  )
}

export default App