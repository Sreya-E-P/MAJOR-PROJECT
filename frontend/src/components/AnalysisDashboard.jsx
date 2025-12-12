import React, { useState, useEffect, useRef } from 'react'
import ScanTypeSelector from './ScanTypeSelector'
import DynamicForm from './DynamicForm'
import ConfigUpload from './ConfigUpload'
import RealTimeLog from './RealTimeLog'
import ToolResults from './ToolResults'
import { analyzeSecurity, getConfigs, healthCheck, detailedHealthCheck, getAnalysisStatus } from '../services/api'
import './AnalysisDashboard.css'

const AnalysisDashboard = ({ onAnalysisComplete, analysisResults }) => {
  // Scan type state
  const [scanType, setScanType] = useState('')
  
  // Form state
  const [prompt, setPrompt] = useState('')
  const [assistantResponse, setAssistantResponse] = useState('')
  const [garakConfigs, setGarakConfigs] = useState([])
  const [pyritConfigs, setPyritConfigs] = useState([])
  const [selectedGarakConfig, setSelectedGarakConfig] = useState('')
  const [selectedPyritConfig, setSelectedPyritConfig] = useState('')
  const [garakLogFile, setGarakLogFile] = useState(null)
  const [pyritLogFile, setPyritLogFile] = useState(null)
  
  // Analysis state
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [error, setError] = useState('')
  const [logs, setLogs] = useState([])
  const [backendStatus, setBackendStatus] = useState('checking')
  const [toolStatus, setToolStatus] = useState({
    garak: 'checking',
    pyrit: 'checking'
  })
  const [currentAnalysisId, setCurrentAnalysisId] = useState(null)
  const [downloadStatus, setDownloadStatus] = useState('')
  const ws = useRef(null)

  useEffect(() => {
    loadConfigs()
    checkBackendStatus()
    checkToolStatus()
    
    // Request notification permission
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission()
    }
  }, [])

  const checkBackendStatus = async () => {
    try {
      await healthCheck()
      setBackendStatus('healthy')
      setLogs(prev => [...prev, '✅ Backend service is healthy'])
    } catch (err) {
      setBackendStatus('unhealthy')
      setLogs(prev => [...prev, '❌ Backend service is unavailable'])
    }
  }

  const checkToolStatus = async () => {
    try {
      const healthData = await detailedHealthCheck()
      setToolStatus({
        garak: healthData.installations?.garak?.status || 'unknown',
        pyrit: healthData.installations?.pyrit?.status || 'unknown'
      })
      
      if (healthData.installations?.garak?.status === 'available') {
        setLogs(prev => [...prev, '✅ Garak is properly installed'])
      } else {
        setLogs(prev => [...prev, '⚠️ Garak is not available - using simulation'])
      }
      
      if (healthData.installations?.pyrit?.status === 'available') {
        setLogs(prev => [...prev, '✅ PyRIT is properly installed'])
      } else {
        setLogs(prev => [...prev, '⚠️ PyRIT is not available - using simulation'])
      }
    } catch (err) {
      setLogs(prev => [...prev, '⚠️ Could not check tool status'])
    }
  }

  const loadConfigs = async () => {
    try {
      const [garakResponse, pyritResponse] = await Promise.all([
        getConfigs('garak'),
        getConfigs('pyrit')
      ])
      
      setGarakConfigs(garakResponse.configs || [])
      setPyritConfigs(pyritResponse.configs || [])
      setLogs(prev => [...prev, '📁 Configurations loaded successfully'])
    } catch (err) {
      setLogs(prev => [...prev, `⚠️ Failed to load configurations: ${err.message}`])
    }
  }

  // 🚨 SIMPLE AUTO-DOWNLOAD - Just download existing PDF
  const triggerAutoDownload = (analysisId) => {
    console.log('🚀 Starting auto-download for existing PDF:', analysisId)
    
    // First show download status
    setDownloadStatus('Downloading report...')
    setLogs(prev => [...prev, '📥 Downloading PDF report...'])
    
    // Wait 1 second to ensure everything is ready
    setTimeout(() => {
      try {
        // Create a simple link to download the existing PDF
        const downloadUrl = `http://localhost:8000/download/report/${analysisId}`
        
        // Method 1: Create a temporary iframe (most reliable)
        const iframe = document.createElement('iframe')
        iframe.style.display = 'none'
        iframe.src = downloadUrl
        document.body.appendChild(iframe)
        
        console.log('✅ Auto-download triggered via iframe')
        setLogs(prev => [...prev, '✅ PDF download started! Check your Downloads folder.'])
        setDownloadStatus('Download complete! Check Downloads folder.')
        
        // Show browser notification if allowed
        if ('Notification' in window && Notification.permission === 'granted') {
          new Notification('LLM Security Report Ready', {
            body: `Report ${analysisId} downloaded to your Downloads folder.`,
            icon: '/favicon.ico'
          })
        }
        
        // Clean up after 3 seconds
        setTimeout(() => {
          if (document.body.contains(iframe)) {
            document.body.removeChild(iframe)
            console.log('🧹 Cleaned up iframe')
          }
        }, 3000)
        
      } catch (error) {
        console.error('Auto-download failed:', error)
        setLogs(prev => [...prev, `⚠️ Auto-download failed: ${error.message}`])
        setDownloadStatus('Download failed - Try manual download')
        
        // Fallback: Open in new tab
        setTimeout(() => {
          const downloadUrl = `http://localhost:8000/download/report/${analysisId}`
          window.open(downloadUrl, '_blank')
          setLogs(prev => [...prev, '📄 PDF opened in new tab as fallback'])
        }, 1000)
      }
    }, 1000)
  }

  const connectWebSocket = (analysisId) => {
    if (ws.current) {
      ws.current.close()
    }

    ws.current = new WebSocket(`ws://localhost:8000/ws/analysis/${analysisId}`)
    
    ws.current.onopen = () => {
      setLogs(prev => [...prev, '🔗 WebSocket connected for real-time updates'])
    }

    ws.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        setLogs(prev => [...prev, `📊 ${data.current_step} - Progress: ${data.progress}%`])
        
        if (data.status === 'completed' && data.results) {
          setLogs(prev => [...prev, '✅ Analysis completed successfully!'])
          onAnalysisComplete(data.results)
          setCurrentAnalysisId(data.results.analysis_id)
          setIsAnalyzing(false)
          
          // 🚨 AUTO-DOWNLOAD THE EXISTING PDF
          setLogs(prev => [...prev, '📥 Auto-downloading PDF report...'])
          triggerAutoDownload(data.results.analysis_id)
        }
      } catch (err) {
        console.error('WebSocket message error:', err)
      }
    }

    ws.current.onclose = () => {
      setLogs(prev => [...prev, '🔌 WebSocket disconnected'])
    }

    ws.current.onerror = (error) => {
      setLogs(prev => [...prev, '❌ WebSocket error occurred'])
      console.error('WebSocket error:', error)
    }
  }

  const handleAnalyze = async () => {
    if (!scanType) {
      setError('Please select an analysis type')
      return
    }

    if (backendStatus !== 'healthy') {
      setError('Backend service is unavailable. Please check if the server is running.')
      return
    }

    setIsAnalyzing(true)
    setError('')
    setCurrentAnalysisId(null)
    setDownloadStatus('')
    setLogs(prev => [...prev, `🚀 Starting ${scanType.replace('_', ' ')}...`])

    try {
      // Prepare form data based on scan type
      const formData = new FormData()
      formData.append('analysis_type', scanType)

      if (scanType === 'full_scan' || scanType === 'fast_analysis') {
        if (!prompt.trim()) {
          throw new Error('User prompt is required for this analysis type')
        }
        formData.append('user_prompt', prompt)
        
        if (scanType === 'fast_analysis' && !assistantResponse.trim()) {
          throw new Error('Assistant response is required for Fast Analysis')
        }
        formData.append('assistant_response', assistantResponse || '')
      }

      if (scanType === 'full_scan') {
        if (selectedGarakConfig) {
          formData.append('garak_config_id', selectedGarakConfig)
        }
        if (selectedPyritConfig) {
          formData.append('pyrit_config_id', selectedPyritConfig)
        }
      }

      if (scanType === 'log_analysis') {
        if (!garakLogFile && !pyritLogFile) {
          throw new Error('At least one log file (Garak or PyRIT) is required for Log Analysis')
        }
        if (garakLogFile) {
          formData.append('garak_log_file', garakLogFile)
        }
        if (pyritLogFile) {
          formData.append('pyrit_log_file', pyritLogFile)
        }
      }

      const response = await analyzeSecurity(formData)

      if (response.analysis_id) {
        setCurrentAnalysisId(response.analysis_id)
        connectWebSocket(response.analysis_id)
        setLogs(prev => [...prev, `📋 Analysis ID: ${response.analysis_id}`])
      }

      // Also poll for status as backup to WebSocket
      if (response.analysis_id) {
        pollAnalysisStatus(response.analysis_id)
      }

    } catch (err) {
      setError(err.message || 'Analysis request failed')
      setLogs(prev => [...prev, `💥 Request failed: ${err.message}`])
      setIsAnalyzing(false)
    }
  }

  const pollAnalysisStatus = async (analysisId) => {
    const maxAttempts = 60 // 5 minutes at 5-second intervals
    let attempts = 0
    
    const poll = async () => {
      if (attempts >= maxAttempts) {
        setLogs(prev => [...prev, '⏰ Analysis timeout - stopping status polling'])
        setIsAnalyzing(false)
        return
      }
      
      try {
        const status = await getAnalysisStatus(analysisId)
        
        if (status.status === 'completed' && status.results) {
          setLogs(prev => [...prev, '✅ Analysis completed via polling!'])
          onAnalysisComplete(status.results)
          setCurrentAnalysisId(status.results.analysis_id)
          setIsAnalyzing(false)
          
          // 🚨 AUTO-DOWNLOAD THE EXISTING PDF
          setLogs(prev => [...prev, '📥 Auto-downloading PDF report...'])
          triggerAutoDownload(status.results.analysis_id)
          return
        } else if (status.status === 'error') {
          setError(status.error || 'Analysis failed')
          setLogs(prev => [...prev, `❌ Analysis failed: ${status.error}`])
          setIsAnalyzing(false)
          return
        } else if (status.status === 'running') {
          setLogs(prev => [...prev, `📊 Polling: ${status.current_step} - ${status.progress}%`])
          attempts++
          setTimeout(poll, 5000)
        } else {
          attempts++
          setTimeout(poll, 5000)
        }
      } catch (err) {
        console.error('Polling error:', err)
        attempts++
        setTimeout(poll, 5000)
      }
    }
    
    poll()
  }

  // 🚨 SIMPLE MANUAL DOWNLOAD - Just download existing PDF
  const handleManualDownload = () => {
    if (!analysisResults?.analysis_id) {
      setError('No analysis results available for download')
      return
    }

    try {
      setLogs(prev => [...prev, '📥 Downloading PDF report to your computer...'])
      setDownloadStatus('Starting download...')
      
      const analysisId = analysisResults.analysis_id
      const downloadUrl = `http://localhost:8000/download/report/${analysisId}`
      
      // Create a link and click it
      const link = document.createElement('a')
      link.href = downloadUrl
      link.download = `security_report_${analysisId}.pdf`
      link.style.display = 'none'
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      
      setLogs(prev => [...prev, '✅ PDF download started! Check your Downloads folder.'])
      setDownloadStatus('Download started! Check Downloads folder.')
      
    } catch (err) {
      setError(`PDF download failed: ${err.message}`)
      setLogs(prev => [...prev, `❌ Download failed: ${err.message}`])
      setDownloadStatus('Download failed')
    }
  }

  const handleConfigsUpdated = () => {
    loadConfigs()
  }

  const getToolStatusBadge = (tool) => {
    const status = toolStatus[tool]
    const statusConfig = {
      available: { emoji: '✅', text: 'Available' },
      unavailable: { emoji: '⚠️', text: 'Unavailable' },
      checking: { emoji: '🔄', text: 'Checking' },
      error: { emoji: '❌', text: 'Error' }
    }
    
    const config = statusConfig[status] || { emoji: '❓', text: 'Unknown' }
    return `${config.emoji} ${config.text}`
  }

  const resetForm = () => {
    setPrompt('')
    setAssistantResponse('')
    setSelectedGarakConfig('')
    setSelectedPyritConfig('')
    setGarakLogFile(null)
    setPyritLogFile(null)
    setError('')
    setDownloadStatus('')
  }

  return (
    <div className="analysis-dashboard">
      <div className="dashboard-header">
        <h2>🔍 LLM Security Analysis Dashboard</h2>
        <div className="status-container">
          <div className={`status-indicator ${backendStatus}`}>
            Backend: {backendStatus === 'healthy' ? '🟢 Healthy' : '🔴 Unhealthy'}
          </div>
          <div className="tool-status">
            <span>Garak: {getToolStatusBadge('garak')}</span>
            <span>PyRIT: {getToolStatusBadge('pyrit')}</span>
          </div>
        </div>
      </div>

      <div className="dashboard-content">
        {/* Download Status Notification */}
        {downloadStatus && (
          <div className="download-notification">
            <div className="download-icon">📥</div>
            <div className="download-message">
              <strong>{downloadStatus}</strong>
              <p>The PDF report will appear in your Downloads folder</p>
            </div>
          </div>
        )}

        {/* Scan Type Selection */}
        <ScanTypeSelector 
          selectedType={scanType}
          onTypeChange={(type) => {
            setScanType(type)
            resetForm()
          }}
        />

        {/* Dynamic Form based on Scan Type */}
        {scanType && (
          <DynamicForm
            scanType={scanType}
            prompt={prompt}
            onPromptChange={setPrompt}
            assistantResponse={assistantResponse}
            onAssistantResponseChange={setAssistantResponse}
            garakConfigs={garakConfigs}
            selectedGarakConfig={selectedGarakConfig}
            onGarakConfigChange={setSelectedGarakConfig}
            pyritConfigs={pyritConfigs}
            selectedPyritConfig={selectedPyritConfig}
            onPyritConfigChange={setSelectedPyritConfig}
            garakLogFile={garakLogFile}
            onGarakLogChange={setGarakLogFile}
            pyritLogFile={pyritLogFile}
            onPyritLogChange={setPyritLogFile}
            isAnalyzing={isAnalyzing}
          />
        )}

        {/* Configuration Manager - Only show for Full Scan */}
        {scanType === 'full_scan' && (
          <div className="config-section">
            <ConfigUpload onConfigsUpdated={handleConfigsUpdated} />
          </div>
        )}

        {/* Analysis Controls */}
        {scanType && (
          <div className="analysis-controls-section">
            <div className="analysis-controls">
              <button 
                onClick={handleAnalyze}
                disabled={isAnalyzing || backendStatus !== 'healthy'}
                className={`analyze-btn ${isAnalyzing ? 'analyzing' : ''}`}
              >
                {isAnalyzing ? (
                  <>
                    <div className="spinner"></div>
                    Analyzing...
                  </>
                ) : (
                  `🚀 Start ${scanType.replace('_', ' ').toUpperCase()}`
                )}
              </button>

              {error && (
                <div className="error-message">
                  ❌ {error}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Results Section */}
        {analysisResults && (
          <div className="results-section">
            <div className="results-header">
              <h3>📊 Analysis Results</h3>
              <div className="results-actions">
                <button 
                  onClick={handleManualDownload}
                  className="download-btn"
                >
                  📥 Download PDF Report Again
                </button>
                {currentAnalysisId && (
                  <div className="analysis-id">
                    ID: {currentAnalysisId}
                  </div>
                )}
              </div>
            </div>

            {/* Tool Results */}
            <ToolResults results={analysisResults} />

            {/* Executive Summary */}
            <div className="executive-summary">
              <h4>📋 Executive Summary</h4>
              <div className={`risk-level ${analysisResults.risk_level?.toLowerCase() || 'unknown'}`}>
                Overall Risk: {analysisResults.risk_level || 'UNKNOWN'}
              </div>
              
              <div className="summary-grid">
                <div className="summary-item">
                  <span className="label">Analysis ID:</span>
                  <span className="value">{analysisResults.analysis_id}</span>
                </div>
                <div className="summary-item">
                  <span className="label">Timestamp:</span>
                  <span className="value">{new Date(analysisResults.timestamp).toLocaleString()}</span>
                </div>
                <div className="summary-item">
                  <span className="label">PDF Ready:</span>
                  <span className="value">
                    {analysisResults.report_generated ? '✅ Yes' : '❌ No'}
                  </span>
                </div>
                <div className="summary-item">
                  <span className="label">Scan Type:</span>
                  <span className="value">{analysisResults.analysis_type}</span>
                </div>
                <div className="summary-item">
                  <span className="label">Report:</span>
                  <span className="value">
                    <a 
                      href={`http://localhost:8000/download/report/${analysisResults.analysis_id}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="report-link"
                      onClick={(e) => {
                        e.preventDefault()
                        handleManualDownload()
                      }}
                    >
                      📄 Click to Download Report
                    </a>
                  </span>
                </div>
              </div>
            </div>

            {/* Recommendations */}
            {analysisResults.recommendations && analysisResults.recommendations.length > 0 && (
              <div className="recommendations">
                <h4>💡 Security Recommendations</h4>
                <div className="recommendations-list">
                  {analysisResults.recommendations.map((rec, index) => (
                    <div key={index} className="recommendation-item">
                      <span className="rec-number">{index + 1}.</span>
                      <span className="rec-text">{rec}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Real-time Logs */}
        <RealTimeLog logs={logs} />
      </div>
    </div>
  )
}

export default AnalysisDashboard