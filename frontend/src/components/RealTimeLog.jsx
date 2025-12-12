import React, { useEffect, useRef } from 'react'
import './RealTimeLog.css'

const RealTimeLog = ({ logs }) => {
  const logEndRef = useRef(null)
  const logContainerRef = useRef(null)

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ 
      behavior: 'smooth',
      block: 'nearest'
    })
  }, [logs])

  const getLogIcon = (log) => {
    if (log.includes('✅') || log.includes('successfully')) return '✅'
    if (log.includes('❌') || log.includes('failed') || log.includes('error')) return '❌'
    if (log.includes('⚠️') || log.includes('warning')) return '⚠️'
    if (log.includes('🚀')) return '🚀'
    if (log.includes('📊')) return '📊'
    if (log.includes('🔗')) return '🔗'
    if (log.includes('🔌')) return '🔌'
    if (log.includes('📁')) return '📁'
    if (log.includes('💥')) return '💥'
    if (log.includes('🔍')) return '🔍'
    return '📝'
  }

  const getLogType = (log) => {
    if (log.includes('✅') || log.includes('successfully')) return 'success'
    if (log.includes('❌') || log.includes('failed') || log.includes('error') || log.includes('💥')) return 'error'
    if (log.includes('⚠️') || log.includes('warning')) return 'warning'
    if (log.includes('🚀')) return 'start'
    if (log.includes('📊') || log.includes('Progress')) return 'progress'
    if (log.includes('🔗') || log.includes('WebSocket')) return 'connection'
    return 'info'
  }

  return (
    <div className="real-time-log">
      <div className="log-header">
        <h4>📋 Analysis Log</h4>
        <div className="log-stats">
          <span className="log-count">{logs.length} entries</span>
          <button 
            className="clear-logs"
            onClick={() => window.location.reload()}
            title="Refresh to clear logs"
          >
            🔄 Refresh
          </button>
        </div>
      </div>

      <div className="log-container" ref={logContainerRef}>
        {logs.length === 0 ? (
          <div className="no-logs">
            <p>No logs yet. Start an analysis to see real-time updates.</p>
          </div>
        ) : (
          logs.map((log, index) => {
            const icon = getLogIcon(log)
            const type = getLogType(log)
            const timestamp = new Date().toLocaleTimeString()
            
            return (
              <div key={index} className={`log-entry ${type}`}>
                <div className="log-timestamp">
                  {timestamp}
                </div>
                <div className="log-icon">
                  {icon}
                </div>
                <div className="log-message">
                  {log.replace(/[✅❌⚠️🚀📊🔗🔌📁💥🔍]/g, '').trim()}
                </div>
              </div>
            )
          })
        )}
        <div ref={logEndRef} />
      </div>

      <div className="log-footer">
        <div className="log-legend">
          <div className="legend-item">
            <span className="legend-icon">✅</span>
            <span>Success</span>
          </div>
          <div className="legend-item">
            <span className="legend-icon">⚠️</span>
            <span>Warning</span>
          </div>
          <div className="legend-item">
            <span className="legend-icon">❌</span>
            <span>Error</span>
          </div>
          <div className="legend-item">
            <span className="legend-icon">🚀</span>
            <span>Progress</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default RealTimeLog