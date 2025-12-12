import React, { useState } from 'react'
import './ToolResults.css'

const ToolResults = ({ results }) => {
  const [expandedTool, setExpandedTool] = useState(null)

  const toggleTool = (toolName) => {
    setExpandedTool(expandedTool === toolName ? null : toolName)
  }

  const renderGarakResults = (garakResults) => {
    if (!garakResults) return null

    return (
      <div className="tool-details">
        <div className="tool-status">
          <strong>Status:</strong> 
          <span className={`status ${garakResults.status}`}>
            {garakResults.status}
          </span>
          {garakResults.execution_time && (
            <span className="execution-time">
              (Time: {garakResults.execution_time.toFixed(2)}s)
            </span>
          )}
        </div>

        {garakResults.vulnerabilities && garakResults.vulnerabilities.length > 0 && (
          <div className="results-content">
            <div className="vulnerabilities">
              <h6>🛡️ Vulnerabilities Detected:</h6>
              <div className="vuln-count">
                {garakResults.vulnerability_count || garakResults.vulnerabilities.length} found
              </div>
              {garakResults.vulnerabilities.slice(0, 5).map((vuln, index) => (
                <div key={index} className="vulnerability-item">
                  <span className="vuln-type">{vuln.type}</span>
                  <span className={`vuln-severity ${vuln.severity}`}>{vuln.severity}</span>
                  <span className="vuln-confidence">{(vuln.confidence * 100).toFixed(1)}%</span>
                </div>
              ))}
              {garakResults.vulnerabilities.length > 5 && (
                <div className="more-vulns">
                  +{garakResults.vulnerabilities.length - 5} more vulnerabilities
                </div>
              )}
            </div>
          </div>
        )}

        {garakResults.vulnerability_count === 0 && (
          <div className="no-vulnerabilities">
            ✅ No vulnerabilities detected by Garak
          </div>
        )}

        {garakResults.error && (
          <div className="tool-error">
            <strong>Error:</strong> {garakResults.error}
          </div>
        )}
      </div>
    )
  }

  const renderPyRITResults = (pyritResults) => {
    if (!pyritResults) return null

    return (
      <div className="tool-details">
        <div className="tool-status">
          <strong>Status:</strong> 
          <span className={`status ${pyritResults.status}`}>
            {pyritResults.status}
          </span>
          {pyritResults.execution_time && (
            <span className="execution-time">
              (Time: {pyritResults.execution_time.toFixed(2)}s)
            </span>
          )}
        </div>

        {pyritResults.results && (
          <div className="results-content">
            <div className="conversation-stats">
              <div className="stat-item">
                <span className="stat-label">Attack Successful:</span>
                <span className="stat-value">
                  {pyritResults.attack_successful ? '✅ Yes' : '❌ No'}
                </span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Vulnerabilities:</span>
                <span className="stat-value">{pyritResults.vulnerability_count || 0}</span>
              </div>
            </div>

            {pyritResults.vulnerabilities && pyritResults.vulnerabilities.length > 0 && (
              <div className="vulnerabilities">
                <h6>⚡ PyRIT Vulnerabilities:</h6>
                {pyritResults.vulnerabilities.map((vuln, index) => (
                  <div key={index} className="vulnerability-item">
                    <span className="vuln-type">{vuln.type}</span>
                    <span className={`vuln-severity ${vuln.severity}`}>{vuln.severity}</span>
                    <span className="vuln-source">Source: {vuln.source}</span>
                  </div>
                ))}
              </div>
            )}

            {pyritResults.attack_successful && (
              <div className="attack-success">
                <div className="warning-icon">⚠️</div>
                <div className="warning-content">
                  <strong>Jailbreak Successful</strong>
                  <p>PyRIT successfully bypassed the model's safety measures.</p>
                </div>
              </div>
            )}
          </div>
        )}

        {pyritResults.vulnerability_count === 0 && !pyritResults.attack_successful && (
          <div className="no-vulnerabilities">
            ✅ No successful attacks detected by PyRIT
          </div>
        )}

        {pyritResults.error && (
          <div className="tool-error">
            <strong>Error:</strong> {pyritResults.error}
          </div>
        )}
      </div>
    )
  }

  const renderEnsembleResults = (ensembleResults) => {
    if (!ensembleResults) return null

    return (
      <div className="tool-details">
        <div className="tool-status">
          <strong>Status:</strong> 
          <span className="status completed">Completed</span>
        </div>

        <div className="results-content">
          <div className="ensemble-metrics">
            <div className="metric-item">
              <span className="metric-label">Total Vulnerabilities:</span>
              <span className="metric-value">{ensembleResults.total_vulnerabilities || 0}</span>
            </div>
            <div className="metric-item">
              <span className="metric-label">Combined Risk:</span>
              <span className={`metric-value risk-${ensembleResults.risk_summary?.overall_risk}`}>
                {ensembleResults.risk_summary?.overall_risk || 'unknown'}
              </span>
            </div>
            <div className="metric-item">
              <span className="metric-label">Ensemble Confidence:</span>
              <span className="metric-value">
                {((ensembleResults.ensemble_detection?.confidence || 0) * 100).toFixed(1)}%
              </span>
            </div>
          </div>

          {ensembleResults.combined_vulnerabilities && ensembleResults.combined_vulnerabilities.length > 0 && (
            <div className="combined-vulnerabilities">
              <h6>🔗 Combined Vulnerabilities:</h6>
              <div className="vulnerabilities-grid">
                {ensembleResults.combined_vulnerabilities.slice(0, 8).map((vuln, index) => (
                  <div key={index} className="combined-vuln-item">
                    <span className="vuln-source">{vuln.source}</span>
                    <span className="vuln-type">{vuln.type}</span>
                    <span className={`vuln-severity ${vuln.severity}`}>{vuln.severity}</span>
                  </div>
                ))}
              </div>
              {ensembleResults.combined_vulnerabilities.length > 8 && (
                <div className="more-vulns">
                  +{ensembleResults.combined_vulnerabilities.length - 8} more vulnerabilities
                </div>
              )}
            </div>
          )}

          {ensembleResults.mitre_attack_matrix && (
            <div className="mitre-coverage">
              <h6>🎯 MITRE ATT&CK Coverage:</h6>
              <div className="coverage-metric">
                <span className="coverage-label">Coverage Score:</span>
                <span className="coverage-value">
                  {ensembleResults.mitre_attack_matrix.coverage_score?.toFixed(1)}%
                </span>
              </div>
              <div className="tactics-list">
                {Object.keys(ensembleResults.mitre_attack_matrix.tactics || {}).map(tactic => (
                  <span key={tactic} className="tactic-tag">{tactic}</span>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    )
  }

  const tools = [
    {
      name: 'garak',
      title: '🛡️ NVIDIA Garak Analysis',
      icon: '🛡️',
      description: 'Comprehensive prompt injection and jailbreak detection',
      render: () => renderGarakResults(results.garak_results)
    },
    {
      name: 'pyrit',
      title: '⚡ Microsoft PyRIT Analysis',
      icon: '⚡',
      description: 'Multi-turn conversation analysis and escalation detection',
      render: () => renderPyRITResults(results.pyrit_results)
    },
    {
      name: 'ensemble',
      title: '🤖 Research Ensemble Analysis',
      icon: '🤖',
      description: 'Advanced ensemble detection with CVSS 4.0 scoring',
      render: () => renderEnsembleResults(results.ensemble_analysis)
    }
  ]

  return (
    <div className="tool-results">
      <h4>🛠️ Tool Analysis Results</h4>
      <div className="tools-grid">
        {tools.map(tool => {
          const hasResults = 
            (tool.name === 'garak' && results.garak_results) ||
            (tool.name === 'pyrit' && results.pyrit_results) ||
            (tool.name === 'ensemble' && results.ensemble_analysis)
          
          return (
            <div key={tool.name} className={`tool-card ${hasResults ? 'has-results' : 'no-results'}`}>
              <div 
                className="tool-header"
                onClick={() => hasResults && toggleTool(tool.name)}
              >
                <div className="tool-info">
                  <div className="tool-icon">{tool.icon}</div>
                  <div className="tool-text">
                    <h5>{tool.title}</h5>
                    <p>{tool.description}</p>
                  </div>
                </div>
                <div className="tool-toggle">
                  {hasResults ? (
                    expandedTool === tool.name ? '▼' : '►'
                  ) : (
                    <span className="no-data">No Data</span>
                  )}
                </div>
              </div>
              
              {hasResults && expandedTool === tool.name && tool.render()}
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default ToolResults