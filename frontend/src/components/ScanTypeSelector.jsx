import React from 'react'
import './ScanTypeSelector.css'

const ScanTypeSelector = ({ selectedType, onTypeChange }) => {
  const scanTypes = [
    {
      id: 'full_scan',
      title: '🔍 Full Security Scan',
      description: 'Complete analysis with Garak, PyRIT, and Research Model',
      features: ['Garak Configuration', 'PyRIT Configuration', 'Research Model Detection', 'CVSS 4.0 Scoring'],
      icon: '🛡️'
    },
    {
      id: 'log_analysis',
      title: '📊 Log Analysis',
      description: 'Analyze existing Garak and PyRIT log files',
      features: ['Garak JSONL Logs', 'PyRIT JSON Logs', 'Multi-turn Analysis', 'Attack Pattern Detection'],
      icon: '📁'
    },
    {
      id: 'fast_analysis',
      title: '⚡ Fast Analysis',
      description: 'Quick analysis using Research Model only',
      features: ['Research Model Only', 'Fast Processing', 'Basic Detection', 'Quick Results'],
      icon: '🚀'
    }
  ]

  return (
    <div className="scan-type-selector">
      <h3>🎯 Select Analysis Type</h3>
      <div className="scan-types-grid">
        {scanTypes.map(scanType => (
          <div
            key={scanType.id}
            className={`scan-type-card ${selectedType === scanType.id ? 'selected' : ''}`}
            onClick={() => onTypeChange(scanType.id)}
          >
            <div className="scan-type-header">
              <div className="scan-type-icon">{scanType.icon}</div>
              <div className="scan-type-info">
                <h4>{scanType.title}</h4>
                <p>{scanType.description}</p>
              </div>
            </div>
            <div className="scan-type-features">
              {scanType.features.map((feature, index) => (
                <span key={index} className="feature-tag">✓ {feature}</span>
              ))}
            </div>
            <div className="selection-indicator">
              {selectedType === scanType.id ? '✅ Selected' : '⬜ Select'}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default ScanTypeSelector