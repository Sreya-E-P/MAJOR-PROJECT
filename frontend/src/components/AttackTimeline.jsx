import React from 'react'
import './AttackTimeline.css'

const AttackTimeline = ({ analysis }) => {
  if (!analysis.turn_analysis || analysis.turn_analysis.length === 0) {
    return (
      <div className="no-analysis">
        <p>No turn analysis available</p>
      </div>
    )
  }

  const getRiskIcon = (riskLevel) => {
    switch (riskLevel.toLowerCase()) {
      case 'critical': return '🔴'
      case 'high': return '🟠'
      case 'medium': return '🟡'
      case 'low': return '🟢'
      default: return '⚪'
    }
  }

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return '#dc3545'
    if (confidence >= 0.6) return '#fd7e14'
    if (confidence >= 0.4) return '#ffc107'
    return '#28a745'
  }

  return (
    <div className="attack-timeline">
      <div className="timeline-header">
        <h5>🔄 Multi-Turn Attack Timeline</h5>
        <div className="timeline-stats">
          <span className="stat">Turns: {analysis.turn_analysis.length}</span>
          <span className="stat">Escalation: {analysis.escalation_detected ? '✅ Detected' : '❌ Not Detected'}</span>
        </div>
      </div>

      <div className="timeline">
        {analysis.turn_analysis.map((turn, index) => (
          <div key={index} className={`timeline-item ${turn.risk_level.toLowerCase()}`}>
            <div className="timeline-marker">
              {getRiskIcon(turn.risk_level)}
            </div>
            
            <div className="timeline-content">
              <div className="turn-header">
                <div className="turn-info">
                  <span className="turn-number">Turn {turn.turn + 1}</span>
                  <span className={`risk-badge ${turn.risk_level.toLowerCase()}`}>
                    {turn.risk_level}
                  </span>
                </div>
                <div className="turn-metrics">
                  <span 
                    className="confidence" 
                    style={{ color: getConfidenceColor(turn.confidence) }}
                  >
                    Confidence: {(turn.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
              
              <div className="turn-text">
                {turn.text}
              </div>
              
              <div className="turn-prediction">
                <strong>Detected:</strong> 
                <span className={`prediction-type ${turn.predicted_class}`}>
                  {turn.predicted_class.replace('_', ' ').toUpperCase()}
                </span>
              </div>

              {turn.all_probabilities && Object.keys(turn.all_probabilities).length > 1 && (
                <div className="probability-breakdown">
                  <div className="probabilities-title">Probability Distribution:</div>
                  <div className="probabilities">
                    {Object.entries(turn.all_probabilities)
                      .sort(([,a], [,b]) => b - a)
                      .slice(0, 3)
                      .map(([category, prob]) => (
                        <div key={category} className="probability-item">
                          <span className="prob-category">{category.replace('_', ' ')}</span>
                          <span className="prob-value">{(prob * 100).toFixed(1)}%</span>
                        </div>
                      ))
                    }
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
      
      {analysis.contextual_patterns && analysis.contextual_patterns.length > 0 && (
        <div className="contextual-patterns">
          <h6>🎯 Contextual Patterns Detected</h6>
          <div className="patterns-grid">
            {analysis.contextual_patterns.map((pattern, index) => (
              <div key={index} className="pattern-card">
                <div className="pattern-header">
                  <strong>{pattern.pattern}</strong>
                  <span className={`pattern-severity ${pattern.severity.toLowerCase()}`}>
                    {pattern.severity}
                  </span>
                </div>
                <div className="pattern-description">
                  {pattern.description}
                </div>
                <div className="pattern-details">
                  <span>Turns: {pattern.turns.join(', ')}</span>
                  <span>Confidence: {(pattern.confidence * 100).toFixed(1)}%</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {analysis.escalation_detected && (
        <div className="escalation-warning">
          <div className="warning-icon">⚠️</div>
          <div className="warning-content">
            <strong>Multi-turn Escalation Detected</strong>
            <p>The attack pattern shows increasing severity across multiple conversation turns, indicating sophisticated attack behavior.</p>
          </div>
        </div>
      )}

      {analysis.scientific_metrics && (
        <div className="scientific-metrics">
          <h6>📈 Detection Metrics</h6>
          <div className="metrics-grid">
            {Object.entries(analysis.scientific_metrics).map(([metric, value]) => (
              <div key={metric} className="metric-item">
                <span className="metric-name">
                  {metric.replace(/_/g, ' ').toUpperCase()}
                </span>
                <span className="metric-value">
                  {typeof value === 'number' ? value.toFixed(3) : value}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default AttackTimeline