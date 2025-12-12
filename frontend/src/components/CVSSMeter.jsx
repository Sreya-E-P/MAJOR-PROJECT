import React from 'react'
import './CVSSMeter.css'

const CVSSMeter = ({ score, cvssAssessment }) => {
  const getSeverity = (score) => {
    if (score >= 9.0) return 'critical'
    if (score >= 7.0) return 'high'
    if (score >= 4.0) return 'medium'
    if (score >= 0.1) return 'low'
    return 'none'
  }

  const severity = getSeverity(score)
  const percentage = Math.min((score / 10) * 100, 100)

  const severityInfo = {
    none: { color: '#28a745', description: 'No security risk detected' },
    low: { color: '#ffc107', description: 'Low security risk' },
    medium: { color: '#fd7e14', description: 'Medium security risk' },
    high: { color: '#dc3545', description: 'High security risk' },
    critical: { color: '#721c24', description: 'Critical security risk' }
  }

  const getLLMRiskScore = () => {
    return cvssAssessment?.llm_supplemental_metrics?.llm_risk_score || 0
  }

  const getOverallSeverity = () => {
    return cvssAssessment?.overall_assessment?.severity || 'NONE'
  }

  const getRemediationPriority = () => {
    return cvssAssessment?.overall_assessment?.priority || 'LOW'
  }

  return (
    <div className="cvss-meter">
      <div className="cvss-header">
        <h4>CVSS 4.0 + LLM Risk Assessment</h4>
        <div className="score-container">
          <span className={`score ${severity}`}>{score.toFixed(1)}</span>
          <span className="score-label">CVSS 4.0</span>
        </div>
      </div>
      
      <div className="meter-container">
        <div className="meter-background">
          <div 
            className={`meter-fill ${severity}`}
            style={{ width: `${percentage}%` }}
          ></div>
        </div>
        <div className="severity-labels">
          <span>None</span>
          <span>Low</span>
          <span>Medium</span>
          <span>High</span>
          <span>Critical</span>
        </div>
      </div>

      {/* LLM Risk Score */}
      <div className="llm-risk-section">
        <div className="llm-risk-score">
          <span className="llm-label">LLM Risk Score:</span>
          <span className={`llm-value ${getSeverity(getLLMRiskScore())}`}>
            {getLLMRiskScore().toFixed(1)}
          </span>
        </div>
      </div>
      
      <div className="severity-info">
        <div className="severity-level">
          Overall Severity: <span className={getOverallSeverity().toLowerCase()}>{getOverallSeverity()}</span>
        </div>
        <div className="remediation-priority">
          Remediation Priority: <span className={`priority-${getRemediationPriority().toLowerCase()}`}>
            {getRemediationPriority()}
          </span>
        </div>
        <div className="severity-description">
          {severityInfo[severity].description}
        </div>
      </div>

      {cvssAssessment && (
        <div className="cvss-details">
          <div className="detail-section">
            <h5>CVSS 4.0 Metrics</h5>
            <div className="metrics-grid">
              {Object.entries(cvssAssessment.cvss_4_0?.metrics || {}).map(([metric, value]) => (
                <div key={metric} className="metric-item">
                  <span className="metric-name">{metric}</span>
                  <span className="metric-value">{value}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="detail-section">
            <h5>LLM Supplemental Metrics</h5>
            <div className="llm-metrics">
              {cvssAssessment.llm_supplemental_metrics && (
                <>
                  <div className="llm-metric">
                    <span className="metric-label">Safety Impact:</span>
                    <span className="metric-value">
                      {cvssAssessment.llm_supplemental_metrics.safety_impact?.level} - 
                      {cvssAssessment.llm_supplemental_metrics.safety_impact?.description}
                    </span>
                  </div>
                  <div className="llm-metric">
                    <span className="metric-label">Automation Potential:</span>
                    <span className="metric-value">
                      {cvssAssessment.llm_supplemental_metrics.automation_potential?.level} - 
                      {cvssAssessment.llm_supplemental_metrics.automation_potential?.description}
                    </span>
                  </div>
                  <div className="llm-metric">
                    <span className="metric-label">Value Density:</span>
                    <span className="metric-value">
                      {cvssAssessment.llm_supplemental_metrics.value_density?.level} - 
                      {cvssAssessment.llm_supplemental_metrics.value_density?.description}
                    </span>
                  </div>
                </>
              )}
            </div>
          </div>

          {cvssAssessment.scoring_breakdown && (
            <div className="scoring-breakdown">
              <h5>Scoring Formula</h5>
              <div className="formula">
                {cvssAssessment.scoring_breakdown.calculation}
              </div>
              <div className="breakdown-details">
                Raw Product: {cvssAssessment.scoring_breakdown.raw_product}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default CVSSMeter