import React from 'react'
import './MITREMatrix.css'

const MITREMatrix = ({ mitreData, detectionResult }) => {
  if (!mitreData && !detectionResult?.mitre_attack) {
    return (
      <div className="mitre-matrix">
        <div className="no-mitre-data">
          <h4>🎯 MITRE ATT&CK Framework</h4>
          <p>No MITRE ATT&CK mapping available for this analysis.</p>
        </div>
      </div>
    )
  }

  const mitreInfo = mitreData || detectionResult?.mitre_attack || {}
  const matrixData = detectionResult?.mitre_attack_matrix || {}

  const getTacticColor = (tactic) => {
    const colors = {
      'Reconnaissance': '#ff6b6b',
      'Resource Development': '#ffa726',
      'Initial Access': '#ffca28',
      'Execution': '#66bb6a',
      'Persistence': '#26a69a',
      'Privilege Escalation': '#26c6da',
      'Defense Evasion': '#29b6f6',
      'Credential Access': '#5c6bc0',
      'Discovery': '#7e57c2',
      'Lateral Movement': '#ab47bc',
      'Collection': '#ec407a',
      'Exfiltration': '#ef5350',
      'Impact': '#78909c'
    }
    return colors[tactic] || '#9e9e9e'
  }

  return (
    <div className="mitre-matrix">
      <h4>🎯 MITRE ATT&CK Framework Mapping</h4>
      
      {/* Techniques Table */}
      {mitreInfo.detailed_techniques && mitreInfo.detailed_techniques.length > 0 && (
        <div className="techniques-section">
          <h5>🛡️ Associated Techniques</h5>
          <div className="techniques-grid">
            {mitreInfo.detailed_techniques.map((technique, index) => (
              <div key={index} className="technique-card">
                <div className="technique-header">
                  <span className="technique-id">{technique.id}</span>
                  <span 
                    className="tactic-badge"
                    style={{ backgroundColor: getTacticColor(technique.tactic) }}
                  >
                    {technique.tactic}
                  </span>
                </div>
                <div className="technique-name">{technique.name}</div>
                <div className="technique-description">{technique.description}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Tactics Overview */}
      {matrixData.tactics && Object.keys(matrixData.tactics).length > 0 && (
        <div className="tactics-section">
          <h5>🎯 Tactical Analysis</h5>
          <div className="tactics-grid">
            {Object.entries(matrixData.tactics).map(([tactic, attacks]) => (
              <div key={tactic} className="tactic-card">
                <div 
                  className="tactic-header"
                  style={{ backgroundColor: getTacticColor(tactic) }}
                >
                  {tactic}
                </div>
                <div className="tactic-content">
                  {attacks.map((attack, idx) => (
                    <div key={idx} className="attack-item">
                      <span className="attack-type">{attack.attack}</span>
                      <div className="technique-list">
                        {attack.techniques.map((tech, techIdx) => (
                          <span key={techIdx} className="technique-tag">{tech}</span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Coverage Score */}
      {matrixData.coverage_score !== undefined && (
        <div className="coverage-section">
          <h5>📊 MITRE ATT&CK Coverage</h5>
          <div className="coverage-score">
            <div className="score-circle">
              <span className="score-value">{matrixData.coverage_score.toFixed(1)}%</span>
            </div>
            <div className="coverage-details">
              <div className="detail-item">
                <span className="label">Unique Techniques:</span>
                <span className="value">{matrixData.techniques?.length || 0}</span>
              </div>
              <div className="detail-item">
                <span className="label">Tactics Covered:</span>
                <span className="value">{Object.keys(matrixData.tactics || {}).length}</span>
              </div>
              <div className="detail-item">
                <span className="label">Recommended Mitigations:</span>
                <span className="value">{matrixData.mitigations?.length || 0}</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Mitigations */}
      {mitreInfo.mitigations && mitreInfo.mitigations.length > 0 && (
        <div className="mitigations-section">
          <h5>🛡️ Recommended MITRE Mitigations</h5>
          <div className="mitigations-list">
            {mitreInfo.mitigations.map((mitigation, index) => (
              <div key={index} className="mitigation-item">
                <span className="mitigation-id">M{mitigation}</span>
                <span className="mitigation-text">
                  {getMitigationDescription(mitigation)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Description */}
      {mitreInfo.description && (
        <div className="description-section">
          <h5>📝 Analysis Description</h5>
          <p className="mitre-description">{mitreInfo.description}</p>
        </div>
      )}
    </div>
  )
}

// Helper function to get mitigation descriptions
const getMitigationDescription = (mitigationId) => {
  const descriptions = {
    'M1015': 'Active Defense: Engage with attackers to gather intelligence',
    'M1016': 'Vulnerability Scanning: Regularly scan for vulnerabilities',
    'M1017': 'User Training: Train users on security awareness',
    'M1026': 'Privileged Account Management: Manage privileged accounts',
    'M1030': 'Network Segmentation: Segment network to limit access',
    'M1032': 'Multi-factor Authentication: Implement MFA for all accounts',
    'M1035': 'Limit Access: Restrict access based on need-to-know',
    'M1036': 'Account Use Policies: Enforce account usage policies',
    'M1037': 'Filter Management: Manage email and web filtering',
    'M1038': 'Execution Prevention: Prevent unauthorized code execution',
    'M1041': 'Encrypt Sensitive Information: Encrypt data at rest and in transit',
    'M1045': 'Filter Network Traffic: Monitor and filter network traffic',
    'M1048': 'Application Isolation: Isolate applications from system',
    'M1049': 'Antivirus/Antimalware: Use antivirus and antimalware solutions',
    'M1050': 'Application Isolation: Isolate untrusted applications',
    'M1051': 'Update Software: Keep all software updated',
    'M1052': 'User Account Control: Implement user account controls',
    'M1053': 'Software Configuration: Harden software configurations',
    'M1056': 'Pre-compromise: Implement pre-compromise protections',
    'M1057': 'DoS Protection: Protect against denial of service attacks'
  }
  return descriptions[mitigationId] || 'General security mitigation'
}

export default MITREMatrix