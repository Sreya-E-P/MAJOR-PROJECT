import React from 'react'
import FileUpload from './FileUpload'
import './DynamicForm.css'

const DynamicForm = ({ 
  scanType, 
  prompt, 
  onPromptChange, 
  assistantResponse, 
  onAssistantResponseChange,
  garakConfigs,
  selectedGarakConfig,
  onGarakConfigChange,
  pyritConfigs,
  selectedPyritConfig,
  onPyritConfigChange,
  garakLogFile,
  onGarakLogChange,
  pyritLogFile,
  onPyritLogChange,
  isAnalyzing 
}) => {
  
  // Render different form sections based on scan type
  const renderFormSections = () => {
    switch (scanType) {
      case 'full_scan':
        return (
          <>
            <div className="form-section">
              <h4>🛠️ Tool Configurations</h4>
              <p className="section-description">
                Garak and PyRIT will generate their own test prompts. You can optionally provide a specific prompt to analyze.
              </p>
              
              <div className="config-grid">
                <div className="input-group">
                  <label>🛡️ Garak Configuration:</label>
                  <select 
                    value={selectedGarakConfig} 
                    onChange={(e) => onGarakConfigChange(e.target.value)}
                    disabled={isAnalyzing}
                  >
                    <option value="">Default Garak Config</option>
                    {garakConfigs.map(config => (
                      <option key={config.config_id} value={config.config_id}>
                        {config.model_info?.model_name || config.config_id}
                      </option>
                    ))}
                  </select>
                  <div className="input-help">
                    Garak will automatically generate jailbreak and prompt injection tests
                  </div>
                </div>

                <div className="input-group">
                  <label>⚡ PyRIT Configuration:</label>
                  <select 
                    value={selectedPyritConfig} 
                    onChange={(e) => onPyritConfigChange(e.target.value)}
                    disabled={isAnalyzing}
                  >
                    <option value="">Default PyRIT Config</option>
                    {pyritConfigs.map(config => (
                      <option key={config.config_id} value={config.config_id}>
                        {config.model_info?.model_name || config.config_id}
                      </option>
                    ))}
                  </select>
                  <div className="input-help">
                    PyRIT will perform multi-turn conversation attacks
                  </div>
                </div>
              </div>
            </div>

            <div className="form-section">
              <h4>💬 Optional Prompt Analysis</h4>
              <p className="section-description">
                Optional: Analyze a specific prompt alongside tool-generated tests
              </p>
              
              <div className="input-group">
                <label>User Prompt (Optional):</label>
                <textarea
                  value={prompt}
                  onChange={(e) => onPromptChange(e.target.value)}
                  placeholder="Optionally enter a specific prompt to analyze alongside automated tests..."
                  rows="3"
                  disabled={isAnalyzing}
                />
                <div className="input-help">
                  If provided, this prompt will be analyzed using the Research Model in addition to tool-generated tests
                </div>
              </div>
              
              <div className="input-group">
                <label>Assistant Response (Optional):</label>
                <textarea
                  value={assistantResponse}
                  onChange={(e) => onAssistantResponseChange(e.target.value)}
                  placeholder="Optionally enter the assistant's response for compliance analysis..."
                  rows="2"
                  disabled={isAnalyzing}
                />
                <div className="input-help">
                  If provided, the response will be analyzed for information leakage
                </div>
              </div>
            </div>
          </>
        )

      case 'log_analysis':
        return (
          <div className="form-section">
            <h4>📁 Log File Upload</h4>
            <p className="section-description">
              Upload Garak JSONL logs and/or PyRIT JSON logs for analysis
            </p>
            
            <div className="file-upload-grid">
              <FileUpload
                label="🛡️ Garak JSONL Log File"
                accept=".jsonl,.json"
                file={garakLogFile}
                onFileChange={onGarakLogChange}
                disabled={isAnalyzing}
                required={!pyritLogFile}
              />
              
              <FileUpload
                label="⚡ PyRIT JSON Log File" 
                accept=".json"
                file={pyritLogFile}
                onFileChange={onPyritLogChange}
                disabled={isAnalyzing}
                required={!garakLogFile}
              />
            </div>
            
            <div className="log-requirements">
              <h5>📋 File Requirements:</h5>
              <ul>
                <li><strong>Garak:</strong> JSONL format with one JSON object per line</li>
                <li><strong>PyRIT:</strong> JSON format with conversation data</li>
                <li>At least one log file is required</li>
              </ul>
            </div>
          </div>
        )

      case 'fast_analysis':
        return (
          <div className="form-section">
            <h4>⚡ Fast Prompt Analysis</h4>
            <div className="input-group">
              <label>User Prompt:</label>
              <textarea
                value={prompt}
                onChange={(e) => onPromptChange(e.target.value)}
                placeholder="Enter the prompt for quick security analysis..."
                rows="4"
                disabled={isAnalyzing}
                required
              />
            </div>
            <div className="input-group">
              <label>Assistant Response:</label>
              <textarea
                value={assistantResponse}
                onChange={(e) => onAssistantResponseChange(e.target.value)}
                placeholder="Enter the assistant's response for compliance analysis..."
                rows="3"
                disabled={isAnalyzing}
                required
              />
            </div>
            <div className="fast-analysis-note">
              <strong>Note:</strong> Fast analysis uses only the Research Model for quick detection 
              without external tool dependencies.
            </div>
          </div>
        )

      default:
        return (
          <div className="no-selection">
            <p>Please select an analysis type to continue.</p>
          </div>
        )
    }
  }

  const getValidationErrors = () => {
    const errors = []

    // Full Scan: No prompt required - tools generate their own
    if (scanType === 'full_scan') {
      // No validation errors - prompt is optional
    }

    if (scanType === 'log_analysis') {
      if (!garakLogFile && !pyritLogFile) {
        errors.push('At least one log file (Garak or PyRIT) is required for Log Analysis')
      }
    }

    if (scanType === 'fast_analysis') {
      if (!prompt.trim()) {
        errors.push('User prompt is required for Fast Analysis')
      }
      if (!assistantResponse.trim()) {
        errors.push('Assistant response is required for Fast Analysis')
      }
    }

    return errors
  }

  const validationErrors = getValidationErrors()

  return (
    <div className="dynamic-form">
      {scanType && (
        <div className="form-header">
          <h3>
            {scanType === 'full_scan' && '🔍 Full Security Scan Configuration'}
            {scanType === 'log_analysis' && '📊 Log Analysis Configuration'} 
            {scanType === 'fast_analysis' && '⚡ Fast Analysis Configuration'}
          </h3>
          {validationErrors.length > 0 && (
            <div className="validation-errors">
              <strong>Please fix the following:</strong>
              <ul>
                {validationErrors.map((error, index) => (
                  <li key={index}>❌ {error}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
      
      <div className="form-content">
        {renderFormSections()}
      </div>
    </div>
  )
}

export default DynamicForm