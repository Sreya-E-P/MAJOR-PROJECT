import React, { useState, useRef } from 'react';
import { uploadConfig, getConfigs } from '../services/api';
import './ConfigUpload.css';

const ConfigUpload = ({ onConfigsUpdated }) => {
  const [uploadStatus, setUploadStatus] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [garakConfigs, setGarakConfigs] = useState([]);
  const [pyritConfigs, setPyritConfigs] = useState([]);
  const fileInputGarak = useRef(null);
  const fileInputPyrit = useRef(null);

  React.useEffect(() => {
    loadConfigs();
  }, []);

  const loadConfigs = async () => {
    try {
      const [garakResponse, pyritResponse] = await Promise.all([
        getConfigs('garak'),
        getConfigs('pyrit')
      ]);
      setGarakConfigs(garakResponse.configs || []);
      setPyritConfigs(pyritResponse.configs || []);
    } catch (err) {
      console.error('Failed to load configs:', err);
    }
  };

  const handleConfigUpload = async (configType, file) => {
    if (!file) {
      setUploadStatus('Please select a configuration file');
      return;
    }

    // Validate file extensions
    const fileExtension = file.name.split('.').pop().toLowerCase();
    if (configType === 'garak' && fileExtension !== 'yaml' && fileExtension !== 'yml') {
      setUploadStatus('❌ Garak configuration must be a YAML file (.yaml or .yml)');
      return;
    }
    if (configType === 'pyrit' && fileExtension !== 'yaml' && fileExtension !== 'yml') {
      setUploadStatus('❌ PyRIT configuration must be a YAML file (.yaml or .yml)');
      return;
    }

    setIsUploading(true);
    setUploadStatus(`Uploading ${configType} configuration...`);

    try {
      const result = await uploadConfig(configType, file);
      setUploadStatus(`✅ ${configType.toUpperCase()} configuration uploaded successfully!`);
      
      // Reload configs
      await loadConfigs();
      onConfigsUpdated?.();
      
      // Clear file input
      if (configType === 'garak') {
        fileInputGarak.current.value = '';
      } else {
        fileInputPyrit.current.value = '';
      }
    } catch (err) {
      setUploadStatus(`❌ Upload failed: ${err.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  const handleFileSelect = (configType, event) => {
    const file = event.target.files[0];
    if (file) {
      handleConfigUpload(configType, file);
    }
  };

  const ConfigList = ({ configs, type }) => (
    <div className="config-list">
      <h4>{type === 'garak' ? '🛡️ Garak' : '⚡ PyRIT'} Configurations</h4>
      {configs.length === 0 ? (
        <p className="no-configs">No configurations uploaded yet</p>
      ) : (
        <div className="config-cards">
          {configs.map(config => (
            <div key={config.config_id} className="config-card">
              <div className="config-header">
                <h5>{config.model_info?.model_name || 'Unnamed Config'}</h5>
                <span className="config-id">{config.config_id}</span>
              </div>
              <div className="config-details">
                <div className="config-detail">
                  <span className="label">Type:</span>
                  <span className="value">{config.config_type}</span>
                </div>
                <div className="config-detail">
                  <span className="label">Uploaded:</span>
                  <span className="value">{new Date(config.upload_time).toLocaleDateString()}</span>
                </div>
                {config.model_info && (
                  <>
                    <div className="config-detail">
                      <span className="label">Model:</span>
                      <span className="value">{config.model_info.model_name}</span>
                    </div>
                    {config.model_info.probes_count && (
                      <div className="config-detail">
                        <span className="label">Probes:</span>
                        <span className="value">{config.model_info.probes_count}</span>
                      </div>
                    )}
                    {config.model_info.scanner_type && (
                      <div className="config-detail">
                        <span className="label">Scanner:</span>
                        <span className="value">{config.model_info.scanner_type}</span>
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );

  return (
    <div className="config-upload">
      <h3>⚙️ Configuration Manager</h3>
      
      <div className="upload-section">
        <div className="upload-zone">
          <div className="upload-item">
            <h4>Upload Garak Configuration</h4>
            <p>Upload YAML configuration files for Garak</p>
            <input
              type="file"
              ref={fileInputGarak}
              accept=".yaml,.yml"
              onChange={(e) => handleFileSelect('garak', e)}
              disabled={isUploading}
            />
            <button 
              onClick={() => fileInputGarak.current?.click()}
              disabled={isUploading}
              className="upload-btn"
            >
              {isUploading ? '📤 Uploading...' : '📁 Choose Garak Config (YAML)'}
            </button>
          </div>

          <div className="upload-item">
            <h4>Upload PyRIT Configuration</h4>
            <p>Upload YAML configuration files for PyRIT</p>
            <input
              type="file"
              ref={fileInputPyrit}
              accept=".yaml,.yml"
              onChange={(e) => handleFileSelect('pyrit', e)}
              disabled={isUploading}
            />
            <button 
              onClick={() => fileInputPyrit.current?.click()}
              disabled={isUploading}
              className="upload-btn"
            >
              {isUploading ? '📤 Uploading...' : '📁 Choose PyRIT Config (YAML)'}
            </button>
          </div>
        </div>

        {uploadStatus && (
          <div className={`upload-status ${uploadStatus.includes('✅') ? 'success' : 'error'}`}>
            {uploadStatus}
          </div>
        )}
      </div>

      <div className="config-lists">
        <ConfigList configs={garakConfigs} type="garak" />
        <ConfigList configs={pyritConfigs} type="pyrit" />
      </div>
    </div>
  );
};

export default ConfigUpload;