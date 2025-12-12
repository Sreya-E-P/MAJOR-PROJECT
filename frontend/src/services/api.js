import axios from 'axios';

const API_BASE = 'http://localhost:8000';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE,
  timeout: 300000, // 5 minutes timeout for long-running analyses
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`Making ${config.method?.toUpperCase()} request to ${config.url}`);
    return config;
  },
  (error) => {
    console.error('Request error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// ==================== AUTO-DOWNLOAD FUNCTIONS ====================

/**
 * Auto-download PDF report immediately after analysis completion
 * This is the function your WebSocket/polling should call
 */
export const autoDownloadPDF = (analysisId) => {
  try {
    console.log('🔽 Starting auto-download for:', analysisId);
    
    // Create hidden link and trigger download
    const downloadUrl = `${API_BASE}/download/report/${analysisId}`;
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = `security_report_${analysisId}.pdf`;
    link.style.display = 'none';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    console.log('✅ Auto-download triggered');
    return { success: true, message: 'Auto-download started' };
    
  } catch (error) {
    console.error('Auto-download failed:', error);
    return { success: false, message: 'Auto-download failed' };
  }
};

/**
 * Simple download function - triggers browser download
 */
export const simpleDownload = (analysisId) => {
  const downloadUrl = `${API_BASE}/download/report/${analysisId}`;
  window.location.href = downloadUrl;
  return { success: true, message: 'Download started' };
};

// ==================== ANALYSIS ENDPOINTS ====================

/**
 * Unified analysis endpoint for all scan types (FormData)
 */
export const analyzeSecurity = async (formData) => {
  try {
    const response = await api.post('/api/analyze', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      }
    });
    
    // ✅ FIX: If analysis is complete and has ID, trigger auto-download
    if (response.data.analysis_id && response.data.status === 'completed') {
      console.log('✅ Analysis complete, triggering auto-download...');
      // Download immediately if results are already complete
      if (response.data.report_generated) {
        autoDownloadPDF(response.data.analysis_id);
      }
    }
    
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Security analysis failed');
  }
};

/**
 * Instant analysis (Fast Analysis)
 */
export const analyzeInstant = async (analysisData) => {
  try {
    const response = await api.post('/api/analyze-instant', analysisData, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      }
    });
    
    // ✅ FIX: Trigger auto-download if analysis is complete
    if (response.data.analysis_id && response.data.report_generated) {
      console.log('✅ Instant analysis complete, triggering auto-download...');
      autoDownloadPDF(response.data.analysis_id);
    }
    
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Instant analysis failed');
  }
};

/**
 * Legacy analyzePrompt for JSON requests
 */
export const analyzePrompt = async (requestData) => {
  try {
    const response = await api.post('/api/analyze', requestData);
    
    // ✅ FIX: Trigger auto-download if analysis is complete
    if (response.data.analysis_id && response.data.report_generated) {
      console.log('✅ Prompt analysis complete, triggering auto-download...');
      autoDownloadPDF(response.data.analysis_id);
    }
    
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Analysis request failed');
  }
};

// ==================== CONFIGURATION MANAGEMENT ====================

/**
 * Upload configuration files
 */
export const uploadConfig = async (configType, file) => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('config_type', configType);
    
    const response = await api.post(`/api/upload-config`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      }
    });
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Config upload failed');
  }
};

/**
 * Get uploaded configurations
 */
export const getConfigs = async (configType = null) => {
  try {
    const params = configType ? { config_type: configType } : {};
    const response = await api.get(`/api/configs`, { params });
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Failed to fetch configs');
  }
};

/**
 * Delete a configuration
 */
export const deleteConfig = async (configId, configType) => {
  try {
    const response = await api.delete(`/api/configs/${configId}`, {
      params: { config_type: configType }
    });
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Failed to delete config');
  }
};

/**
 * Validate a configuration
 */
export const validateConfig = async (configId, configType) => {
  try {
    const response = await api.get(`/api/configs/${configId}/validate`, {
      params: { config_type: configType }
    });
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Config validation failed');
  }
};

/**
 * Get configuration templates
 */
export const getConfigTemplates = async (configType) => {
  try {
    const response = await api.get(`/api/config-templates/${configType}`);
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Failed to fetch config templates');
  }
};

// ==================== ANALYSIS STATUS & RESULTS ====================

/**
 * Get analysis status by ID
 */
export const getAnalysisStatus = async (analysisId) => {
  try {
    const response = await api.get(`/api/analysis/${analysisId}`);
    
    // ✅ FIX: If status is completed and report is generated, trigger auto-download
    if (response.data.status === 'completed' && response.data.report_generated) {
      console.log('✅ Analysis completed, checking for auto-download...');
      
      // Check if we should auto-download (track in localStorage to prevent duplicates)
      const hasDownloaded = localStorage.getItem(`downloaded_${analysisId}`);
      if (!hasDownloaded) {
        setTimeout(() => {
          autoDownloadPDF(analysisId);
          // Mark as downloaded
          localStorage.setItem(`downloaded_${analysisId}`, 'true');
        }, 1000);
      }
    }
    
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Failed to fetch analysis status');
  }
};

/**
 * Get analysis history
 */
export const getAnalysisHistory = async (limit = 10) => {
  try {
    const response = await api.get(`/api/analysis-history`, {
      params: { limit }
    });
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Failed to fetch analysis history');
  }
};

/**
 * Get debug info about analysis store
 */
export const getAnalysisDebugInfo = async () => {
  try {
    const response = await api.get(`/api/debug/analysis-store`);
    return response.data;
  } catch (error) {
    throw new Error('Failed to fetch analysis debug info');
  }
};

// ==================== PDF REPORT MANAGEMENT ====================

/**
 * Generate PDF report for analysis
 */
export const generatePDFReport = async (analysisId) => {
  try {
    const response = await api.post(`/api/generate-pdf/${analysisId}`);
    
    // ✅ FIX: After generating, trigger download
    if (response.data.status === 'success') {
      console.log('✅ PDF generated, triggering download...');
      setTimeout(() => autoDownloadPDF(analysisId), 500);
    }
    
    return response.data;
  } catch (error) {
    // If PDF already exists, that's okay
    if (error.response?.status === 409) {
      console.log('PDF already exists for analysis:', analysisId);
      // Still trigger download since it exists
      setTimeout(() => autoDownloadPDF(analysisId), 500);
      return { status: 'success', message: 'PDF already exists' };
    }
    throw new Error(error.response?.data?.detail || 'Failed to generate PDF report');
  }
};

/**
 * Download PDF report (blob-based approach)
 */
export const downloadPDFReport = async (analysisId) => {
  try {
    const response = await axios.get(`${API_BASE}/download/report/${analysisId}`, {
      responseType: 'blob',
    });
    
    // Create a blob URL and trigger download
    const blob = new Blob([response.data], { type: 'application/pdf' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `security_report_${analysisId}_${Date.now()}.pdf`;
    link.style.display = 'none';
    document.body.appendChild(link);
    link.click();
    
    // Clean up
    setTimeout(() => {
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    }, 100);
    
    console.log('✅ PDF downloaded successfully via blob');
    return { success: true, message: 'PDF downloaded successfully' };
    
  } catch (error) {
    console.error('PDF Download Error:', error);
    
    // Fallback to simple download
    try {
      autoDownloadPDF(analysisId);
      return { success: true, message: 'PDF download triggered (fallback)' };
    } catch (fallbackError) {
      throw new Error('Failed to download PDF report');
    }
  }
};

/**
 * View PDF report in new tab
 */
export const viewPDFReport = (analysisId) => {
  try {
    const url = `${API_BASE}/download/report/${analysisId}`;
    window.open(url, '_blank');
    return { success: true, message: 'PDF opened in new tab' };
  } catch (error) {
    console.error('PDF View Error:', error);
    throw new Error(error.response?.data?.detail || 'Failed to open PDF report');
  }
};

// ==================== SECURITY TOOLS ENDPOINTS ====================

/**
 * Run full security scan (Garak + PyRIT)
 */
export const runFullScan = async (scanConfig) => {
  try {
    const response = await api.post('/api/security/full-scan', scanConfig);
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Full scan failed to start');
  }
};

/**
 * Analyze log files
 */
export const analyzeLogs = async (logConfig) => {
  try {
    const formData = new FormData();
    
    if (logConfig.garakLog) {
      formData.append('garak_log', logConfig.garakLog);
    }
    if (logConfig.pyritLog) {
      formData.append('pyrit_log', logConfig.pyritLog);
    }
    
    const response = await api.post('/api/security/log-analysis', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      }
    });
    
    // ✅ FIX: Trigger auto-download for log analysis
    if (response.data.analysis_id && response.data.report_generated) {
      console.log('✅ Log analysis complete, triggering auto-download...');
      setTimeout(() => autoDownloadPDF(response.data.analysis_id), 1000);
    }
    
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Log analysis failed');
  }
};

/**
 * Quick prompt analysis
 */
export const analyzePromptQuick = async (promptConfig) => {
  try {
    const response = await api.post('/api/security/prompt-analysis', promptConfig);
    
    // ✅ FIX: Trigger auto-download
    if (response.data.analysis_id && response.data.report_generated) {
      console.log('✅ Quick prompt analysis complete, triggering auto-download...');
      setTimeout(() => autoDownloadPDF(response.data.analysis_id), 1000);
    }
    
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Prompt analysis failed');
  }
};

/**
 * Get scan status for long-running operations
 */
export const getScanStatus = async (analysisId) => {
  try {
    const response = await api.get(`/api/security/scan-status/${analysisId}`);
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Failed to fetch scan status');
  }
};

// ==================== HEALTH & DIAGNOSTICS ====================

/**
 * Basic health check
 */
export const healthCheck = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    throw new Error('Backend service is unavailable. Please ensure the server is running on localhost:8000');
  }
};

/**
 * Detailed health check with system status
 */
export const detailedHealthCheck = async () => {
  try {
    const response = await api.get('/health/detailed');
    return response.data;
  } catch (error) {
    throw new Error('Detailed health check failed. Backend service may be unavailable.');
  }
};

// ==================== MODEL TESTING ENDPOINTS ====================

/**
 * Test single-turn model directly
 */
export const testSingleTurnModel = async (prompt) => {
  try {
    const formData = new FormData();
    formData.append('prompt', prompt);
    
    const response = await api.post('/api/test-single-turn', formData, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      }
    });
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Single-turn model test failed');
  }
};

/**
 * Test multi-turn model directly
 */
export const testMultiTurnModel = async (prompt) => {
  try {
    const formData = new FormData();
    formData.append('prompt', prompt);
    
    const response = await api.post('/api/test-multi-turn', formData, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      }
    });
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Multi-turn model test failed');
  }
};

/**
 * Test pattern detection model
 */
export const testPatternModel = async (prompt) => {
  try {
    const formData = new FormData();
    formData.append('prompt', prompt);
    
    const response = await api.post('/api/test-pattern-model', formData, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      }
    });
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Pattern model test failed');
  }
};

/**
 * Direct test endpoint
 */
export const testDirectDetection = async (prompt) => {
  try {
    const formData = new FormData();
    formData.append('prompt', prompt);
    
    const response = await api.post('/api/test-direct', formData, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      }
    });
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Direct test failed');
  }
};

// ==================== WEBHOOKS & REAL-TIME UPDATES ====================

/**
 * Create WebSocket connection for real-time updates
 */
export const createWebSocketConnection = (analysisId, onMessage, onError) => {
  try {
    const ws = new WebSocket(`ws://localhost:8000/ws/analysis/${analysisId}`);
    
    ws.onopen = () => {
      console.log(`WebSocket connected for analysis: ${analysisId}`);
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
        
        // ✅ FIX: Auto-download when analysis is complete
        if (data.type === 'completed' && data.results) {
          console.log('✅ WebSocket: Analysis complete, auto-downloading...');
          setTimeout(() => autoDownloadPDF(data.results.analysis_id), 500);
        }
      } catch (error) {
        console.error('WebSocket message parsing error:', error);
      }
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      if (onError) onError(error);
    };
    
    ws.onclose = () => {
      console.log(`WebSocket disconnected for analysis: ${analysisId}`);
    };
    
    return ws;
  } catch (error) {
    console.error('WebSocket creation failed:', error);
    throw error;
  }
};

// ==================== UTILITY FUNCTIONS ====================

/**
 * Convert FormData to object for debugging
 */
export const formDataToObject = (formData) => {
  const object = {};
  formData.forEach((value, key) => {
    object[key] = value;
  });
  return object;
};

/**
 * Create analysis request payload
 */
export const createAnalysisPayload = (scanType, formData) => {
  const payload = new FormData();
  payload.append('analysis_type', scanType);
  
  Object.keys(formData).forEach(key => {
    if (formData[key] !== null && formData[key] !== undefined) {
      payload.append(key, formData[key]);
    }
  });
  
  return payload;
};

export default api;