import React, { useState, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Download, Shield, AlertTriangle, CheckCircle, Upload, Settings } from 'lucide-react';
import CVSSMeter from '../components/CVSSMeter';
import AttackTimeline from '../components/AttackTimeline';
import RealTimeLog from '../components/RealTimeLog';
import { analysisAPI } from '../services/api';

const startAnalysis = async () => {
  if (!prompt.trim()) {
    alert('Please enter a prompt to analyze');
    return;
  }

  setLoading(true);
  try {
    // Process config files first
    let garakConfigData = null;
    let pyritConfigData = null;

    if (garakConfig || pyritConfig) {
      const configResponse = await analysisAPI.uploadConfigs(garakConfig, pyritConfig);
      garakConfigData = configResponse.config_data.garak_config;
      pyritConfigData = configResponse.config_data.pyrit_config;
    }

    // ✅ CORRECT Request Format matching backend AnalysisRequest model
    const analysisRequest = {
      prompt: prompt.trim(),
      max_turns: 5,
      garak_probes: ["all"],  // ✅ Added missing field
      pyrit_strategies: ["all"],  // ✅ Added missing field
      enable_roberta_detection: true,
      target_model: "gpt-3.5-turbo",
      garak_config: garakConfigData,  // ✅ Now it's a Dict
      pyrit_config: pyritConfigData,  // ✅ Now it's a Dict
      context: {}  // ✅ Added missing optional field
    };

    console.log('Starting analysis with:', analysisRequest);
    const response = await analysisAPI.startAnalysis(analysisRequest);
    
    navigate(`/analysis/${response.analysis_id}`);
    
  } catch (error) {
    console.error('Error starting analysis:', error);
    alert('Failed to start analysis. Please check the console for details.');
  } finally {
    setLoading(false);
  }
};

  const startAnalysis = async () => {
    if (!prompt.trim()) {
      alert('Please enter a prompt to analyze');
      return;
    }

    setLoading(true);
    try {
      // First upload config files if provided
      let garakConfigData = null;
      let pyritConfigData = null;

      if (garakConfig || pyritConfig) {
        const configResponse = await analysisAPI.uploadConfigs(garakConfig, pyritConfig);
        garakConfigData = configResponse.config_data.garak_config;
        pyritConfigData = configResponse.config_data.pyrit_config;
      }

      // Start analysis with the processed config data
      const analysisRequest = {
        prompt: prompt.trim(),
        max_turns: 5,
        enable_roberta_detection: true,
        target_model: "gpt-3.5-turbo",
        garak_config: garakConfigData,
        pyrit_config: pyritConfigData
      };

      console.log('Starting analysis with:', analysisRequest);
      const response = await analysisAPI.startAnalysis(analysisRequest);
      
      // Navigate to the new analysis page
      navigate(`/analysis/${response.analysis_id}`);
      
    } catch (error) {
      console.error('Error starting analysis:', error);
      alert('Failed to start analysis. Please check the console for details.');
    } finally {
      setLoading(false);
    }
  };

  const downloadReport = async () => {
    if (!analysisId) return;
    
    try {
      const blob = await analysisAPI.downloadReport(analysisId);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `security-report-${analysisId}.pdf`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error downloading report:', error);
      alert('Failed to download report');
    }
  };

  const handleAnalysisComplete = (data) => {
    setAnalysisData(data.results);
  };

  // If we have an analysis ID, show the results
  if (analysisId && analysisId !== 'new') {
    return (
      <div style={{ minHeight: '100vh', background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', padding: '20px' }}>
        <div className="fade-in" style={{ maxWidth: '1200px', margin: '0 auto' }}>
          {/* Header */}
          <div className="flex items-center justify-between mb-30">
            <button 
              onClick={() => navigate('/')}
              className="btn-secondary flex items-center gap-10"
            >
              <ArrowLeft size={16} />
              Back to Analysis
            </button>
            
            <h1 style={{ color: 'white', margin: 0 }}>Security Analysis Results</h1>
            
            <button 
              onClick={downloadReport}
              className="btn-primary flex items-center gap-10"
            >
              <Download size={16} />
              Download PDF Report
            </button>
          </div>

          {/* Analysis ID */}
          <div style={{ background: 'rgba(255, 255, 255, 0.1)', padding: '15px', borderRadius: '8px', marginBottom: '20px', color: 'white' }}>
            Analysis ID: <strong>{analysisId}</strong>
          </div>

          {/* Real-time Logs */}
          <RealTimeLog 
            analysisId={analysisId} 
            onAnalysisComplete={handleAnalysisComplete}
          />

          {/* Results Section */}
          {analysisData && (
            <div style={{ marginTop: '30px' }}>
              {/* Executive Summary */}
              <div className="card slide-up">
                <h2 style={{ marginBottom: '20px', color: '#333' }}>Executive Summary</h2>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '20px', marginBottom: '20px' }}>
                  <div className={`card risk-${analysisData.executive_summary?.overall_risk_level?.toLowerCase()}`} style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '2rem', fontWeight: 'bold' }}>
                      {analysisData.executive_summary?.overall_risk_level || 'LOW'}
                    </div>
                    <div style={{ fontSize: '0.9rem' }}>Overall Risk</div>
                  </div>
                  
                  <div className="card" style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#667eea' }}>
                      {analysisData.executive_summary?.cvss_score?.toFixed(1) || '0.0'}
                    </div>
                    <div style={{ fontSize: '0.9rem', color: '#666' }}>CVSS Score</div>
                  </div>
                  
                  <div className="card" style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#ff6b6b' }}>
                      {analysisData.executive_summary?.total_vulnerabilities || 0}
                    </div>
                    <div style={{ fontSize: '0.9rem', color: '#666' }}>Vulnerabilities</div>
                  </div>
                  
                  <div className="card" style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#ff9ff3' }}>
                      {analysisData.executive_summary?.critical_findings || 0}
                    </div>
                    <div style={{ fontSize: '0.9rem', color: '#666' }}>Critical Findings</div>
                  </div>
                </div>
                
                <div style={{ padding: '15px', background: '#f8f9fa', borderRadius: '8px', borderLeft: '4px solid #667eea' }}>
                  <strong>Summary:</strong> {analysisData.executive_summary?.summary}
                </div>
              </div>

              {/* CVSS Meter */}
              {analysisData.cvss_assessment && (
                <CVSSMeter 
                  cvssData={analysisData.cvss_assessment}
                  vulnerabilities={[
                    ...(analysisData.detailed_findings?.garak_analysis?.vulnerabilities || []),
                    ...(analysisData.detailed_findings?.pyrit_analysis?.vulnerabilities || [])
                  ]}
                />
              )}

              {/* Attack Timeline */}
              <AttackTimeline analysisData={analysisData} />

              {/* Tool Findings */}
              <div className="card slide-up">
                <h3 style={{ marginBottom: '20px', color: '#333' }}>Tool Findings</h3>
                <div style={{ display: 'grid', gap: '15px' }}>
                  {analysisData.detailed_findings?.garak_analysis && (
                    <div style={{ padding: '15px', background: '#f0f8ff', borderRadius: '8px', border: '1px solid #667eea30' }}>
                      <h4 style={{ color: '#333', marginBottom: '10px' }}>🛡️ NVIDIA Garak Analysis</h4>
                      <div>Vulnerabilities Found: <strong>{analysisData.detailed_findings.garak_analysis.total_vulnerabilities}</strong></div>
                      <div>Risk Level: <strong>{analysisData.detailed_findings.garak_analysis.risk_level}</strong></div>
                    </div>
                  )}
                  
                  {analysisData.detailed_findings?.pyrit_analysis && (
                    <div style={{ padding: '15px', background: '#fff0f5', borderRadius: '8px', border: '1px solid #ff9ff330' }}>
                      <h4 style={{ color: '#333', marginBottom: '10px' }}>🔍 Microsoft PyRIT Analysis</h4>
                      <div>Escalation Detected: <strong>{analysisData.detailed_findings.pyrit_analysis.escalation_detected ? 'Yes' : 'No'}</strong></div>
                      <div>Risk Level: <strong>{analysisData.detailed_findings.pyrit_analysis.risk_level}</strong></div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }

  // New Analysis Form
  return (
    <div style={{ minHeight: '100vh', background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', padding: '20px' }}>
      <div className="fade-in" style={{ maxWidth: '800px', margin: '0 auto' }}>
        {/* Header */}
        <div className="text-center mb-40">
          <h1 style={{ color: 'white', fontSize: '2.5rem', marginBottom: '10px' }}>
            🛡️ LLM Security Framework
          </h1>
          <p style={{ color: 'rgba(255, 255, 255, 0.8)', fontSize: '1.1rem' }}>
            Advanced Security Assessment for Large Language Models
          </p>
        </div>

        <div className="card" style={{ background: 'white', padding: '40px' }}>
          <h2 style={{ marginBottom: '30px', color: '#333', textAlign: 'center' }}>
            Start New Security Analysis
          </h2>

          {/* Prompt Input */}
          <div className="form-group">
            <label className="form-label">
              Prompt to Analyze *
            </label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Enter the prompt you want to analyze for security vulnerabilities..."
              className="form-textarea"
            />
          </div>

          {/* Configuration Upload */}
          <div style={{ marginBottom: '30px' }}>
            <div className="flex items-center gap-10 mb-15">
              <Settings size={20} />
              <h3 style={{ margin: 0, color: '#333' }}>Tool Configurations (Optional)</h3>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
              {/* Garak Config */}
              <div>
                <label className="form-label">
                  NVIDIA Garak Config
                </label>
                <input
                  ref={garakFileRef}
                  type="file"
                  accept=".yaml,.yml,.json"
                  onChange={(e) => handleConfigUpload('garak', e.target.files[0])}
                  style={{ display: 'none' }}
                />
                <button
                  onClick={() => garakFileRef.current?.click()}
                  className="btn-secondary"
                  style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}
                >
                  <Upload size={16} />
                  {garakConfig ? garakConfig.name : 'Upload YAML Config'}
                </button>
              </div>

              {/* PyRIT Config */}
              <div>
                <label className="form-label">
                  Microsoft PyRIT Config
                </label>
                <input
                  ref={pyritFileRef}
                  type="file"
                  accept=".json"
                  onChange={(e) => handleConfigUpload('pyrit', e.target.files[0])}
                  style={{ display: 'none' }}
                />
                <button
                  onClick={() => pyritFileRef.current?.click()}
                  className="btn-secondary"
                  style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}
                >
                  <Upload size={16} />
                  {pyritConfig ? pyritConfig.name : 'Upload JSON Config'}
                </button>
              </div>
            </div>
          </div>

          {/* Analysis Options */}
          <div style={{ background: '#f8f9fa', padding: '20px', borderRadius: '8px', marginBottom: '30px' }}>
            <h4 style={{ marginBottom: '15px', color: '#333' }}>Analysis Options</h4>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
              <div>
                <label className="form-label">
                  Max Conversation Turns
                </label>
                <select className="form-select" defaultValue="5">
                  <option value="3">3 Turns</option>
                  <option value="5">5 Turns</option>
                  <option value="10">10 Turns</option>
                </select>
              </div>
              <div>
                <label className="form-label">
                  Target Model
                </label>
                <select className="form-select" defaultValue="gpt-3.5-turbo">
                  <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
                  <option value="gpt-4">GPT-4</option>
                  <option value="claude-2">Claude 2</option>
                  <option value="llama-2">Llama 2</option>
                </select>
              </div>
            </div>
          </div>

          {/* Start Analysis Button */}
          <button
            onClick={startAnalysis}
            disabled={loading || !prompt.trim()}
            className="btn-primary"
            style={{ width: '100%', fontSize: '1.1rem', padding: '15px' }}
          >
            {loading ? '🔄 Starting Analysis...' : '🚀 Start Security Analysis'}
          </button>

          {/* Features List */}
          <div style={{ marginTop: '40px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '20px' }}>
            <div className="text-center">
              <Shield size={32} color="#667eea" />
              <h4 style={{ margin: '10px 0 5px 0', color: '#333' }}>NVIDIA Garak</h4>
              <p style={{ color: '#666', fontSize: '0.9rem' }}>Comprehensive prompt injection detection</p>
            </div>
            <div className="text-center">
              <AlertTriangle size={32} color="#ff9ff3" />
              <h4 style={{ margin: '10px 0 5px 0', color: '#333' }}>Microsoft PyRIT</h4>
              <p style={{ color: '#666', fontSize: '0.9rem' }}>Multi-turn attack simulation</p>
            </div>
            <div className="text-center">
              <CheckCircle size={32} color="#1dd1a1" />
              <h4 style={{ margin: '10px 0 5px 0', color: '#333' }}>Custom RoBERTa</h4>
              <p style={{ color: '#666', fontSize: '0.9rem' }}>AI-powered attack pattern detection</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Analysis;