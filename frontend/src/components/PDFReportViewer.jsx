import React, { useState, useEffect } from 'react'
import { Document, Page, pdfjs } from 'react-pdf'
import { downloadPDFReport } from '../services/api'
import './PDFReportViewer.css'

// Configure PDF.js worker
pdfjs.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.js`

const PDFReportViewer = ({ analysisResults }) => {
  const [pdfUrl, setPdfUrl] = useState(null)
  const [numPages, setNumPages] = useState(null)
  const [pageNumber, setPageNumber] = useState(1)
  const [scale, setScale] = useState(1.2)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (analysisResults?.analysis_id) {
      loadPDFReport()
    }
  }, [analysisResults])

  const loadPDFReport = async () => {
    if (!analysisResults?.analysis_id) return

    setIsLoading(true)
    setError(null)

    try {
      // Download and create blob URL for viewing
      const pdfBlob = await downloadPDFReport(analysisResults.analysis_id)
      const url = URL.createObjectURL(pdfBlob)
      setPdfUrl(url)
    } catch (err) {
      setError(err.message)
      console.error('Failed to load PDF report:', err)
    } finally {
      setIsLoading(false)
    }
  }

  const onDocumentLoadSuccess = ({ numPages }) => {
    setNumPages(numPages)
  }

  const goToPreviousPage = () => {
    setPageNumber(prev => Math.max(prev - 1, 1))
  }

  const goToNextPage = () => {
    setPageNumber(prev => Math.min(prev + 1, numPages))
  }

  const zoomIn = () => {
    setScale(prev => Math.min(prev + 0.2, 3))
  }

  const zoomOut = () => {
    setScale(prev => Math.max(prev - 0.2, 0.5))
  }

  const resetZoom = () => {
    setScale(1.2)
  }

  const handleDownload = async () => {
    if (!analysisResults?.analysis_id) return

    try {
      await downloadPDFReport(analysisResults.analysis_id)
    } catch (err) {
      setError(`Download failed: ${err.message}`)
    }
  }

  if (isLoading) {
    return (
      <div className="pdf-viewer">
        <div className="pdf-loading">
          <div className="spinner"></div>
          <p>Loading PDF Report...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="pdf-viewer">
        <div className="pdf-error">
          <div className="error-icon">❌</div>
          <h3>Failed to Load PDF Report</h3>
          <p>{error}</p>
          <button onClick={loadPDFReport} className="retry-btn">
            🔄 Retry
          </button>
        </div>
      </div>
    )
  }

  if (!pdfUrl) {
    return (
      <div className="pdf-viewer">
        <div className="pdf-placeholder">
          <div className="placeholder-icon">📊</div>
          <h3>PDF Report Viewer</h3>
          <p>No PDF report available. Complete an analysis to generate a report.</p>
          {analysisResults?.analysis_id && (
            <button onClick={handleDownload} className="download-btn">
              📥 Download Report
            </button>
          )}
        </div>
      </div>
    )
  }

  return (
    <div className="pdf-viewer">
      <div className="pdf-controls">
        <div className="controls-left">
          <button 
            onClick={goToPreviousPage} 
            disabled={pageNumber <= 1}
            className="control-btn"
          >
            ◀ Previous
          </button>
          
          <span className="page-info">
            Page {pageNumber} of {numPages || '--'}
          </span>
          
          <button 
            onClick={goToNextPage} 
            disabled={pageNumber >= numPages}
            className="control-btn"
          >
            Next ▶
          </button>
        </div>

        <div className="controls-center">
          <span className="zoom-info">
            Zoom: {Math.round(scale * 100)}%
          </span>
        </div>

        <div className="controls-right">
          <button onClick={zoomOut} className="control-btn" disabled={scale <= 0.5}>
            🔍-
          </button>
          <button onClick={resetZoom} className="control-btn">
            🔄
          </button>
          <button onClick={zoomIn} className="control-btn" disabled={scale >= 3}>
            🔍+
          </button>
          <button onClick={handleDownload} className="control-btn download">
            📥 Download
          </button>
        </div>
      </div>

      <div className="pdf-container">
        <Document
          file={pdfUrl}
          onLoadSuccess={onDocumentLoadSuccess}
          onLoadError={(error) => setError(error.message)}
          loading={
            <div className="pdf-loading-inline">
              <div className="spinner"></div>
              <p>Loading PDF document...</p>
            </div>
          }
        >
          <Page 
            pageNumber={pageNumber} 
            scale={scale}
            loading={
              <div className="page-loading">
                Loading page {pageNumber}...
              </div>
            }
          />
        </Document>
      </div>

      <div className="pdf-footer">
        <div className="report-info">
          <strong>Report ID:</strong> {analysisResults.analysis_id}
          <strong>Generated:</strong> {new Date(analysisResults.timestamp).toLocaleString()}
        </div>
      </div>
    </div>
  )
}

export default PDFReportViewer