import React, { useRef } from 'react'
import './FileUpload.css'

const FileUpload = ({ 
  label, 
  accept, 
  file, 
  onFileChange, 
  disabled = false, 
  required = false 
}) => {
  const fileInputRef = useRef(null)

  const handleFileSelect = (event) => {
    const selectedFile = event.target.files[0]
    if (selectedFile) {
      onFileChange(selectedFile)
    }
  }

  const handleDrop = (event) => {
    event.preventDefault()
    const droppedFile = event.dataTransfer.files[0]
    if (droppedFile) {
      onFileChange(droppedFile)
    }
  }

  const handleDragOver = (event) => {
    event.preventDefault()
  }

  const removeFile = () => {
    onFileChange(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const getFileInfo = (file) => {
    if (!file) return null
    
    const sizeInMB = (file.size / (1024 * 1024)).toFixed(2)
    return {
      name: file.name,
      size: `${sizeInMB} MB`,
      type: file.type
    }
  }

  const fileInfo = file ? getFileInfo(file) : null

  return (
    <div className="file-upload">
      <label className="file-upload-label">
        {label}
        {required && <span className="required-asterisk">*</span>}
      </label>
      
      <div 
        className={`file-drop-zone ${file ? 'has-file' : ''} ${disabled ? 'disabled' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onClick={() => !disabled && fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept={accept}
          onChange={handleFileSelect}
          disabled={disabled}
          className="file-input"
        />
        
        {fileInfo ? (
          <div className="file-info">
            <div className="file-icon">📄</div>
            <div className="file-details">
              <div className="file-name">{fileInfo.name}</div>
              <div className="file-size">{fileInfo.size}</div>
            </div>
            {!disabled && (
              <button 
                type="button" 
                className="remove-file-btn"
                onClick={(e) => {
                  e.stopPropagation()
                  removeFile()
                }}
              >
                ❌
              </button>
            )}
          </div>
        ) : (
          <div className="file-placeholder">
            <div className="upload-icon">📁</div>
            <div className="upload-text">
              <strong>Click to select or drag and drop</strong>
              <span>Accepted format: {accept}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default FileUpload