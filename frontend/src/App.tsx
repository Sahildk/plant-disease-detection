import { useState, useRef } from 'react';
import type { DragEvent, ChangeEvent } from 'react';
import './App.css';

interface PredictionResult {
  disease: string;
  raw_class: string;
  confidence: number;
  status: string;
}

interface Advice {
  summary: string;
  tips: string[];
}

export default function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Parse prediction results and return care advice
  const getCareAdvice = (rawClass: string, status: string): Advice => {
    const cls = rawClass.toLowerCase();
    
    if (status === 'Healthy') {
      return {
        summary: 'Excellent! Your plant is healthy. Maintain its health with these best practices:',
        tips: [
          'Ensure regular watering based on the plant species’ needs; let the topsoil dry out slightly between waterings.',
          'Provide adequate sunlight (typically 6-8 hours for vegetable crops).',
          'Periodically inspect the underside of leaves for any signs of early pests or spots.',
          'Apply balanced organic compost once a month during the growing season.'
        ]
      };
    }

    if (cls.includes('blight')) {
      return {
        summary: 'Blight is a common fungal or bacterial disease that spreads rapidly in humid conditions. Recommended actions:',
        tips: [
          'Immediately prune and destroy infected leaves. Do not compost them as spores can survive.',
          'Avoid overhead watering; water plants directly at the root to keep foliage dry.',
          'Improve air circulation by spacing plants properly and pruning lower leaves.',
          'Apply an appropriate copper-based fungicide or organic bio-fungicide early in the morning.'
        ]
      };
    }

    if (cls.includes('rust')) {
      return {
        summary: 'Rust is a fungal infection recognizable by orange, yellow, or brown spore-filled pustules. Recommended actions:',
        tips: [
          'Remove and dispose of all affected leaves and ground debris around the plant.',
          'Apply sulfur-based fungicides or neem oil early to prevent further spore propagation.',
          'Keep foliage dry and water only at the soil level.',
          'Ensure adequate spacing between plants to lower humidity levels.'
        ]
      };
    }

    if (cls.includes('spot') || cls.includes('scab')) {
      return {
        summary: 'Leaf spots/scabs are usually caused by bacteria or fungi that thrive on wet foliage. Recommended actions:',
        tips: [
          'Prune the lower, highly infected branches to limit splash-up infection from the soil.',
          'Avoid handling plants when they are wet to prevent spreading the pathogen.',
          'Sanitize gardening tools with rubbing alcohol between uses.',
          'Apply copper-based sprays or organic sulfur powders to protect uninfected leaves.'
        ]
      };
    }

    if (cls.includes('mildew')) {
      return {
        summary: 'Powdery mildew is a fungal disease that creates a white, powdery coating on leaf surfaces. Recommended actions:',
        tips: [
          'Place the plant in a sunny area; mildew thrives in damp, shaded conditions.',
          'Prune crowded areas to increase air circulation and reduce humidity.',
          'Spray leaves with a dilute mixture of milk and water (1:9 ratio) or use neem oil.',
          'Water in the morning so excess moisture evaporates during the day.'
        ]
      };
    }

    if (cls.includes('virus')) {
      return {
        summary: 'Viral infections (like Mosaic Virus) cannot be cured. The focus is to contain the spread. Recommended actions:',
        tips: [
          'Isolate or remove the infected plant entirely to protect neighboring healthy plants.',
          'Control insect vectors (like aphids or whiteflies) that spread the virus using insecticidal soap.',
          'Always wash hands and sanitize tools thoroughly after handling infected plants.',
          'Ensure future plantings use certified virus-free seeds and disease-resistant varieties.'
        ]
      };
    }

    if (cls.includes('mite') || cls.includes('insect') || cls.includes('pest')) {
      return {
        summary: 'Pests or spider mites detected. They suck plant sap, leading to yellowing and leaf loss. Recommended actions:',
        tips: [
          'Blast the underside of the leaves with a strong stream of water to dislodge pests.',
          'Apply organic insecticidal soap or neem oil spray, coating both tops and bottoms of leaves.',
          'Introduce natural predators like ladybugs or predatory mites to your garden.',
          'Keep the plant well-hydrated to reduce susceptibility to mite stress.'
        ]
      };
    }

    // Default warning advice
    return {
      summary: 'General plant anomaly detected. To safeguard your crop, follow these steps:',
      tips: [
        'Isolate the plant from other crops to prevent potential spread of disease.',
        'Trim and dispose of highly symptomatic leaves showing spots, decay, or discoloration.',
        'Check soil moisture levels to ensure the plant is neither underwatered nor waterlogged.',
        'Treat with a broad-spectrum organic fungicide or neem oil as a protective measure.'
      ]
    };
  };

  // Drag and drop handlers
  const handleDrag = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    setError(null);
    setResult(null);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('image/')) {
        setSelectedFile(file);
        setPreviewUrl(URL.createObjectURL(file));
      } else {
        setError('Please upload an image file (PNG, JPG, JPEG).');
      }
    }
  };

  // File input change handler
  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    setError(null);
    setResult(null);

    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (file.type.startsWith('image/')) {
        setSelectedFile(file);
        setPreviewUrl(URL.createObjectURL(file));
      } else {
        setError('Please select an image file (PNG, JPG, JPEG).');
      }
    }
  };

  const onButtonClick = () => {
    fileInputRef.current?.click();
  };

  // Clear states
  const clearSelection = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Submit and query FastAPI backend
  const analyzeImage = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Prediction failed. Please try again.');
      }

      const data: PredictionResult = await response.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'An error occurred while contacting the server.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <div className="title-container">
          <img src="/logo.jpg" alt="LeafShield AI Logo" className="app-logo" />
          <h1>LeafShield AI</h1>
        </div>
        <p>Instant Plant Disease Detection via Deep Learning Model. Upload a clear picture of a plant leaf to analyze its health status.</p>
      </header>

      <main className="glass-card">
        {/* Upload Section */}
        {!previewUrl ? (
          <div 
            className={`upload-zone ${dragActive ? 'active' : ''}`}
            onDragEnter={handleDrag}
            onDragOver={handleDrag}
            onDragLeave={handleDrag}
            onDrop={handleDrop}
            onClick={onButtonClick}
          >
            <input 
              type="file" 
              className="file-input" 
              ref={fileInputRef} 
              onChange={handleFileChange}
              accept="image/*"
            />
            <svg className="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
              <polyline points="17 8 12 3 7 8"/>
              <line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
            <div className="upload-text">
              <p>Drag and drop your leaf image here</p>
              <span>or click to browse from files</span>
            </div>
          </div>
        ) : (
          /* Preview Section */
          <div className="preview-container">
            <div className="image-wrapper">
              <img src={previewUrl} alt="Leaf Preview" className="preview-image" />
              <button className="change-image-btn" onClick={clearSelection} title="Remove image">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="18" y1="6" x2="6" y2="18"></line>
                  <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
              </button>
            </div>
            
            {!result && (
              <button 
                className="btn btn-primary" 
                onClick={analyzeImage} 
                disabled={isLoading}
              >
                {isLoading ? (
                  <>
                    <span className="spinner"></span>
                    Analyzing Leaf...
                  </>
                ) : (
                  'Analyze Health Status'
                )}
              </button>
            )}
          </div>
        )}

        {/* Error Message */}
        {error && (
          <div className="status-badge danger" style={{ width: '100%', borderRadius: '12px', justifyContent: 'center', padding: '12px' }}>
            <span>Error: {error}</span>
          </div>
        )}

        {/* Prediction Results Panel */}
        {result && (
          <div className="results-panel">
            <div className="results-header">
              <h2>Diagnostic Report</h2>
              <span className={`status-badge ${result.status === 'Healthy' ? 'healthy' : 'danger'}`}>
                {result.status === 'Healthy' ? (
                  <>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ marginRight: '4px' }}>
                      <polyline points="20 6 9 17 4 12"></polyline>
                    </svg>
                    Healthy
                  </>
                ) : (
                  <>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ marginRight: '4px' }}>
                      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                      <line x1="12" y1="9" x2="12" y2="13"></line>
                      <line x1="12" y1="17" x2="12.01" y2="17"></line>
                    </svg>
                    Disease Detected
                  </>
                )}
              </span>
            </div>

            <div>
              <p className="confidence-header" style={{ marginBottom: '4px', fontSize: '0.85rem' }}>IDENTIFIED CONDITION</p>
              <div className="disease-name">{result.disease}</div>
            </div>

            {/* Confidence Score Bar */}
            <div className="confidence-container">
              <div className="confidence-header">
                <span>Model Confidence</span>
                <span className="confidence-value">{(result.confidence * 100).toFixed(1)}%</span>
              </div>
              <div className="confidence-track">
                <div 
                  className="confidence-fill" 
                  style={{ width: `${result.confidence * 100}%` }}
                ></div>
              </div>
            </div>

            {/* Care / Action Recommendations */}
            <div className="advice-section">
              <h3>Recommended Action Plan</h3>
              <div className="advice-card">
                <p style={{ color: '#fff', fontWeight: '500', marginBottom: '12px', fontSize: '0.95rem' }}>
                  {getCareAdvice(result.raw_class, result.status).summary}
                </p>
                <ul>
                  {getCareAdvice(result.raw_class, result.status).tips.map((tip, idx) => (
                    <li key={idx}>{tip}</li>
                  ))}
                </ul>
              </div>
            </div>
            
            <button className="btn btn-primary" onClick={clearSelection} style={{ marginTop: '12px' }}>
              Analyze Another Leaf
            </button>
          </div>
        )}
      </main>
    </div>
  );
}
