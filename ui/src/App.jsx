import React, { useState, Suspense } from 'react'
import TypingTitle from './components/TypingTitle'
import PredictionForm from './components/PredictionForm'
import PredictionResult from './components/PredictionResult'
import FeatureChart from './components/FeatureChart'
import ExportPanel from './components/ExportPanel'
import Dashboard from './components/Dashboard'
const ReportViewer = React.lazy(() => import('./components/ReportViewer'))

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, error: null }
  }
  static getDerivedStateFromError(error) {
    return { hasError: true, error }
  }
  componentDidCatch(error, info) {
    console.error('UI render error:', error, info)
  }
  render() {
    if (this.state.hasError) {
      return (
        <div style={{padding:'1.5rem', background:'var(--card)', border:'1px solid var(--border)', borderRadius:12, margin:'1rem 0'}}>
          <div style={{color:'#f87171', fontWeight:600, marginBottom:'0.5rem'}}>⚠ Display error — results are still available</div>
          <div style={{color:'var(--sub)', fontSize:'0.8rem', fontFamily:'monospace', whiteSpace:'pre-wrap'}}>{this.state.error?.message}</div>
          <button onClick={()=>this.setState({hasError:false,error:null})} style={{marginTop:'1rem', padding:'0.4rem 1rem', background:'var(--surface)', border:'1px solid var(--border)', borderRadius:6, color:'var(--text)', cursor:'pointer', fontSize:'0.8rem'}}>Retry display</button>
        </div>
      )
    }
    return this.props.children
  }
}

export default function App(){
  const [prediction, setPrediction] = useState(null)
  const [featureData, setFeatureData] = useState(null)
  const [reportMarkdown, setReportMarkdown] = useState(null)

  return (
    <div className="min-h-screen">

      {/* Landing hero: put an image at /landing.jpg (ui/public/landing.jpg) */}
      <section className="hero min-h-screen relative flex items-center justify-center text-center">
        <div className="quad-grid">
          <div className="quad q-tl">
            <div className="hero-title"><TypingTitle /></div>
          </div>
          <div className="quad q-tr" />
          <div className="quad q-bl"><img src="/quad-bl.jpg" alt="" className="quad-img"/></div>
          <div className="quad q-br" />
        </div>
        <div className="hero-overlay" />
        <div className="hero-top-right">
          <div className="logo-container">
            <img src="/logo.svg" alt="Logo" className="hero-logo"/>
          </div>
        </div>

        <div className="hero-bottom-right">
          <button className="btn hero-explore" onClick={()=>{document.getElementById('app').scrollIntoView({behavior:'smooth'})}}>Explore</button>
        </div>

        <div className="hero-bottom-left">
          <div className="hero-subtitle">
            <span className="subtitle-line">privacy-first patient reports</span>
            <span className="subtitle-dot">·</span>
            <span className="subtitle-line">shap biomarkers</span>
            <span className="subtitle-dot">·</span>
            <span className="subtitle-line">llm-backed biomedical audit</span>
          </div>
        </div>
        
      </section>



      {/* Main app content */}
      <main id="app" className="main-app">
        <div className="app-container">

          {/* Dashboard section */}
          <div className="section-header">
            <div className="section-badge">Dashboard</div>
            <h2 className="section-heading">Training Results</h2>
            <p className="section-desc">Performance metrics across 12 TCGA cancer cohorts — XGBoost models trained on 500 top-variance genes from UCSC Xena expression data.</p>
          </div>

          <Dashboard />

          <div className="section-divider" />

          {/* Prediction section */}
          <div className="section-header">
            <div className="section-badge">Pipeline</div>
            <h2 className="section-heading">Run a Prediction</h2>
            <p className="section-desc">Upload gene expression data or paste a JSON sample to get cancer‑type predictions, SHAP feature importance, and an AI‑generated biomedical audit report.</p>
          </div>

          <div className="card p-6 fade-in">
            <h2 className="section-title">Input</h2>
            <PredictionForm onResult={(res)=>{ setPrediction(res.prediction); setFeatureData(res.features); setReportMarkdown(res.report) }} />
          </div>

          {prediction && (
            <ErrorBoundary>
              <div className="section-divider" />
              <div className="results-grid">
                <div className="space-y-6">
                  <div className="card p-6 fade-in prediction-result card-glow">
                    <div className="card-tag">Result</div>
                    <PredictionResult prediction={prediction} />
                  </div>

                  <div className="card p-6 fade-in card-glow">
                    <div className="card-tag">Features</div>
                    <h3 className="text-lg font-medium">Feature Importance</h3>
                    <div className="feature-chart">
                      <FeatureChart data={featureData} />
                    </div>
                  </div>
                </div>

                <div className="space-y-6">
                  <div className="card p-6 fade-in card-glow">
                    <div className="card-tag">AI Report</div>
                    <h3 className="text-lg font-medium">Biomedical Audit</h3>
                    <Suspense fallback={<div className="subtle">Loading report…</div>}>
                      <div className="report-viewer">
                        <ReportViewer markdown={reportMarkdown} />
                      </div>
                    </Suspense>
                  </div>
                  <div className="card p-6 fade-in">
                    <div className="card-tag">Export</div>
                    <ExportPanel data={{prediction, features: featureData, report: reportMarkdown}} />
                  </div>
                </div>
              </div>
            </ErrorBoundary>
          )}
        </div>
      </main>

      <footer className="site-footer">
        <div className="footer-inner">
          <div className="footer-brand">
            <img src="/logo.svg" alt="GUIDO" className="footer-logo" />
            <span className="footer-name">GUIDO</span>
          </div>
          <div className="footer-desc">
            Genomic Unified Intelligence for Disease Oncology —
            privacy-first cancer prediction powered by XGBoost &amp; SHAP biomarkers.
          </div>
          <div className="footer-links">
            <a href="https://github.com" target="_blank" rel="noreferrer">GitHub</a>
            <span className="footer-sep">·</span>
            <a href="#app">Run Prediction</a>
            <span className="footer-sep">·</span>
            <a href="#app">Dashboard</a>
          </div>
          <div className="footer-legal">
            For research purposes only — not a substitute for clinical diagnosis.
            &copy; {new Date().getFullYear()} GUIDO. All rights reserved.
          </div>
        </div>
      </footer>

    </div>
  )
}
