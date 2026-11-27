import React, { useState } from 'react'
import axios from 'axios'
import IntakeSpecModal from './IntakeSpecModal'

const API_BASE = import.meta.env.VITE_API_BASE || ''

const COHORT_OPTIONS = [
  { value: 'PAN-CANCER-ALL', label: '★ TCGA-PAN-CANCER-ALL (Master 33-Cohort Foundation Engine)', projectId: 'TCGA-PAN-CANCER-ALL' },
  { value: 'BRCA', label: 'Breast Invasive Carcinoma (BRCA)', projectId: 'TCGA-BRCA' },
  { value: 'LUAD', label: 'Lung Adenocarcinoma (LUAD)', projectId: 'TCGA-LUAD' },
  { value: 'LUSC', label: 'Lung Squamous Cell Carcinoma (LUSC)', projectId: 'TCGA-LUSC' },
  { value: 'KIRC', label: 'Kidney Renal Clear Cell Carcinoma (KIRC)', projectId: 'TCGA-KIRC' },
  { value: 'KIRP', label: 'Kidney Renal Papillary Cell Carcinoma (KIRP)', projectId: 'TCGA-KIRP' },
  { value: 'COAD', label: 'Colon Adenocarcinoma (COAD)', projectId: 'TCGA-COAD' },
  { value: 'STAD', label: 'Stomach Adenocarcinoma (STAD)', projectId: 'TCGA-STAD' },
  { value: 'LIHC', label: 'Liver Hepatocellular Carcinoma (LIHC)', projectId: 'TCGA-LIHC' },
  { value: 'GBM', label: 'Glioblastoma Multiforme (GBM)', projectId: 'TCGA-GBM' },
  { value: 'LGG', label: 'Brain Lower Grade Glioma (LGG)', projectId: 'TCGA-LGG' },
  { value: 'OV', label: 'Ovarian Serous Cystadenocarcinoma (OV)', projectId: 'TCGA-OV' },
  { value: 'UCEC', label: 'Uterine Corpus Endometrial Carcinoma (UCEC)', projectId: 'TCGA-UCEC' },
  { value: 'BLCA', label: 'Bladder Urothelial Carcinoma (BLCA)', projectId: 'TCGA-BLCA' },
  { value: 'PRAD', label: 'Prostate Adenocarcinoma (PRAD)', projectId: 'TCGA-PRAD' },
  { value: 'HNSC', label: 'Head and Neck Squamous Cell Carcinoma (HNSC)', projectId: 'TCGA-HNSC' },
  { value: 'THCA', label: 'Thyroid Carcinoma (THCA)', projectId: 'TCGA-THCA' },
  { value: 'SARC', label: 'Sarcoma (SARC)', projectId: 'TCGA-SARC' },
  { value: 'SKCM', label: 'Skin Cutaneous Melanoma (SKCM)', projectId: 'TCGA-SKCM' },
  { value: 'LAML', label: 'Acute Myeloid Leukemia (LAML)', projectId: 'TCGA-LAML' },
]

export default function PredictionForm({ onResult }) {
  const [selectedCohort, setSelectedCohort] = useState('LUAD')
  const [fileMode, setFileMode] = useState('synthetic') // 'synthetic' | 'own'
  const [testRiskProfile, setTestRiskProfile] = useState('positive') // 'positive' | 'negative'
  const [csvFile, setCsvFile] = useState(null)
  const [jsonPayload, setJsonPayload] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [showSpecModal, setShowSpecModal] = useState(false)

  const currentCohort = COHORT_OPTIONS.find(c => c.value === selectedCohort) || COHORT_OPTIONS[1]
  const posFileName = `${selectedCohort}-positive.csv`
  const negFileName = `${selectedCohort}-negative.csv`
  const activeTestFile = testRiskProfile === 'positive' ? posFileName : negFileName
  const activeTestUrl = `/samples/${activeTestFile}`

  async function handleSubmit(e) {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      const formData = new FormData()
      formData.append('project_id', currentCohort.projectId)
      formData.append('disease_name', selectedCohort.toLowerCase())

      if (fileMode === 'own') {
        if (csvFile) {
          formData.append('file', csvFile)
        } else if (jsonPayload) {
          formData.append('json_payload', jsonPayload)
        } else {
          throw new Error('Please select a CSV file or paste gene expression JSON.')
        }
      } else {
        // Fetch the calibrated synthetic benchmark file
        let resp
        try {
          resp = await fetch(activeTestUrl)
          if (!resp.ok) {
            // Fallback to root if samples subfolder doesn't resolve
            resp = await fetch(`/${activeTestFile}`)
          }
        } catch {
          resp = await fetch(`/${activeTestFile}`)
        }

        if (!resp || !resp.ok) {
          throw new Error(`Synthetic test file ${activeTestFile} could not be loaded.`)
        }
        const blob = await resp.blob()
        const file = new File([blob], activeTestFile, { type: 'text/csv' })
        formData.append('file', file)
      }

      const apiResp = await axios.post(`${API_BASE}/api/ui/predict`, formData)
      onResult(apiResp.data)
    } catch (err) {
      const detail = err?.response?.data?.detail || err?.response?.data?.message || err?.message || 'Unknown error occurred during prediction.'
      setError(`Prediction pipeline error: ${detail}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-4">
      {/* Intake Specification Modal */}
      <IntakeSpecModal isOpen={showSpecModal} onClose={() => setShowSpecModal(false)} />

      {/* Clinical Guidance Banner with Spec PDF Link */}
      <div style={{
        background: 'rgba(90, 106, 83, 0.08)',
        border: '1px solid rgba(90, 106, 83, 0.25)',
        borderRadius: 10,
        padding: '1.25rem',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: 12,
      }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--green, #5A6A53)', textTransform: 'uppercase', letterSpacing: 1 }}>
              Clinical Intake Standard & Parameters Guide
            </span>
            <span style={{ fontSize: 10, padding: '2px 6px', borderRadius: 4, background: 'rgba(90,106,83,0.2)', color: '#AABAA3' }}>
              HGNC / log₂(TPM+1)
            </span>
          </div>
          <p style={{ fontSize: 12, color: 'var(--sub, #888)', margin: '4px 0 0', maxWidth: 540 }}>
            Review sample quality requirements (DV200 ≥ 30%), gene normalization protocols, and multi-omics formatting specifications before running clinical patient data.
          </p>
        </div>
        <button
          type="button"
          onClick={() => setShowSpecModal(true)}
          style={{
            background: 'rgba(90, 106, 83, 0.2)',
            border: '1px solid rgba(90, 106, 83, 0.4)',
            color: '#F2EAE6',
            borderRadius: 8,
            padding: '0.5rem 1rem',
            fontSize: 12,
            fontWeight: 600,
            cursor: 'pointer',
            display: 'inline-flex',
            alignItems: 'center',
            gap: 6,
            transition: 'all 0.15s ease',
          }}
        >
          📄 View / Download Intake Spec (PDF)
        </button>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Step 1: Select Cancer Cohort */}
        <div>
          <label className="text-sm subtle" style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span>1. Cancer Type & Organ Cohort</span>
            <span style={{ color: 'var(--green, #5A6A53)', fontWeight: 600 }}>Active Model: {currentCohort.projectId}</span>
          </label>
          <select
            className="w-full p-2.5 rounded bg-[var(--surface)] border border-[var(--border)] text-[var(--text)]"
            value={selectedCohort}
            onChange={e => setSelectedCohort(e.target.value)}
            style={{ fontSize: 13 }}
          >
            {COHORT_OPTIONS.map(opt => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
        </div>

        {/* Step 2: Intake Method Selector */}
        <div>
          <label className="text-sm subtle">2. Data Intake Method</label>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
            <button
              type="button"
              onClick={() => setFileMode('synthetic')}
              style={{
                padding: '0.75rem',
                borderRadius: 8,
                textAlign: 'left',
                background: fileMode === 'synthetic' ? 'rgba(90, 106, 83, 0.2)' : 'var(--surface)',
                border: `1px solid ${fileMode === 'synthetic' ? 'var(--green, #5A6A53)' : 'var(--border)'}`,
                cursor: 'pointer',
                color: 'var(--text)',
              }}
            >
              <div style={{ fontSize: 13, fontWeight: 700, color: fileMode === 'synthetic' ? '#F2EAE6' : '#CCC' }}>
                ⚡ Test with Calibrated Benchmark File
              </div>
              <div style={{ fontSize: 11, color: 'var(--sub, #888)', marginTop: 2 }}>
                Pre-calibrated {selectedCohort} synthetic positive & negative risk profiles
              </div>
            </button>

            <button
              type="button"
              onClick={() => setFileMode('own')}
              style={{
                padding: '0.75rem',
                borderRadius: 8,
                textAlign: 'left',
                background: fileMode === 'own' ? 'rgba(90, 106, 83, 0.2)' : 'var(--surface)',
                border: `1px solid ${fileMode === 'own' ? 'var(--green, #5A6A53)' : 'var(--border)'}`,
                cursor: 'pointer',
                color: 'var(--text)',
              }}
            >
              <div style={{ fontSize: 13, fontWeight: 700, color: fileMode === 'own' ? '#F2EAE6' : '#CCC' }}>
                📁 Upload My Own Patient Data
              </div>
              <div style={{ fontSize: 11, color: 'var(--sub, #888)', marginTop: 2 }}>
                Custom RNA-seq CSV normalized to log₂(TPM + 1)
              </div>
            </button>
          </div>
        </div>

        {/* Dynamic Panel: Synthetic Benchmark File Selector */}
        {fileMode === 'synthetic' ? (
          <div style={{
            background: 'var(--surface, #252525)',
            border: '1px solid var(--border, #2E2E2E)',
            borderRadius: 10,
            padding: '1.25rem',
          }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: '#AABAA3', marginBottom: 10 }}>
              Select Calibrated Test Profile for <span style={{ color: '#F2EAE6' }}>{selectedCohort}</span>:
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              {/* Positive Risk Profile */}
              <div
                onClick={() => setTestRiskProfile('positive')}
                style={{
                  padding: '1rem',
                  borderRadius: 8,
                  background: testRiskProfile === 'positive' ? 'rgba(184, 74, 74, 0.15)' : 'rgba(255,255,255,0.02)',
                  border: `1px solid ${testRiskProfile === 'positive' ? 'rgba(184, 74, 74, 0.5)' : 'rgba(255,255,255,0.05)'}`,
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: 11, fontWeight: 700, color: '#e06c75', textTransform: 'uppercase' }}>
                    ● High Progression Risk (&gt;65%)
                  </span>
                  <span style={{ fontSize: 10, color: 'var(--sub, #888)', fontFamily: 'monospace' }}>.CSV</span>
                </div>
                <div style={{ fontSize: 13, fontWeight: 600, color: '#EDEDED', marginTop: 4 }}>
                  {posFileName}
                </div>
                <p style={{ fontSize: 11, color: 'var(--sub, #888)', margin: '4px 0 0' }}>
                  Elevated Proliferation, EMT Invasion, Hypoxia, and Glycolysis metagenes.
                </p>
                <div style={{ marginTop: 8 }}>
                  <a
                    href={`/samples/${posFileName}`}
                    download
                    onClick={e => e.stopPropagation()}
                    style={{ fontSize: 11, color: '#AABAA3', textDecoration: 'underline' }}
                  >
                    ⬇ Download {posFileName}
                  </a>
                </div>
              </div>

              {/* Negative Risk Profile */}
              <div
                onClick={() => setTestRiskProfile('negative')}
                style={{
                  padding: '1rem',
                  borderRadius: 8,
                  background: testRiskProfile === 'negative' ? 'rgba(90, 106, 83, 0.2)' : 'rgba(255,255,255,0.02)',
                  border: `1px solid ${testRiskProfile === 'negative' ? 'rgba(90, 106, 83, 0.6)' : 'rgba(255,255,255,0.05)'}`,
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: 11, fontWeight: 700, color: '#7A9A78', textTransform: 'uppercase' }}>
                    ● Favorable / Low Risk (&lt;35%)
                  </span>
                  <span style={{ fontSize: 10, color: 'var(--sub, #888)', fontFamily: 'monospace' }}>.CSV</span>
                </div>
                <div style={{ fontSize: 13, fontWeight: 600, color: '#EDEDED', marginTop: 4 }}>
                  {negFileName}
                </div>
                <p style={{ fontSize: 11, color: 'var(--sub, #888)', margin: '4px 0 0' }}>
                  Suppressed oncogenic drivers; intact DNA damage repair and differentiation.
                </p>
                <div style={{ marginTop: 8 }}>
                  <a
                    href={`/samples/${negFileName}`}
                    download
                    onClick={e => e.stopPropagation()}
                    style={{ fontSize: 11, color: '#AABAA3', textDecoration: 'underline' }}
                  >
                    ⬇ Download {negFileName}
                  </a>
                </div>
              </div>
            </div>

            <div style={{ fontSize: 11, color: 'var(--sub, #888)', marginTop: 10, display: 'flex', alignItems: 'center', gap: 6 }}>
              <span>ℹ</span>
              <span>Loaded file: <strong style={{ color: '#F2EAE6' }}>{activeTestFile}</strong> · Evaluated with the 70/30 Dual-Consensus Engine.</span>
            </div>
          </div>
        ) : (
          /* Dynamic Panel: Upload Own File */
          <div style={{
            background: 'var(--surface, #252525)',
            border: '1px solid var(--border, #2E2E2E)',
            borderRadius: 10,
            padding: '1.25rem',
            display: 'flex',
            flexDirection: 'column',
            gap: 12,
          }}>
            <div>
              <label className="text-sm subtle">Upload CSV File (Single patient row or samples × features)</label>
              <div className="upload-zone mt-1" style={{ border: '1px dashed var(--border)', padding: '1rem', borderRadius: 8, textAlign: 'center' }}>
                <input
                  type="file"
                  accept=".csv,.tsv,.txt"
                  onChange={e => setCsvFile(e.target.files[0])}
                  style={{ fontSize: 12 }}
                />
              </div>
              {csvFile && (
                <div style={{ fontSize: 11, color: '#AABAA3', marginTop: 4 }}>
                  ✓ Selected file: <strong>{csvFile.name}</strong> ({(csvFile.size / 1024).toFixed(1)} KB)
                </div>
              )}
            </div>

            <div>
              <label className="text-sm subtle">Or Paste JSON Patient Profile</label>
              <textarea
                rows={3}
                className="w-full p-2.5 rounded bg-[#121212] border border-[var(--border)] text-[var(--text)] font-mono text-xs"
                placeholder='{"MKI67": 10.4, "TP53": 8.2, "VEGFA": 9.1, "BRCA1": 5.4}'
                value={jsonPayload}
                onChange={e => setJsonPayload(e.target.value)}
              />
            </div>
          </div>
        )}

        {/* Submit Execution */}
        <div className="flex items-center gap-3 pt-2">
          <button
            type="submit"
            className="submit-btn"
            disabled={loading}
            style={{
              background: 'var(--green, #5A6A53)',
              color: '#F2EAE6',
              fontWeight: 700,
              fontSize: 13,
              letterSpacing: 0.5,
              padding: '0.7rem 1.8rem',
              borderRadius: 8,
              border: 'none',
              cursor: loading ? 'not-allowed' : 'pointer',
              boxShadow: '0 4px 14px rgba(90,106,83,0.3)',
            }}
          >
            {loading ? 'Executing Dual-Consensus Engine…' : `Run Prediction (${selectedCohort})`}
          </button>
          {loading && <div className="subtle" style={{ fontSize: 12 }}>Extracting 12 Hallmark Metagenes & running consensus inference…</div>}
        </div>

        {error && (
          <div style={{
            background: 'rgba(232, 112, 112, 0.1)',
            border: '1px solid rgba(232, 112, 112, 0.3)',
            borderRadius: 8,
            padding: '0.75rem',
            color: '#f87171',
            fontSize: 12,
          }}>
            {error}
          </div>
        )}
      </form>
    </div>
  )
}
