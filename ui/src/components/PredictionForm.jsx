import React, { useState } from 'react'
import axios from 'axios'

// Empty string = relative URLs (/api/…) — works in Docker where API and UI share the same origin.
// Vite's dev-server proxy handles /api/* → http://127.0.0.1:8000 during local development.
const API_BASE = import.meta.env.VITE_API_BASE || ''

const diseaseOptions = [
  { value: 'breast_invasive_carcinoma', label: 'Breast Invasive Carcinoma' },
  { value: 'colon_adenocarcinoma', label: 'Colon Adenocarcinoma' },
  { value: 'glioblastoma_multiforme', label: 'Glioblastoma Multiforme' },
  { value: 'kidney_chromophobe', label: 'Kidney Chromophobe' },
  { value: 'kidney_renal_clear_cell_carcinoma', label: 'Kidney Renal Clear Cell Carcinoma' },
  { value: 'kidney_renal_papillary_cell_carcinoma', label: 'Kidney Renal Papillary Cell Carcinoma' },
  { value: 'brain_lower_grade_glioma', label: 'Brain Lower Grade Glioma' },
  { value: 'liver_hepatocellular_carcinoma', label: 'Liver Hepatocellular Carcinoma' },
  { value: 'lung_adenocarcinoma', label: 'Lung Adenocarcinoma' },
  { value: 'lung_squamous_cell_carcinoma', label: 'Lung Squamous Cell Carcinoma' },
  { value: 'pancreatic_adenocarcinoma', label: 'Pancreatic Adenocarcinoma' },
  { value: 'prostate_adenocarcinoma', label: 'Prostate Adenocarcinoma' },
  { value: 'skin_cutaneous_melanoma', label: 'Skin Cutaneous Melanoma' }
]

const SAMPLE_FILES = [
  {
    value: 'positive',
    label: 'Sample — Positive prediction (LUAD)',
    url: '/sample_positive_LUAD.csv',
    filename: 'sample_positive_LUAD.csv',
    badge: 'POSITIVE',
    badgeColor: 'text-[var(--green)]',
    note: 'Synthetic TCGA-LUAD tumor profile · select Lung Adenocarcinoma',
  },
  {
    value: 'negative',
    label: 'Sample — Negative prediction (LUAD)',
    url: '/sample_negative_LUAD.csv',
    filename: 'sample_negative_LUAD.csv',
    badge: 'NEGATIVE',
    badgeColor: 'text-[var(--sub)]',
    note: 'Synthetic TCGA-LUAD normal profile · select Lung Adenocarcinoma',
  },
]

export default function PredictionForm({onResult}){
  const [csv, setCsv] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [projectId, setProjectId] = useState('TCGA-LUAD')
  const [disease, setDisease] = useState('lung_adenocarcinoma')
  const [jsonPayload, setJsonPayload] = useState('')
  const [fileSource, setFileSource] = useState('own')   // 'own' | 'positive' | 'negative'

  const handleDiseaseChange = (e) => {
    const val = e.target.value;
    setDisease(val);
    const mapping = {
      'breast_invasive_carcinoma': 'TCGA-BRCA',
      'colon_adenocarcinoma': 'TCGA-COAD',
      'glioblastoma_multiforme': 'TCGA-GBM',
      'kidney_chromophobe': 'TCGA-KICH',
      'kidney_renal_clear_cell_carcinoma': 'TCGA-KIRC',
      'kidney_renal_papillary_cell_carcinoma': 'TCGA-KIRP',
      'brain_lower_grade_glioma': 'TCGA-LGG',
      'liver_hepatocellular_carcinoma': 'TCGA-LIHC',
      'lung_adenocarcinoma': 'TCGA-LUAD',
      'lung_squamous_cell_carcinoma': 'TCGA-LUSC',
      'pancreatic_adenocarcinoma': 'TCGA-PAAD',
      'prostate_adenocarcinoma': 'TCGA-PRAD',
      'skin_cutaneous_melanoma': 'TCGA-SKCM'
    };
    if (mapping[val]) setProjectId(mapping[val]);
  };

  async function handleSubmit(e){
    e.preventDefault()
    setLoading(true); setError(null)
    try{
      const formData = new FormData()
      formData.append('project_id', projectId)
      formData.append('disease_name', disease)

      if (fileSource === 'own') {
        if (csv) formData.append('file', csv)
        if (!csv && jsonPayload) formData.append('json_payload', jsonPayload)
      } else {
        const sample = SAMPLE_FILES.find(s => s.value === fileSource)
        if (sample) {
          const resp = await fetch(sample.url)
          const blob = await resp.blob()
          const file = new File([blob], sample.filename, { type: 'text/csv' })
          formData.append('file', file)
        }
      }

      const resp = await axios.post(`${API_BASE}/api/ui/predict`, formData)
      onResult(resp.data)
    }catch(err){
      const detail = err?.response?.data?.detail || err?.response?.data?.message || err?.message || 'Unknown error'
      setError(`Prediction failed: ${detail}`)
    }finally{ setLoading(false) }
  }

  const selectedSample = SAMPLE_FILES.find(s => s.value === fileSource)

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="mb-6 p-4 rounded-lg bg-[rgba(90,106,83,0.05)] border border-[rgba(90,106,83,0.2)]">
        <h4 className="text-sm font-semibold text-[var(--green)] mb-3 uppercase tracking-wider">How to use your own clinical data</h4>
        <ul className="text-sm text-[var(--sub)] space-y-2 list-disc pl-4">
          <li><strong>1. Tissue Biopsy:</strong> Obtain a tumor tissue sample via biopsy or surgical resection.</li>
          <li><strong>2. RNA Sequencing:</strong> Send the sample to a clinical genomics lab (e.g., Illumina, Foundation Medicine) for bulk RNA-Sequencing.</li>
          <li><strong>3. Data Processing:</strong> Request raw gene expression counts and normalize using <code className="text-[var(--text)] bg-[rgba(255,255,255,0.05)] px-1 rounded">log2(count + 1)</code>.</li>
          <li><strong>4. Formatting:</strong> Format as a CSV where the header row contains gene names and values are normalized expression levels.</li>
        </ul>
      </div>

      <div className="grid grid-cols-1 gap-3">
        <div>
          <label className="text-sm subtle">Disease</label>
          <select className="w-full p-2 rounded bg-[var(--surface)] border border-[var(--border)] text-[var(--text)]" value={disease} onChange={handleDiseaseChange}>
            {diseaseOptions.map(opt => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
        </div>

        <div>
          <label className="text-sm subtle">File source</label>
          <select
            className="w-full p-2 rounded bg-[var(--surface)] border border-[var(--border)] text-[var(--text)]"
            value={fileSource}
            onChange={e => { setFileSource(e.target.value); setCsv(null) }}
          >
            <option value="own">Upload my own file</option>
            <optgroup label="── Sample files ──">
              {SAMPLE_FILES.map(s => (
                <option key={s.value} value={s.value}>{s.label}</option>
              ))}
            </optgroup>
          </select>
        </div>

        {fileSource === 'own' ? (
          <>
            <div>
              <label className="text-sm subtle">Upload CSV (samples × features)</label>
              <div className="upload-zone mt-1">
                <input type="file" accept="text/csv" onChange={e => setCsv(e.target.files[0])} />
              </div>
            </div>
            <div>
              <label className="text-sm subtle">Or paste JSON sample (object or array)</label>
              <textarea rows={4} className="w-full p-2 mt-2 rounded" placeholder='{"GENE1":1.2,"GENE2":0.4}' value={jsonPayload} onChange={e=>setJsonPayload(e.target.value)} />
            </div>
          </>
        ) : (
          <div className="p-3 rounded-lg bg-[rgba(255,255,255,0.03)] border border-[var(--border)] flex items-start gap-3">
            <div className="mt-0.5 text-lg">🧬</div>
            <div>
              <p className="text-sm font-medium text-[var(--text)]">{selectedSample?.filename}</p>
              <p className="text-xs mt-0.5">
                Expected result: <span className={`font-semibold ${selectedSample?.badgeColor}`}>{selectedSample?.badge}</span>
              </p>
              <p className="text-[11px] text-[var(--sub)] mt-1">{selectedSample?.note}</p>
            </div>
          </div>
        )}
      </div>

      <div className="flex items-center gap-3">
        <button type="submit" className="submit-btn" disabled={loading}>
          {loading ? 'Processing…' : 'Submit'}
        </button>
        {loading && <div className="subtle">This may take a moment…</div>}
      </div>
      {error && <div className="text-sm text-red-400">{error}</div>}
    </form>
  )
}
