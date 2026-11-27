import React from 'react'

export default function IntakeSpecModal({ isOpen, onClose }) {
  if (!isOpen) return null

  const handlePrint = () => {
    window.print()
  }

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 9999,
        background: 'rgba(0, 0, 0, 0.85)',
        backdropFilter: 'blur(8px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '1rem',
      }}
      onClick={onClose}
    >
      <div
        style={{
          background: 'var(--card, #1E1E1E)',
          border: '1px solid var(--border, #2E2E2E)',
          borderRadius: 14,
          maxWidth: '780px',
          width: '100%',
          maxHeight: '90vh',
          overflowY: 'auto',
          padding: '2rem',
          boxShadow: '0 20px 60px rgba(0,0,0,0.6)',
          color: 'var(--text, #EDEDED)',
          position: 'relative',
        }}
        onClick={e => e.stopPropagation()}
      >
        {/* Header Bar */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', borderBottom: '1px solid var(--border, #2E2E2E)', paddingBottom: '1rem', marginBottom: '1.5rem' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontFamily: "'Negan', Inter, sans-serif", fontSize: 24, fontWeight: 800, color: 'var(--green, #5A6A53)', letterSpacing: 1 }}>
                GUIDO
              </span>
              <span style={{ fontSize: 11, padding: '2px 8px', borderRadius: 99, background: 'rgba(90,106,83,0.2)', color: '#AABAA3', border: '1px solid rgba(90,106,83,0.3)', fontWeight: 600 }}>
                SPECIFICATION v2.4
              </span>
            </div>
            <h3 style={{ fontSize: 16, fontWeight: 700, margin: '6px 0 2px', color: '#F2EAE6' }}>
              Clinical RNA-Seq & Multi-Omics Data Intake Specification
            </h3>
            <p style={{ fontSize: 11, color: 'var(--sub, #888)', margin: 0 }}>
              Hospital-Standard Reference Guide for Molecular Oncology Decision Support
            </p>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button
              onClick={handlePrint}
              style={{
                background: 'var(--surface, #252525)',
                border: '1px solid var(--border, #2E2E2E)',
                borderRadius: 6,
                padding: '0.4rem 0.8rem',
                color: '#EDEDED',
                fontSize: 11,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: 4,
              }}
            >
              🖨 Print / Save PDF
            </button>
            <button
              onClick={onClose}
              style={{
                background: 'transparent',
                border: 'none',
                color: 'var(--sub, #888)',
                fontSize: 18,
                cursor: 'pointer',
                padding: '0 4px',
              }}
            >
              ✕
            </button>
          </div>
        </div>

        {/* Section 1: Specimen Quality Requirements */}
        <div style={{ marginBottom: '1.5rem' }}>
          <h4 style={{ fontSize: 13, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1, color: '#AABAA3', marginBottom: 8 }}>
            1. Pre-Analytical Specimen & Quality Criteria
          </h4>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.75rem', fontSize: 12 }}>
            <div style={{ background: 'rgba(255,255,255,0.02)', padding: '0.75rem', borderRadius: 8, border: '1px solid rgba(255,255,255,0.04)' }}>
              <div style={{ fontWeight: 600, color: '#F2EAE6', marginBottom: 2 }}>Neoplastic Cellularity</div>
              <div style={{ color: 'var(--sub, #888)' }}>Minimum ≥ 20% viable tumor cell content verified by board-certified pathologist on H&E frozen section or FFPE block.</div>
            </div>
            <div style={{ background: 'rgba(255,255,255,0.02)', padding: '0.75rem', borderRadius: 8, border: '1px solid rgba(255,255,255,0.04)' }}>
              <div style={{ fontWeight: 600, color: '#F2EAE6', marginBottom: 2 }}>RNA Integrity & Quality Gate</div>
              <div style={{ color: 'var(--sub, #888)' }}>DV200 ≥ 30% for FFPE specimens, or RIN ≥ 5.0 for fresh-frozen tissue. Samples with DV200 &lt; 30% are flagged as degraded.</div>
            </div>
            <div style={{ background: 'rgba(255,255,255,0.02)', padding: '0.75rem', borderRadius: 8, border: '1px solid rgba(255,255,255,0.04)' }}>
              <div style={{ fontWeight: 600, color: '#F2EAE6', marginBottom: 2 }}>Sequencing Depth</div>
              <div style={{ color: 'var(--sub, #888)' }}>Minimum ≥ 20 million paired-end reads (Illumina NovaSeq / NextSeq, poly-A selected or rRNA-depleted library).</div>
            </div>
            <div style={{ background: 'rgba(255,255,255,0.02)', padding: '0.75rem', borderRadius: 8, border: '1px solid rgba(255,255,255,0.04)' }}>
              <div style={{ fontWeight: 600, color: '#F2EAE6', marginBottom: 2 }}>Data Completeness Gate</div>
              <div style={{ color: 'var(--sub, #888)' }}>Zero-fraction &lt; 60% across the 130 core features. Missing values are imputed with cohort-calibrated median expressions.</div>
            </div>
          </div>
        </div>

        {/* Section 2: Data Normalization & Formatting */}
        <div style={{ marginBottom: '1.5rem' }}>
          <h4 style={{ fontSize: 13, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1, color: '#AABAA3', marginBottom: 8 }}>
            2. Normalization & Gene Naming Standard
          </h4>
          <div style={{ fontSize: 12, lineHeight: 1.6, color: '#CCCCCC', background: 'rgba(255,255,255,0.02)', padding: '0.85rem', borderRadius: 8, border: '1px solid rgba(255,255,255,0.04)' }}>
            <p style={{ margin: '0 0 8px' }}>
              <strong>Gene Identifiers:</strong> Must use official HUGO Gene Nomenclature Committee (HGNC) approved gene symbols (e.g., <code style={{ color: '#F2EAE6', background: 'rgba(255,255,255,0.08)', padding: '1px 5px', borderRadius: 4 }}>MKI67</code>, <code style={{ color: '#F2EAE6', background: 'rgba(255,255,255,0.08)', padding: '1px 5px', borderRadius: 4 }}>TP53</code>, <code style={{ color: '#F2EAE6', background: 'rgba(255,255,255,0.08)', padding: '1px 5px', borderRadius: 4 }}>EGFR</code>, <code style={{ color: '#F2EAE6', background: 'rgba(255,255,255,0.08)', padding: '1px 5px', borderRadius: 4 }}>VEGFA</code>). Ensembl IDs (ENSG) should be mapped prior to upload.
            </p>
            <p style={{ margin: '0 0 8px' }}>
              <strong>Normalization:</strong> Raw RNA-Seq fragment counts must be converted to Transcripts Per Million (TPM) or FPKM-UQ and transformed via <code style={{ color: '#AABAA3', background: 'rgba(90,106,83,0.2)', padding: '2px 6px', borderRadius: 4, fontFamily: 'monospace' }}>log2(TPM + 1)</code>.
            </p>
            <p style={{ margin: 0 }}>
              <strong>DNA Methylation:</strong> Optional beta-values ($0.00$ to $1.00$) from Illumina HumanMethylation450/EPIC arrays with prefix <code style={{ color: '#C99E52', background: 'rgba(201,158,82,0.15)', padding: '1px 5px', borderRadius: 4, fontFamily: 'monospace' }}>METH_cg...</code>. If absent, the engine computes baseline methylation from cohort averages.
            </p>
          </div>
        </div>

        {/* Section 3: File Format Specification */}
        <div style={{ marginBottom: '1.5rem' }}>
          <h4 style={{ fontSize: 13, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1, color: '#AABAA3', marginBottom: 8 }}>
            3. File Format & Example CSV Structure
          </h4>
          <div style={{ background: '#121212', border: '1px solid var(--border, #2E2E2E)', borderRadius: 8, padding: '0.85rem', fontFamily: 'monospace', fontSize: 11, color: '#AABAA3', overflowX: 'auto' }}>
            <div>MKI67,AURKA,BIRC5,TOP2A,VIM,CDH2,SNAI1,VEGFA,HIF1A,PKM,PDCD1,SOX2,BRCA1,BCL2</div>
            <div style={{ color: '#F2EAE6', marginTop: 4 }}>10.42,9.81,8.95,11.20,12.35,8.12,7.40,9.15,10.22,11.80,6.54,8.10,5.20,9.45</div>
          </div>
          <div style={{ fontSize: 11, color: 'var(--sub, #888)', marginTop: 6 }}>
            • Headers must contain gene symbols. A single row containing numerical expression values represents one patient sample.
          </div>
        </div>

        {/* Section 4: 12 Hallmark Metagenes */}
        <div style={{ marginBottom: '1.5rem' }}>
          <h4 style={{ fontSize: 13, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1, color: '#AABAA3', marginBottom: 8 }}>
            4. Dynamic Hallmark Metagene Extraction
          </h4>
          <div style={{ fontSize: 11, color: '#CCCCCC', lineHeight: 1.5 }}>
            You do not need to pre-compute the 12 Hallmark Oncology Metagenes. The GUIDO clinical pipeline automatically extracts and z-score standardizes all 12 pathways (Proliferation, EMT Invasion, Angiogenesis, Hypoxia, Glycolysis, Immune Exhaustion, Stemness, DNA Repair, Apoptosis Evasion, Inflammation, ECM Remodeling, and Oxidative Stress) at runtime.
          </div>
        </div>

        {/* Regulatory RUO Banner */}
        <div style={{ borderTop: '1px solid var(--border, #2E2E2E)', paddingTop: '1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ fontSize: 10, color: 'var(--sub, #888)', maxWidth: 540 }}>
            <strong>Regulatory Status:</strong> Research Use Only (RUO). Investigational decision-support software. Not cleared by US FDA, EMA, or CDSCO for primary diagnostic or independent treatment prescribing.
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'var(--green, #5A6A53)',
              color: '#F2EAE6',
              border: 'none',
              borderRadius: 6,
              padding: '0.5rem 1.25rem',
              fontSize: 12,
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            Close Guide
          </button>
        </div>
      </div>
    </div>
  )
}
