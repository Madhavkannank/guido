import React, { useState } from 'react'

export default function PredictionResult({ prediction }) {
  const [activeTab, setActiveTab] = useState('summary')

  const prob = prediction.prob !== undefined ? prediction.prob : 0.5
  const pct = Math.round(prob * 100)
  const tier = prediction.risk_tier || (pct < 35 ? 'Low Risk' : pct < 65 ? 'Moderate Risk' : 'High Risk')
  const horizon = prediction.time_horizon || '3-Year Progression & Recurrence Risk'
  const qc = prediction.sample_qc || null

  const pOrgan = prediction.organ_risk_score !== undefined ? prediction.organ_risk_score : null
  const pPancan = prediction.pancan_risk_score !== undefined ? prediction.pancan_risk_score : null
  const consensusMode = prediction.consensus_mode || 'Dual-Consensus (70% Organ Specialist / 30% Pan-Cancer Master)'
  const concordance = prediction.model_concordance || 'Concordant'

  const tierColors = {
    'Low Risk': { text: '#34d399', bg: 'rgba(52, 211, 153, 0.12)', border: 'rgba(52, 211, 153, 0.35)', bar: '#34d399' },
    'Moderate Risk': { text: '#fbbf24', bg: 'rgba(251, 191, 36, 0.12)', border: 'rgba(251, 191, 36, 0.35)', bar: '#fbbf24' },
    'High Risk': { text: '#f87171', bg: 'rgba(248, 113, 113, 0.15)', border: 'rgba(248, 113, 113, 0.4)', bar: '#f87171' },
  }

  const currentTier = tierColors[tier] || tierColors['Moderate Risk']

  // Strictly observed clinical interpretation based on model metrics
  const clinicalObservation = tier === 'Low Risk'
    ? 'Genomic and hallmark metagene profile demonstrates suppressed proliferation and angiogenic signaling, consistent with a favorable progression-free trajectory. In validation cohorts, patients in this tier exhibited lower longitudinal recurrence hazards.'
    : tier === 'Moderate Risk'
    ? 'Genomic profile reflects intermediate oncogenic pathway activation. Balanced activation of repair and proliferative metagenes indicates potential risk of progression requiring correlated clinical staging and close surveillance.'
    : 'Genomic profile demonstrates significant elevation in aggressive cancer hallmarks (e.g. sustained proliferation, EMT invasion, hypoxia signaling). In empirical cohorts, this profile is associated with elevated 3-year recurrence hazard.'

  return (
    <div className="risk-scorecard" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      {/* Hospital Report Header Bar */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: 8,
        paddingBottom: '0.75rem',
        borderBottom: '1px solid var(--border)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{
            fontSize: 10,
            fontWeight: 800,
            letterSpacing: 1.5,
            padding: '2px 8px',
            borderRadius: 4,
            background: 'rgba(201, 158, 82, 0.15)',
            color: '#E8CA8B',
            border: '1px solid rgba(201, 158, 82, 0.35)',
            textTransform: 'uppercase',
          }}>
            RESEARCH USE ONLY (RUO)
          </span>
          {qc && (
            <span style={{
              fontSize: 10,
              fontWeight: 600,
              padding: '2px 8px',
              borderRadius: 4,
              background: qc.status === 'PASS' ? 'rgba(52, 211, 153, 0.12)' : 'rgba(251, 191, 36, 0.12)',
              color: qc.status === 'PASS' ? '#34d399' : '#fbbf24',
              border: `1px solid ${qc.status === 'PASS' ? 'rgba(52, 211, 153, 0.3)' : 'rgba(251, 191, 36, 0.3)'}`,
            }}>
              Pre-Analytical QC: {qc.status} ({qc.quality_score}/100)
            </span>
          )}
        </div>
        <div style={{ fontSize: 11, color: 'var(--sub, #888)', fontFamily: 'monospace' }}>
          {prediction.model || 'GUIDO Multi-Omics Engine'}
        </div>
      </div>

      {/* QC Warning Banner if present */}
      {qc && qc.warnings && qc.warnings.length > 0 && (
        <div style={{
          padding: '0.5rem 0.75rem',
          borderRadius: 6,
          background: 'rgba(251, 191, 36, 0.08)',
          border: '1px solid rgba(251, 191, 36, 0.25)',
          fontSize: 11,
          color: '#fbbf24',
        }}>
          <strong>Pre-analytical Quality Notice:</strong> {qc.warnings.join(' ')}
        </div>
      )}

      {/* Primary Scorecard Metric Display */}
      <div style={{
        background: 'rgba(255, 255, 255, 0.02)',
        border: '1px solid var(--border)',
        borderRadius: 10,
        padding: '1.25rem',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 8 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{
              fontSize: 12,
              fontWeight: 700,
              textTransform: 'uppercase',
              letterSpacing: 1.2,
              padding: '3px 10px',
              borderRadius: 99,
              color: currentTier.text,
              background: currentTier.bg,
              border: `1px solid ${currentTier.border}`,
            }}>
              {tier}
            </span>
            <span style={{ fontSize: 12, color: 'var(--sub)' }}>{horizon}</span>
          </div>
          <div style={{ fontSize: 11, color: '#AABAA3', background: 'rgba(90, 106, 83, 0.15)', padding: '2px 8px', borderRadius: 4 }}>
            Concordance: <strong>{concordance}</strong>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, margin: '1rem 0 0.5rem' }}>
          <span style={{ fontSize: '2.8rem', fontWeight: 800, color: currentTier.text, lineHeight: 1 }}>
            {pct}%
          </span>
          <span style={{ fontSize: 13, color: 'var(--sub)' }}>
            Consensus Genomic Progression Probability (70/30 Dual Blend)
          </span>
        </div>

        {/* Tri-Zone Risk Track */}
        <div style={{ marginTop: 8 }}>
          <div style={{ height: 10, background: 'var(--surface)', borderRadius: 99, overflow: 'hidden', position: 'relative' }}>
            <div
              style={{
                width: `${pct}%`,
                height: '100%',
                background: currentTier.bar,
                borderRadius: 99,
                transition: 'width 0.6s cubic-bezier(0.16, 1, 0.3, 1)',
              }}
            />
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--sub)', marginTop: 4, fontFamily: 'monospace' }}>
            <span>0% (Favorable &lt;35%)</span>
            <span>Intermediate (35%–65%)</span>
            <span>Aggressive (&gt;65%) 100%</span>
          </div>
        </div>

        {/* Dual-Consensus Model Breakdown Sub-Scores */}
        {pOrgan !== null && pPancan !== null && (
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(3, 1fr)',
            gap: 10,
            marginTop: '1.25rem',
            paddingTop: '1rem',
            borderTop: '1px solid rgba(255,255,255,0.05)',
          }}>
            <div style={{ background: 'var(--surface)', padding: '0.6rem 0.8rem', borderRadius: 6, border: '1px solid var(--border)' }}>
              <div style={{ fontSize: 10, color: 'var(--sub)' }}>Organ Specialist Model (70%)</div>
              <div style={{ fontSize: 16, fontWeight: 700, color: '#AABAA3', marginTop: 2 }}>
                {(pOrgan * 100).toFixed(1)}%
              </div>
            </div>
            <div style={{ background: 'var(--surface)', padding: '0.6rem 0.8rem', borderRadius: 6, border: '1px solid var(--border)' }}>
              <div style={{ fontSize: 10, color: 'var(--sub)' }}>Pan-Cancer Master Model (30%)</div>
              <div style={{ fontSize: 16, fontWeight: 700, color: '#C0D5BE', marginTop: 2 }}>
                {(pPancan * 100).toFixed(1)}%
              </div>
            </div>
            <div style={{ background: 'var(--surface)', padding: '0.6rem 0.8rem', borderRadius: 6, border: '1px solid var(--border)' }}>
              <div style={{ fontSize: 10, color: 'var(--sub)' }}>Dual-Consensus Score</div>
              <div style={{ fontSize: 16, fontWeight: 700, color: currentTier.text, marginTop: 2 }}>
                {(prob * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Interactive Tabs for Hospital Sections */}
      <div style={{ display: 'flex', gap: 6, borderBottom: '1px solid var(--border)', paddingBottom: 6 }}>
        {[
          { id: 'summary', label: 'Clinical Assessment' },
          { id: 'nutrition', label: 'Supportive Oncology & Nutrition' },
          { id: 'advisory', label: 'Physician & MDT Advisory' },
        ].map(tab => {
          const isActive = activeTab === tab.id
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                padding: '0.4rem 0.8rem',
                borderRadius: 6,
                fontSize: 11,
                fontWeight: 600,
                cursor: 'pointer',
                background: isActive ? 'rgba(90, 106, 83, 0.25)' : 'transparent',
                color: isActive ? '#F2EAE6' : 'var(--sub)',
                border: `1px solid ${isActive ? 'rgba(90, 106, 83, 0.4)' : 'transparent'}`,
              }}
            >
              {tab.label}
            </button>
          )
        })}
      </div>

      {/* TAB CONTENT 1: Clinical Assessment */}
      {activeTab === 'summary' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <div style={{
            background: 'var(--surface)',
            border: '1px solid var(--border)',
            borderLeft: `4px solid ${currentTier.text}`,
            borderRadius: 8,
            padding: '1rem',
            fontSize: 12,
            lineHeight: 1.6,
          }}>
            <div style={{ fontWeight: 700, color: currentTier.text, marginBottom: 4, textTransform: 'uppercase', letterSpacing: 0.5, fontSize: 11 }}>
              Observed Molecular Phenotype
            </div>
            <p style={{ margin: 0, color: '#EDEDED' }}>
              {clinicalObservation}
            </p>
          </div>

          <div style={{ background: 'rgba(255,255,255,0.02)', padding: '0.85rem', borderRadius: 8, border: '1px solid var(--border)', fontSize: 11, color: 'var(--sub)', lineHeight: 1.5 }}>
            <strong style={{ color: '#F2EAE6' }}>Architecture Provenance:</strong> Evaluated using the GUIDO Dual-Consensus architecture blending organ-specific sensitivity with cross-tissue stability from 5,259 TCGA patients and external hospital benchmarks (METABRIC and GSE31210).
          </div>
        </div>
      )}

      {/* TAB CONTENT 2: Evidence-Based Supportive Oncology & Nutrition */}
      {activeTab === 'nutrition' && (
        <div style={{
          background: 'var(--surface)',
          border: '1px solid var(--border)',
          borderRadius: 8,
          padding: '1.25rem',
          display: 'flex',
          flexDirection: 'column',
          gap: 12,
          fontSize: 12,
          lineHeight: 1.6,
        }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
              <span style={{ fontSize: 14 }}>🥗</span>
              <strong style={{ color: '#F2EAE6', fontSize: 13 }}>Evidence-Based Clinical Nutrition Guidance (ESPEN Guidelines)</strong>
            </div>
            <p style={{ margin: 0, color: 'var(--sub)' }}>
              Metabolic and nutritional support tailored to maintain lean muscle mass and support immune function during systemic or localized anti-cancer therapy:
            </p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 10 }}>
            <div style={{ background: 'rgba(255,255,255,0.02)', padding: '0.75rem', borderRadius: 6, border: '1px solid rgba(255,255,255,0.04)' }}>
              <strong style={{ color: '#AABAA3', display: 'block', marginBottom: 2 }}>Caloric &amp; Energy Targets</strong>
              <span style={{ color: '#CCC', fontSize: 11 }}>25–30 kcal/kg/day to prevent cancer cachexia, involuntary weight loss, and sarcopenia during active therapy.</span>
            </div>
            <div style={{ background: 'rgba(255,255,255,0.02)', padding: '0.75rem', borderRadius: 6, border: '1px solid rgba(255,255,255,0.04)' }}>
              <strong style={{ color: '#AABAA3', display: 'block', marginBottom: 2 }}>Protein Requirements</strong>
              <span style={{ color: '#CCC', fontSize: 11 }}>1.2–1.5 g/kg/day of high-biological-value protein to support tissue healing and immune cell regeneration.</span>
            </div>
            <div style={{ background: 'rgba(255,255,255,0.02)', padding: '0.75rem', borderRadius: 6, border: '1px solid rgba(255,255,255,0.04)' }}>
              <strong style={{ color: '#AABAA3', display: 'block', marginBottom: 2 }}>Hydration Protocol</strong>
              <span style={{ color: '#CCC', fontSize: 11 }}>30–35 mL/kg/day of clear fluids, unless restricted by treating physician due to renal, cardiopulmonary, or hepatic factors.</span>
            </div>
            <div style={{ background: 'rgba(255,255,255,0.02)', padding: '0.75rem', borderRadius: 6, border: '1px solid rgba(255,255,255,0.04)' }}>
              <strong style={{ color: '#C99E52', display: 'block', marginBottom: 2 }}>Supplement Safety Advisory</strong>
              <span style={{ color: '#CCC', fontSize: 11 }}>Avoid unverified high-dose antioxidant megadoses (e.g. mega-dose Vitamin C, E) during radiation or alkylating chemotherapy, as they may impair therapeutic reactive oxygen species (ROS) pathways.</span>
            </div>
          </div>

          <div style={{ fontSize: 11, color: 'var(--sub)', borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: 8 }}>
            ℹ All dietary modifications must be overseen by a Registered Dietitian / Certified Specialist in Oncology Nutrition (CSO) in coordination with the primary oncology care team.
          </div>
        </div>
      )}

      {/* TAB CONTENT 3: Physician & Multidisciplinary Team Advisory */}
      {activeTab === 'advisory' && (
        <div style={{
          background: 'var(--surface)',
          border: '1px solid var(--border)',
          borderRadius: 8,
          padding: '1.25rem',
          display: 'flex',
          flexDirection: 'column',
          gap: 12,
          fontSize: 12,
          lineHeight: 1.6,
        }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
              <span style={{ fontSize: 14 }}>🩺</span>
              <strong style={{ color: '#F2EAE6', fontSize: 13 }}>Standard Clinical Advisory &amp; Multidisciplinary Consultation</strong>
            </div>
            <p style={{ margin: 0, color: 'var(--sub)' }}>
              Molecular risk scores represent computational decision-support indicators and must be contextualized within formal clinical evaluation:
            </p>
          </div>

          <ul style={{ margin: 0, paddingLeft: '1.25rem', color: '#CCC', fontSize: 11, display: 'flex', flexDirection: 'column', gap: 6 }}>
            <li>
              <strong>Multidisciplinary Tumor Board (MDT) Review:</strong> Genomic risk estimates should be reviewed alongside clinical oncologists, surgical oncologists, radiation oncologists, and pathologists.
            </li>
            <li>
              <strong>Histopathological Correlation:</strong> Correlate with formal surgical pathology reports (tumor grade, lymphovascular invasion, perineural invasion, and margin status).
            </li>
            <li>
              <strong>Anatomic Staging (AJCC/UICC TNM 8th Edition):</strong> Incorporate radiologic imaging (CT, PET-CT, MRI) and TNM nodal stage before altering treatment intensity.
            </li>
            <li>
              <strong>Somatic &amp; Germline Molecular Biomarkers:</strong> Verify actionable mutations (e.g., EGFR, ALK, KRAS, BRAF, BRCA1/2, MSI-H/dMMR, PD-L1) for standard-of-care targeted therapies.
            </li>
          </ul>

          <div style={{
            background: 'rgba(201, 158, 82, 0.1)',
            border: '1px solid rgba(201, 158, 82, 0.25)',
            borderRadius: 6,
            padding: '0.6rem 0.8rem',
            fontSize: 11,
            color: '#E8CA8B',
          }}>
            <strong>Physician Consultation Notice:</strong> Patients must never modify, discontinue, or alter prescribed cancer therapies based on this computational report without direct consultation with their treating oncologist.
          </div>
        </div>
      )}

      {/* Regulatory Limitation & Provenance Footer */}
      <div style={{
        padding: '0.75rem 1rem',
        borderRadius: 8,
        background: 'rgba(255,255,255,0.015)',
        border: '1px solid var(--border)',
        fontSize: 10,
        color: 'var(--sub)',
        lineHeight: 1.5,
      }}>
        <div style={{ fontWeight: 700, color: '#AABAA3', marginBottom: 2 }}>
          REGULATORY DISCLAIMER — RESEARCH USE ONLY (RUO):
        </div>
        GUIDO is an investigational decision-support algorithm developed for exploratory and retrospective research. It is not cleared, approved, or CE-marked by the US FDA, EMA, or CDSCO as a Software as a Medical Device (SaMD) for primary diagnostic confirmation or autonomous prescription of therapy.
      </div>
    </div>
  )
}
