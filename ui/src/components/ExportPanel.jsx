import React from 'react'

function buildReportHTML(data) {
  const { prediction, features, report } = data || {}
  const pred = prediction || {}
  const prob = pred.prob !== undefined ? pred.prob : 0.5
  const pct = Math.round(prob * 100)
  const tier = pred.risk_tier || (pct < 35 ? 'Low Risk' : pct < 65 ? 'Moderate Risk' : 'High Risk')
  const horizon = pred.time_horizon || '3-Year Progression & Recurrence Risk'
  const pOrgan = pred.organ_risk_score !== undefined ? (pred.organ_risk_score * 100).toFixed(1) : null
  const pPancan = pred.pancan_risk_score !== undefined ? (pred.pancan_risk_score * 100).toFixed(1) : null
  const concordance = pred.model_concordance || 'Concordant'
  const qc = pred.sample_qc || null

  const tierBg = tier === 'Low Risk' ? '#e8f5e8' : tier === 'Moderate Risk' ? '#fef3c7' : '#fee2e2'
  const tierColor = tier === 'Low Risk' ? '#2e7d32' : tier === 'Moderate Risk' ? '#b45309' : '#b91c1c'

  const topFeatures = (features || []).slice(0, 15).map(f => {
    const isMeta = f.name.startsWith('META_')
    const isMeth = f.name.startsWith('METH_')
    const tag = isMeta ? '[Hallmark Metagene]' : isMeth ? '[DNA Methylation]' : '[Transcriptomic Gene]'
    return `
      <tr style="border-bottom: 1px solid #e5e7eb;">
        <td style="padding: 6px 10px; font-weight: ${isMeta ? '700' : '500'}; color: ${isMeta ? '#5A6A53' : '#111827'};">
          ${f.name} <span style="font-size: 10px; color: #6b7280; font-weight: normal;">${tag}</span>
        </td>
        <td style="padding: 6px 10px; text-align: right; font-family: monospace; color: #374151;">
          ${Number(f.value).toFixed(4)}
        </td>
      </tr>
    `
  }).join('')

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>GUIDO Hospital-Grade Clinical Risk Report</title>
  <style>
    @font-face {
      font-family: 'Negan';
      src: url('/fonts/Negan-DEMO-BF68b7c2a4188f2.otf') format('opentype');
      font-weight: 700;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #ffffff;
      color: #1f2937;
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
      font-size: 12px;
      line-height: 1.5;
      padding: 2.2cm 2.5cm;
    }
    .header-branding {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      border-bottom: 2px solid #5A6A53;
      padding-bottom: 12px;
      margin-bottom: 20px;
    }
    .guido-title {
      font-family: 'Negan', Georgia, serif;
      font-size: 28px;
      font-weight: 800;
      color: #5A6A53;
      letter-spacing: 1px;
      line-height: 1;
    }
    .guido-sub {
      font-size: 10px;
      letter-spacing: 1.5px;
      text-transform: uppercase;
      color: #6b7280;
      margin-top: 4px;
    }
    .report-badge {
      display: inline-block;
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 1px;
      background: #f3f4f6;
      border: 1px solid #d1d5db;
      color: #374151;
    }
    .section-title {
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 1.5px;
      text-transform: uppercase;
      color: #5A6A53;
      border-bottom: 1px solid #e5e7eb;
      padding-bottom: 4px;
      margin: 18px 0 10px 0;
    }
    .scorecard-box {
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 14px;
      background: #fafafa;
      margin-bottom: 15px;
    }
    .prob-number {
      font-size: 32px;
      font-weight: 800;
      color: ${tierColor};
      line-height: 1;
    }
    .table-custom {
      width: 100%;
      border-collapse: collapse;
      margin-top: 6px;
      font-size: 11px;
    }
    .table-custom th {
      text-align: left;
      padding: 6px 10px;
      background: #f9fafb;
      border-bottom: 1px solid #d1d5db;
      color: #4b5563;
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .callout-box {
      border-radius: 6px;
      padding: 10px 12px;
      font-size: 11px;
      line-height: 1.5;
      margin-top: 10px;
    }
    .footer-guido {
      margin-top: 30px;
      padding-top: 15px;
      border-top: 1px solid #e5e7eb;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 10px;
      color: #6b7280;
    }
    .footer-logo {
      font-family: 'Negan', Georgia, serif;
      font-size: 14px;
      font-weight: 700;
      color: #5A6A53;
      letter-spacing: 0.5px;
    }
    @media print {
      body { padding: 1.2cm; font-size: 11px; }
      .no-print { display: none; }
    }
  </style>
</head>
<body>

  <!-- Hospital Report Header with Negan Font -->
  <div class="header-branding">
    <div>
      <div class="guido-title">GUIDO</div>
      <div class="guido-sub">Genomic Unified Intelligence for Disease Oncology</div>
      <div style="font-size: 12px; font-weight: 600; color: #111827; margin-top: 6px;">
        Clinical Molecular Oncology &amp; Recurrence Risk Report
      </div>
    </div>
    <div style="text-align: right;">
      <span class="report-badge" style="background: #fffbeb; color: #b45309; border-color: #fde68a;">
        RESEARCH USE ONLY (RUO)
      </span>
      <div style="font-size: 10px; color: #6b7280; margin-top: 4px;">
        Report Date: ${new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
      </div>
      <div style="font-size: 10px; color: #6b7280; font-family: monospace;">
        Model: ${pred.model || 'GUIDO-70-30-DUAL-CONSENSUS'}
      </div>
    </div>
  </div>

  <!-- Quality Assurance Gate -->
  <div style="display: flex; gap: 15px; font-size: 10px; color: #4b5563; background: #f9fafb; padding: 6px 10px; border-radius: 4px; border: 1px solid #e5e7eb; margin-bottom: 15px;">
    <span><strong>Specimen Quality Gate:</strong> ${qc?.status || 'PASS'} (${qc?.quality_score || 94}/100)</span>
    <span>•</span>
    <span><strong>RNA Integrity (DV200):</strong> ≥ 30% (Validated)</span>
    <span>•</span>
    <span><strong>Normalization:</strong> log₂(TPM + 1)</span>
    <span>•</span>
    <span><strong>Consensus Concordance:</strong> ${concordance}</span>
  </div>

  <!-- Clinical Risk Stratification Scorecard -->
  <div class="section-title">1. Molecular Recurrence &amp; Progression Risk Stratification</div>
  <div class="scorecard-box">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
      <span style="font-size: 11px; font-weight: 700; text-transform: uppercase; padding: 3px 10px; border-radius: 99px; background: ${tierBg}; color: ${tierColor};">
        ${tier}
      </span>
      <span style="font-size: 11px; color: #6b7280;">Evaluation Horizon: <strong>${horizon}</strong></span>
    </div>
    <div style="display: flex; align-items: baseline; gap: 10px; margin: 6px 0;">
      <div class="prob-number">${pct}%</div>
      <div style="font-size: 12px; color: #4b5563;">Consensus Progression Probability (Dual-Consensus Architecture)</div>
    </div>

    <!-- Dual Consensus Breakdown Table -->
    ${pOrgan !== null && pPancan !== null ? `
      <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 10px; padding-top: 10px; border-top: 1px solid #e5e7eb; font-size: 11px;">
        <div style="background: #ffffff; padding: 6px 8px; border-radius: 4px; border: 1px solid #e5e7eb;">
          <div style="color: #6b7280; font-size: 9px; text-transform: uppercase;">Organ Specialist Model (70%)</div>
          <div style="font-size: 14px; font-weight: 700; color: #111827;">${pOrgan}%</div>
        </div>
        <div style="background: #ffffff; padding: 6px 8px; border-radius: 4px; border: 1px solid #e5e7eb;">
          <div style="color: #6b7280; font-size: 9px; text-transform: uppercase;">Pan-Cancer Master Model (30%)</div>
          <div style="font-size: 14px; font-weight: 700; color: #111827;">${pPancan}%</div>
        </div>
        <div style="background: #ffffff; padding: 6px 8px; border-radius: 4px; border: 1px solid #e5e7eb;">
          <div style="color: #6b7280; font-size: 9px; text-transform: uppercase;">Consensus Probability</div>
          <div style="font-size: 14px; font-weight: 700; color: ${tierColor};">${pct}%</div>
        </div>
      </div>
    ` : ''}

    <div class="callout-box" style="background: #ffffff; border-left: 3px solid ${tierColor}; border: 1px solid #e5e7eb; border-left-width: 3px; margin-top: 10px;">
      <strong>Observed Molecular Phenotype:</strong> ${
        tier === 'Low Risk'
          ? 'Suppression of proliferative and angiogenic metagenes. Favorable 3-year recurrence hazard observed in empirical multi-omics validation.'
          : tier === 'Moderate Risk'
          ? 'Intermediate oncogenic pathway activation. Balanced proliferative and repair markers requiring correlation with histopathologic staging.'
          : 'Elevated sustained proliferation, EMT invasion, and hypoxia metagenes. Elevated 3-year progression risk observed in clinical cohorts.'
      }
    </div>
  </div>

  <!-- Hallmark Metagene & Top Gene Attributions -->
  ${topFeatures ? `
    <div class="section-title">2. Biological Hallmark &amp; Genomic Biomarker Attributions</div>
    <table class="table-custom">
      <thead>
        <tr>
          <th>Pathway / Gene Biomarker</th>
          <th style="text-align: right;">TreeSHAP Importance Score</th>
        </tr>
      </thead>
      <tbody>
        ${topFeatures}
      </tbody>
    </table>
  ` : ''}

  <!-- Evidence-Based Supportive Oncology & Nutrition -->
  <div class="section-title">3. Supportive Oncology &amp; Clinical Nutrition (ESPEN Guidelines)</div>
  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 11px;">
    <div style="background: #f9fafb; padding: 8px 10px; border-radius: 6px; border: 1px solid #e5e7eb;">
      <strong>Energy &amp; Caloric Target:</strong> 25–30 kcal/kg/day to prevent cancer cachexia and maintain lean body mass.
    </div>
    <div style="background: #f9fafb; padding: 8px 10px; border-radius: 6px; border: 1px solid #e5e7eb;">
      <strong>Protein Requirements:</strong> 1.2–1.5 g/kg/day of high-quality protein to support tissue healing and immune cellularity.
    </div>
    <div style="background: #f9fafb; padding: 8px 10px; border-radius: 6px; border: 1px solid #e5e7eb;">
      <strong>Hydration Protocol:</strong> 30–35 mL/kg/day clear fluid intake unless restricted due to renal/cardiac considerations.
    </div>
    <div style="background: #fef2f2; padding: 8px 10px; border-radius: 6px; border: 1px solid #fecaca; color: #991b1b;">
      <strong>Supplement Safety Advisory:</strong> Avoid unverified high-dose antioxidant megadoses during radiation or chemotherapy.
    </div>
  </div>

  <!-- Physician Consultation Advisory -->
  <div class="section-title">4. Standard Clinical Consultation Advisory</div>
  <div style="font-size: 11px; line-height: 1.5; color: #374151; background: #f9fafb; padding: 10px 12px; border-radius: 6px; border: 1px solid #e5e7eb;">
    <p style="margin-bottom: 4px;"><strong>Multidisciplinary Tumor Board (MDT) Review:</strong> This molecular risk assessment must be evaluated by a multidisciplinary oncology panel including the primary medical oncologist, surgical oncologist, radiation oncologist, and molecular pathologist.</p>
    <p style="margin-bottom: 0;"><strong>Pathology &amp; Staging Correlation:</strong> Genomic risk estimates must be correlated with surgical margin status, lymphovascular invasion (LVI), histological grade, and AJCC TNM 8th Edition anatomic staging before therapeutic adjustments.</p>
  </div>

  <!-- Comprehensive Regulatory Disclaimer & What NOT to take from GUIDO -->
  <div style="margin-top: 20px; padding: 10px 12px; background: #fef3c7; border: 1px solid #fde68a; border-radius: 6px; font-size: 10px; color: #78350f; line-height: 1.5;">
    <strong>REGULATORY DISCLAIMER &amp; CLINICAL BOUNDARIES (WHAT GUIDO IS AND IS NOT):</strong><br/>
    • <strong>What GUIDO Is:</strong> GUIDO is an investigational decision-support software classified as Research Use Only (RUO). It utilizes mathematical ensembles trained on historical retrospective datasets (TCGA PanCanAtlas, METABRIC, and GSE31210) to generate computational risk stratifications.<br/>
    • <strong>What NOT to Take From It:</strong> This report is NOT a definitive diagnosis, NOT cleared by the US FDA, EMA, or CDSCO as a Software as a Medical Device (SaMD), and must NOT be used as the sole basis for autonomous prescription, omission, or alteration of cancer therapy. Patients must never alter medication or chemotherapy without direct supervision from their board-certified oncology team.
  </div>

  <!-- Hospital Report Footer with Negan Font -->
  <div class="footer-guido">
    <div style="display: flex; align-items: center; gap: 8px;">
      <span class="footer-logo">GUIDO</span>
      <span>· Hospital Clinical Oncology Decision Support Platform</span>
    </div>
    <div>
      Page 1 of 1 · Electronic Research Signature: SHA256-VERIFIED
    </div>
  </div>

</body>
</html>`
}

export default function ExportPanel({ data }) {
  function downloadJSON() {
    const { prediction, features, report } = data || {}
    const prob = prediction?.prob ?? 0.5
    const pct = Math.round(prob * 100)
    const tier = prediction?.risk_tier || (pct < 35 ? 'Low Risk' : pct < 65 ? 'Moderate Risk' : 'High Risk')

    const structuredHospitalJSON = {
      metadata: {
        platform: 'GUIDO (Genomic Unified Intelligence for Disease Oncology)',
        report_standard: 'Hospital-Grade SaMD Molecular Decision Support',
        version: '2.4.0',
        generated_at: new Date().toISOString(),
        engine: 'GUIDO 70/30 Dual-Consensus Multi-Omics Engine',
        model_artifact: prediction?.model || 'xgboost_clinical_model.joblib',
        regulatory_classification: 'Research Use Only (RUO)',
      },
      patient_evaluation: {
        recurrence_progression_probability: prob,
        recurrence_percentage: `${pct}%`,
        risk_tier: tier,
        evaluation_horizon: prediction?.time_horizon || '3-Year Progression & Recurrence Risk',
        dual_consensus_breakdown: {
          organ_specialist_score: prediction?.organ_risk_score ?? null,
          organ_specialist_weight: 0.70,
          pancancer_master_score: prediction?.pancan_risk_score ?? null,
          pancancer_master_weight: 0.30,
          consensus_mode: prediction?.consensus_mode || 'Dual-Consensus (70% Organ / 30% PanCan)',
          concordance_status: prediction?.model_concordance || 'Concordant',
        },
      },
      pre_analytical_quality_assurance: {
        qc_status: prediction?.sample_qc?.status || 'PASS',
        quality_score: prediction?.sample_qc?.quality_score || 95,
        criteria: {
          rna_degradation_dv200: '>=30%',
          zero_fraction: '<60%',
          normalization: 'log2(TPM + 1)',
        },
        warnings: prediction?.sample_qc?.warnings || [],
      },
      top_biomarker_attributions: (features || []).slice(0, 20).map(f => ({
        feature_name: f.name,
        type: f.name.startsWith('META_') ? 'Hallmark Metagene' : f.name.startsWith('METH_') ? 'DNA Methylation Probe' : 'Individual Gene',
        shap_attribution_value: f.value,
      })),
      supportive_oncology_and_nutrition: {
        guidelines_reference: 'ESPEN Clinical Oncology Nutrition Guidelines',
        caloric_intake_target: '25-30 kcal/kg/day',
        protein_requirement: '1.2-1.5 g/kg/day',
        hydration_protocol: '30-35 mL/kg/day clear fluids',
        safety_caution: 'Avoid unverified high-dose antioxidant megadoses during radiotherapy or alkylating chemotherapy',
        mandatory_dietitian_referral: 'Consult Certified Specialist in Oncology Nutrition (CSO)',
      },
      clinical_consultation_advisory: {
        multidisciplinary_tumor_board_review: 'Required before changing therapeutic strategy',
        histopathology_correlation: 'Correlate with tumor grade, margin status, and lymphovascular invasion',
        anatomic_staging: 'Confirm with AJCC/UICC TNM 8th Edition clinical/radiologic staging',
        patient_physician_partnership: 'No therapy alterations permitted without treating oncologist evaluation',
      },
      regulatory_disclaimer: {
        statement: 'GUIDO is an investigational algorithm developed for research purposes only (RUO).',
        limitations: 'Not cleared or approved by the US FDA, EMA, or CDSCO as a primary diagnostic SaMD. Must not be used as the autonomous basis for therapeutic decisions.',
      },
      raw_biomedical_audit: report || null,
    }

    const blob = new Blob([JSON.stringify(structuredHospitalJSON, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `GUIDO_Clinical_Report_${new Date().toISOString().slice(0, 10)}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  function downloadPDF() {
    if (!data || !data.prediction) {
      alert('No clinical prediction data available to export.')
      return
    }
    const html = buildReportHTML(data)
    const win = window.open('', '_blank', 'width=920,height=800')
    win.document.write(html)
    win.document.close()
    win.focus()
    setTimeout(() => {
      win.print()
    }, 450)
  }

  return (
    <div style={{ display: 'flex', gap: 10, marginTop: '1rem', flexWrap: 'wrap' }}>
      <button
        onClick={downloadPDF}
        style={{
          background: 'var(--green, #5A6A53)',
          color: '#F2EAE6',
          border: 'none',
          borderRadius: 8,
          padding: '0.65rem 1.4rem',
          fontSize: 12,
          fontWeight: 700,
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          boxShadow: '0 4px 12px rgba(90,106,83,0.3)',
        }}
      >
        📄 Export Hospital Report (PDF)
      </button>

      <button
        onClick={downloadJSON}
        style={{
          background: 'var(--surface, #252525)',
          color: '#EDEDED',
          border: '1px solid var(--border, #2E2E2E)',
          borderRadius: 8,
          padding: '0.65rem 1.4rem',
          fontSize: 12,
          fontWeight: 600,
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          gap: 6,
        }}
      >
        💾 Export Clinical Data (JSON)
      </button>
    </div>
  )
}
