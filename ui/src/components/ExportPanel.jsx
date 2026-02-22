import React from 'react'

function buildReportHTML(data) {
  const { prediction, features, report } = data || {}

  const pred = prediction || {}
  const pct = Math.round((pred.prob || 0) * 100)
  const labelColor = pred.label === 'positive' ? '#e87070' : '#5A6A53'

  // Render report object as table rows
  function renderSection(obj, depth = 0) {
    if (!obj || typeof obj !== 'object') return `<span>${obj ?? '—'}</span>`
    if (Array.isArray(obj)) {
      return '<ul style="margin:0;padding-left:1.2em">' +
        obj.map(v => `<li>${typeof v === 'object' ? renderSection(v, depth+1) : v}</li>`).join('') +
        '</ul>'
    }
    return '<table style="width:100%;border-collapse:collapse;margin-top:4px">' +
      Object.entries(obj).map(([k, v]) => {
        const label = k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
        const val = typeof v === 'object' ? renderSection(v, depth+1) : String(v ?? '—')
        // colour-code pass/fail
        const style = /PASS|APPROVED|VALIDATED/i.test(String(v)) ? 'color:#5A6A53;font-weight:600' :
                      /FAIL|REJECT|INSUFFICIENT/i.test(String(v)) ? 'color:#e87070;font-weight:600' : ''
        return `<tr style="border-bottom:1px solid #2E2E2E">
          <td style="padding:5px 8px;color:#888;white-space:nowrap;width:38%;vertical-align:top">${label}</td>
          <td style="padding:5px 8px;${style}">${val}</td>
        </tr>`
      }).join('') +
      '</table>'
  }

  const reportSections = report
    ? Object.entries(report).map(([k, v]) => {
        const title = k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
        return `<div style="margin-bottom:1.5rem">
          <h3 style="font-size:0.8rem;letter-spacing:1.5px;text-transform:uppercase;color:#888;margin:0 0 0.5rem 0">${title}</h3>
          ${typeof v === 'object' ? renderSection(v) : `<p style="margin:0">${v}</p>`}
        </div>`
      }).join('')
    : ''

  const featureRows = (features || []).slice(0, 15).map(f =>
    `<tr style="border-bottom:1px solid #2E2E2E">
      <td style="padding:4px 8px">${f.name}</td>
      <td style="padding:4px 8px;color:#888">${f.value.toFixed(4)}</td>
    </tr>`
  ).join('')

  return `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>GUIDO Prediction Report</title>
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    body{background:#fff;color:#111;font-family:Arial,sans-serif;font-size:13px;line-height:1.6;padding:2.5cm}
    h1{font-size:22px;font-weight:700;margin-bottom:0.25rem}
    h2{font-size:13px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#555;margin:1.5rem 0 0.75rem 0;border-bottom:1px solid #ddd;padding-bottom:4px}
    h3{font-size:11px;letter-spacing:1px;text-transform:uppercase;color:#888;margin:0 0 0.4rem 0}
    table{width:100%;border-collapse:collapse}
    td{vertical-align:top}
    ul{padding-left:1.2em}
    li{margin-bottom:2px}
    .badge{display:inline-block;padding:2px 10px;border-radius:99px;font-weight:700;font-size:14px}
    .meta{color:#888;font-size:11px;margin-top:4px}
    @media print{body{padding:1.5cm}}
  </style>
</head>
<body>
  <div style="margin-bottom:2rem">
    <div style="color:#5A6A53;font-size:10px;letter-spacing:2px;text-transform:uppercase;margin-bottom:0.5rem">GUIDO · Genomic Unified Intelligence for Disease Oncology</div>
    <h1>Patient Prediction Report</h1>
    <div class="meta">Generated: ${new Date().toLocaleString()} &nbsp;·&nbsp; Model: ${pred.model || '—'}</div>
  </div>

  <h2>Prediction</h2>
  <div style="display:flex;align-items:center;gap:1.5rem;margin-bottom:1rem">
    <span class="badge" style="background:${pred.label === 'positive' ? '#fde8e8' : '#e8f5e8'};color:${labelColor}">${(pred.label || '—').toUpperCase()}</span>
    <div>
      <div style="font-size:20px;font-weight:700">${pct}% confidence</div>
      <div class="meta">Processing time: ${pred.time || '—'}</div>
    </div>
  </div>

  ${featureRows ? `
  <h2>Top Feature Importances</h2>
  <table><thead><tr style="border-bottom:2px solid #ddd">
    <th style="text-align:left;padding:4px 8px">Gene</th>
    <th style="text-align:left;padding:4px 8px">Importance</th>
  </tr></thead><tbody>${featureRows}</tbody></table>` : ''}

  ${reportSections ? `<h2 style="margin-top:2rem">Biomedical Audit</h2>${reportSections}` : ''}
</body>
</html>`
}

export default function ExportPanel({data}){
  function downloadJSON(){
    const blob = new Blob([JSON.stringify(data, null, 2)], {type:'application/json'})
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = 'prediction.json'; a.click()
    URL.revokeObjectURL(url)
  }

  function downloadPDF(){
    if(!data || !data.prediction){
      alert('No prediction data available to export')
      return
    }
    const html = buildReportHTML(data)
    const win = window.open('', '_blank', 'width=900,height=700')
    win.document.write(html)
    win.document.close()
    win.focus()
    // slight delay so styles render before print dialog opens
    setTimeout(() => { win.print() }, 400)
  }

  return (
    <div className="flex gap-3 mt-4">
      <button onClick={downloadPDF} className="export-btn">Download PDF</button>
      <button onClick={downloadJSON} className="export-btn">Download JSON</button>
    </div>
  )
}
