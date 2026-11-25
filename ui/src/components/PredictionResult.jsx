import React from 'react'

export default function PredictionResult({ prediction }) {
  const prob = prediction.prob !== undefined ? prediction.prob : 0.5
  const pct = Math.round(prob * 100)
  const tier = prediction.risk_tier || (pct < 35 ? 'Low Risk' : pct < 65 ? 'Moderate Risk' : 'High Risk')
  const horizon = prediction.time_horizon || '3-Year Progression & Recurrence Risk'
  const recommendation = prediction.recommendation || (
    pct < 35
      ? 'Favorable prognosis. Routine clinical surveillance recommended.'
      : pct < 65
      ? 'Intermediate risk profile. Close periodic surveillance recommended.'
      : 'High disease progression risk. Consideration for intensified adjuvant therapy.'
  )

  const tierColors = {
    'Low Risk': { text: '#34d399', bg: 'rgba(52, 211, 153, 0.12)', border: 'rgba(52, 211, 153, 0.35)' },
    'Moderate Risk': { text: '#fbbf24', bg: 'rgba(251, 191, 36, 0.12)', border: 'rgba(251, 191, 36, 0.35)' },
    'High Risk': { text: '#f87171', bg: 'rgba(248, 113, 113, 0.15)', border: 'rgba(248, 113, 113, 0.4)' },
  }

  const currentTier = tierColors[tier] || tierColors['Moderate Risk']

  return (
    <div className="risk-scorecard">
      <div className="flex items-center justify-between flex-wrap gap-2 mb-2">
        <div className="flex items-center gap-3">
          <span
            className="px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider"
            style={{
              color: currentTier.text,
              backgroundColor: currentTier.bg,
              border: `1px solid ${currentTier.border}`,
            }}
          >
            {tier}
          </span>
          <span className="text-xs text-sub">{horizon}</span>
        </div>
        <div className="text-xs text-sub font-mono">{prediction.model || 'Risk Model'}</div>
      </div>

      <div className="my-3 flex items-baseline gap-2">
        <span className="text-4xl font-extrabold tracking-tight" style={{ color: currentTier.text }}>
          {pct}%
        </span>
        <span className="text-sm text-sub">Absolute Disease Risk Score</span>
      </div>

      {/* Tri-Zone Risk Spectrum Track */}
      <div className="mt-2">
        <div className="progress-track relative overflow-hidden h-3 rounded-full" style={{ background: 'var(--surface)' }}>
          <div
            className="progress-fill h-full rounded-full transition-all duration-500"
            style={{
              width: `${pct}%`,
              background: pct < 35 ? '#34d399' : pct < 65 ? '#fbbf24' : '#f87171',
            }}
          />
        </div>
        <div className="flex justify-between text-[11px] text-sub mt-1 font-mono">
          <span>0% (Low &lt;35%)</span>
          <span>Moderate (35-65%)</span>
          <span>High (&gt;65%) 100%</span>
        </div>
      </div>

      {/* Clinical Guidance Box */}
      <div
        className="mt-4 p-3 rounded-lg text-xs leading-relaxed"
        style={{
          background: 'var(--surface)',
          border: `1px solid var(--border)`,
          borderLeft: `4px solid ${currentTier.text}`,
        }}
      >
        <span className="font-semibold block mb-0.5" style={{ color: currentTier.text }}>
          Clinical Interpretation:
        </span>
        <span className="text-muted">{recommendation}</span>
      </div>

      <div className="mt-3 flex justify-between items-center text-[11px] text-sub">
        <span>Model latency: {prediction.time || '0.0s'}</span>
        <span className="text-[10px] opacity-75">Research &amp; Prototype Decision Engine</span>
      </div>
    </div>
  )
}
