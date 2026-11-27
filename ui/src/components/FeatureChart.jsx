import React from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

const GREEN = '#5A6A53' // Individual genes
const SAGE  = '#7A9A78' // Hallmark Metagenes
const AMBER = '#C99E52' // DNA Methylation probes

export default function FeatureChart({ data }) {
  if (!data || !Array.isArray(data) || data.length === 0) {
    return <div className="subtle" style={{ padding: '1rem 0' }}>No feature importance available for this patient run.</div>
  }

  const top = data.slice(0, 15)
  const h = Math.max(220, top.length * 30)

  const getFeatureColor = (name) => {
    if (name.startsWith('META_')) return SAGE
    if (name.startsWith('METH_') || name.startsWith('cg')) return AMBER
    return GREEN
  }

  const getFeatureType = (name) => {
    if (name.startsWith('META_')) return 'Hallmark Metagene'
    if (name.startsWith('METH_') || name.startsWith('cg')) return 'DNA Methylation'
    return 'Transcriptomic Gene'
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      {/* Legend chips */}
      <div style={{ display: 'flex', gap: 12, fontSize: 11, color: '#888', flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ width: 10, height: 10, borderRadius: 2, background: SAGE }} />
          <span>Hallmark Metagene (META_)</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ width: 10, height: 10, borderRadius: 2, background: AMBER }} />
          <span>DNA Methylation (METH_)</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ width: 10, height: 10, borderRadius: 2, background: GREEN }} />
          <span>Individual Gene Expression</span>
        </div>
      </div>

      <div style={{ height: h }}>
        <ResponsiveContainer>
          <BarChart data={top} layout="vertical" margin={{ left: 110, right: 20, top: 5, bottom: 5 }}>
            <XAxis
              type="number"
              tick={{ fill: 'var(--sub)', fontSize: 11 }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              type="category"
              dataKey="name"
              tick={{ fill: 'var(--text)', fontSize: 11 }}
              width={105}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              contentStyle={{
                background: 'rgba(30, 30, 30, 0.95)',
                border: '1px solid var(--border)',
                borderRadius: 8,
                color: 'var(--text)',
                fontSize: 12,
                backdropFilter: 'blur(8px)',
              }}
              formatter={(v, name, item) => [
                `${Number(v).toFixed(4)} (TreeSHAP attribution) — ${getFeatureType(item.payload.name)}`,
                'Impact Score',
              ]}
            />
            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
              {top.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={getFeatureColor(entry.name)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
