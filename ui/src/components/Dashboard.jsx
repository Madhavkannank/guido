import React, { useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  PieChart, Pie, Cell, Sector, Legend,
} from 'recharts'
import {
  cancerTypes,
  modelDiscrimination, riskStratificationData, cohortPieData,
  clinicalMetrics, organSpecialists, consensusComparison, hallmarkPathways,
  totalPatients, totalCancerCohorts, totalOrganSpecialists, totalClinicalModels,
  masterModelAUROC, masterModelCIndex, dualConsensusAUROC, dualConsensusCIndex,
  totalHallmarks, featuresPerModel,
} from '../data/dashboardData'

/* ---- Dark Oncology Color Palette ---- */
const GREEN   = '#5A6A53' // Brand primary olive green
const SAGE    = '#7A9A78' // Secondary soft green
const PALE    = '#AABAA3' // Tertiary light sage
const CREAM   = '#F2EAE6' // Warm cream highlight
const AMBER   = '#C99E52' // Moderate risk amber
const CRIMSON = '#B84A4A' // High risk crimson
const MUTED   = '#888888' // Neutral secondary text
const CARD    = '#1E1E1E' // Dark card surface
const BORDER  = '#2E2E2E' // Card border
const SURFACE = '#252525' // Elevated input/chip surface

const PIE_COLORS = ['#5A6A53', '#7A9A78', '#AABAA3']

const tooltipStyle = {
  background: 'rgba(30, 30, 30, 0.95)',
  backdropFilter: 'blur(10px)',
  border: `1px solid ${BORDER}`,
  color: '#EDEDED',
  borderRadius: 8,
  fontSize: 12,
  boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
}
const tooltipLabelStyle = { color: '#F2EAE6', fontWeight: 600, marginBottom: 4 }
const tooltipItemStyle  = { color: '#EDEDED' }

/* ---- Pie active shape ---- */
function renderActiveShape(props) {
  const { cx, cy, innerRadius, outerRadius, startAngle, endAngle, fill } = props
  return (
    <Sector
      cx={cx} cy={cy}
      innerRadius={innerRadius - 3}
      outerRadius={outerRadius + 6}
      startAngle={startAngle}
      endAngle={endAngle}
      fill={fill}
    />
  )
}

/* ---- Stat KPI Box ---- */
function Stat({ value, label, subtext, highlight }) {
  return (
    <div className={`dash-stat ${highlight ? 'dash-stat-highlight' : ''}`}>
      <div className="dash-stat-value" style={highlight ? { color: '#F2EAE6' } : {}}>{value}</div>
      <div className="dash-stat-label">{label}</div>
      {subtext && <div style={{ fontSize: 10, color: MUTED, marginTop: 4 }}>{subtext}</div>}
    </div>
  )
}

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState('benchmarks')
  const [selectedCohort, setSelectedCohort] = useState('Dual-Consensus (70/30 Blend)')
  const [hoveredPieIdx, setHoveredPieIdx] = useState(null)

  const selectedMetrics = clinicalMetrics[selectedCohort] || clinicalMetrics['Dual-Consensus (70/30 Blend)']

  const radarData = [
    { metric: 'AUROC', value: selectedMetrics.auroc },
    { metric: 'C-Index', value: selectedMetrics.c_index },
    { metric: 'Accuracy', value: selectedMetrics.accuracy },
    { metric: 'Calibration', value: +(1.0 - (selectedMetrics.brier || 0.2)).toFixed(3) },
    { metric: 'Sensitivity', value: selectedMetrics.sensitivity },
    { metric: 'Specificity', value: selectedMetrics.specificity },
  ]

  return (
    <section className="dashboard">
      {/* Real Clinical Stats row */}
      <div className="dash-stats-row" style={{ gridTemplateColumns: 'repeat(5, 1fr)' }}>
        <Stat
          value={totalPatients.toLocaleString()}
          label="Total Real Patients"
          subtext="5,259 TCGA + 2,206 External"
        />
        <Stat
          value="14 + 1 Models"
          label="Multi-Model Suite"
          subtext="14 Specialists + 1 PanCan Master"
        />
        <Stat
          value="70 / 30"
          label="Dual-Consensus"
          subtext="Organ (70%) + PanCan (30%)"
          highlight
        />
        <Stat
          value={dualConsensusAUROC.toFixed(3)}
          label="Consensus AUROC"
          subtext="Held-out multi-cohort blend"
        />
        <Stat
          value={dualConsensusCIndex.toFixed(3)}
          label="Harrell's C-Index"
          subtext="Longitudinal concordance"
        />
      </div>

      {/* Tab navigation for interactive exploration */}
      <div style={{
        display: 'flex',
        gap: 8,
        borderBottom: `1px solid ${BORDER}`,
        paddingBottom: 10,
        overflowX: 'auto',
      }}>
        {[
          { id: 'benchmarks', label: 'Consensus & Hospital Benchmarks' },
          { id: 'specialists', label: '14 Organ Specialist Models' },
          { id: 'architecture', label: '70/30 Dual-Consensus Engine' },
          { id: 'hallmarks', label: '12 Hallmark Oncology Metagenes' },
        ].map(tab => {
          const isActive = activeTab === tab.id
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                padding: '0.45rem 0.95rem',
                borderRadius: 8,
                fontSize: 12,
                fontWeight: 600,
                letterSpacing: '0.3px',
                cursor: 'pointer',
                transition: 'all 0.15s ease',
                background: isActive ? 'rgba(90,106,83,0.3)' : 'rgba(255,255,255,0.02)',
                color: isActive ? '#F2EAE6' : MUTED,
                border: `1px solid ${isActive ? 'rgba(90,106,83,0.5)' : BORDER}`,
              }}
            >
              {tab.label}
            </button>
          )
        })}
      </div>

      {/* TAB 1: Consensus & Hospital Benchmarks */}
      {activeTab === 'benchmarks' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {/* Chart 1 — Discrimination: AUROC & C-Index across cohorts */}
          <div className="dash-card">
            <div className="dash-card-header">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 10 }}>
                <div>
                  <h3 className="dash-card-title">Dual-Consensus & Independent Hospital Benchmarks</h3>
                  <p className="dash-card-desc">
                    Harrell's Concordance Index (C-Index) and 3-Year Recurrence AUROC evaluated across held-out splits and live hospital validation cohorts.
                  </p>
                </div>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  <span style={{ padding: '4px 10px', borderRadius: 6, fontSize: 11, background: 'rgba(90,106,83,0.25)', color: '#AABAA3', border: '1px solid rgba(90,106,83,0.4)' }}>
                    Dual-Consensus AUROC: 0.819
                  </span>
                  <span style={{ padding: '4px 10px', borderRadius: 6, fontSize: 11, background: 'rgba(122,154,120,0.25)', color: '#C0D5BE', border: '1px solid rgba(122,154,120,0.4)' }}>
                    METABRIC: AUROC 0.749 (N=1,980)
                  </span>
                  <span style={{ padding: '4px 10px', borderRadius: 6, fontSize: 11, background: 'rgba(201,158,82,0.25)', color: '#E8CA8B', border: '1px solid rgba(201,158,82,0.4)' }}>
                    GEO GSE31210: C-Index 0.968 (N=226)
                  </span>
                </div>
              </div>
            </div>
            <div className="dash-chart" style={{ height: 320 }}>
              <ResponsiveContainer>
                <BarChart data={modelDiscrimination} margin={{ top: 15, right: 20, left: 0, bottom: 15 }}>
                  <XAxis dataKey="cohort" tick={{ fill: MUTED, fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis domain={[0.4, 1.05]} tick={{ fill: MUTED, fontSize: 11 }} axisLine={false} tickLine={false}
                    tickFormatter={v => v.toFixed(2)} />
                  <Tooltip
                    contentStyle={tooltipStyle}
                    labelStyle={tooltipLabelStyle}
                    itemStyle={tooltipItemStyle}
                    cursor={{ fill: 'rgba(255,255,255,0.03)' }}
                    formatter={(v, name) => [v.toFixed(3), name === 'AUROC' ? '3-Year Recurrence AUROC' : "Harrell's C-Index"]}
                  />
                  <ReferenceLine y={0.5} stroke="rgba(255,255,255,0.15)" strokeDasharray="4 3"
                    label={{ value: 'Random Guess (0.50)', position: 'insideBottomRight', fill: 'rgba(255,255,255,0.3)', fontSize: 10 }} />
                  <ReferenceLine y={0.7} stroke="rgba(122,154,120,0.4)" strokeDasharray="3 3"
                    label={{ value: 'Clinical Benchmark (0.70)', position: 'insideTopRight', fill: '#7A9A78', fontSize: 10 }} />
                  <Bar dataKey="AUROC" fill={GREEN} radius={[4, 4, 0, 0]} barSize={26} name="AUROC" />
                  <Bar dataKey="C_Index" fill={SAGE} radius={[4, 4, 0, 0]} barSize={26} name="C_Index" />
                  <Legend
                    wrapperStyle={{ fontSize: 12, color: MUTED, paddingTop: 10 }}
                    formatter={(value) => value === 'AUROC' ? '3-Year Recurrence AUROC' : "Harrell's Concordance Index (C-Index)"}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Row: Tri-Zone Stratification + Metric Radar */}
          <div className="dash-row-2col">
            {/* Tri-Zone Risk Stratification */}
            <div className="dash-card">
              <div className="dash-card-header">
                <h3 className="dash-card-title">Tri-Zone Clinical Risk Stratification</h3>
                <p className="dash-card-desc">
                  Patient tier distribution: Low Risk (&lt;35%), Moderate (35–65%), and High Risk (&gt;65%) progression probability.
                </p>
              </div>
              <div className="dash-chart" style={{ height: 300 }}>
                <ResponsiveContainer>
                  <BarChart data={riskStratificationData} layout="vertical" margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
                    <XAxis type="number" tick={{ fill: MUTED, fontSize: 11 }} axisLine={false} tickLine={false} />
                    <YAxis type="category" dataKey="cohort" tick={{ fill: '#EDEDED', fontSize: 10 }} width={175}
                      axisLine={false} tickLine={false} />
                    <Tooltip
                      contentStyle={tooltipStyle}
                      labelStyle={tooltipLabelStyle}
                      itemStyle={tooltipItemStyle}
                      cursor={{ fill: 'rgba(255,255,255,0.03)' }}
                      formatter={(v, name, item) => [
                        `${v.toLocaleString()} patients (${((v / item.payload.total) * 100).toFixed(1)}%)`,
                        name,
                      ]}
                    />
                    <Bar dataKey="Low Risk (<35%)" stackId="a" fill={GREEN} />
                    <Bar dataKey="Moderate Risk (35-65%)" stackId="a" fill={AMBER} />
                    <Bar dataKey="High Risk (>65%)" stackId="a" fill={CRIMSON} radius={[0, 4, 4, 0]} />
                    <Legend wrapperStyle={{ fontSize: 11, color: MUTED }} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Radar metrics */}
            <div className="dash-card">
              <div className="dash-card-header">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
                  <div>
                    <h3 className="dash-card-title">Clinical Metric Radar</h3>
                    <p className="dash-card-desc">6-axis discrimination, calibration, and sensitivity</p>
                  </div>
                  <div className="dash-card-controls">
                    <select
                      value={selectedCohort}
                      onChange={e => setSelectedCohort(e.target.value)}
                      className="dash-select"
                    >
                      {cancerTypes.map(c => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </div>
                </div>
              </div>
              <div className="dash-chart" style={{ height: 240 }}>
                <ResponsiveContainer>
                  <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="68%">
                    <PolarGrid stroke="rgba(255,255,255,0.08)" />
                    <PolarAngleAxis dataKey="metric" tick={{ fill: MUTED, fontSize: 11 }} />
                    <PolarRadiusAxis domain={[0, 1]} tick={false} axisLine={false} />
                    <Radar dataKey="value" stroke={GREEN} fill={GREEN} fillOpacity={0.3} strokeWidth={2} />
                    <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle}
                      formatter={v => [(v * 100).toFixed(1) + '%', 'Value']} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
              {/* Metric value chips */}
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px 6px', padding: '0 0.5rem 0.5rem', justifyContent: 'center' }}>
                {radarData.map(item => {
                  const pct = (item.value * 100).toFixed(1)
                  return (
                    <div key={item.metric} style={{
                      padding: '3px 8px', borderRadius: 99, fontSize: 10,
                      background: 'rgba(90,106,83,0.18)',
                      border: '1px solid rgba(90,106,83,0.35)',
                      color: '#AABAA3',
                    }}>
                      <span style={{ color: MUTED, marginRight: 4 }}>{item.metric}</span>
                      {pct}%
                    </div>
                  )
                })}
              </div>
            </div>
          </div>

          {/* Corpus Distribution Pie */}
          <div className="dash-card">
            <div className="dash-card-header">
              <h3 className="dash-card-title">Multi-Consortium Patient Corpus (7,465 Longitudinal Patients)</h3>
              <p className="dash-card-desc">Curated across NIH TCGA PanCanAtlas (33 cohorts), METABRIC breast cancer, and NCBI GEO GSE31210</p>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.3fr', gap: '1.5rem', alignItems: 'center' }}>
              <div className="dash-chart" style={{ height: 230 }}>
                <ResponsiveContainer>
                  <PieChart>
                    <Pie
                      data={cohortPieData} dataKey="value" nameKey="name"
                      cx="50%" cy="50%" outerRadius={85} innerRadius={48}
                      stroke={CARD} strokeWidth={2}
                      activeIndex={hoveredPieIdx}
                      activeShape={renderActiveShape}
                      onMouseEnter={(_, i) => setHoveredPieIdx(i)}
                      onMouseLeave={() => setHoveredPieIdx(null)}
                    >
                      {cohortPieData.map((_, i) => <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />)}
                    </Pie>
                    <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle}
                      formatter={(v, name, item) => [
                        `${v.toLocaleString()} patients (${((v / totalPatients) * 100).toFixed(1)}%) — ${item.payload.type}`,
                        item.payload.source,
                      ]}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {cohortPieData.map((d, i) => (
                  <div
                    key={d.name}
                    style={{
                      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                      padding: '8px 12px', borderRadius: 8,
                      background: hoveredPieIdx === i ? 'rgba(90,106,83,0.2)' : 'rgba(255,255,255,0.02)',
                      border: `1px solid ${hoveredPieIdx === i ? 'rgba(90,106,83,0.4)' : BORDER}`,
                      cursor: 'pointer', transition: 'all .15s',
                    }}
                    onMouseEnter={() => setHoveredPieIdx(i)}
                    onMouseLeave={() => setHoveredPieIdx(null)}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <span style={{ width: 10, height: 10, borderRadius: 2, background: PIE_COLORS[i % PIE_COLORS.length] }} />
                      <div>
                        <div style={{ fontSize: 12, fontWeight: 600, color: hoveredPieIdx === i ? '#F2EAE6' : '#EDEDED' }}>{d.name}</div>
                        <div style={{ fontSize: 10, color: MUTED }}>{d.source}</div>
                      </div>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <div style={{ fontSize: 13, fontWeight: 700, color: '#EDEDED' }}>{d.value.toLocaleString()}</div>
                      <div style={{ fontSize: 10, color: MUTED }}>{((d.value / totalPatients) * 100).toFixed(1)}%</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: 14 Organ Specialist Models Breakdown */}
      {activeTab === 'specialists' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div className="dash-card">
            <div className="dash-card-header">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 10 }}>
                <div>
                  <h3 className="dash-card-title">14 Organ Specialist Models (Cross-Cohort Discrimination)</h3>
                  <p className="dash-card-desc">
                    Each specialist model was trained with 130 features (100 top prognostic individual genes, 18 DNA methylation probes, and 12 Hallmark Metagenes) dedicated to organ-specific oncogenic pathways.
                  </p>
                </div>
                <span style={{ padding: '4px 10px', borderRadius: 6, fontSize: 11, background: 'rgba(90,106,83,0.25)', color: '#AABAA3', border: '1px solid rgba(90,106,83,0.4)' }}>
                  14 Specialist Models Verified
                </span>
              </div>
            </div>
            <div className="dash-chart" style={{ height: 420 }}>
              <ResponsiveContainer>
                <BarChart data={organSpecialists} layout="vertical" margin={{ top: 10, right: 25, left: 40, bottom: 10 }}>
                  <XAxis type="number" domain={[0.5, 0.95]} tick={{ fill: MUTED, fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis type="category" dataKey="cohort" tick={{ fill: '#EDEDED', fontSize: 11 }} width={160} axisLine={false} tickLine={false} />
                  <Tooltip
                    contentStyle={tooltipStyle}
                    labelStyle={tooltipLabelStyle}
                    itemStyle={tooltipItemStyle}
                    cursor={{ fill: 'rgba(255,255,255,0.03)' }}
                    formatter={(v, name) => [v.toFixed(3), name === 'auroc' ? '3-Year Recurrence AUROC' : "Harrell's C-Index"]}
                  />
                  <ReferenceLine x={0.7} stroke="rgba(122,154,120,0.4)" strokeDasharray="3 3"
                    label={{ value: 'Clinical Benchmark (0.70)', position: 'insideTopRight', fill: '#7A9A78', fontSize: 10 }} />
                  <Bar dataKey="auroc" fill={GREEN} barSize={12} name="auroc" radius={[0, 4, 4, 0]} />
                  <Bar dataKey="c_index" fill={SAGE} barSize={12} name="c_index" radius={[0, 4, 4, 0]} />
                  <Legend
                    wrapperStyle={{ fontSize: 11, color: MUTED, paddingTop: 10 }}
                    formatter={(value) => value === 'auroc' ? '3-Year Recurrence AUROC' : "Harrell's Concordance Index (C-Index)"}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Grid of 14 Specialist Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))', gap: '1rem' }}>
            {organSpecialists.map(m => (
              <div
                key={m.code}
                style={{
                  background: CARD,
                  border: `1px solid ${BORDER}`,
                  borderRadius: 10,
                  padding: '1rem',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 8,
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: '#F2EAE6' }}>{m.cohort}</div>
                  <span style={{ fontSize: 9, padding: '2px 6px', borderRadius: 4, background: 'rgba(90,106,83,0.2)', color: PALE, fontFamily: 'monospace' }}>
                    {m.samples} pts
                  </span>
                </div>
                <div style={{ fontSize: 10, color: MUTED }}>{m.organs}</div>
                <div style={{ display: 'flex', justifyContent: 'space-between', paddingTop: 6, borderTop: `1px solid rgba(255,255,255,0.05)` }}>
                  <div>
                    <div style={{ fontSize: 9, color: MUTED }}>AUROC</div>
                    <div style={{ fontSize: 14, fontWeight: 700, color: '#AABAA3' }}>{m.auroc.toFixed(3)}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: 9, color: MUTED }}>C-Index</div>
                    <div style={{ fontSize: 14, fontWeight: 700, color: '#C0D5BE' }}>{m.c_index.toFixed(3)}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: 9, color: MUTED }}>Brier</div>
                    <div style={{ fontSize: 14, fontWeight: 700, color: MUTED }}>{m.brier.toFixed(3)}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* TAB 3: 70/30 Dual-Consensus Engine */}
      {activeTab === 'architecture' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div className="dash-card">
            <div className="dash-card-header">
              <h3 className="dash-card-title">Dual-Consensus Architecture: Organ-Specific Sensitivity + Pan-Cancer Stability</h3>
              <p className="dash-card-desc">
                Mathematical synthesis blending 70% weight on the organ specialist model with 30% weight on the 5,259-patient Pan-Cancer Foundation model.
              </p>
            </div>

            {/* Formula Display Box */}
            <div style={{
              background: 'rgba(90,106,83,0.12)',
              border: '1px solid rgba(90,106,83,0.3)',
              borderRadius: 10,
              padding: '1.25rem',
              margin: '0.75rem 0 1.25rem',
              textAlign: 'center',
            }}>
              <div style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: 1.5, color: '#AABAA3', marginBottom: 6, fontWeight: 600 }}>
                Consensus Probability Blending Formula
              </div>
              <div style={{
                fontFamily: 'monospace',
                fontSize: 18,
                fontWeight: 700,
                color: '#F2EAE6',
                letterSpacing: 0.5,
              }}>
                p_consensus = 0.70 · p_organ + 0.30 · p_pancan
              </div>
              <div style={{ fontSize: 11, color: MUTED, marginTop: 8, maxWidth: 600, margin: '8px auto 0' }}>
                Where <code style={{ color: '#AABAA3' }}>p_organ</code> captures unique primary organ hallmarks (e.g., HER2, IDH1, GATA3) and <code style={{ color: '#AABAA3' }}>p_pancan</code> anchors calibration across the broader oncological continuum.
              </div>
            </div>

            {/* Strategy comparison table */}
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: `1px solid ${BORDER}`, color: MUTED, textAlign: 'left' }}>
                    <th style={{ padding: '8px 12px' }}>Inference Strategy</th>
                    <th style={{ padding: '8px 12px' }}>AUROC</th>
                    <th style={{ padding: '8px 12px' }}>C-Index</th>
                    <th style={{ padding: '8px 12px' }}>Brier Score</th>
                    <th style={{ padding: '8px 12px' }}>Expected Cal. Error</th>
                    <th style={{ padding: '8px 12px' }}>Clinical Assessment</th>
                  </tr>
                </thead>
                <tbody>
                  {consensusComparison.map(s => {
                    const isOptimal = s.strategy.includes('70% Organ')
                    return (
                      <tr
                        key={s.strategy}
                        style={{
                          borderBottom: `1px solid rgba(255,255,255,0.04)`,
                          background: isOptimal ? 'rgba(90,106,83,0.15)' : 'transparent',
                        }}
                      >
                        <td style={{ padding: '10px 12px', fontWeight: isOptimal ? 700 : 500, color: isOptimal ? '#F2EAE6' : '#EDEDED' }}>
                          {s.strategy}
                          {isOptimal && (
                            <span style={{ marginLeft: 8, fontSize: 9, padding: '2px 6px', borderRadius: 4, background: GREEN, color: '#F2EAE6' }}>
                              PRODUCTION
                            </span>
                          )}
                        </td>
                        <td style={{ padding: '10px 12px', color: '#AABAA3', fontWeight: 600 }}>{s.auroc.toFixed(3)}</td>
                        <td style={{ padding: '10px 12px', color: '#C0D5BE', fontWeight: 600 }}>{s.c_index.toFixed(3)}</td>
                        <td style={{ padding: '10px 12px', color: MUTED }}>{s.brier.toFixed(3)}</td>
                        <td style={{ padding: '10px 12px', color: MUTED }}>{s.ece.toFixed(3)}</td>
                        <td style={{ padding: '10px 12px', color: isOptimal ? '#AABAA3' : MUTED, fontSize: 11 }}>{s.desc}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* TAB 4: 12 Hallmark Oncology Metagenes */}
      {activeTab === 'hallmarks' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div className="dash-card">
            <div className="dash-card-header">
              <h3 className="dash-card-title">12 Hallmark Oncology Metagenes (Biological Pathway Architecture)</h3>
              <p className="dash-card-desc">
                Derived from the Hanahan & Weinberg Hallmarks of Cancer. Each metagene is computed dynamically across incoming patient RNA-seq profiles using leak-free baseline z-score normalization.
              </p>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(290px, 1fr))', gap: '1rem', marginTop: 12 }}>
              {hallmarkPathways.map(h => (
                <div
                  key={h.id}
                  style={{
                    background: CARD,
                    border: `1px solid ${BORDER}`,
                    borderRadius: 10,
                    padding: '1rem',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 6,
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: 12, fontWeight: 700, color: '#AABAA3' }}>{h.label}</span>
                    <span style={{ fontSize: 9, padding: '2px 6px', borderRadius: 4, background: 'rgba(255,255,255,0.05)', color: MUTED, fontFamily: 'monospace' }}>
                      {h.id}
                    </span>
                  </div>
                  <div style={{ fontSize: 10, color: '#CCCCCC', fontStyle: 'italic', margin: '4px 0' }}>
                    Marker Genes: {h.genes}
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingTop: 6, borderTop: `1px solid rgba(255,255,255,0.05)`, fontSize: 10 }}>
                    <span style={{ color: MUTED }}>High-Risk Calibration Delta:</span>
                    <span style={{ color: h.weight.startsWith('+') ? '#E8CA8B' : '#7A9A78', fontWeight: 600 }}>
                      {h.weight}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </section>
  )
}
