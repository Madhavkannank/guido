import React, { useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  PieChart, Pie, Cell, Sector, Legend,
} from 'recharts'
import {
  modelDiscrimination, riskStratificationData, cohortPieData,
  clinicalMetrics, hallmarkPathways,
  totalPatients, totalCancerCohorts, totalClinicalModels,
  masterModelAUROC, masterModelCIndex, totalHallmarks, featuresPerModel,
} from '../data/dashboardData'

/* ---- colour palette ---- */
const GREEN  = '#5A6A53'
const SAGE   = '#7A9A78'
const AMBER  = '#C99E52'
const CRIMSON= '#B84A4A'
const CREAM  = '#F2EAE6'
const MUTED  = '#888888'
const CARD   = '#1E1E1E'
const BORDER = '#2E2E2E'
const SURFACE= '#252525'

const PIE_COLORS = ['#5A6A53', '#7A9A78', '#A2BAA0']

const tooltipStyle     = { background: CARD, border: `1px solid ${BORDER}`, color: '#EDEDED', borderRadius: 8, fontSize: 12 }
const tooltipLabelStyle = { color: '#EDEDED', fontWeight: 600 }
const tooltipItemStyle  = { color: '#EDEDED' }

/* ---- pie active shape: expands slice when hovered ---- */
function renderActiveShape(props) {
  const { cx, cy, innerRadius, outerRadius, startAngle, endAngle, fill } = props
  return (
    <Sector cx={cx} cy={cy}
            innerRadius={innerRadius - 4} outerRadius={outerRadius + 8}
            startAngle={startAngle} endAngle={endAngle}
            fill={fill} />
  )
}

/* ---- stat card ---- */
function Stat({ value, label, subtext }){
  return (
    <div className="dash-stat">
      <div className="dash-stat-value">{value}</div>
      <div className="dash-stat-label">{label}</div>
      {subtext && <div style={{fontSize: 10, color: MUTED, marginTop: 2}}>{subtext}</div>}
    </div>
  )
}

/* ---- main dashboard ---- */
export default function Dashboard(){
  const [selectedCohort, setSelectedCohort] = useState('PAN-CANCER-ALL')
  const [hoveredPieIdx, setHoveredPieIdx] = useState(null)

  const selectedMetrics = clinicalMetrics[selectedCohort] || clinicalMetrics['PAN-CANCER-ALL']

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
      <div className="dash-stats-row">
        <Stat value={totalPatients.toLocaleString()} label="Total Real Patients" subtext="TCGA + METABRIC + GEO" />
        <Stat value={totalCancerCohorts} label="Pan-Cancer Cohorts" subtext="All 33 TCGA types" />
        <Stat value={masterModelAUROC.toFixed(3)} label="Master 3-Yr AUROC" subtext="Held-out 15% test split" />
        <Stat value={masterModelCIndex.toFixed(3)} label="Harrell's C-Index" subtext="Concordance index" />
        <Stat value={`${totalHallmarks} Pathways`} label="Hallmark Metagenes" subtext={`${featuresPerModel} features/model`} />
      </div>

      {/* Chart 1 — Discrimination: AUROC & C-Index across cohorts */}
      <div className="dash-card">
        <div className="dash-card-header">
          <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 8}}>
            <div>
              <h3 className="dash-card-title">Genomic Discrimination & Independent Hospital Generalization</h3>
              <p className="dash-card-desc">
                Harrell's Concordance Index (C-Index) and 3-Year Recurrence AUROC evaluated across held-out splits and live external hospital cohorts.
              </p>
            </div>
            <div style={{display: 'flex', gap: 8}}>
              <span style={{padding: '3px 8px', borderRadius: 4, fontSize: 11, background: 'rgba(90,106,83,0.25)', color: '#AABAA3', border: '1px solid rgba(90,106,83,0.4)'}}>
                METABRIC Hospital: AUROC 0.749 (EXCELLENT)
              </span>
              <span style={{padding: '3px 8px', borderRadius: 4, fontSize: 11, background: 'rgba(122,154,120,0.25)', color: '#C0D5BE', border: '1px solid rgba(122,154,120,0.4)'}}>
                GEO GSE31210: C-Index 0.968 (EXCELLENT)
              </span>
            </div>
          </div>
        </div>
        <div className="dash-chart" style={{height: 320}}>
          <ResponsiveContainer>
            <BarChart data={modelDiscrimination} margin={{top:15, right:20, left:0, bottom:15}}>
              <XAxis dataKey="cohort" tick={{fill: MUTED, fontSize:11}} axisLine={false} tickLine={false} />
              <YAxis domain={[0.4, 1.05]} tick={{fill: MUTED, fontSize:11}} axisLine={false} tickLine={false}
                     tickFormatter={v => v.toFixed(2)} />
              <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle}
                       cursor={{fill:'rgba(255,255,255,0.03)'}}
                       formatter={(v, name) => [v.toFixed(3), name === 'AUROC' ? '3-Year Risk AUROC' : "Harrell's C-Index"]} />
              <ReferenceLine y={0.5} stroke="rgba(255,255,255,0.15)" strokeDasharray="4 3"
                             label={{value:'Random Guess (0.50)', position:'insideBottomRight', fill:'rgba(255,255,255,0.3)', fontSize:10}} />
              <ReferenceLine y={0.7} stroke="rgba(122,154,120,0.4)" strokeDasharray="3 3"
                             label={{value:'Clinical Benchmark (0.70)', position:'insideTopRight', fill:'#7A9A78', fontSize:10}} />
              <Bar dataKey="AUROC" fill={GREEN} radius={[4,4,0,0]} barSize={22} name="AUROC" />
              <Bar dataKey="C_Index" fill={SAGE} radius={[4,4,0,0]} barSize={22} name="C_Index" />
              <Legend wrapperStyle={{fontSize:12, color: MUTED}}
                      formatter={(value) => value === 'AUROC' ? '3-Year Recurrence AUROC' : "Harrell's Concordance Index (C-Index)"} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Row: Chart 2 (Tri-Zone Stratification) + Chart 3 (Radar Performance) */}
      <div className="dash-row-2col">
        {/* Chart 2 — Tri-Zone Risk Stratification */}
        <div className="dash-card">
          <div className="dash-card-header">
            <h3 className="dash-card-title">Tri-Zone Clinical Risk Stratification</h3>
            <p className="dash-card-desc">
              Patient tier distribution: Low Risk (&lt;35%), Moderate (35–65%), and High Risk (&gt;65%) progression probability.
            </p>
          </div>
          <div className="dash-chart" style={{height: 310}}>
            <ResponsiveContainer>
              <BarChart data={riskStratificationData} layout="vertical" margin={{top:5, right:20, left:20, bottom:5}}>
                <XAxis type="number" tick={{fill: MUTED, fontSize:11}} axisLine={false} tickLine={false} />
                <YAxis type="category" dataKey="cohort" tick={{fill:'#EDEDED', fontSize:11}} width={110}
                       axisLine={false} tickLine={false} />
                <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle}
                         cursor={{fill:'rgba(255,255,255,0.03)'}}
                         formatter={(v, name, item) => [
                           `${v.toLocaleString()} patients (${((v / item.payload.total) * 100).toFixed(1)}%)`,
                           name
                         ]} />
                <Bar dataKey="Low Risk (<35%)" stackId="a" fill={GREEN} radius={[0,0,0,0]} />
                <Bar dataKey="Moderate Risk (35-65%)" stackId="a" fill={AMBER} radius={[0,0,0,0]} />
                <Bar dataKey="High Risk (>65%)" stackId="a" fill={CRIMSON} radius={[0,4,4,0]} />
                <Legend wrapperStyle={{fontSize:11, color: MUTED}} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart 3 — Radar metrics */}
        <div className="dash-card">
          <div className="dash-card-header">
            <div style={{display:'flex', justifyContent:'space-between', alignItems:'center', width:'100%'}}>
              <div>
                <h3 className="dash-card-title">Clinical Metric Radar</h3>
                <p className="dash-card-desc">6-axis discrimination, calibration, and sensitivity</p>
              </div>
              <div className="dash-card-controls">
                <select value={selectedCohort} onChange={e => setSelectedCohort(e.target.value)}
                        className="dash-select">
                  {cancerTypes.map(c => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
            </div>
          </div>
          <div className="dash-chart" style={{height: 250}}>
            <ResponsiveContainer>
              <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="68%">
                <PolarGrid stroke="rgba(255,255,255,0.08)" />
                <PolarAngleAxis dataKey="metric" tick={{fill: MUTED, fontSize:11}} />
                <PolarRadiusAxis domain={[0, 1]} tick={false} axisLine={false} />
                <Radar dataKey="value" stroke={GREEN} fill={GREEN} fillOpacity={0.3} strokeWidth={2} />
                <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle}
                         formatter={v => [(v * 100).toFixed(1) + '%', 'Value']} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
          {/* Metric value chips */}
          <div style={{display:'flex', flexWrap:'wrap', gap:'4px 6px', padding:'0 1rem 0.75rem', justifyContent:'center'}}>
            {radarData.map(item => {
              const pct = (item.value * 100).toFixed(1)
              const dim = item.value < 0.65
              return (
                <div key={item.metric} style={{
                  padding:'3px 9px', borderRadius:99, fontSize:11,
                  background: dim ? 'rgba(232,112,112,0.12)' : 'rgba(90,106,83,0.18)',
                  border: `1px solid ${dim ? 'rgba(232,112,112,0.25)' : 'rgba(90,106,83,0.35)'}`,
                  color: dim ? '#e87070' : '#AABAA3',
                }}>
                  <span style={{color: MUTED, marginRight:4}}>{item.metric}</span>
                  {pct}%
                </div>
              )
            })}
          </div>
        </div>
      </div>

      {/* Row: Chart 4 (Cohort Size Pie) + Section 5 (Hallmark Metagene Architecture) */}
      <div className="dash-row-2col">
        {/* Chart 4 — Multi-Consortium Cohort Breakdown */}
        <div className="dash-card">
          <div className="dash-card-header">
            <h3 className="dash-card-title">Multi-Consortium Training Corpus</h3>
            <p className="dash-card-desc">7,781 real clinical oncology patients across 75+ GB genomics</p>
          </div>
          <div className="dash-chart" style={{height: 230}}>
            <ResponsiveContainer>
              <PieChart>
                <Pie data={cohortPieData} dataKey="value" nameKey="name"
                     cx="50%" cy="50%" outerRadius={85} innerRadius={48}
                     stroke={CARD} strokeWidth={2}
                     activeIndex={hoveredPieIdx}
                     activeShape={renderActiveShape}
                     onMouseEnter={(_, i) => setHoveredPieIdx(i)}
                     onMouseLeave={() => setHoveredPieIdx(null)}>
                  {cohortPieData.map((_, i) => <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />)}
                </Pie>
                <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle}
                         formatter={(v, name, item) => [
                           `${v.toLocaleString()} patients (${((v / totalPatients) * 100).toFixed(1)}%) — ${item.payload.type}`,
                           item.payload.source
                         ]} />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div style={{display:'flex', flexDirection:'column', gap:6, padding:'0 1rem 1rem'}}>
            {cohortPieData.map((d, i) => (
              <div key={d.name}
                   style={{
                     display:'flex', justifyContent:'space-between', alignItems:'center',
                     padding:'4px 8px', borderRadius:6,
                     background: hoveredPieIdx === i ? 'rgba(255,255,255,0.07)' : 'rgba(255,255,255,0.02)',
                     border: `1px solid ${hoveredPieIdx === i ? 'rgba(255,255,255,0.15)' : 'rgba(255,255,255,0.05)'}`,
                     cursor:'pointer', transition:'all .15s',
                   }}
                   onMouseEnter={() => setHoveredPieIdx(i)}
                   onMouseLeave={() => setHoveredPieIdx(null)}>
                <div style={{display:'flex', alignItems:'center', gap:8}}>
                  <span style={{width:10, height:10, borderRadius:2, background: PIE_COLORS[i % PIE_COLORS.length]}} />
                  <span style={{fontSize:11, color: hoveredPieIdx === i ? '#EDEDED' : '#CCCCCC'}}>{d.name}</span>
                </div>
                <div style={{display:'flex', alignItems:'center', gap:8}}>
                  <span style={{fontSize:10, color: MUTED}}>{d.type}</span>
                  <span style={{fontSize:11, fontWeight:600, color:'#EDEDED'}}>{d.value.toLocaleString()}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Section 5 — 12 Hallmark Metagene Pathways */}
        <div className="dash-card">
          <div className="dash-card-header">
            <h3 className="dash-card-title">12 Hallmark Oncology Metagenes</h3>
            <p className="dash-card-desc">Leak-free biological pathway features computed across every patient</p>
          </div>
          <div style={{maxHeight: 330, overflowY: 'auto', padding: '0 1rem 1rem', display: 'flex', flexDirection: 'column', gap: 6}}>
            {hallmarkPathways.map((h, idx) => (
              <div key={h.id} style={{
                padding: '6px 10px', borderRadius: 6,
                background: 'rgba(255,255,255,0.02)',
                border: '1px solid rgba(255,255,255,0.05)',
                display: 'flex', flexDirection: 'column', gap: 2,
              }}>
                <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
                  <span style={{fontSize: 12, fontWeight: 600, color: '#AABAA3'}}>{h.label}</span>
                  <span style={{fontSize: 10, color: MUTED, fontFamily: 'monospace'}}>{h.id}</span>
                </div>
                <div style={{fontSize: 10, color: '#888888', fontStyle: 'italic'}}>
                  Genes: {h.genes}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}
