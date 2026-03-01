import React, { useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  PieChart, Pie, Cell, Sector, Legend,
} from 'recharts'
import {
  aurocComparison, sampleDistribution,
  totalPatients, totalCancerTypes, avgClinicalAUROC,
  clinicalMetrics,
} from '../data/dashboardData'

/* ---- colour palette ---- */
const GREEN  = '#5A6A53'
const CREAM  = '#F2EAE6'
const MUTED  = '#888'
const CARD   = '#1E1E1E'
const BORDER = '#2E2E2E'
const SURFACE= '#252525'

const PIE_COLORS = ['#5A6A53','#7A8A73','#4A5A43','#8A9A83','#3A4A33','#6A7A63',
                    '#9AAA93','#2A3A23','#AABAA3','#5A6A53','#B0C0A9','#4A5A43']

const tooltipStyle     = { background: CARD, border: `1px solid ${BORDER}`, color: '#EDEDED', borderRadius: 8, fontSize: 12 }
const tooltipLabelStyle = { color: '#EDEDED', fontWeight: 600 }
const tooltipItemStyle  = { color: '#EDEDED' }

/* ---- pie active shape: expands slice when hovered (via legend or direct) ---- */
function renderActiveShape(props) {
  const { cx, cy, innerRadius, outerRadius, startAngle, endAngle, fill } = props
  return (
    <Sector cx={cx} cy={cy}
            innerRadius={innerRadius - 4} outerRadius={outerRadius + 10}
            startAngle={startAngle} endAngle={endAngle}
            fill={fill} />
  )
}

/* ---- stat card ---- */
function Stat({ value, label }){
  return (
    <div className="dash-stat">
      <div className="dash-stat-value">{value}</div>
      <div className="dash-stat-label">{label}</div>
    </div>
  )
}

/* ---- main dashboard ---- */
export default function Dashboard(){
  const [radarCancer, setRadarCancer] = useState('PRAD')
  const [hoveredPieIdx, setHoveredPieIdx] = useState(null)
  const radarKeys = ['accuracy','precision','recall','f1','specificity','sensitivity']

  const radarData = radarKeys.map(k => ({
    metric: k.charAt(0).toUpperCase() + k.slice(1),
    value: clinicalMetrics[radarCancer]?.[k] ?? 0,
  }))

  // Prepare pie data for the selected cancer from sampleDistribution
  const pieData = sampleDistribution.filter(d => d.Tumor + d.Normal > 0)

  return (
    <section className="dashboard">
      {/* Stats row */}
      <div className="dash-stats-row">
        <Stat value={totalPatients.toLocaleString()} label="Total Patients" />
        <Stat value={totalCancerTypes}              label="Cancer Types" />
        <Stat value={avgClinicalAUROC}              label="Avg Clinical AUROC" />
        <Stat value="500"                           label="Genes per Model" />
      </div>

      {/* Chart 1 — AUROC comparison */}
      <div className="dash-card">
        <div className="dash-card-header">
          <h3 className="dash-card-title">Model Performance — Test AUROC</h3>
          <p className="dash-card-desc">Clinical (tumor vs normal) vs synthetic labels across 10 TCGA cohorts · leak-free · class-balanced</p>
        </div>
        <div className="dash-chart" style={{height: 320}}>
          <ResponsiveContainer>
            <BarChart data={aurocComparison} margin={{top:10, right:20, left:0, bottom:5}}>
              <XAxis dataKey="cancer" tick={{fill: MUTED, fontSize:11}} axisLine={false} tickLine={false} />
              <YAxis domain={[0,1.05]} tick={{fill: MUTED, fontSize:11}} axisLine={false} tickLine={false}
                     tickFormatter={v => v.toFixed(1)} />
              <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} cursor={{fill:'rgba(255,255,255,0.03)'}} />
              <Bar dataKey="Clinical"  fill={GREEN}  radius={[4,4,0,0]} barSize={18} />
              <Bar dataKey="Synthetic" fill={MUTED}   radius={[4,4,0,0]} barSize={18} />
              <Legend wrapperStyle={{fontSize:12, color: MUTED}} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Chart 2 — Sample distribution */}
      <div className="dash-row-2col">
        <div className="dash-card">
          <div className="dash-card-header">
            <h3 className="dash-card-title">Sample Distribution</h3>
            <p className="dash-card-desc">Tumor vs normal samples per cancer type</p>
          </div>
          <div className="dash-chart" style={{height: 300}}>
            <ResponsiveContainer>
              <BarChart data={sampleDistribution} layout="vertical"
                        margin={{top:5, right:20, left:5, bottom:5}}>
                <XAxis type="number" tick={{fill: MUTED, fontSize:11}} axisLine={false} tickLine={false} />
                <YAxis type="category" dataKey="cancer" tick={{fill:'#EDEDED', fontSize:11}} width={48}
                       axisLine={false} tickLine={false} />
                <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} cursor={{fill:'rgba(255,255,255,0.03)'}} />
                <Bar dataKey="Tumor"  stackId="a" fill={GREEN}  radius={[0,0,0,0]} />
                <Bar dataKey="Normal" stackId="a" fill={CREAM}  radius={[0,4,4,0]} />
                <Legend wrapperStyle={{fontSize:12, color: MUTED}} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart 3 — Radar metrics */}
        <div className="dash-card">
          <div className="dash-card-header">
            <h3 className="dash-card-title">Clinical Model Metrics</h3>
            <div className="dash-card-controls">
              <select value={radarCancer} onChange={e => setRadarCancer(e.target.value)}
                      className="dash-select">
                {Object.keys(clinicalMetrics).map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
          </div>
          <div className="dash-chart" style={{height: 300}}>
            <ResponsiveContainer>
              <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="70%">
                <PolarGrid stroke="rgba(255,255,255,0.08)" />
                <PolarAngleAxis dataKey="metric" tick={{fill: MUTED, fontSize:11}} />
                <PolarRadiusAxis domain={[0,1]} tick={false} axisLine={false} />
                <Radar dataKey="value" stroke={GREEN} fill={GREEN} fillOpacity={0.25} strokeWidth={2} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Chart 4 — Cohort size pie */}
      <div className="dash-card">
        <div className="dash-card-header">
          <h3 className="dash-card-title">Cohort Size Breakdown</h3>
          <p className="dash-card-desc">Patient distribution across {totalCancerTypes} TCGA cancer types</p>
        </div>
        <div className="dash-chart" style={{height: 260}}>
          <ResponsiveContainer>
            <PieChart>
              <Pie data={pieData} dataKey="Tumor" nameKey="cancer"
                   cx="50%" cy="50%" outerRadius={95} innerRadius={52}
                   stroke={CARD} strokeWidth={2}
                   activeIndex={hoveredPieIdx}
                   activeShape={renderActiveShape}
                   onMouseEnter={(_, i) => setHoveredPieIdx(i)}
                   onMouseLeave={() => setHoveredPieIdx(null)}>
                {pieData.map((_,i) => <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />)}
              </Pie>
              <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} />
            </PieChart>
          </ResponsiveContainer>
        </div>
        {/* Interactive legend — hover highlights corresponding slice */}
        <div style={{display:'flex', flexWrap:'wrap', gap:'4px 6px', padding:'0 1rem 1rem', justifyContent:'center'}}>
          {pieData.map((d, i) => (
            <div key={d.cancer}
                 style={{
                   display:'flex', alignItems:'center', gap:5, cursor:'pointer',
                   padding:'3px 8px', borderRadius:6,
                   background: hoveredPieIdx === i ? 'rgba(255,255,255,0.07)' : 'transparent',
                   transition:'background .15s',
                 }}
                 onMouseEnter={() => setHoveredPieIdx(i)}
                 onMouseLeave={() => setHoveredPieIdx(null)}>
              <span style={{width:10, height:10, borderRadius:2, background: PIE_COLORS[i % PIE_COLORS.length], flexShrink:0, display:'inline-block'}} />
              <span style={{fontSize:11, color: hoveredPieIdx === i ? '#EDEDED' : MUTED, transition:'color .15s'}}>
                {d.cancer}
              </span>
              <span style={{fontSize:10, color:'rgba(255,255,255,0.3)'}}>
                {(d.Tumor + d.Normal).toLocaleString()}
              </span>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
