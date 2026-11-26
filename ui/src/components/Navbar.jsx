import React from 'react'

export default function Navbar(){
  return (
    <header className="w-full border-b" style={{borderColor:'var(--border)'}}>
      <div className="max-w-4xl mx-auto px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-lg font-bold tracking-tight text-[var(--text)]">GUIDO</span>
          <span className="text-xs px-2 py-0.5 rounded font-mono font-bold bg-amber-500/10 text-amber-400 border border-amber-500/30">
            RUO
          </span>
          <span className="text-xs text-[var(--sub)] hidden sm:inline">
            Clinical Cancer Recurrence &amp; Progression Risk Engine
          </span>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-[11px] font-mono text-[var(--sub)] hidden md:inline">
            SaMD Class C / CDSCO Investigational
          </span>
        </div>
      </div>
    </header>
  )
}
