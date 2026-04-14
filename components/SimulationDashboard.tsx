'use client';

import { SimulationRun } from '@/lib/types';

export function SimulationDashboard({ run }: { run?: SimulationRun }) {
  if (!run) return <section className="glass rounded-2xl p-5">Run a simulation to view dashboard outputs.</section>;
  const cards = [
    ['Political Score', run.politicalScore],
    ['Risk Score', run.riskScore],
    ['Stability Score', run.stabilityScore],
    ['Vote Swing %', run.voteSwingPct]
  ];
  return (
    <section className="glass rounded-2xl p-5 space-y-3">
      <h2 className="text-xl font-semibold">2) Simulation Dashboard</h2>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">{cards.map(([k, v]) => <div key={k} className="rounded-xl bg-slate-900/70 p-3"><p className="text-xs text-slate-400">{k}</p><p className="text-2xl font-bold">{v}</p></div>)}</div>
      <p className="text-sm text-slate-300">Foresight: Best {run.foresight.best} • Probable {run.foresight.probable} • Worst {run.foresight.worst} • Confidence {run.confidence.toUpperCase()}</p>
    </section>
  );
}
