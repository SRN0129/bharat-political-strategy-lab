'use client';

import { useMemo, useState } from 'react';
import { ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis, CartesianGrid } from 'recharts';
import { useLabStore } from '@/lib/store';

const metrics = ['politicalScore', 'riskScore', 'stabilityScore', 'voteSwingPct'] as const;

type Metric = (typeof metrics)[number];

export function ScenarioComparison() {
  const runs = useLabStore((s) => s.runs);
  const [xMetric, setXMetric] = useState<Metric>('politicalScore');
  const [yMetric, setYMetric] = useState<Metric>('stabilityScore');

  const top = useMemo(() => runs.slice(0, 8), [runs]);

  if (runs.length < 2) return <section className="glass rounded-2xl p-5">5) Scenario Comparison (run at least 2 simulations).</section>;

  const scatterData = top.map((r) => ({
    name: r.input.title.slice(0, 22),
    x: Number(r[xMetric]),
    y: Number(r[yMetric]),
    z: Math.max(10, r.politicalScore - r.riskScore + 30)
  }));

  const a = top[0];
  const b = top[1];

  return (
    <section className="glass rounded-2xl p-5 space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-xl font-semibold">5) Scenario Comparison — Multi-Run Intelligence</h2>
        <div className="flex gap-2 text-sm">
          <label>X:
            <select className="ml-1 rounded bg-slate-900/70 p-1" value={xMetric} onChange={(e) => setXMetric(e.target.value as Metric)}>
              {metrics.map((m) => <option key={m} value={m}>{m}</option>)}
            </select>
          </label>
          <label>Y:
            <select className="ml-1 rounded bg-slate-900/70 p-1" value={yMetric} onChange={(e) => setYMetric(e.target.value as Metric)}>
              {metrics.map((m) => <option key={m} value={m}>{m}</option>)}
            </select>
          </label>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-4">
        <div className="rounded-xl bg-slate-900/60 p-3 h-72">
          <p className="text-sm text-slate-300 mb-1">Scenario Positioning Map ({xMetric} vs {yMetric})</p>
          <ResponsiveContainer>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" stroke="#233" />
              <XAxis dataKey="x" name={xMetric} />
              <YAxis dataKey="y" name={yMetric} />
              <ZAxis dataKey="z" range={[80, 350]} />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} />
              <Scatter data={scatterData} fill="#22d3ee" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        <div className="rounded-xl bg-slate-900/60 p-3 overflow-auto">
          <p className="text-sm text-slate-300 mb-1">Top 8 Runs Rankboard</p>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-slate-400">
                <th>#</th><th>Title</th><th>PS</th><th>Risk</th><th>Stab</th><th>Swing%</th>
              </tr>
            </thead>
            <tbody>
              {top.map((r, idx) => (
                <tr key={r.id} className="border-t border-white/10">
                  <td>{idx + 1}</td>
                  <td>{r.input.title.slice(0, 26)}</td>
                  <td>{r.politicalScore}</td>
                  <td>{r.riskScore}</td>
                  <td>{r.stabilityScore}</td>
                  <td>{r.voteSwingPct}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="overflow-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-slate-400">
              <th>Metric</th><th>{a.input.title}</th><th>{b.input.title}</th><th>Diff (A-B)</th><th>Interpretation</th>
            </tr>
          </thead>
          <tbody>
            {[
              ['Political Score', a.politicalScore, b.politicalScore, 'Higher means stronger electoral positioning'],
              ['Risk Score', a.riskScore, b.riskScore, 'Lower means safer execution/political downside'],
              ['Stability Score', a.stabilityScore, b.stabilityScore, 'Higher means better resilience across shocks'],
              ['Vote Swing %', a.voteSwingPct, b.voteSwingPct, 'Higher indicates better expected conversion to votes'],
              ['Perception Gap', a.perception.perceptionGapIndex, b.perception.perceptionGapIndex, 'Lower means communication aligns with outcomes']
            ].map((r) => (
              <tr key={String(r[0])} className="border-t border-white/10">
                <td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{(Number(r[1]) - Number(r[2])).toFixed(2)}</td><td>{r[3]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
