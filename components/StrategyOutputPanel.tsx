'use client';

import { useMemo, useState } from 'react';
import { SimulationRun } from '@/lib/types';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  PolarAngleAxis,
  PolarGrid,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from 'recharts';

const tabs = ['Narratives', 'Risk Matrix', 'Segment Intelligence', 'AI Explainer', 'Monte Carlo + Message Lab'] as const;

type Tab = (typeof tabs)[number];

export function StrategyOutputPanel({ run }: { run?: SimulationRun }) {
  const [activeTab, setActiveTab] = useState<Tab>('Narratives');
  if (!run) return null;

  const riskMatrix = useMemo(
    () => [
      { axis: 'Political', value: run.politicalScore },
      { axis: 'Risk', value: 100 - run.riskScore },
      { axis: 'Stability', value: run.stabilityScore },
      { axis: 'Visibility', value: run.perception.visibilityScore },
      { axis: 'Perception Align', value: 100 - run.perception.perceptionGapIndex }
    ],
    [run]
  );

  const segmentBars = useMemo(
    () => run.segmentImpact.map((s) => ({ segment: s.segment.slice(0, 24), impact: s.impactScore, contribution: s.contribution })),
    [run]
  );

  return (
    <section className="glass rounded-2xl p-5 space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-xl font-semibold">4) Strategy Output Panel — Advanced Command Center</h2>
        <div className="flex gap-2 flex-wrap">
          {tabs.map((t) => (
            <button
              key={t}
              onClick={() => setActiveTab(t)}
              className={`rounded-lg px-3 py-1.5 text-sm ${activeTab === t ? 'bg-cyan-400 text-slate-950 font-semibold' : 'bg-white/10'}`}
            >
              {t}
            </button>
          ))}
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-4">
        <div className="rounded-xl bg-slate-900/60 p-3 h-64">
          <p className="text-sm mb-1 text-slate-300">Approval vs Time Curve (Launch sensitivity)</p>
          <ResponsiveContainer>
            <LineChart data={run.timeCurve}>
              <CartesianGrid strokeDasharray="3 3" stroke="#233" />
              <XAxis dataKey="monthToElection" />
              <YAxis domain={[35, 80]} />
              <Tooltip />
              <Line dataKey="approval" stroke="#22d3ee" strokeWidth={2.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="rounded-xl bg-slate-900/60 p-3 h-64">
          <p className="text-sm mb-1 text-slate-300">Scenario Power Polygon</p>
          <ResponsiveContainer>
            <RadarChart data={riskMatrix}>
              <PolarGrid />
              <PolarAngleAxis dataKey="axis" />
              <Radar dataKey="value" stroke="#a78bfa" fill="#a78bfa" fillOpacity={0.35} />
              <Tooltip />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {activeTab === 'Narratives' && (
        <div className="grid md:grid-cols-3 gap-3 text-sm">
          <div><h3 className="font-semibold text-emerald-300">Pro-policy narrative</h3><ul className="list-disc pl-4">{run.narratives.pro.map((n) => <li key={n}>{n}</li>)}</ul></div>
          <div><h3 className="font-semibold text-rose-300">Opposition attack lines</h3><ul className="list-disc pl-4">{run.narratives.opposition.map((n) => <li key={n}>{n}</li>)}</ul></div>
          <div><h3 className="font-semibold text-slate-300">Neutral media framing</h3><ul className="list-disc pl-4">{run.narratives.media.map((n) => <li key={n}>{n}</li>)}</ul></div>
        </div>
      )}

      {activeTab === 'Risk Matrix' && (
        <div className="grid md:grid-cols-2 gap-4">
          <div className="rounded-xl bg-slate-900/60 p-3 h-64">
            <p className="text-sm mb-1 text-slate-300">Stress/Score Heat Bar</p>
            <ResponsiveContainer>
              <BarChart data={[{ m: 'Political', v: run.politicalScore }, { m: 'Risk', v: run.riskScore }, { m: 'Stability', v: run.stabilityScore }, { m: 'Gap', v: run.perception.perceptionGapIndex }]}>
                <CartesianGrid strokeDasharray="3 3" stroke="#233" />
                <XAxis dataKey="m" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="v" fill="#22d3ee" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="rounded-xl bg-slate-900/60 p-3">
            <p className="font-semibold mb-2">Critical Stress Notes</p>
            <ul className="list-disc pl-4 text-sm space-y-1">{run.stress.thresholds.map((t) => <li key={t}>{t}</li>)}</ul>
            <p className="font-semibold mt-3 mb-2">Risk Flags</p>
            <ul className="list-disc pl-4 text-sm space-y-1">{run.stress.flags.map((f) => <li key={f}>{f}</li>)}</ul>
          </div>
        </div>
      )}

      {activeTab === 'Segment Intelligence' && (
        <div className="rounded-xl bg-slate-900/60 p-3 h-72">
          <p className="text-sm mb-1 text-slate-300">Segment Impact + Contribution Intelligence</p>
          <ResponsiveContainer>
            <AreaChart data={segmentBars}>
              <CartesianGrid strokeDasharray="3 3" stroke="#233" />
              <XAxis dataKey="segment" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Area type="monotone" dataKey="impact" stroke="#22d3ee" fill="#22d3ee" fillOpacity={0.2} />
              <Area type="monotone" dataKey="contribution" stroke="#a78bfa" fill="#a78bfa" fillOpacity={0.2} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      {activeTab === 'AI Explainer' && (
        <div className="grid md:grid-cols-2 gap-3 text-sm">
          <div className="rounded-xl bg-slate-900/60 p-3">
            <h3 className="font-semibold text-cyan-300">Executive AI Brief</h3>
            <p className="mt-2">{run.aiBrief?.executiveSummary ?? 'No AI brief available for this run.'}</p>
            <ul className="list-disc pl-4 mt-2">{(run.aiBrief?.plainLanguage ?? []).map((l) => <li key={l}>{l}</li>)}</ul>
          </div>
          <div className="rounded-xl bg-slate-900/60 p-3">
            <h3 className="font-semibold text-violet-300">Strategic Q&A + Recommended Actions</h3>
            <ul className="list-disc pl-4 mt-2">{(run.aiBrief?.strategicQA ?? []).map((qa) => <li key={qa.question}><strong>{qa.question}</strong> {qa.answer}</li>)}</ul>
            <ul className="list-disc pl-4 mt-2">{(run.aiBrief?.recommendedActions ?? []).map((a) => <li key={a}>{a}</li>)}</ul>
          </div>
        </div>
      )}


      {activeTab === 'Monte Carlo + Message Lab' && (
        <div className="grid md:grid-cols-2 gap-3 text-sm">
          <div className="rounded-xl bg-slate-900/60 p-3">
            <h3 className="font-semibold text-cyan-300">Monte Carlo Uncertainty Layer</h3>
            <p>Simulations: {run.monteCarlo?.simulations ?? 0}</p>
            <p>P10 / P50 / P90: {run.monteCarlo?.p10 ?? '--'} / {run.monteCarlo?.p50 ?? '--'} / {run.monteCarlo?.p90 ?? '--'}</p>
            <p>Win Probability (score ≥ 55): {run.monteCarlo?.winProbability ?? '--'}%</p>
          </div>
          <div className="rounded-xl bg-slate-900/60 p-3">
            <h3 className="font-semibold text-violet-300">Message Testing Lab</h3>
            <p className="mb-2">Best Variant: {run.messageTesting?.bestVariant ?? '--'}</p>
            <ul className="list-disc pl-4">{(run.messageTesting?.variants ?? []).map((v) => <li key={v.message}><strong>{v.persuasionScore}</strong> persuasion / <strong>{v.backlashRisk}</strong> backlash — {v.message}</li>)}</ul>
            <p className="mt-3 font-semibold text-cyan-200">Supporter Funnel</p>
            <p>Aware {run.supporterFunnel?.awareness ?? '--'}% → Consider {run.supporterFunnel?.consideration ?? '--'}% → Support {run.supporterFunnel?.support ?? '--'}% → Turnout Intent {run.supporterFunnel?.turnoutIntent ?? '--'}%</p>
            <p className="mt-2 font-semibold text-cyan-200">Consensus Map Bridge</p>
            <p>{run.consensusMap?.bridgingNarrative ?? '--'}</p>
            <ul className="list-disc pl-4">{(run.consensusMap?.clusters ?? []).map((c) => <li key={c.name}>{c.name}: size {c.size}, alignment {c.alignment}</li>)}</ul>
            <p className="mt-3 font-semibold text-amber-200">Commitment Tracker (Campaign Issue Tickets)</p>
            <p>Unresolved: {run.commitmentTracker?.unresolvedCount ?? 0}</p>
            <ul className="list-disc pl-4">{(run.commitmentTracker?.tickets ?? []).map((t) => <li key={t.id}>{t.id} [{t.priority}] {t.issue} — {t.owner} ({t.status})</li>)}</ul>
          </div>
        </div>
      )}

      <p className="text-sm">Visibility Score: {run.perception.visibilityScore} | Perception Gap Index: {run.perception.perceptionGapIndex} | Risk Flag: {run.perception.politicallyRisky ? 'Economically sound, politically weak perception' : 'Balanced perception'}</p>
    </section>
  );
}
