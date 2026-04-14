'use client';

import { useEffect, useState } from 'react';
import { AdversarialTimeline } from '@/components/AdversarialTimeline';
import { ApiStatusPanel } from '@/components/ApiStatusPanel';
import { CatalogBrowser } from '@/components/CatalogBrowser';
import { OfflineRecoveryPanel } from '@/components/OfflineRecoveryPanel';
import { ScenarioBuilder } from '@/components/ScenarioBuilder';
import { ScenarioComparison } from '@/components/ScenarioComparison';
import { SimulationDashboard } from '@/components/SimulationDashboard';
import { StrategyOutputPanel } from '@/components/StrategyOutputPanel';
import {
  buildOfflineRunFromScenario,
  DEFAULT_OFFLINE_STATE,
  detectOfflineStatus,
  loadLastScenario,
  loadRunHistorySnapshot,
  resilientFetch,
  saveLastScenario,
  saveRunSnapshot
} from '@/lib/offlineRecovery';
import { useLabStore } from '@/lib/store';
import { ScenarioInput, SimulationRun } from '@/lib/types';

export default function HomePage() {
  const { currentInput, setInput, runs, setRuns, addRun } = useLabStore();
  const [loading, setLoading] = useState(false);
  const currentRun = runs[0];
  const [sentimentSummary, setSentimentSummary] = useState<{ latest?: number; topic?: string }>({});
  const [error, setError] = useState<string>('');
  const [offlineState, setOfflineState] = useState(DEFAULT_OFFLINE_STATE);

  useEffect(() => {
    const offline = detectOfflineStatus();
    setOfflineState(offline);

    const snapshotRuns = loadRunHistorySnapshot();
    if (snapshotRuns.length > 0) setRuns(snapshotRuns);

    const lastScenario = loadLastScenario();
    if (lastScenario) setInput(lastScenario);

    resilientFetch<SimulationRun[]>('/api/scenarios', snapshotRuns).then((data) => {
      if (Array.isArray(data) && data.length > 0) setRuns(data);
    });

    resilientFetch<{ yearly: { sentiment: number }[]; topicWise: { topic: string }[] }>(
      '/api/history',
      { yearly: [{ sentiment: 58 }], topicWise: [{ topic: 'subsidy' }] }
    ).then((d) => {
      setSentimentSummary({ latest: d.yearly.at(-1)?.sentiment, topic: d.topicWise[0]?.topic });
    });
  }, [setInput, setRuns]);

  const runUnifiedSimulation = async () => {
    setLoading(true);
    setError('');
    saveLastScenario(currentInput);

    const fallbackRun = buildOfflineRunFromScenario(currentInput);

    const data = await resilientFetch<SimulationRun>(
      '/api/simulate',
      fallbackRun,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(currentInput)
      },
      7000
    );

    if (data.id.startsWith('offline_')) {
      setOfflineState({ isOffline: true, fallbackReason: 'Primary simulation endpoint unavailable. Offline safe simulation generated.' });
      setError('You are in offline-safe mode. Results are synthetic fallback estimates.');
    }

    addRun(data);
    saveRunSnapshot(data);
    setLoading(false);
  };

  return (
    <main className="mx-auto max-w-7xl p-4 md:p-8 space-y-4">
      <header className="glass rounded-2xl p-5">
        <h1 className="text-3xl font-bold">Bharat Political Strategy Lab</h1>
        <p className="text-slate-300">AI-powered political, policy, and governance simulation sandbox (India, aggregated, 2000–2025 opinion layer).</p>
        <p className="text-xs text-slate-400 mt-2">Current public mood proxy: {sentimentSummary.latest ?? '--'} | Dominant baseline topic: {sentimentSummary.topic ?? '--'}</p>
      </header>

      <OfflineRecoveryPanel offline={offlineState} />
      <ApiStatusPanel />
      <CatalogBrowser onSelect={(scenario: ScenarioInput) => setInput(scenario)} />

      <ScenarioBuilder onRun={runUnifiedSimulation} />

      {error && <div className="glass rounded-xl p-3 border border-rose-300/40 text-rose-200">{error}</div>}
      {loading && <div className="glass rounded-xl p-3">Running simulation...</div>}

      <SimulationDashboard run={currentRun} />
      <AdversarialTimeline run={currentRun} />
      <StrategyOutputPanel run={currentRun} />

      {currentRun && (
        <section className="glass rounded-2xl p-5 grid md:grid-cols-2 gap-4 text-sm">
          <div>
            <h3 className="font-semibold mb-2">Segment Impact Table</h3>
            <table className="w-full">
              <thead><tr className="text-left text-slate-400"><th>Segment</th><th>Impact</th><th>Contribution</th></tr></thead>
              <tbody>{currentRun.segmentImpact.map((s) => <tr className="border-t border-white/10" key={s.segment}><td>{s.segment}</td><td>{s.impactScore}</td><td>{s.contribution}</td></tr>)}</tbody>
            </table>
          </div>
          <div>
            <h3 className="font-semibold mb-2">Risk Analysis & Stress Points</h3>
            <p>Thresholds:</p><ul className="list-disc pl-4">{currentRun.stress.thresholds.map((t) => <li key={t}>{t}</li>)}</ul>
            <p className="mt-2">Flags:</p><ul className="list-disc pl-4">{currentRun.stress.flags.map((f) => <li key={f}>{f}</li>)}</ul>
            <p className="mt-2">Weak points:</p><ul className="list-disc pl-4">{currentRun.adversarial.weakPoints.map((w) => <li key={w}>{w}</li>)}</ul>
            {currentRun.manifesto && <><p className="mt-2">Manifesto Feasibility: {currentRun.manifesto.feasibilityScore}</p><ul className="list-disc pl-4">{currentRun.manifesto.optimized.map((o) => <li key={o}>{o}</li>)}</ul></>}
            {currentRun.aiBrief && <>
              <p className="mt-3 font-semibold">AI Strategy Bot Brief</p>
              <p>{currentRun.aiBrief.executiveSummary}</p>
              <ul className="list-disc pl-4">{currentRun.aiBrief.recommendedActions.map((a) => <li key={a}>{a}</li>)}</ul>
            </>}
          </div>
        </section>
      )}

      <ScenarioComparison />
    </main>
  );
}
