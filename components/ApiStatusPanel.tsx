'use client';

import { useEffect, useState } from 'react';

type Health = {
  status: string;
  now: string;
  datasets: { scenarios: number; sentimentYears: number; topics: number; storedRuns: number };
  freeTier: { paidApis: boolean; localStorage: boolean; cloudRequired: boolean };
};

export function ApiStatusPanel() {
  const [health, setHealth] = useState<Health | null>(null);

  useEffect(() => {
    fetch('/api/health').then((r) => r.json()).then(setHealth).catch(() => setHealth(null));
  }, []);

  return (
    <section className="glass rounded-2xl p-5">
      <h2 className="text-xl font-semibold">Platform Health & Free-Tier Diagnostics</h2>
      {!health ? (
        <p className="text-sm text-slate-400 mt-2">Loading health diagnostics…</p>
      ) : (
        <div className="mt-3 grid md:grid-cols-2 gap-4 text-sm">
          <div className="rounded-xl bg-slate-900/60 p-3">
            <p>Status: <strong className="text-emerald-400">{health.status.toUpperCase()}</strong></p>
            <p>Data catalog size: {health.datasets.scenarios} scenarios</p>
            <p>Sentiment years: {health.datasets.sentimentYears}</p>
            <p>Topic sentiment entries: {health.datasets.topics}</p>
            <p>Stored simulation memory: {health.datasets.storedRuns} runs</p>
          </div>
          <div className="rounded-xl bg-slate-900/60 p-3">
            <p>Paid APIs used: <strong>{health.freeTier.paidApis ? 'Yes' : 'No'}</strong></p>
            <p>Local JSON storage: <strong>{health.freeTier.localStorage ? 'Enabled' : 'Disabled'}</strong></p>
            <p>Cloud storage required: <strong>{health.freeTier.cloudRequired ? 'Yes' : 'No'}</strong></p>
            <p className="text-slate-400 mt-2">Timestamp: {new Date(health.now).toLocaleString()}</p>
          </div>
        </div>
      )}
    </section>
  );
}
