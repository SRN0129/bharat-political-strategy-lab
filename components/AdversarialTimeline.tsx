'use client';

import { SimulationRun } from '@/lib/types';

export function AdversarialTimeline({ run }: { run?: SimulationRun }) {
  if (!run) return null;
  return (
    <section className="glass rounded-2xl p-5 space-y-2">
      <h2 className="text-xl font-semibold">3) Adversarial Timeline View</h2>
      {run.adversarial.timeline.map((t) => (
        <div key={t.round} className="rounded-lg bg-slate-900/60 p-3">
          <div className="flex justify-between"><span>{t.round}</span><span className={t.effect >= 0 ? 'text-emerald-400' : 'text-rose-400'}>{t.effect.toFixed(2)}</span></div>
          <p className="text-sm text-slate-400">{t.note}</p>
        </div>
      ))}
    </section>
  );
}
