'use client';

import { motion } from 'framer-motion';
import { useLabStore } from '@/lib/store';

export function ScenarioBuilder({ onRun }: { onRun: () => void }) {
  const { currentInput, setInput } = useLabStore();

  return (
    <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="glass rounded-2xl p-5 space-y-4">
      <h2 className="text-xl font-semibold">1) Scenario Builder</h2>
      <input className="w-full rounded-lg bg-slate-900/70 p-2" value={currentInput.title} onChange={(e) => setInput({ title: e.target.value })} placeholder="Scenario title" />
      <textarea className="w-full min-h-24 rounded-lg bg-slate-900/70 p-2" value={currentInput.policyText} onChange={(e) => setInput({ policyText: e.target.value })} placeholder="Policy / campaign idea" />
      <textarea className="w-full min-h-20 rounded-lg bg-slate-900/70 p-2" value={currentInput.manifestoText} onChange={(e) => setInput({ manifestoText: e.target.value })} placeholder="Manifesto text (optional)" />
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <label className="text-sm">Visibility: {currentInput.visibility}
          <input type="range" min={0} max={100} value={currentInput.visibility} onChange={(e) => setInput({ visibility: Number(e.target.value) })} className="w-full" />
        </label>
        <label className="text-sm">Cost (₹ crore)
          <input className="w-full rounded bg-slate-900/70 p-2" type="number" value={currentInput.estimatedCost} onChange={(e) => setInput({ estimatedCost: Number(e.target.value) })} />
        </label>
        <label className="text-sm">Beneficiaries (M)
          <input className="w-full rounded bg-slate-900/70 p-2" type="number" value={currentInput.beneficiariesM} onChange={(e) => setInput({ beneficiariesM: Number(e.target.value) })} />
        </label>
        <label className="text-sm">Launch Timing
          <select className="w-full rounded bg-slate-900/70 p-2" value={currentInput.launchWindow} onChange={(e) => setInput({ launchWindow: e.target.value as any })}>
            <option value="12m">1 year</option><option value="6m">6 months</option><option value="3m">3 months</option><option value="1m">1 month</option>
          </select>
        </label>
      </div>
      <div className="flex flex-wrap gap-2 items-center">
        <button onClick={onRun} className="rounded-lg bg-cyan-500 px-4 py-2 text-slate-950 font-semibold">Run Unified Simulation</button>
        <p className="text-xs text-slate-400">Tip: Use the Scenario Catalog above to load from 4,003 prebuilt samples.</p>
      </div>
    </motion.section>
  );
}
