'use client';

import { OfflineState } from '@/lib/offlineRecovery';

export function OfflineRecoveryPanel({ offline }: { offline: OfflineState }) {
  if (!offline.isOffline) return null;

  return (
    <section className="glass rounded-2xl p-4 border border-amber-300/40">
      <h2 className="text-lg font-semibold text-amber-200">Offline Resilience Mode Active</h2>
      <p className="text-sm text-amber-100 mt-1">{offline.fallbackReason}</p>
      <ul className="list-disc pl-5 mt-2 text-sm text-slate-200">
        <li>Using local catalog fallback and snapshot memory.</li>
        <li>Synthetic simulation output is provided so user flow never blocks.</li>
        <li>Reconnect to internet for full neural + adversarial precision.</li>
      </ul>
    </section>
  );
}
