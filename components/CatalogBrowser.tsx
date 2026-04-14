'use client';

import { useEffect, useState } from 'react';
import { ScenarioInput } from '@/lib/types';

type CatalogResponse = {
  page: number;
  pageSize: number;
  total: number;
  totalPages: number;
  hasNext: boolean;
  hasPrev: boolean;
  data: ScenarioInput[];
};

export function CatalogBrowser({ onSelect }: { onSelect: (scenario: ScenarioInput) => void }) {
  const [search, setSearch] = useState('');
  const [query, setQuery] = useState('');
  const [page, setPage] = useState(1);
  const [catalog, setCatalog] = useState<CatalogResponse | null>(null);

  useEffect(() => {
    const url = `/api/catalog?page=${page}&pageSize=12&q=${encodeURIComponent(query)}`;
    fetch(url).then((r) => r.json()).then(setCatalog);
  }, [page, query]);

  return (
    <section className="glass rounded-2xl p-5 space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-xl font-semibold">Scenario Catalog (4K+ Library)</h2>
        <div className="flex gap-2">
          <input
            className="rounded-lg bg-slate-900/70 p-2 text-sm"
            placeholder="Search scenarios..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <button
            onClick={() => {
              setPage(1);
              setQuery(search.trim());
            }}
            className="rounded-lg bg-white/10 px-3"
          >
            Search
          </button>
        </div>
      </div>

      <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-3">
        {(catalog?.data ?? []).map((s) => (
          <button
            key={`${s.title}-${s.estimatedCost}-${s.beneficiariesM}`}
            onClick={() => onSelect(s)}
            className="text-left rounded-xl bg-slate-900/60 p-3 border border-white/10 hover:border-cyan-300/40"
          >
            <p className="font-semibold">{s.title}</p>
            <p className="text-xs text-slate-400 mt-1 line-clamp-3">{s.policyText}</p>
            <p className="text-xs mt-2">{s.regionFocus.toUpperCase()} • {s.launchWindow} • ₹{s.estimatedCost}cr</p>
          </button>
        ))}
      </div>

      <div className="flex items-center justify-between text-sm">
        <p className="text-slate-400">{catalog ? `Page ${catalog.page}/${catalog.totalPages} • ${catalog.total} results` : 'Loading...'}</p>
        <div className="flex gap-2">
          <button disabled={!catalog?.hasPrev} onClick={() => setPage((p) => Math.max(1, p - 1))} className="rounded bg-white/10 px-3 py-1 disabled:opacity-40">Prev</button>
          <button disabled={!catalog?.hasNext} onClick={() => setPage((p) => p + 1)} className="rounded bg-white/10 px-3 py-1 disabled:opacity-40">Next</button>
        </div>
      </div>
    </section>
  );
}
