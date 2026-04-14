import { NextResponse } from 'next/server';
import scenarios from '@/data/sample_scenarios.json';

function parseIntSafe(v: string | null, fallback: number) {
  const n = Number(v);
  return Number.isFinite(n) ? Math.floor(n) : fallback;
}

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const page = Math.max(1, parseIntSafe(searchParams.get('page'), 1));
  const pageSize = Math.min(50, Math.max(5, parseIntSafe(searchParams.get('pageSize'), 12)));
  const q = (searchParams.get('q') ?? '').toLowerCase().trim();

  const filtered = q
    ? scenarios.filter((s) => s.title.toLowerCase().includes(q) || s.policyText.toLowerCase().includes(q))
    : scenarios;

  const total = filtered.length;
  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  const boundedPage = Math.min(page, totalPages);
  const start = (boundedPage - 1) * pageSize;
  const data = filtered.slice(start, start + pageSize);

  return NextResponse.json({
    page: boundedPage,
    pageSize,
    total,
    totalPages,
    hasNext: boundedPage < totalPages,
    hasPrev: boundedPage > 1,
    data
  });
}
