import { NextResponse } from 'next/server';
import { runSimulation } from '@/lib/simulationEngine';

export async function POST(req: Request) {
  const { manifestoText } = await req.json();
  const run = runSimulation({
    title: 'Manifesto draft',
    policyText: manifestoText,
    launchWindow: '6m',
    regionFocus: 'national',
    visibility: 60,
    estimatedCost: 25000,
    beneficiariesM: 20,
    manifestoText
  });
  return NextResponse.json(run.manifesto ?? null);
}
