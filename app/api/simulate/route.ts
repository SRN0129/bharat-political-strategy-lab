import { NextResponse } from 'next/server';
import { runSimulation } from '@/lib/simulationEngine';
import { saveRun } from '@/lib/storage';
import { validateScenarioInput } from '@/lib/validators';

export async function POST(req: Request) {
  try {
    const payload = await req.json();
    const validated = validateScenarioInput(payload);

    if (!validated.ok) {
      return NextResponse.json(
        { error: 'Invalid scenario input', details: validated.errors },
        { status: 400 }
      );
    }

    const run = runSimulation(validated.data);
    saveRun(run);
    return NextResponse.json(run);
  } catch {
    return NextResponse.json(
      { error: 'Simulation request could not be processed.' },
      { status: 500 }
    );
  }
}
