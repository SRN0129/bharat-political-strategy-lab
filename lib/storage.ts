import fs from 'fs';
import path from 'path';
import { SimulationRun } from './types';

const dbPath = path.join(process.cwd(), 'data', 'simulations.json');

export function readRuns(): SimulationRun[] {
  try {
    const raw = fs.readFileSync(dbPath, 'utf-8');
    return JSON.parse(raw) as SimulationRun[];
  } catch {
    return [];
  }
}

export function saveRun(run: SimulationRun) {
  const runs = readRuns();
  runs.unshift(run);
  fs.writeFileSync(dbPath, JSON.stringify(runs.slice(0, 50), null, 2));
  return runs;
}
