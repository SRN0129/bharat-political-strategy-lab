import { create } from 'zustand';
import { ScenarioInput, SimulationRun } from './types';

type LabState = {
  currentInput: ScenarioInput;
  runs: SimulationRun[];
  setInput: (patch: Partial<ScenarioInput>) => void;
  setRuns: (runs: SimulationRun[]) => void;
  addRun: (run: SimulationRun) => void;
};

export const defaultInput: ScenarioInput = {
  title: 'Free Electricity Policy',
  policyText: 'Provide up to 200 units of free electricity per household for low-income families.',
  launchWindow: '6m',
  regionFocus: 'national',
  visibility: 80,
  estimatedCost: 38000,
  beneficiariesM: 42,
  manifestoText: ''
};

export const useLabStore = create<LabState>((set) => ({
  currentInput: defaultInput,
  runs: [],
  setInput: (patch) => set((state) => ({ currentInput: { ...state.currentInput, ...patch } })),
  setRuns: (runs) => set({ runs }),
  addRun: (run) => set((state) => ({ runs: [run, ...state.runs].slice(0, 50) }))
}));
