import { ScenarioInput, SimulationRun } from './types';

const STORAGE_KEYS = {
  LAST_RUN: 'bpsl:last-run',
  RUN_HISTORY: 'bpsl:run-history',
  LAST_SCENARIO: 'bpsl:last-scenario'
} as const;

export const OFFLINE_SCENARIO_FALLBACKS: ScenarioInput[] = [
  {
    title: 'Offline Fallback: Free Electricity Lite',
    policyText: 'Deliver 100 units free electricity for low-income households with prepaid smart verification.',
    launchWindow: '6m',
    regionFocus: 'national',
    visibility: 72,
    estimatedCost: 21000,
    beneficiariesM: 26,
    manifestoText: ''
  },
  {
    title: 'Offline Fallback: Jobs + Skill Blend',
    policyText: 'District jobs and skilling mix with verified attendance and digital wage transfer.',
    launchWindow: '3m',
    regionFocus: 'north',
    visibility: 69,
    estimatedCost: 33000,
    beneficiariesM: 19,
    manifestoText: ''
  },
  {
    title: 'Offline Fallback: Urban Cost Relief',
    policyText: 'Transit + rental support for urban working families using Aadhaar-seeded eligibility checks.',
    launchWindow: '1m',
    regionFocus: 'west',
    visibility: 66,
    estimatedCost: 28000,
    beneficiariesM: 14,
    manifestoText: ''
  }
];

export type OfflineState = {
  isOffline: boolean;
  fallbackReason: string;
};

export const DEFAULT_OFFLINE_STATE: OfflineState = {
  isOffline: false,
  fallbackReason: ''
};

function isBrowser() {
  return typeof window !== 'undefined';
}

export function saveLastScenario(input: ScenarioInput) {
  if (!isBrowser()) return;
  try {
    localStorage.setItem(STORAGE_KEYS.LAST_SCENARIO, JSON.stringify(input));
  } catch {
    // ignore storage failures in restrictive browsers
  }
}

export function loadLastScenario(): ScenarioInput | null {
  if (!isBrowser()) return null;
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.LAST_SCENARIO);
    if (!raw) return null;
    return JSON.parse(raw) as ScenarioInput;
  } catch {
    return null;
  }
}

export function saveRunSnapshot(run: SimulationRun) {
  if (!isBrowser()) return;
  try {
    localStorage.setItem(STORAGE_KEYS.LAST_RUN, JSON.stringify(run));
    const existingRaw = localStorage.getItem(STORAGE_KEYS.RUN_HISTORY);
    const existing = existingRaw ? (JSON.parse(existingRaw) as SimulationRun[]) : [];
    const merged = [run, ...existing].slice(0, 20);
    localStorage.setItem(STORAGE_KEYS.RUN_HISTORY, JSON.stringify(merged));
  } catch {
    // ignore
  }
}

export function loadRunSnapshot(): SimulationRun | null {
  if (!isBrowser()) return null;
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.LAST_RUN);
    if (!raw) return null;
    return JSON.parse(raw) as SimulationRun;
  } catch {
    return null;
  }
}

export function loadRunHistorySnapshot(): SimulationRun[] {
  if (!isBrowser()) return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.RUN_HISTORY);
    if (!raw) return [];
    return JSON.parse(raw) as SimulationRun[];
  } catch {
    return [];
  }
}

export function detectOfflineStatus(): OfflineState {
  if (!isBrowser()) return DEFAULT_OFFLINE_STATE;
  if (navigator.onLine) return DEFAULT_OFFLINE_STATE;
  return {
    isOffline: true,
    fallbackReason: 'Network unavailable. Running from local snapshots and fallback catalog.'
  };
}

export function resilientFetch<T>(
  url: string,
  fallback: T,
  options?: RequestInit,
  timeoutMs = 5000
): Promise<T> {
  return new Promise((resolve) => {
    if (!isBrowser()) {
      resolve(fallback);
      return;
    }

    const timer = setTimeout(() => resolve(fallback), timeoutMs);

    fetch(url, options)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error('bad response'))))
      .then((data) => {
        clearTimeout(timer);
        resolve(data as T);
      })
      .catch(() => {
        clearTimeout(timer);
        resolve(fallback);
      });
  });
}

export function buildOfflineRunFromScenario(input: ScenarioInput): SimulationRun {
  const syntheticScore = Math.max(35, Math.min(75, Math.round(50 + (input.visibility - 50) * 0.2)));
  const risk = Math.max(25, Math.min(80, Math.round(55 - (input.visibility - 50) * 0.15)));
  const stability = Math.max(35, Math.min(85, Math.round(48 + (input.beneficiariesM / 2))));

  return {
    id: `offline_${Date.now()}`,
    createdAt: new Date().toISOString(),
    input,
    politicalScore: syntheticScore,
    riskScore: risk,
    stabilityScore: stability,
    voteSwingPct: Number(((syntheticScore - 50) * 0.2).toFixed(2)),
    confidence: 'low',
    foresight: {
      best: Math.min(100, syntheticScore + 8),
      probable: syntheticScore,
      worst: Math.max(0, syntheticScore - 10)
    },
    adversarial: {
      approvalDelta: Number(((syntheticScore - 50) / 5).toFixed(2)),
      weakPoints: ['Offline synthetic mode: live adversarial intelligence unavailable.'],
      counters: ['Reconnect to network for full simulation fidelity.'],
      timeline: [
        { round: 'Policy Announcement', effect: 1, note: 'Offline estimated announcement effect.' },
        { round: 'Opposition Counter', effect: -0.8, note: 'Offline estimated counter effect.' },
        { round: 'Media Framing', effect: 0.4, note: 'Offline estimated media effect.' },
        { round: 'Public Sentiment Shift', effect: 0.2, note: 'Offline estimated sentiment effect.' }
      ]
    },
    stress: {
      thresholds: ['Offline mode threshold placeholder: reconnect for precise stress limits.'],
      flags: ['Offline simulation mode active'],
      scenarios: ['Limited-mode scenario']
    },
    narratives: {
      pro: ['Offline mode: broad policy support narrative generated.'],
      opposition: ['Offline mode: fiscal sustainability challenge likely.'],
      media: ['Offline mode: execution and transparency remain core framing.']
    },
    timeCurve: [12, 9, 6, 3, 1].map((m) => ({ monthToElection: m, approval: Number((50 + (syntheticScore - 50) * (1 - m / 20)).toFixed(2)) })),
    segmentImpact: [
      { segment: 'Welfare-sensitive (low income)', impactScore: syntheticScore, contribution: Number((syntheticScore * 0.3).toFixed(2)) },
      { segment: 'Aspirational middle class', impactScore: syntheticScore - 3, contribution: Number(((syntheticScore - 3) * 0.2).toFixed(2)) }
    ],
    perception: {
      visibilityScore: input.visibility,
      perceptionGapIndex: 25,
      politicallyRisky: true
    },
    manifesto: {
      feasibilityScore: 55,
      contradictions: ['Offline estimate; full contradiction analysis unavailable.'],
      optimized: ['Reconnect for full manifesto optimization output.']
    },
    aiBrief: {
      executiveSummary: 'Offline contingency summary generated from local fallback logic.',
      plainLanguage: ['This result is generated in offline safe mode with simplified assumptions.'],
      strategicQA: [{ question: 'How reliable is this?', answer: 'Use this as temporary fallback only.' }],
      recommendedActions: ['Retry simulation when network is available.']
    }
  };
}
