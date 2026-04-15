export type Confidence = 'low' | 'medium' | 'high';
export type LaunchWindow = '12m' | '6m' | '3m' | '1m';

export type ScenarioInput = {
  title: string;
  policyText: string;
  launchWindow: LaunchWindow;
  regionFocus: 'national' | 'north' | 'south' | 'east' | 'west';
  visibility: number;
  estimatedCost: number;
  beneficiariesM: number;
  manifestoText?: string;
};

export type FactorWeight = {
  key: string;
  label: string;
  category: 'economic' | 'demographic' | 'political' | 'behavioral' | 'media' | 'timing' | 'implementation';
  weight: number;
};

export type SegmentImpact = {
  segment: string;
  impactScore: number;
  contribution: number;
};

export type SimulationRun = {
  id: string;
  createdAt: string;
  input: ScenarioInput;
  politicalScore: number;
  riskScore: number;
  stabilityScore: number;
  voteSwingPct: number;
  confidence: Confidence;
  foresight: { best: number; probable: number; worst: number };
  adversarial: {
    approvalDelta: number;
    weakPoints: string[];
    counters: string[];
    timeline: { round: string; effect: number; note: string }[];
  };
  stress: {
    thresholds: string[];
    flags: string[];
    scenarios: string[];
  };
  narratives: {
    pro: string[];
    opposition: string[];
    media: string[];
  };
  timeCurve: { monthToElection: number; approval: number }[];
  segmentImpact: SegmentImpact[];
  perception: {
    visibilityScore: number;
    perceptionGapIndex: number;
    politicallyRisky: boolean;
  };
  manifesto?: {
    feasibilityScore: number;
    contradictions: string[];
    optimized: string[];
  };

  aiBrief?: {
    executiveSummary: string;
    plainLanguage: string[];
    strategicQA: { question: string; answer: string }[];
    recommendedActions: string[];
  };

  monteCarlo?: {
    simulations: number;
    p10: number;
    p50: number;
    p90: number;
    winProbability: number;
  };
  messageTesting?: {
    variants: { message: string; persuasionScore: number; backlashRisk: number }[];
    bestVariant: string;
  };

  supporterFunnel?: {
    awareness: number;
    consideration: number;
    support: number;
    turnoutIntent: number;
  };
  consensusMap?: {
    clusters: { name: string; size: number; alignment: number }[];
    bridgingNarrative: string;
  };

  commitmentTracker?: {
    tickets: { id: string; issue: string; owner: string; priority: 'low' | 'medium' | 'high'; status: 'open' | 'in_progress' | 'resolved' }[];
    unresolvedCount: number;
  };
};
