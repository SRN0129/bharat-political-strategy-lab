import { SimulationRun } from '@/lib/types';

type Tone = 'direct' | 'balanced' | 'cautious';

const clamp = (n: number, min: number, max: number) => Math.max(min, Math.min(max, n));

function inferTone(run: SimulationRun): Tone {
  if (run.riskScore >= 65) return 'cautious';
  if (run.politicalScore >= 60 && run.stabilityScore >= 58) return 'direct';
  return 'balanced';
}

function explainScoreBand(score: number): string {
  if (score >= 75) return 'very strong';
  if (score >= 60) return 'strong';
  if (score >= 50) return 'competitive';
  if (score >= 40) return 'fragile';
  return 'weak';
}

function explainRiskBand(score: number): string {
  if (score >= 70) return 'high';
  if (score >= 50) return 'medium';
  return 'low';
}

function explainConfidence(run: SimulationRun): string {
  if (run.confidence === 'high') return 'Model confidence is high because risk concentration is controlled.';
  if (run.confidence === 'medium') return 'Model confidence is moderate; gains depend on narrative discipline and implementation speed.';
  return 'Model confidence is low due to concentrated downside and volatility risk.';
}

function openingLine(run: SimulationRun, tone: Tone): string {
  const scoreBand = explainScoreBand(run.politicalScore);
  if (tone === 'direct') {
    return `This scenario is currently ${scoreBand}: it can win narrative momentum if execution remains visible and disciplined.`;
  }
  if (tone === 'cautious') {
    return `This scenario is ${scoreBand} but vulnerable; political upside exists only if risk controls are activated immediately.`;
  }
  return `This scenario is ${scoreBand} and contestable; outcome quality depends on timing, framing, and operational credibility.`;
}

function buildPlainLanguage(run: SimulationRun): string[] {
  const lines: string[] = [];
  lines.push(`Political score is ${run.politicalScore}/100, which indicates ${explainScoreBand(run.politicalScore)} political positioning.`);
  lines.push(`Risk score is ${run.riskScore}/100 (${explainRiskBand(run.riskScore)} risk); biggest pressure points are fiscal stress, leakage, and narrative volatility.`);
  lines.push(`Expected vote swing proxy is ${run.voteSwingPct}% under current assumptions and region weighting.`);
  lines.push(`Best/probable/worst foresight path is ${run.foresight.best}/${run.foresight.probable}/${run.foresight.worst}.`);
  lines.push(explainConfidence(run));
  if (run.perception.politicallyRisky) {
    lines.push('Perception gap warning: economic logic is stronger than public visibility, so communication and proof-of-delivery must be intensified.');
  } else {
    lines.push('Perception gap is manageable: communication and delivery are reasonably aligned.');
  }
  return lines;
}

function buildStrategicQA(run: SimulationRun): { question: string; answer: string }[] {
  return [
    {
      question: 'What can make this fail fastest?',
      answer: run.stress.flags[0] ?? 'Execution slippage and narrative fragmentation.'
    },
    {
      question: 'What should be done in the first 30 days?',
      answer: run.adversarial.counters[0] ?? 'Publish a transparent fiscal and delivery roadmap.'
    },
    {
      question: 'How should opposition attacks be neutralized?',
      answer: 'Use audited metrics, beneficiary evidence, and constituency-level grievance closure reports every week.'
    },
    {
      question: 'When should launch be prioritized?',
      answer: `Use the time-curve peak and current model confidence (${run.confidence.toUpperCase()}) to finalize launch timing.`
    },
    {
      question: 'How should this be explained to undecided voters?',
      answer: 'One promise, one number, one deadline, and one local beneficiary proof story.'
    }
  ];
}

function recommendedActions(run: SimulationRun): string[] {
  const actions = [
    'Publish a 90-day delivery scorecard with state and district cuts.',
    'Run a narrative matrix: pro-message, opposition rebuttal, and neutral media proof points.',
    'Create a fiscal guardrail note with explicit annual caps and fallback triggers.',
    'Instrument rapid feedback loops to reduce leakage and complaints.'
  ];

  if (run.riskScore >= 60) {
    actions.push('Activate a war-room model for risk hotspots and grievance redressal response within 72 hours.');
  }
  if (run.perception.perceptionGapIndex >= 20) {
    actions.push('Prioritize visibility interventions: local demos, transparent beneficiary lists, and simple eligibility messaging.');
  }
  if (run.voteSwingPct < 0) {
    actions.push('Delay rollout or redesign policy packaging before high-visibility announcement.');
  }
  return actions;
}

function synthesizeExecutiveSummary(run: SimulationRun): string {
  const tone = inferTone(run);
  const intro = openingLine(run, tone);
  const risk = `Current risk is ${explainRiskBand(run.riskScore)} and stability is ${run.stabilityScore}/100.`;
  const window = `Time-shift outputs indicate that execution quality and narrative discipline are critical across the pre-election window.`;
  return `${intro} ${risk} ${window}`;
}

export function generateAIBrief(run: SimulationRun) {
  const executiveSummary = synthesizeExecutiveSummary(run);
  const plainLanguage = buildPlainLanguage(run);
  const strategicQA = buildStrategicQA(run);
  const recommended = recommendedActions(run);

  return {
    executiveSummary,
    plainLanguage,
    strategicQA,
    recommendedActions: recommended
  };
}
