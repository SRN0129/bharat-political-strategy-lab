import fs from 'fs';
import path from 'path';
import sentimentSeries from '@/data/sentiment_2000_2025.json';
import topicSentiment from '@/data/topic_sentiment.json';

export type NeuralInput = {
  fiscalCost: number;
  visibility: number;
  beneficiaries: number;
  controversy: number;
  adminCapacity: number;
  recencyBoost: number;
  mood: number;
  topicSensitivity: number;
};

export type NeuralOutput = {
  scoreDelta: number;
  riskDelta: number;
  stabilityDelta: number;
  confidence: number;
};

export type NeuralFeedback = {
  input: NeuralInput;
  heuristicTarget: {
    scoreDelta: number;
    riskDelta: number;
    stabilityDelta: number;
    confidence: number;
  };
};

type DatasetRow = {
  features: number[];
  target: number[];
};

type NetworkParams = {
  w1: number[][];
  b1: number[];
  w2: number[][];
  b2: number[];
  w3: number[][];
  b3: number[];
};

type NeuralMemoryStore = {
  version: string;
  rows: DatasetRow[];
  updates: number;
  lastUpdatedAt: string;
};

const INPUT_SIZE = 8;
const H1 = 20;
const H2 = 12;
const OUTPUT_SIZE = 4;
const MEMORY_PATH = path.join(process.cwd(), 'data', 'neural_memory.json');

const clamp = (v: number, min = -1, max = 1) => Math.max(min, Math.min(max, v));
const relu = (x: number) => (x > 0 ? x : 0);
const drelu = (x: number) => (x > 0 ? 1 : 0);
const tanh = (x: number) => Math.tanh(x);
const dtanh = (x: number) => 1 - Math.tanh(x) ** 2;

function seededRandom(seed: number) {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) % 4294967296;
    return s / 4294967296;
  };
}

function initParams(seed = 2026): NetworkParams {
  const rand = seededRandom(seed);
  const mat = (r: number, c: number, scale: number) =>
    Array.from({ length: r }, () =>
      Array.from({ length: c }, () => (rand() * 2 - 1) * scale)
    );
  const vec = (n: number) => Array.from({ length: n }, () => 0);

  return {
    w1: mat(INPUT_SIZE, H1, 0.3),
    b1: vec(H1),
    w2: mat(H1, H2, 0.25),
    b2: vec(H2),
    w3: mat(H2, OUTPUT_SIZE, 0.2),
    b3: vec(OUTPUT_SIZE)
  };
}

function dot(a: number[], b: number[]) {
  let s = 0;
  for (let i = 0; i < a.length; i += 1) s += a[i] * b[i];
  return s;
}

function matMul(x: number[], w: number[][], b: number[]) {
  const out: number[] = [];
  for (let j = 0; j < w[0].length; j += 1) {
    const col = w.map((row) => row[j]);
    out.push(dot(x, col) + b[j]);
  }
  return out;
}

function baseDataset(): DatasetRow[] {
  const moodSeries = sentimentSeries.map((s) => s.sentiment / 100);
  const rows: DatasetRow[] = [];

  for (let i = 0; i < moodSeries.length; i += 1) {
    const mood = moodSeries[i];
    const prev = moodSeries[Math.max(0, i - 1)] ?? mood;
    const topic = topicSentiment[i % topicSentiment.length];

    const features = [
      clamp(0.2 + (1 - mood) * 0.6, 0, 1),
      clamp(0.3 + mood * 0.7, 0, 1),
      clamp(0.2 + topic.sensitivity * 0.7, 0, 1),
      clamp(0.25 + (1 - mood) * 0.55, 0, 1),
      clamp(0.35 + mood * 0.45, 0, 1),
      clamp(0.25 + (mood - prev + 0.5) * 0.4, 0, 1),
      clamp(mood, 0, 1),
      clamp(topic.sensitivity, 0, 1)
    ];

    const target = [
      clamp((mood - 0.5) * 0.35 + (topic.sentiment / 100 - 0.5) * 0.2, -1, 1),
      clamp((0.55 - mood) * 0.25 + (1 - topic.sensitivity) * 0.1, -1, 1),
      clamp((mood - 0.45) * 0.28 + 0.03, -1, 1),
      clamp(0.55 + (mood - 0.5) * 0.35, 0, 1)
    ];

    rows.push({ features, target });
  }

  return rows;
}

function readMemory(): NeuralMemoryStore {
  try {
    const raw = fs.readFileSync(MEMORY_PATH, 'utf-8');
    const parsed = JSON.parse(raw) as NeuralMemoryStore;
    if (!Array.isArray(parsed.rows)) throw new Error('invalid rows');
    return parsed;
  } catch {
    return {
      version: '1.0',
      rows: [],
      updates: 0,
      lastUpdatedAt: new Date(0).toISOString()
    };
  }
}

function writeMemory(store: NeuralMemoryStore) {
  fs.writeFileSync(MEMORY_PATH, JSON.stringify(store, null, 2));
}

function normalizeInput(input: NeuralInput): number[] {
  return [
    clamp(input.fiscalCost, 0, 1),
    clamp(input.visibility, 0, 1),
    clamp(input.beneficiaries, 0, 1),
    clamp(input.controversy, 0, 1),
    clamp(input.adminCapacity, 0, 1),
    clamp(input.recencyBoost, 0, 1),
    clamp(input.mood, 0, 1),
    clamp(input.topicSensitivity, 0, 1)
  ];
}

function normalizeTarget(fb: NeuralFeedback['heuristicTarget']): number[] {
  return [
    clamp(fb.scoreDelta / 10, -1, 1),
    clamp(fb.riskDelta / 10, -1, 1),
    clamp(fb.stabilityDelta / 10, -1, 1),
    clamp(fb.confidence, 0, 1)
  ];
}

function buildDatasetFromMemory(): DatasetRow[] {
  const base = baseDataset();
  const memory = readMemory();
  const rows = [...base, ...memory.rows].slice(-1500);
  return rows;
}

function forward(params: NetworkParams, x: number[]) {
  const z1 = matMul(x, params.w1, params.b1);
  const a1 = z1.map(relu);
  const z2 = matMul(a1, params.w2, params.b2);
  const a2 = z2.map(relu);
  const z3 = matMul(a2, params.w3, params.b3);
  const y = [tanh(z3[0]), tanh(z3[1]), tanh(z3[2]), 1 / (1 + Math.exp(-z3[3]))];
  return { z1, a1, z2, a2, z3, y };
}

function train(params: NetworkParams, data: DatasetRow[], epochs = 600, lr = 0.01) {
  for (let ep = 0; ep < epochs; ep += 1) {
    for (const row of data) {
      const { z1, a1, z2, a2, z3, y } = forward(params, row.features);

      const dz3 = y.map((pred, i) => {
        if (i < 3) return (pred - row.target[i]) * dtanh(z3[i]);
        const sig = y[3];
        return (pred - row.target[3]) * sig * (1 - sig);
      });

      const dz2 = Array(H2).fill(0);
      for (let j = 0; j < H2; j += 1) {
        let grad = 0;
        for (let k = 0; k < OUTPUT_SIZE; k += 1) grad += dz3[k] * params.w3[j][k];
        dz2[j] = grad * drelu(z2[j]);
      }

      const dz1 = Array(H1).fill(0);
      for (let j = 0; j < H1; j += 1) {
        let grad = 0;
        for (let k = 0; k < H2; k += 1) grad += dz2[k] * params.w2[j][k];
        dz1[j] = grad * drelu(z1[j]);
      }

      for (let i = 0; i < H2; i += 1) {
        for (let j = 0; j < OUTPUT_SIZE; j += 1) params.w3[i][j] -= lr * dz3[j] * a2[i];
      }
      for (let j = 0; j < OUTPUT_SIZE; j += 1) params.b3[j] -= lr * dz3[j];

      for (let i = 0; i < H1; i += 1) {
        for (let j = 0; j < H2; j += 1) params.w2[i][j] -= lr * dz2[j] * a1[i];
      }
      for (let j = 0; j < H2; j += 1) params.b2[j] -= lr * dz2[j];

      for (let i = 0; i < INPUT_SIZE; i += 1) {
        for (let j = 0; j < H1; j += 1) params.w1[i][j] -= lr * dz1[j] * row.features[i];
      }
      for (let j = 0; j < H1; j += 1) params.b1[j] -= lr * dz1[j];
    }
  }
  return params;
}

let cachedParams: NetworkParams | null = null;
let cachedVersion = '';

function memoryVersionKey(store: NeuralMemoryStore): string {
  return `${store.version}_${store.updates}_${store.lastUpdatedAt}`;
}

function getParams() {
  const mem = readMemory();
  const v = memoryVersionKey(mem);
  if (cachedParams && cachedVersion === v) return cachedParams;

  const params = initParams(1947);
  const dataset = [...baseDataset(), ...mem.rows].slice(-1500);
  const dynamicEpochs = 600 + Math.min(500, mem.updates * 10);
  cachedParams = train(params, dataset, dynamicEpochs, 0.008);
  cachedVersion = v;
  return cachedParams;
}

export function recordNeuralFeedback(feedback: NeuralFeedback) {
  const mem = readMemory();
  const newRow: DatasetRow = {
    features: normalizeInput(feedback.input),
    target: normalizeTarget(feedback.heuristicTarget)
  };
  const rows = [...mem.rows, newRow].slice(-1200);
  const updated: NeuralMemoryStore = {
    ...mem,
    rows,
    updates: mem.updates + 1,
    lastUpdatedAt: new Date().toISOString()
  };
  writeMemory(updated);
  cachedParams = null;
  cachedVersion = '';
}

export function selfCorrectNeuralNetwork(cycles = 2) {
  const mem = readMemory();
  if (mem.rows.length === 0) return;
  const params = getParams();
  const dataset = buildDatasetFromMemory();
  train(params, dataset, Math.max(20, cycles * 15), 0.004);
  mem.lastUpdatedAt = new Date().toISOString();
  writeMemory(mem);
  cachedParams = params;
  cachedVersion = memoryVersionKey(mem);
}

export function inferPoliticalNeuralDelta(input: NeuralInput): NeuralOutput {
  const params = getParams();
  const x = normalizeInput(input);
  const { y } = forward(params, x);

  return {
    scoreDelta: Number((y[0] * 8).toFixed(2)),
    riskDelta: Number((y[1] * 6).toFixed(2)),
    stabilityDelta: Number((y[2] * 5).toFixed(2)),
    confidence: Number(y[3].toFixed(3))
  };
}
