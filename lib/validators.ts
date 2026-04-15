import { ScenarioInput } from './types';

const launchSet = new Set(['12m', '6m', '3m', '1m']);
const regionSet = new Set(['national', 'north', 'south', 'east', 'west']);

export type ValidationResult<T> =
  | { ok: true; data: T }
  | { ok: false; errors: string[] };

function isString(v: unknown): v is string {
  return typeof v === 'string';
}

function isNumber(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}

export function validateScenarioInput(raw: unknown): ValidationResult<ScenarioInput> {
  const errors: string[] = [];
  const obj = raw as Partial<ScenarioInput>;

  if (!obj || typeof obj !== 'object') {
    return { ok: false, errors: ['Payload must be a JSON object.'] };
  }

  if (!isString(obj.title) || obj.title.trim().length < 3) {
    errors.push('title must be a string with at least 3 characters.');
  }
  if (!isString(obj.policyText) || obj.policyText.trim().length < 10) {
    errors.push('policyText must be a string with at least 10 characters.');
  }
  if (!isString(obj.launchWindow) || !launchSet.has(obj.launchWindow)) {
    errors.push('launchWindow must be one of 12m, 6m, 3m, 1m.');
  }
  if (!isString(obj.regionFocus) || !regionSet.has(obj.regionFocus)) {
    errors.push('regionFocus must be one of national, north, south, east, west.');
  }
  if (!isNumber(obj.visibility) || obj.visibility < 0 || obj.visibility > 100) {
    errors.push('visibility must be a number between 0 and 100.');
  }
  if (!isNumber(obj.estimatedCost) || obj.estimatedCost < 0 || obj.estimatedCost > 500000) {
    errors.push('estimatedCost must be a number between 0 and 500000 (₹ crore).');
  }
  if (!isNumber(obj.beneficiariesM) || obj.beneficiariesM < 0 || obj.beneficiariesM > 200) {
    errors.push('beneficiariesM must be a number between 0 and 200 million.');
  }
  if (obj.manifestoText !== undefined && !isString(obj.manifestoText)) {
    errors.push('manifestoText must be a string when provided.');
  }

  if (errors.length > 0) return { ok: false, errors };

  return {
    ok: true,
    data: {
      title: obj.title!.trim(),
      policyText: obj.policyText!.trim(),
      launchWindow: obj.launchWindow as ScenarioInput['launchWindow'],
      regionFocus: obj.regionFocus as ScenarioInput['regionFocus'],
      visibility: Math.round(obj.visibility!),
      estimatedCost: Math.round(obj.estimatedCost!),
      beneficiariesM: Number(obj.beneficiariesM!.toFixed(2)),
      manifestoText: obj.manifestoText?.trim() || ''
    }
  };
}
