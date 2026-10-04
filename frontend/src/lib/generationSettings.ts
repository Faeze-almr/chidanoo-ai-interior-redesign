export interface TechnicalSettings {
  depthScale: number;
  cannyScale: number;
  guidanceScale: number;
  numInferenceSteps: number;
  seed: number;
}

export type ChangeAmount = 'light' | 'complete' | 'major';
export type StructureProtection = 'strong' | 'balanced' | 'flexible';
export type ReplacementStrength = 'subtle' | 'balanced' | 'complete';
export type ReferenceInfluence = 'inspired' | 'balanced' | 'strong';

export const FULL_DEFAULTS: TechnicalSettings = {
  depthScale: 0.5,
  cannyScale: 0.2,
  guidanceScale: 7.5,
  numInferenceSteps: 30,
  seed: 42,
};

export const TARGETED_DEFAULTS: TechnicalSettings = {
  depthScale: 0.4,
  cannyScale: 0.1,
  guidanceScale: 7.5,
  numInferenceSteps: 30,
  seed: 42,
};

export const REFERENCE_DEFAULTS: TechnicalSettings = { ...FULL_DEFAULTS };

const CHANGE_FACTORS: Record<ChangeAmount, number> = {
  light: 1.15,
  complete: 1,
  major: 0.8,
};

const STRUCTURE_FACTORS: Record<StructureProtection, number> = {
  strong: 1,
  balanced: 0.85,
  flexible: 0.7,
};

const roundScale = (value: number) => Math.round(Math.min(1, Math.max(0, value)) * 100) / 100;

export function fullPresetSettings(
  change: ChangeAmount,
  protection: StructureProtection,
  current: TechnicalSettings = FULL_DEFAULTS,
): TechnicalSettings {
  const factor = CHANGE_FACTORS[change] * STRUCTURE_FACTORS[protection];
  return {
    ...current,
    depthScale: roundScale(FULL_DEFAULTS.depthScale * factor),
    cannyScale: roundScale(FULL_DEFAULTS.cannyScale * factor),
  };
}

export const REPLACEMENT_VALUES: Record<ReplacementStrength, number> = {
  subtle: 0.65,
  balanced: 0.82,
  complete: 1,
};

export const REFERENCE_VALUES: Record<ReferenceInfluence, number> = {
  inspired: 0.4,
  balanced: 0.6,
  strong: 0.8,
};

export function randomVariation(): number {
  return Math.floor(Math.random() * 1_000_000);
}

