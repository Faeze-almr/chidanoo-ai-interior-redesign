import { Dictionary } from '../locales/en';

export function localizeGenerationError(error: unknown, t: Dictionary, fallback = t.generationFailed): string {
  const message = error instanceof Error ? error.message : '';
  const normalized = message.toLowerCase();
  if (normalized.includes('cuda') || normalized.includes('gpu')) {
    return normalized.includes('memory') ? t.gpuMemoryFull : t.cudaRequired;
  }
  if (normalized.includes('mask')) return t.invalidMask;
  if (normalized.includes('image') || normalized.includes('dimension') || normalized.includes('contrast') || normalized.includes('dark')) {
    return t.invalidImage;
  }
  if (normalized.includes('setting') || normalized.includes('parameter')) return t.invalidSettings;
  if (normalized.includes('model') || normalized.includes('adapter') || normalized.includes('encoder')) return t.modelUnavailable;
  if (normalized.includes('save') || normalized.includes('storage') || normalized.includes('history')) return t.storageFailed;
  if (normalized.includes('fetch') || normalized.includes('connect') || normalized.includes('network')) return t.serverConnectionFailed;
  return fallback;
}
