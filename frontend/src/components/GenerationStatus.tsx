import { useI18n } from '../lib/i18n';
import { IconLoader } from './Icons';

interface Props {
  status: 'idle' | 'submitting' | 'queued' | 'running' | 'succeeded' | 'failed';
  error: string | null;
}

export function GenerationStatus({ status, error }: Props) {
  const { t } = useI18n();

  if (status === 'idle' || status === 'succeeded') return null;

  if (status === 'failed') {
    return (
      <div className="w-full p-4 mt-4 bg-error/10 border border-error/20 rounded-xl flex flex-col items-center justify-center text-center">
        <p className="text-error font-semibold mb-1">{error ?? t.generationFailed}</p>
        <p className="mt-2 text-sm text-text-secondary">{t.recoveryHint}</p>
      </div>
    );
  }

  // Active generating states
  let message = t.preparingImage;
  if (status === 'queued') message = t.loadingModels;
  if (status === 'running') message = t.generatingRedesign;

  return (
    <div className="w-full p-8 mt-4 bg-surface border border-border rounded-2xl flex flex-col items-center justify-center text-center shadow-sm">
      <div className="relative mb-6">
        <div className="absolute inset-0 bg-primary/20 blur-xl rounded-full"></div>
        <IconLoader className="w-12 h-12 text-primary animate-spin relative z-10" />
      </div>
      <h3 className="text-lg font-serif font-semibold text-text-main mb-2">
        {message}
      </h3>
      <p className="text-sm text-text-muted">
        {t.generationTimeHint}
      </p>
      {error && <p className="mt-3 text-sm text-warning">{error}</p>}
    </div>
  );
}
