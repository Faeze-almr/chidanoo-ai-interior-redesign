import { useState, useCallback, useRef, useEffect } from 'react';
import { useI18n } from '../lib/i18n';
import { ImageUploader } from './ImageUploader';
import { GenerationStatus } from './GenerationStatus';
import { BeforeAfterSlider } from './BeforeAfterSlider';
import { submitReferenceRedesign, getJobStatus, jobImageUrl } from '../lib/api';
import { IconDownload } from './Icons';
import { localizeGenerationError } from '../lib/errorMessages';
import { AdvancedSettings, PresetGroup } from './GenerationControls';
import {
  REFERENCE_DEFAULTS,
  REFERENCE_VALUES,
  ReferenceInfluence,
  TechnicalSettings,
} from '../lib/generationSettings';

const MAX_POLL_FAILURES = 3;

export function ReferenceRedesignView() {
  const { t } = useI18n();
  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const [refFile, setRefFile] = useState<File | null>(null);
  const [prompt, setPrompt] = useState('');
  const [influence, setInfluence] = useState<ReferenceInfluence>('balanced');
  const [settings, setSettings] = useState<TechnicalSettings>({ ...REFERENCE_DEFAULTS });
  
  const [uiState, setUiState] = useState<'idle' | 'submitting' | 'queued' | 'running' | 'succeeded' | 'failed'>('idle');
  const [error, setError] = useState<string | null>(null);
  const [resultJobId, setResultJobId] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const pollTimeoutRef = useRef<number | null>(null);
  const pollFailuresRef = useRef(0);

  useEffect(() => {
    if (sourceFile) {
      const url = URL.createObjectURL(sourceFile);
      setPreviewUrl(url);
      return () => URL.revokeObjectURL(url);
    } else {
      setPreviewUrl(null);
    }
  }, [sourceFile]);

  useEffect(() => {
    return () => {
      if (pollTimeoutRef.current !== null) window.clearTimeout(pollTimeoutRef.current);
    };
  }, []);

  const poll = useCallback((jobId: string) => {
    const tick = async () => {
      try {
        const job = await getJobStatus(jobId);
        pollFailuresRef.current = 0;
        setError(null);
        if (job.status === 'succeeded') {
          setUiState('succeeded');
          setResultJobId(job.job_id);
          return;
        }
        if (job.status === 'failed') {
          setUiState('failed');
          setError(localizeGenerationError(job.error ? new Error(job.error) : null, t, t.referenceGenerationFailed));
          return;
        }
        setUiState(job.status);
        pollTimeoutRef.current = window.setTimeout(tick, 1500);
      } catch (err) {
        pollFailuresRef.current += 1;
        if (pollFailuresRef.current <= MAX_POLL_FAILURES) {
          setError(t.connectionRetrying);
          pollTimeoutRef.current = window.setTimeout(tick, 3000);
          return;
        }
        setUiState('failed');
        setError(localizeGenerationError(err, t, t.referenceGenerationFailed));
      }
    };
    void tick();
  }, [t]);

  const handleSubmit = async () => {
    if (!sourceFile || !refFile) return;
    setError(null);
    pollFailuresRef.current = 0;
    setResultJobId(null);
    setUiState('submitting');
    try {
      const response = await submitReferenceRedesign({
        image: sourceFile,
        reference: refFile,
        userText: prompt.trim() ? prompt.trim() : undefined,
        depthScale: settings.depthScale,
        cannyScale: settings.cannyScale,
        guidanceScale: settings.guidanceScale,
        numInferenceSteps: settings.numInferenceSteps,
        seed: settings.seed,
        referenceScale: REFERENCE_VALUES[influence],
      });
      setUiState('queued');
      poll(response.job_id);
    } catch (err) {
      setUiState('failed');
      setError(localizeGenerationError(err, t, t.referenceGenerationFailed));
    }
  };

  const isBusy = uiState === 'submitting' || uiState === 'queued' || uiState === 'running';

  return (
    <div className="w-full max-w-4xl mx-auto flex flex-col gap-8">
      <div>
        <h1 className="text-3xl font-serif font-bold text-text-main mb-2">{t.referenceRedesign}</h1>
        <p className="text-text-secondary">{t.referenceIntro}</p>
      </div>

      {!resultJobId ? (
        <div className="flex flex-col gap-8">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <ImageUploader 
              file={sourceFile} 
              onFileSelect={setSourceFile} 
              label={t.sourceImage}
              description={t.sourceImageDesc}
            />
            <ImageUploader 
              file={refFile} 
              onFileSelect={setRefFile} 
              label={t.referenceImage}
              description={t.referenceImageDesc}
            />
          </div>
          
          <div className="rounded-2xl border border-primary/25 bg-primary/10 p-5">
            <p className="text-sm font-semibold text-text-main">{t.referenceStyleSource}</p>
            <p className="mt-1 text-sm leading-6 text-text-secondary">{t.fromReferenceImage}</p>
          </div>

          <PresetGroup
            title={t.referenceInfluence}
            description={t.referenceInfluenceDesc}
            value={influence}
            onChange={setInfluence}
            options={[
              { value: 'inspired', label: t.inspiredInfluence, description: t.inspiredInfluenceDesc },
              { value: 'balanced', label: t.balancedInfluence, description: t.balancedInfluenceDesc },
              { value: 'strong', label: t.strongInfluence, description: t.strongInfluenceDesc },
            ]}
          />

          <div className="w-full">
            <label className="block text-sm font-semibold text-text-main mb-2">{t.optionalPrompt}</label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              maxLength={400}
              placeholder={t.referencePromptPlaceholder}
              className="w-full p-4 rounded-2xl border-2 border-border bg-surface focus:border-primary focus:ring-4 focus:ring-primary/10 transition-all outline-none resize-none h-24 text-text-main"
            />
          </div>

          <AdvancedSettings settings={settings} onChange={setSettings} />

          <button
            onClick={handleSubmit}
            disabled={!sourceFile || !refFile || isBusy}
            className="tactile-btn tactile-btn-primary w-full md:w-auto md:self-end px-8 py-4 rounded-xl font-semibold text-lg"
          >
            {t.generate}
          </button>

          <GenerationStatus status={uiState} error={error} />
        </div>
      ) : (
        <div className="flex flex-col gap-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
          <h2 className="text-2xl font-serif font-semibold text-text-main">{t.resultTitle}</h2>
          
          {previewUrl && (
            <BeforeAfterSlider 
              beforeImage={previewUrl} 
              afterImage={jobImageUrl(resultJobId)} 
            />
          )}

          <div className="flex flex-wrap gap-4 mt-2">
            <a
              href={jobImageUrl(resultJobId, true)}
              download={`chidanoo-${resultJobId}.png`}
              className="flex items-center justify-center gap-2 px-6 py-3 bg-surface border-2 border-border rounded-xl font-semibold text-text-main hover:bg-bg-soft transition-colors"
            >
              <IconDownload className="w-5 h-5" />
              {t.download}
            </a>
            <button
              onClick={() => {
                setResultJobId(null);
                setUiState('idle');
                setSourceFile(null);
                setRefFile(null);
              }}
              className="px-6 py-3 bg-surface border-2 border-border rounded-xl font-semibold text-text-main hover:bg-bg-soft transition-colors"
            >
              {t.generateAgain}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
