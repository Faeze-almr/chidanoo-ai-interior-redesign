import { useI18n } from '../lib/i18n';
import { randomVariation, TechnicalSettings } from '../lib/generationSettings';

interface AdvancedSettingsProps {
  settings: TechnicalSettings;
  onChange: (settings: TechnicalSettings) => void;
}

export function AdvancedSettings({ settings, onChange }: AdvancedSettingsProps) {
  const { t } = useI18n();
  const set = (key: keyof TechnicalSettings, value: number) => onChange({ ...settings, [key]: value });

  return (
    <details className="rounded-2xl border border-border bg-surface p-5">
      <summary className="cursor-pointer font-semibold text-text-main">{t.advancedSettings}</summary>
      <p className="mt-2 text-sm text-text-muted">{t.advancedSettingsHelp}</p>
      <div className="mt-5 grid gap-5 md:grid-cols-2">
        <RangeField label={t.roomShapePreservation} description={t.roomShapePreservationDesc} value={settings.depthScale} min={0} max={1} step={0.05} onChange={(value) => set('depthScale', value)} />
        <RangeField label={t.boundaryPreservation} description={t.boundaryPreservationDesc} value={settings.cannyScale} min={0} max={1} step={0.05} onChange={(value) => set('cannyScale', value)} />
        <RangeField label={t.promptGuidance} description={t.promptGuidanceDesc} value={settings.guidanceScale} min={1} max={12} step={0.5} onChange={(value) => set('guidanceScale', value)} />
        <RangeField label={t.qualitySteps} description={t.qualityStepsDesc} value={settings.numInferenceSteps} min={10} max={50} step={1} onChange={(value) => set('numInferenceSteps', value)} />
      </div>
      <div className="mt-5 flex flex-col gap-3 sm:flex-row sm:items-end">
        <label className="flex-1 text-sm font-medium text-text-main">
          {t.variationNumber}
          <input
            type="number"
            min={0}
            max={999999}
            value={settings.seed}
            onChange={(event) => set('seed', Math.max(0, Math.min(999999, Number(event.target.value))))}
            className="mt-2 w-full rounded-xl border border-border bg-bg-soft px-3 py-2 text-text-main"
          />
          <span className="mt-2 block text-xs leading-5 text-text-muted">{t.variationNumberDesc}</span>
        </label>
        <button
          type="button"
          onClick={() => set('seed', randomVariation())}
          className="rounded-xl border border-border bg-bg-soft px-4 py-2 font-semibold text-text-main hover:border-primary"
        >
          {t.chooseAnotherVariation}
        </button>
      </div>
    </details>
  );
}

interface RangeFieldProps {
  label: string;
  description: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}

function RangeField({ label, description, value, min, max, step, onChange }: RangeFieldProps) {
  return (
    <label className="text-sm font-medium text-text-main">
      <span className="flex justify-between gap-3"><span>{label}</span><span dir="ltr">{value}</span></span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="mt-3 w-full accent-primary"
      />
      <span className="mt-2 block text-xs leading-5 text-text-muted">{description}</span>
    </label>
  );
}

interface PresetOption<T extends string> {
  value: T;
  label: string;
  description: string;
}

interface PresetGroupProps<T extends string> {
  title: string;
  description?: string;
  value: T;
  options: PresetOption<T>[];
  onChange: (value: T) => void;
}

export function PresetGroup<T extends string>({ title, description, value, options, onChange }: PresetGroupProps<T>) {
  return (
    <fieldset className="rounded-2xl border border-border bg-surface p-5">
      <legend className="px-2 text-sm font-semibold text-text-main">{title}</legend>
      {description && <p className="mb-4 text-sm leading-6 text-text-muted">{description}</p>}
      <div className="grid gap-3 md:grid-cols-3">
        {options.map((option) => (
          <button
            key={option.value}
            type="button"
            aria-pressed={value === option.value}
            onClick={() => onChange(option.value)}
            className={`rounded-xl border p-4 text-start transition-colors ${value === option.value ? 'border-primary bg-surface-lavender' : 'border-border bg-bg-soft hover:border-primary-light'}`}
          >
            <span className="block font-semibold text-text-main">{option.label}</span>
            <span className="mt-1 block text-xs leading-5 text-text-muted">{option.description}</span>
          </button>
        ))}
      </div>
    </fieldset>
  );
}
