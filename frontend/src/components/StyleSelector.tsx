import { useI18n } from '../lib/i18n';
import { IconCheck } from './Icons';

export const STYLE_KEYS = ['modern', 'scandinavian', 'industrial', 'bohemian', 'classic', 'japandi', 'luxury'] as const;
export type StyleKey = typeof STYLE_KEYS[number];

const STYLE_DESCRIPTION_KEYS: Record<StyleKey, 'modernDesc' | 'scandinavianDesc' | 'industrialDesc' | 'bohemianDesc' | 'classicDesc' | 'japandiDesc' | 'luxuryDesc'> = {
  modern: 'modernDesc',
  scandinavian: 'scandinavianDesc',
  industrial: 'industrialDesc',
  bohemian: 'bohemianDesc',
  classic: 'classicDesc',
  japandi: 'japandiDesc',
  luxury: 'luxuryDesc',
};

// The original mockups used room thumbnails whose redistribution rights were
// not documented. Public builds use self-contained abstract previews.
const STYLE_PREVIEWS: Record<StyleKey, string> = {
  modern: 'linear-gradient(135deg, #ede9fe 0%, #c4b5fd 45%, #64748b 100%)',
  scandinavian: 'linear-gradient(135deg, #f8fafc 0%, #e2e8f0 50%, #d6d3d1 100%)',
  industrial: 'linear-gradient(135deg, #1f2937 0%, #57534e 48%, #b45309 100%)',
  bohemian: 'linear-gradient(135deg, #fef3c7 0%, #f59e0b 45%, #9f1239 100%)',
  classic: 'linear-gradient(135deg, #f5f5f4 0%, #d6d3d1 45%, #78716c 100%)',
  japandi: 'linear-gradient(135deg, #fafaf9 0%, #d6d3d1 45%, #a8a29e 100%)',
  luxury: 'linear-gradient(135deg, #312e81 0%, #7c3aed 48%, #d8b4fe 100%)',
};

interface Props {
  selectedStyle: StyleKey;
  onSelect: (style: StyleKey) => void;
}

export function StyleSelector({ selectedStyle, onSelect }: Props) {
  const { t } = useI18n();

  return (
    <div className="w-full">
      <label className="block text-sm font-semibold text-text-main mb-4">{t.style}</label>
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {STYLE_KEYS.map((style) => (
          <button
            key={style}
            onClick={() => onSelect(style)}
            className={`relative flex flex-col p-3 rounded-2xl border-2 text-start transition-all ${
              selectedStyle === style
                ? 'border-primary bg-surface-lavender shadow-sm'
                : 'border-border bg-surface hover:border-primary-light hover:bg-bg-soft'
            }`}
          >
            <div
              className="relative w-full aspect-[4/3] rounded-xl mb-3 overflow-hidden bg-bg-soft"
              dir="ltr"
              role="img"
              aria-label={`${t[style]} style preview`}
              style={{ backgroundImage: STYLE_PREVIEWS[style] }}
            >
              {selectedStyle === style && (
                <div className="absolute inset-0 border-4 border-primary rounded-xl" />
              )}
              {selectedStyle === style && (
                <div className="absolute top-2 right-2 bg-primary text-white rounded-full p-1 shadow-sm z-10">
                  <IconCheck className="w-3 h-3" />
                </div>
              )}
            </div>
            <span className={`block w-full text-sm font-semibold ${selectedStyle === style ? 'text-primary-light' : 'text-text-main'}`}>
              {t[style as keyof typeof t]}
            </span>
            <span className="mt-1 block w-full text-xs leading-5 text-text-muted">
              {t[STYLE_DESCRIPTION_KEYS[style]]}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
