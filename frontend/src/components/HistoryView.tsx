import { useEffect, useState } from 'react';
import { useI18n } from '../lib/i18n';
import { HistoryEntry, historyExportUrl, historyImageUrl, listHistory } from '../lib/api';
import { IconDownload, IconLoader } from './Icons';

export function HistoryView() {
  const { t, lang } = useI18n();
  const [items, setItems] = useState<HistoryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    void listHistory()
      .then((entries) => {
        if (!cancelled) setItems(entries);
      })
      .catch(() => {
        if (!cancelled) setError(t.historyLoadFailed);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [t.historyLoadFailed]);

  const getModeLabel = (mode: HistoryEntry['mode']) => {
    if (mode === 'full_redesign') return t.modeFull;
    if (mode === 'targeted_inpaint') return t.modeTargeted;
    return t.modeReference;
  };

  if (loading) {
    return (
      <div className="flex min-h-[320px] items-center justify-center">
        <IconLoader className="h-10 w-10 animate-spin text-primary" />
      </div>
    );
  }

  if (items.length === 0) {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center py-20 text-center">
        <div className="w-24 h-24 bg-bg-soft rounded-full flex items-center justify-center mb-6 border border-border">
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-border-strong">
            <rect width="18" height="18" x="3" y="3" rx="2" ry="2"/>
            <circle cx="9" cy="9" r="2"/>
            <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>
          </svg>
        </div>
        <h2 className="text-xl font-serif font-semibold text-text-main mb-2">{t.noHistory}</h2>
        <p className="text-text-muted">{error ?? t.noHistoryDesc}</p>
      </div>
    );
  }

  return (
    <div className="w-full flex flex-col gap-8">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <h1 className="text-3xl font-serif font-bold text-text-main">{t.history}</h1>
        <a
          href={historyExportUrl()}
          className="inline-flex items-center justify-center gap-2 rounded-xl border border-border bg-surface px-4 py-2 font-semibold text-text-main hover:border-primary"
        >
          <IconDownload className="h-5 w-5" />
          {t.exportHistory}
        </a>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {items.map((item) => (
          <div key={item.id} className="bg-surface border border-border rounded-2xl overflow-hidden group hover:shadow-md transition-shadow">
            <div className="relative aspect-[4/3] bg-bg-soft">
              <img
                src={historyImageUrl(item.id)}
                alt={t.historyImageAlt}
                className="w-full h-full object-cover"
                loading="lazy"
              />
              <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                <a
                  href={historyImageUrl(item.id, 'output', true)}
                  download={`chidanoo-${item.id}.png`}
                  aria-label={t.download}
                  className="p-3 bg-white text-slate-900 rounded-full shadow-lg hover:scale-105 transition-transform"
                >
                  <IconDownload className="w-5 h-5" />
                </a>
              </div>
            </div>
            <div className="p-4 flex flex-col gap-1">
              <div className="flex items-center justify-between gap-3">
                <span className="font-semibold text-text-main">{getModeLabel(item.mode)}</span>
                {item.style && (
                  <span className="text-xs font-medium text-primary bg-primary/10 px-2 py-1 rounded-md">
                    {item.style in t ? t[item.style as keyof typeof t] : item.style}
                  </span>
                )}
              </div>
              {item.user_request && <p className="line-clamp-2 text-sm text-text-secondary">{item.user_request}</p>}
              <span className="text-xs text-text-muted" dir="ltr">
                {new Date(item.created_at).toLocaleString(lang === 'fa' ? 'fa-IR' : 'en-US', {
                  year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
                })}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
