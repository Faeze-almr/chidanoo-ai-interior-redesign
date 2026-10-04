import React, { createContext, useContext, useEffect, useState } from 'react';
import { en, Dictionary } from '../locales/en';
import { fa } from '../locales/fa';

type Language = 'en' | 'fa';

interface I18nContextType {
  lang: Language;
  setLang: (lang: Language) => void;
  t: Dictionary;
  dir: 'ltr' | 'rtl';
}

const I18nContext = createContext<I18nContextType | undefined>(undefined);

export function I18nProvider({ children }: { children: React.ReactNode }) {
  const [lang, setLang] = useState<Language>(() => {
    const stored = localStorage.getItem('app-lang');
    return (stored === 'fa' || stored === 'en') ? stored : 'en'; // default to en if not selected, though gateway will handle
  });

  const t = lang === 'fa' ? fa : en;
  const dir = lang === 'fa' ? 'rtl' : 'ltr';

  useEffect(() => {
    document.documentElement.lang = lang;
    document.documentElement.dir = dir;
    localStorage.setItem('app-lang', lang);
  }, [lang, dir]);

  return (
    <I18nContext.Provider value={{ lang, setLang, t, dir }}>
      {children}
    </I18nContext.Provider>
  );
}

export function useI18n() {
  const context = useContext(I18nContext);
  if (context === undefined) {
    throw new Error('useI18n must be used within an I18nProvider');
  }
  return context;
}
