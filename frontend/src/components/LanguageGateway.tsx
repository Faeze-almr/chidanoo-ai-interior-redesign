import { useState } from 'react';
import { useI18n } from '../lib/i18n';
import { motion } from 'framer-motion';
import { en } from '../locales/en';
import { fa } from '../locales/fa';

interface Props {
  onComplete: () => void;
}

export function LanguageGateway({ onComplete }: Props) {
  const { lang, setLang } = useI18n();
  const [selected, setSelected] = useState<'en' | 'fa' | null>(null);

  const handleSelect = (choice: 'en' | 'fa') => {
    setSelected(choice);
    setLang(choice);
  };

  const handleContinue = () => {
    if (selected) {
      localStorage.setItem('lang-selected', 'true');
      onComplete();
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-bg" dir="ltr">
      <div className="absolute top-8 left-1/2 -translate-x-1/2 w-full px-6 flex justify-center">
        <img
          src={lang === 'fa' ? '/branding/logo-fa.png' : '/branding/logo-en.png'}
          alt={lang === 'fa' ? fa.appName : en.appName}
          className="h-12 sm:h-16 w-auto max-w-[min(80vw,28rem)] object-contain"
        />
      </div>
      
      <motion.div 
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-md px-6 text-center"
      >
        <h1 className="text-2xl font-serif text-text-main font-semibold mb-2">
          {en.chooseLanguage}
        </h1>
        <h2 className="text-xl font-fa text-text-main font-semibold mb-2" dir="rtl">
          {fa.chooseLanguage}
        </h2>
        <p className="text-text-muted text-sm mb-1">
          {en.changeLanguageLater}
        </p>
        <p className="text-text-muted text-sm font-fa mb-8" dir="rtl">
          {fa.changeLanguageLater}
        </p>

        <div className="grid grid-cols-2 gap-4 mb-8">
          <button
            onClick={() => handleSelect('en')}
            className={`flex flex-col items-center justify-center p-6 rounded-2xl border-2 transition-all ${
              selected === 'en' 
                ? 'border-primary bg-surface-lavender shadow-sm' 
                : 'border-border bg-surface hover:border-primary-light hover:bg-bg-soft'
            }`}
          >
            <div className="text-3xl mb-3 font-semibold text-text-main">EN</div>
            <div className="font-semibold text-text-main">{en.english}</div>
            <div className="text-sm text-text-muted font-fa mt-1">{fa.english}</div>
          </button>

          <button
            onClick={() => handleSelect('fa')}
            className={`flex flex-col items-center justify-center p-6 rounded-2xl border-2 transition-all ${
              selected === 'fa' 
                ? 'border-primary bg-surface-lavender shadow-sm' 
                : 'border-border bg-surface hover:border-primary-light hover:bg-bg-soft'
            }`}
          >
            <div className="text-3xl mb-3 font-semibold text-text-main font-fa">FA</div>
            <div className="font-semibold text-text-main font-fa">{fa.persian}</div>
            <div className="text-sm text-text-muted mt-1">{en.persian}</div>
          </button>
        </div>

        <button
          onClick={handleContinue}
          disabled={!selected}
          className="tactile-btn tactile-btn-primary w-full py-4 rounded-xl font-semibold text-lg flex items-center justify-center"
        >
          {selected === 'fa' ? (
            <span className="font-fa">{fa.continue}</span>
          ) : (
            <span>{en.continue}</span>
          )}
        </button>
      </motion.div>
    </div>
  );
}
