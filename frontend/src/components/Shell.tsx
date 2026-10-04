import React from 'react';
import { useI18n } from '../lib/i18n';
import { IconWand, IconBrush, IconImage, IconHistory, IconHelp } from './Icons';
import { motion, AnimatePresence } from 'framer-motion';

interface Props {
  children: React.ReactNode;
  activeView: 'full' | 'targeted' | 'reference' | 'history' | 'help';
  onViewChange: (view: 'full' | 'targeted' | 'reference' | 'history' | 'help') => void;
}

export function Shell({ children, activeView, onViewChange }: Props) {
  const { t, lang, setLang, dir } = useI18n();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = React.useState(false);

  const menuItems = [
    { id: 'full', label: t.fullRedesign, icon: IconWand },
    { id: 'targeted', label: t.targetedRedesign, icon: IconBrush },
    { id: 'reference', label: t.referenceRedesign, icon: IconImage },
    { id: 'history', label: t.history, icon: IconHistory },
    { id: 'help', label: t.help, icon: IconHelp },
  ] as const;

  const toggleLang = () => {
    setLang(lang === 'en' ? 'fa' : 'en');
  };

  return (
    <div className="min-h-screen bg-bg flex flex-col md:flex-row">
      {/* Mobile Header */}
      <div className="md:hidden flex items-center justify-between p-4 bg-surface border-b border-border">
        <img src={lang === 'fa' ? '/branding/logo-fa.png' : '/branding/logo-en.png'} alt={t.appName} className="h-8 w-auto max-w-[12rem] object-contain" />
        <div className="flex items-center gap-3">
          <button onClick={toggleLang} className="text-sm font-medium px-2 py-1 bg-bg-soft rounded-lg">
            {lang === 'en' ? 'FA' : 'EN'}
          </button>
          <button 
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            className="p-2 bg-bg-soft rounded-lg"
          >
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="3" y1="12" x2="21" y2="12"></line>
              <line x1="3" y1="6" x2="21" y2="6"></line>
              <line x1="3" y1="18" x2="21" y2="18"></line>
            </svg>
          </button>
        </div>
      </div>

      {/* Sidebar Desktop */}
      <div className="hidden md:flex flex-col w-64 bg-surface border-e border-border h-screen sticky top-0 shrink-0">
        <div className="p-6">
          <img src={lang === 'fa' ? '/branding/logo-fa.png' : '/branding/logo-en.png'} alt={t.appName} className="h-12 w-auto max-w-full object-contain mb-8" />
          <nav className="space-y-2">
            {menuItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeView === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => onViewChange(item.id)}
                  className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-colors ${
                    isActive 
                      ? 'bg-surface-lavender text-primary font-semibold shadow-sm border border-primary/20' 
                      : 'text-text-secondary hover:bg-bg-soft hover:text-text-main'
                  }`}
                >
                  <Icon className={`w-5 h-5 ${isActive ? 'text-primary' : 'text-text-muted'}`} />
                  <span className={lang === 'fa' ? 'font-fa mt-1' : ''}>{item.label}</span>
                </button>
              );
            })}
          </nav>
        </div>
        <div className="mt-auto p-6">
          <button 
            onClick={toggleLang}
            className="w-full py-2 bg-bg-soft hover:bg-border border border-border rounded-lg text-sm font-medium transition-colors"
          >
            {lang === 'en' ? 'فارسی' : 'English'}
          </button>
        </div>
      </div>

      {/* Mobile Drawer */}
      <AnimatePresence>
        {isMobileMenuOpen && (
          <>
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsMobileMenuOpen(false)}
              className="fixed inset-0 bg-black/50 z-40 md:hidden"
            />
            <motion.div 
              initial={{ x: dir === 'rtl' ? '100%' : '-100%' }}
              animate={{ x: 0 }}
              exit={{ x: dir === 'rtl' ? '100%' : '-100%' }}
              transition={{ type: 'spring', damping: 25, stiffness: 200 }}
              className={`fixed top-0 bottom-0 ${dir === 'rtl' ? 'right-0' : 'left-0'} w-64 bg-surface z-50 p-6 shadow-xl flex flex-col`}
            >
              <img src={lang === 'fa' ? '/branding/logo-fa.png' : '/branding/logo-en.png'} alt={t.appName} className="h-12 w-auto max-w-full object-contain mb-8" />
              <nav className="space-y-2">
                {menuItems.map((item) => {
                  const Icon = item.icon;
                  const isActive = activeView === item.id;
                  return (
                    <button
                      key={item.id}
                      onClick={() => {
                        onViewChange(item.id);
                        setIsMobileMenuOpen(false);
                      }}
                      className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-colors ${
                        isActive 
                          ? 'bg-surface-lavender text-primary font-semibold shadow-sm border border-primary/20' 
                          : 'text-text-secondary hover:bg-bg-soft hover:text-text-main'
                      }`}
                    >
                      <Icon className={`w-5 h-5 ${isActive ? 'text-primary' : 'text-text-muted'}`} />
                      <span className={lang === 'fa' ? 'font-fa mt-1' : ''}>{item.label}</span>
                    </button>
                  );
                })}
              </nav>
            </motion.div>
          </>
        )}
      </AnimatePresence>

      {/* Main Content Area */}
      <div className="flex-1 overflow-x-hidden min-h-0 bg-bg">
        <main className="max-w-6xl mx-auto p-4 md:p-8 lg:p-12 min-h-[calc(100vh-80px)] md:min-h-screen">
          {children}
        </main>
      </div>
    </div>
  );
}
