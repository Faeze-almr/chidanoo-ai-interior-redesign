import { useI18n } from '../lib/i18n';
import { IconWand, IconBrush, IconImage } from './Icons';

export function HelpView() {
  const { t } = useI18n();
  const content = [
    {
      title: t.fullHelpTitle,
      desc: t.fullHelpDesc,
      when: t.fullHelpWhen,
      icon: IconWand
    },
    {
      title: t.targetedHelpTitle,
      desc: t.targetedHelpDesc,
      when: t.targetedHelpWhen,
      icon: IconBrush
    },
    {
      title: t.referenceHelpTitle,
      desc: t.referenceHelpDesc,
      when: t.referenceHelpWhen,
      icon: IconImage
    }
  ];

  return (
    <div className="w-full max-w-4xl mx-auto flex flex-col gap-8">
      <div>
        <h1 className="text-3xl font-serif font-bold text-text-main mb-2">{t.help}</h1>
      </div>

      <div className="flex flex-col gap-6">
        {content.map((item, i) => {
          const Icon = item.icon;
          return (
            <div key={i} className="bg-surface border border-border p-6 rounded-2xl flex gap-6">
              <div className="w-12 h-12 shrink-0 bg-primary/10 rounded-full flex items-center justify-center text-primary">
                <Icon className="w-6 h-6" />
              </div>
              <div className="flex flex-col gap-2">
                <h3 className="text-xl font-bold text-text-main">{item.title}</h3>
                <p className="text-text-secondary">{item.desc}</p>
                <div className="mt-2 bg-bg-soft p-4 rounded-xl border border-border">
                  <span className="font-semibold text-text-main block mb-1">
                    {t.whenToUse}
                  </span>
                  <p className="text-sm text-text-secondary">{item.when}</p>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
