import React from 'react';
import { Sparkles } from 'lucide-react';

type Tone = 'amber' | 'emerald' | 'sky';

interface CaseItem {
  title: string;
  description?: string;
  image: string;
}

interface CasesSectionProps {
  title: string;
  subtitle?: string;
  feishuLabel: string;
  feishuUrl: string;
  cases: CaseItem[];
  tone?: Tone;
  columns?: 1 | 2 | 3;
}

const toneStyles: Record<Tone, { border: string; glow: string; text: string; sparkle: string }> = {
  amber: {
    border: 'hover:border-amber-300/60',
    glow: 'hover:shadow-[0_0_20px_rgba(217,167,95,0.30)]',
    text: 'text-amber-800 group-hover:text-amber-900',
    sparkle: 'text-amber-300',
  },
  emerald: {
    border: 'hover:border-primary-300/60',
    glow: 'hover:shadow-[0_0_20px_rgba(143,49,71,0.28)]',
    text: 'text-primary-800 group-hover:text-primary-900',
    sparkle: 'text-primary-300',
  },
  sky: {
    border: 'hover:border-primary-300/60',
    glow: 'hover:shadow-[0_0_20px_rgba(143,49,71,0.28)]',
    text: 'text-primary-800 group-hover:text-primary-900',
    sparkle: 'text-primary-300',
  },
};

const CasesSection: React.FC<CasesSectionProps> = ({
  title,
  subtitle,
  feishuLabel,
  feishuUrl,
  cases,
  tone = 'sky',
  columns = 2,
}) => {
  const toneClass = toneStyles[tone] || toneStyles.sky;
  const gridClass =
    columns === 1
      ? 'grid-cols-1'
      : columns === 3
        ? 'grid-cols-1 md:grid-cols-3'
        : 'grid-cols-1 md:grid-cols-2';

  return (
    <div className="mt-10">
      <div className="flex items-center justify-between flex-wrap gap-3 mb-4">
        <div className="flex items-center gap-3">
          <h3 className="text-sm font-semibold text-[var(--text-primary)]">{title}</h3>
          <a
            href={feishuUrl}
            target="_blank"
            rel="noopener noreferrer"
            className={`group inline-flex items-center gap-2 rounded-full bg-white/80 border border-primary-200 px-3 py-1.5 text-xs font-medium text-primary-800 transition-all ${toneClass.border} ${toneClass.glow}`}
          >
            <Sparkles size={12} className={`animate-pulse ${toneClass.sparkle}`} />
            <span className={toneClass.text}>{feishuLabel}</span>
          </a>
        </div>
        {subtitle && <span className="text-xs text-[var(--text-secondary)]">{subtitle}</span>}
      </div>

      <div className={`grid ${gridClass} gap-4`}>
        {cases.map((item) => (
          <div
            key={`${item.title}-${item.image}`}
            className="portal-card-soft rounded-2xl p-3 transition-all duration-300 hover:bg-white/90"
          >
            <div className="rounded-xl overflow-hidden border border-primary-100 bg-[#faf4ee]">
              <img
                src={item.image}
                alt={item.title}
                className="w-full h-auto object-contain"
                loading="lazy"
              />
            </div>
            <div className="mt-3">
              <p className="text-sm text-[var(--text-primary)] font-medium">{item.title}</p>
              {item.description && (
                <p className="mt-1 text-xs leading-relaxed text-[var(--text-secondary)]">{item.description}</p>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default CasesSection;
