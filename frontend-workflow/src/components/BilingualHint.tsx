import React from 'react';
import { Info } from 'lucide-react';

type HintTone = 'sky' | 'violet' | 'emerald';

interface BilingualHintProps {
  title: string;
  zh: string;
  en: string;
  tone?: HintTone;
  className?: string;
}

const toneStyles: Record<HintTone, { ring: string; glow: string; badge: string; icon: string }> = {
  sky: {
    ring: 'ring-primary-400/30',
    glow: 'from-primary-500/18 via-amber-400/10 to-transparent',
    badge: 'border-primary-300/35 bg-primary-500/12 text-[#f8d8df]',
    icon: 'text-[#f4c0cc]',
  },
  violet: {
    ring: 'ring-primary-400/30',
    glow: 'from-primary-600/18 via-primary-400/12 to-transparent',
    badge: 'border-primary-300/35 bg-primary-500/12 text-[#f8d8df]',
    icon: 'text-[#f4c0cc]',
  },
  emerald: {
    ring: 'ring-amber-300/25',
    glow: 'from-amber-400/16 via-primary-500/10 to-transparent',
    badge: 'border-amber-300/30 bg-amber-400/12 text-[#ffe7c7]',
    icon: 'text-[#ffd9a7]',
  },
};

const BilingualHint: React.FC<BilingualHintProps> = ({ title, zh, en, tone = 'sky', className }) => {
  const styles = toneStyles[tone];

  return (
    <div className={`portal-card-soft relative overflow-hidden rounded-2xl p-4 ring-1 ${styles.ring} ${className || ''}`}>
      <div className={`pointer-events-none absolute inset-0 bg-gradient-to-br ${styles.glow}`} />
      <div className="relative flex items-start gap-3">
        <div className={`mt-0.5 flex h-9 w-9 items-center justify-center rounded-xl border ${styles.badge}`}>
          <Info size={16} className={styles.icon} />
        </div>
        <div className="flex-1 space-y-1">
          <p className="text-sm font-semibold text-[var(--text-primary)]">{title}</p>
          <p className="text-xs text-[var(--text-secondary)]">{zh}</p>
          <p className="text-[11px] text-[#8b726b]">{en}</p>
        </div>
      </div>
    </div>
  );
};

export default BilingualHint;
