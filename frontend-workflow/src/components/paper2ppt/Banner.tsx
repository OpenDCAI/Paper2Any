import React from 'react';
import { useTranslation } from 'react-i18next';
import { Github, Star, X } from 'lucide-react';

interface BannerProps {
  show: boolean;
  onClose: () => void;
  stars?: {
    dataflow: number | null;
    agent: number | null;
    dataflex: number | null;
  };
}

const Banner: React.FC<BannerProps> = ({ show, onClose, stars }) => {
  const { t } = useTranslation(['common']);

  if (!show) return null;

  return (
    <div className="relative z-10 w-full flex-shrink-0 border-b border-[rgba(110,76,55,0.12)] bg-[linear-gradient(135deg,rgba(140,29,64,0.95),rgba(108,22,52,0.96))] text-[#fff8f1]">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(197,155,91,0.28),transparent_26%)]" />

      <div className="relative mx-auto flex max-w-7xl flex-col items-center justify-between gap-3 px-4 py-3 sm:flex-row">
        <div className="flex items-center gap-3 flex-wrap justify-center sm:justify-start">
          <a
            href="https://github.com/OpenDCAI"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 rounded-full border border-white/20 bg-white/10 px-3 py-1 transition-colors hover:bg-white/16"
          >
            <Star size={16} className="fill-[#f1cb82] text-[#f1cb82]" />
            <span className="text-xs font-bold text-[#fff8f1]">{t('app.githubProject')}</span>
          </a>

          <span className="text-sm font-medium text-[#fff8f1]">
            {t('app.exploreMore')}
          </span>
        </div>

        <div className="flex items-center gap-2 flex-wrap justify-center">
          <a
            href="https://github.com/OpenDCAI/DataFlow"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 rounded-full border border-white/35 bg-[rgba(255,250,245,0.96)] px-4 py-1.5 text-xs font-semibold text-[#1d1c1a] shadow-[0_14px_30px_rgba(57,24,24,0.16)] transition-all hover:-translate-y-0.5 hover:bg-white"
          >
            <Github size={14} />
            <span>DataFlow</span>
            <span className="flex items-center gap-0.5 rounded-full bg-[rgba(140,29,64,0.08)] px-1.5 py-0.5 text-[10px] text-[#6c1634]"><Star size={8} fill="currentColor" /> {stars?.dataflow || 'Star'}</span>
            <span className="rounded-full bg-[#8c1d40] px-2 py-0.5 text-[10px] text-white">HOT</span>
          </a>

          <a
            href="https://github.com/OpenDCAI/Paper2Any"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 rounded-full border border-white/35 bg-[rgba(255,250,245,0.96)] px-4 py-1.5 text-xs font-semibold text-[#1d1c1a] shadow-[0_14px_30px_rgba(57,24,24,0.16)] transition-all hover:-translate-y-0.5 hover:bg-white"
          >
            <Github size={14} />
            <span>Paper2Any</span>
            <span className="flex items-center gap-0.5 rounded-full bg-[rgba(140,29,64,0.08)] px-1.5 py-0.5 text-[10px] text-[#6c1634]"><Star size={8} fill="currentColor" /> {stars?.agent || 'Star'}</span>
            <span className="rounded-full bg-[#c59b5b] px-2 py-0.5 text-[10px] text-[#4e3420]">NEW</span>
          </a>

          <a
            href="https://github.com/OpenDCAI/DataFlex"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 rounded-full border border-white/35 bg-[rgba(255,250,245,0.96)] px-4 py-1.5 text-xs font-semibold text-[#1d1c1a] shadow-[0_14px_30px_rgba(57,24,24,0.16)] transition-all hover:-translate-y-0.5 hover:bg-white"
          >
            <Github size={14} />
            <span>DataFlex</span>
            <span className="flex items-center gap-0.5 rounded-full bg-[rgba(140,29,64,0.08)] px-1.5 py-0.5 text-[10px] text-[#6c1634]"><Star size={8} fill="currentColor" /> {stars?.dataflex || 'Star'}</span>
            <span className="rounded-full bg-[#c59b5b] px-2 py-0.5 text-[10px] text-[#4e3420]">NEW</span>
          </a>

          <button
            onClick={onClose}
            className="rounded-full p-1 transition-colors hover:bg-white/12"
            aria-label="关闭"
          >
            <X size={16} className="text-[#fff8f1]" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default Banner;
