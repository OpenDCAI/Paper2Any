import React from 'react';
import { Github, Star, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';

interface BannerProps {
  show: boolean;
  onClose: () => void;
  stars: {
    dataflow: number | null;
    agent: number | null;
    dataflex: number | null;
  };
}

const Banner: React.FC<BannerProps> = ({ show, onClose, stars }) => {
  const { t } = useTranslation('common');
  if (!show) return null;

  return (
    <div className="w-full bg-gradient-to-r from-purple-600 via-pink-600 to-orange-500 relative overflow-hidden">
      <div className="absolute inset-0 bg-black opacity-20"></div>
      <div className="absolute inset-0 animate-pulse">
        <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-r from-transparent via-white to-transparent opacity-10 animate-shimmer"></div>
      </div>
      
      <div className="relative max-w-7xl mx-auto px-4 py-3 flex flex-col sm:flex-row items-center justify-between gap-3">
        <div className="flex items-center gap-3 flex-wrap justify-center sm:justify-start">
          <a
            href="https://github.com/OpenDCAI"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 bg-white/20 backdrop-blur-sm rounded-full px-3 py-1 hover:bg-white/30 transition-colors"
          >
            <Star size={16} className="text-yellow-300 fill-yellow-300 animate-pulse" />
            <span className="text-xs font-bold text-white">{t('app.githubProject')}</span>
          </a>
          
          <span className="text-sm font-medium text-white">
            {t('app.exploreMore')}
          </span>
        </div>

        <div className="flex items-center gap-2 flex-wrap justify-center">
          <a
            href="https://github.com/OpenDCAI/DataFlow"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-4 py-1.5 bg-white/95 hover:bg-white text-gray-900 rounded-full text-xs font-semibold transition-all hover:scale-105 shadow-lg"
          >
            <Github size={14} />
            <span>DataFlow</span>
            <span className="bg-gray-200 text-gray-800 px-1.5 py-0.5 rounded-full text-[10px] flex items-center gap-0.5"><Star size={8} fill="currentColor" /> {stars.dataflow || 'Star'}</span>
            <span className="bg-purple-600 text-white px-2 py-0.5 rounded-full text-[10px]">HOT</span>
          </a>

          <a
            href="https://github.com/OpenDCAI/Paper2Any"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-4 py-1.5 bg-white/95 hover:bg-white text-gray-900 rounded-full text-xs font-semibold transition-all hover:scale-105 shadow-lg"
          >
            <Github size={14} />
            <span>Paper2Any</span>
            <span className="bg-gray-200 text-gray-800 px-1.5 py-0.5 rounded-full text-[10px] flex items-center gap-0.5"><Star size={8} fill="currentColor" /> {stars.agent || 'Star'}</span>
            <span className="bg-pink-600 text-white px-2 py-0.5 rounded-full text-[10px]">NEW</span>
          </a>

          <a
            href="https://github.com/OpenDCAI/DataFlex"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-4 py-1.5 bg-white/95 hover:bg-white text-gray-900 rounded-full text-xs font-semibold transition-all hover:scale-105 shadow-lg"
          >
            <Github size={14} />
            <span>DataFlex</span>
            <span className="bg-gray-200 text-gray-800 px-1.5 py-0.5 rounded-full text-[10px] flex items-center gap-0.5"><Star size={8} fill="currentColor" /> {stars.dataflex || 'Star'}</span>
            <span className="bg-sky-600 text-white px-2 py-0.5 rounded-full text-[10px]">NEW</span>
          </a>

          <button
            onClick={onClose}
            className="p-1 hover:bg-white/20 rounded-full transition-colors"
            aria-label="关闭"
          >
            <X size={16} className="text-white" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default Banner;
