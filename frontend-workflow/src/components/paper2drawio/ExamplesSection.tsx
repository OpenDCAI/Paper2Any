import React from 'react';
import { Sparkles } from 'lucide-react';
import { useTranslation } from 'react-i18next';

interface DemoCardProps {
  title: string;
  desc: string;
  inputImg?: string;
  outputImg?: string;
}

const DemoCard = ({ title, desc, inputImg, outputImg }: DemoCardProps) => {
  return (
    <div className="rounded-2xl bg-white/5 border border-white/10 p-4 transition-all duration-300 hover:bg-white/10">
      <div className="flex gap-3 mb-3">
        {/* 左侧：输入示例图片 */}
        <div className="flex-1 rounded-xl bg-white/5 border border-dashed border-white/10 flex items-center justify-center min-h-[120px] overflow-hidden">
          {inputImg ? (
            <img
              src={inputImg}
              alt="输入示例图"
              className="w-full h-full object-cover rounded-xl"
            />
          ) : (
            <span className="text-xs text-slate-400">输入示例图（待替换）</span>
          )}
        </div>
        {/* 右侧：输出 Drawio 示例图片 */}
        <div className="flex-1 rounded-xl bg-sky-500/10 border border-dashed border-sky-300/40 flex items-center justify-center min-h-[120px] overflow-hidden">
          {outputImg ? (
            <img
              src={outputImg}
              alt="Drawio 示例图"
              className="w-full h-full object-cover rounded-xl"
            />
          ) : (
            <span className="text-xs text-sky-200">Drawio 示例图（待替换）</span>
          )}
        </div>
      </div>
      <div>
        <p className="text-sm text-white font-medium mb-1">{title}</p>
        <p className="text-xs text-slate-400 leading-relaxed">{desc}</p>
      </div>
    </div>
  );
};

const ExamplesSection = () => {
  const { t } = useTranslation('paper2drawio');

  return (
    <div className="relative z-0 mt-8 pb-4">
      <div className="flex items-center justify-between flex-wrap gap-3 mb-4">
        <div className="flex items-center gap-3">
          <h3 className="text-sm font-semibold text-slate-200">{t('examples.sectionTitle')}</h3>
          <a
            href="https://wcny4qa9krto.feishu.cn/wiki/VXKiwYndwiWAVmkFU6kcqsTenWh"
            target="_blank"
            rel="noopener noreferrer"
            className="group relative inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/5 border border-white/10 text-xs font-medium text-white overflow-hidden transition-all hover:border-sky-400/60 hover:shadow-[0_0_20px_rgba(14,165,233,0.4)]"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-sky-500/20 via-cyan-500/20 to-blue-500/20 opacity-0 group-hover:opacity-100 transition-opacity" />
            <Sparkles size={12} className="text-yellow-300 animate-pulse relative z-10" />
            <span className="bg-gradient-to-r from-sky-300 via-cyan-300 to-blue-300 bg-clip-text text-transparent group-hover:from-sky-200 group-hover:via-cyan-200 group-hover:to-blue-200 relative z-10">
              {t('examples.feishuLink')}
            </span>
          </a>
        </div>
        <span className="text-xs text-slate-500">
          {t('examples.sectionSubtitle')}
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <DemoCard
          title={t('examples.cards.pdfToDiagramTitle')}
          desc={t('examples.cards.pdfToDiagramDesc')}
        />
        <DemoCard
          title={t('examples.cards.textToDiagramTitle')}
          desc={t('examples.cards.textToDiagramDesc')}
        />
      </div>
    </div>
  );
};

export default ExamplesSection;
