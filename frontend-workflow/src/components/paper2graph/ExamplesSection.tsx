import React from 'react';
import { Sparkles } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { GraphType } from './types';

interface DemoCardProps {
  title: string;
  desc: string;
  inputImg?: string;
  outputImg?: string;
}

const DemoCard = ({ title, desc, inputImg, outputImg }: DemoCardProps) => {
  return (
    <div className="portal-card-soft rounded-xl border border-primary-100 p-3 flex flex-col gap-2 transition-colors hover:border-primary-300/30 hover:bg-white/92">
      <div className="flex gap-2">
        {/* 左侧：输入示例图片 */}
        <div className="demo-input-placeholder flex flex-1 items-center justify-center overflow-hidden rounded-md border border-dashed border-primary-100 bg-white/75">
          {inputImg ? (
            <img
              src={inputImg}
              alt="输入示例图"
              className="w-full h-full object-cover"
            />
          ) : (
            <span className="text-[10px] text-[var(--text-secondary)]">输入示例图（待替换）</span>
          )}
        </div>
        {/* 右侧：输出 PPTX 示例图片 */}
        <div className="flex-1 rounded-md bg-gradient-to-br from-primary-500/12 to-amber-500/10 border border-dashed border-primary-300/35 flex items-center justify-center demo-output-placeholder overflow-hidden">
          {outputImg ? (
            <img
              src={outputImg}
              alt="PPTX 示例图"
              className="w-full h-full object-cover"
            />
          ) : (
            <span className="text-[10px] text-[#fff1e6]">PPTX 示例图（待替换）</span>
          )}
        </div>
      </div>
      <div>
        <p className="mb-1 text-[13px] font-medium text-[var(--text-primary)]">{title}</p>
        <p className="text-[11px] leading-snug text-[var(--text-secondary)]">{desc}</p>
      </div>
    </div>
  );
};

interface ExamplesSectionProps {
  visibleTypes?: GraphType[];
}

const ExamplesSection: React.FC<ExamplesSectionProps> = ({ visibleTypes }) => {
  const { t } = useTranslation('paper2graph');
  const allowed = visibleTypes && visibleTypes.length ? new Set(visibleTypes) : null;
  const examples = [
    {
      type: 'model_arch' as GraphType,
      title: t('examples.cards.paperPdfToFigureTitle'),
      desc: t('examples.cards.paperPdfToFigureDesc'),
      inputImg: '/p2f_paper_pdf_img.png',
      outputImg: '/p2f_paper_pdf_img_2.png',
    },
    {
      type: 'model_arch' as GraphType,
      title: t('examples.cards.figureScreenshotToPptTitle'),
      desc: t('examples.cards.figureScreenshotToPptDesc'),
      inputImg: '/p2f_paper_model_img.png',
      outputImg: '/p2f_paper_modle_img_2.png',
    },
    {
      type: 'model_arch' as GraphType,
      title: t('examples.cards.abstractTextToPptTitle'),
      desc: t('examples.cards.abstractTextToPptDesc'),
      inputImg: '/p2f_paper_content.png',
      outputImg: '/p2f_paper_content_2.png',
    },
    {
      type: 'tech_route' as GraphType,
      title: t('examples.cards.pdfToTechRouteTitle'),
      desc: t('examples.cards.pdfToTechRouteDesc'),
      inputImg: '/p2t_paper_img.png',
      outputImg: '/p2t_paper_img_2.png',
    },
    {
      type: 'tech_route' as GraphType,
      title: t('examples.cards.textToTechRouteTitle'),
      desc: t('examples.cards.textToTechRouteDesc'),
      inputImg: '/p2t_paper_text.png',
      outputImg: '/p2t_paper_text_2.png',
    },
    {
      type: 'exp_data' as GraphType,
      title: t('examples.cards.pdfToExpDataTitle'),
      desc: t('examples.cards.pdfToExpDataDesc'),
      inputImg: '/p2e_paper_1.png',
      outputImg: '/p2e_paper_2.png',
    },
    {
      type: 'exp_data' as GraphType,
      title: t('examples.cards.tableTextToExpDataTitle'),
      desc: t('examples.cards.tableTextToExpDataDesc'),
      inputImg: '/p2f_exp_content_1.png',
      outputImg: '/p2f_exp_content_2.png',
    },
  ];

  const visibleExamples = allowed ? examples.filter(example => allowed.has(example.type)) : examples;

  return (
    <div className="space-y-4 mb-2">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-3">
          <h3 className="text-sm font-medium text-[var(--text-primary)]">{t('examples.sectionTitle')}</h3>
          <a
            href="https://wcny4qa9krto.feishu.cn/wiki/VXKiwYndwiWAVmkFU6kcqsTenWh"
            target="_blank"
            rel="noopener noreferrer"
            className="group relative inline-flex items-center gap-2 overflow-hidden rounded-full border border-primary-200 bg-white/82 px-3 py-1 text-xs font-medium text-primary-800 transition-all hover:border-primary-300/50 hover:shadow-[0_0_18px_rgba(143,49,71,0.18)]"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-primary-500/22 via-primary-400/16 to-amber-500/18 opacity-0 group-hover:opacity-100 transition-opacity" />
            <Sparkles size={12} className="text-amber-300 animate-pulse" />
            <span className="bg-gradient-to-r from-[#f5d1c1] via-[#f8e1b8] to-[#f7cfa4] bg-clip-text text-transparent">
              {t('examples.feishuLink')}
            </span>
          </a>
        </div>
        <span className="text-[11px] text-[var(--text-secondary)]">
          {t('examples.sectionSubtitle')}
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
        {visibleExamples.map((example) => (
          <DemoCard
            key={`${example.type}-${example.title}`}
            title={example.title}
            desc={example.desc}
            inputImg={example.inputImg}
            outputImg={example.outputImg}
          />
        ))}
      </div>
    </div>
  );
};

export default ExamplesSection;
