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
    <div className="glass rounded-lg border border-white/10 p-3 flex flex-col gap-2 hover:bg-white/5 transition-colors">
      <div className="flex gap-2">
        {/* 左侧：输入示例图片 */}
        <div className="flex-1 rounded-md bg-white/5 border border-dashed border-white/10 flex items-center justify-center demo-input-placeholder overflow-hidden">
          {inputImg ? (
            <img
              src={inputImg}
              alt="输入示例图"
              className="w-full h-full object-cover"
            />
          ) : (
            <span className="text-[10px] text-gray-400">输入示例图（待替换）</span>
          )}
        </div>
        {/* 右侧：输出 PPTX 示例图片 */}
        <div className="flex-1 rounded-md bg-primary-500/10 border border-dashed border-primary-300/40 flex items-center justify-center demo-output-placeholder overflow-hidden">
          {outputImg ? (
            <img
              src={outputImg}
              alt="PPTX 示例图"
              className="w-full h-full object-cover"
            />
          ) : (
            <span className="text-[10px] text-primary-200">PPTX 示例图（待替换）</span>
          )}
        </div>
      </div>
      <div>
        <p className="text-[13px] text-white font-medium mb-1">{title}</p>
        <p className="text-[11px] text-gray-400 leading-snug">{desc}</p>
      </div>
    </div>
  );
};

const ExamplesSection = () => {
  const { t } = useTranslation('paper2graph');

  return (
    <div className="space-y-4 mb-2">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-3">
          <h3 className="text-sm font-medium text-gray-200">{t('examples.sectionTitle')}</h3>
          <a
            href="https://wcny4qa9krto.feishu.cn/wiki/VXKiwYndwiWAVmkFU6kcqsTenWh"
            target="_blank"
            rel="noopener noreferrer"
            className="group relative inline-flex items-center gap-2 px-3 py-1 rounded-full bg-black/50 border border-white/10 text-xs font-medium text-white overflow-hidden transition-all hover:border-white/30 hover:shadow-[0_0_15px_rgba(168,85,247,0.5)]"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 via-purple-500/20 to-pink-500/20 opacity-0 group-hover:opacity-100 transition-opacity" />
            <Sparkles size={12} className="text-yellow-300 animate-pulse" />
            <span className="bg-gradient-to-r from-blue-300 via-purple-300 to-pink-300 bg-clip-text text-transparent group-hover:from-blue-200 group-hover:via-purple-200 group-hover:to-pink-200">
              {t('examples.feishuLink')}
            </span>
          </a>
        </div>
        <span className="text-[11px] text-gray-500">
          {t('examples.sectionSubtitle')}
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
        <DemoCard
          title={t('examples.cards.paperPdfToFigureTitle')}
          desc={t('examples.cards.paperPdfToFigureDesc')}
          inputImg="/p2f_paper_pdf_img.png"
          outputImg="/p2f_paper_pdf_img_2.png"
        />
        <DemoCard
          title={t('examples.cards.figureScreenshotToPptTitle')}
          desc={t('examples.cards.figureScreenshotToPptDesc')}
          inputImg="/p2f_paper_model_img.png"
          outputImg="/p2f_paper_modle_img_2.png"
        />
        <DemoCard
          title={t('examples.cards.abstractTextToPptTitle')}
          desc={t('examples.cards.abstractTextToPptDesc')}
          inputImg="/p2f_paper_content.png"
          outputImg="/p2f_paper_content_2.png"
        />
        <DemoCard
          title={t('examples.cards.pdfToTechRouteTitle')}
          desc={t('examples.cards.pdfToTechRouteDesc')}
          inputImg="/p2t_paper_img.png"
          outputImg="/p2t_paper_img_2.png"
        />
        <DemoCard
          title={t('examples.cards.textToTechRouteTitle')}
          desc={t('examples.cards.textToTechRouteDesc')}
          inputImg="/p2t_paper_text.png"
          outputImg="/p2t_paper_text_2.png"
        />
        <DemoCard
          title={t('examples.cards.pdfToExpDataTitle')}
          desc={t('examples.cards.pdfToExpDataDesc')}
          inputImg="/p2e_paper_1.png"
          outputImg="/p2e_paper_2.png"
        />
        <DemoCard
          title={t('examples.cards.tableTextToExpDataTitle')}
          desc={t('examples.cards.tableTextToExpDataDesc')}
          inputImg="/p2f_exp_content_1.png"
          outputImg="/p2f_exp_content_2.png"
        />
      </div>
    </div>
  );
};

export default ExamplesSection;
