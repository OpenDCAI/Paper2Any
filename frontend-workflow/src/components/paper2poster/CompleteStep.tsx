import React from 'react';
import { useTranslation } from 'react-i18next';
import { Download, FileImage, FileText, RotateCcw, CheckCircle, Github, Star } from 'lucide-react';
import { GenerateResult } from './types';

interface CompleteStepProps {
  result: GenerateResult;
  handleReset: () => void;
  handleCopyShareText: () => void;
  copySuccess: string;
  stars?: {
    dataflow: number | null;
    agent: number | null;
    dataflex: number | null;
  };
}

const CompleteStep: React.FC<CompleteStepProps> = ({
  result,
  handleReset,
  handleCopyShareText,
  copySuccess,
  stars
}) => {
  const { t } = useTranslation(['paper2poster', 'common']);

  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-10 text-center">
        <CheckCircle className="w-16 h-16 text-green-400 mx-auto mb-4" />
        <h1 className="text-4xl md:text-5xl font-bold mb-4">
          <span className="bg-gradient-to-r from-green-400 via-emerald-400 to-teal-400 bg-clip-text text-transparent">
            {t('complete.title', '生成完成！')}
          </span>
        </h1>
        <p className="text-base text-gray-300 max-w-2xl mx-auto leading-relaxed">
          {t('complete.desc', '您的学术海报已成功生成')}
        </p>
      </div>

      {/* 下载区域 */}
      <div className="glass rounded-2xl p-8 mb-6 border border-white/10">
        <h2 className="text-xl font-bold text-white mb-4">{t('complete.download', '下载文件')}</h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {result.pptxUrl && (
            <a
              href={result.pptxUrl}
              download
              className="flex items-center gap-3 p-4 bg-gradient-to-r from-green-500/20 to-emerald-500/20 border border-green-500/30 rounded-xl hover:border-green-400 transition-all group"
            >
              <div className="w-12 h-12 rounded-lg bg-green-500/30 flex items-center justify-center group-hover:scale-110 transition-transform">
                <FileText className="w-6 h-6 text-green-300" />
              </div>
              <div className="flex-1">
                <p className="text-white font-medium">{t('complete.downloadPptx', '下载PPTX')}</p>
                <p className="text-sm text-gray-400">{t('complete.pptxDesc', '可编辑格式')}</p>
              </div>
              <Download className="w-5 h-5 text-green-300" />
            </a>
          )}

          {result.pngUrl && (
            <a
              href={result.pngUrl}
              download
              className="flex items-center gap-3 p-4 bg-gradient-to-r from-blue-500/20 to-cyan-500/20 border border-blue-500/30 rounded-xl hover:border-blue-400 transition-all group"
            >
              <div className="w-12 h-12 rounded-lg bg-blue-500/30 flex items-center justify-center group-hover:scale-110 transition-transform">
                <FileImage className="w-6 h-6 text-blue-300" />
              </div>
              <div className="flex-1">
                <p className="text-white font-medium">{t('complete.downloadPng', '下载PNG')}</p>
                <p className="text-sm text-gray-400">{t('complete.pngDesc', '预览图片')}</p>
              </div>
              <Download className="w-5 h-5 text-blue-300" />
            </a>
          )}
        </div>
      </div>

      {/* 分享区域 */}
      <div className="glass rounded-2xl p-8 mb-6 border border-white/10">
        <h2 className="text-xl font-bold text-white mb-4">{t('complete.share', '分享获取免费Key')}</h2>
        <p className="text-gray-300 text-sm mb-4">
          {t('complete.shareDesc', '复制文案并分享到社交媒体，联系管理员即可获取免费API Key')}
        </p>
        <button
          onClick={handleCopyShareText}
          className="w-full py-3 px-4 bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 text-white font-medium rounded-xl transition-all"
        >
          {copySuccess || t('complete.copyShareText', '复制分享文案')}
        </button>
      </div>

      {/* GitHub Stars */}
      <div className="glass rounded-2xl p-6 mb-6 border border-white/10">
        <div className="flex flex-wrap gap-3 justify-center">
          <a
            href="https://github.com/OpenDCAI/DataFlow"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg transition-all"
          >
            <Github size={16} className="text-white" />
            <span className="text-white text-sm font-medium">DataFlow</span>
            <span className="bg-white/20 text-white px-2 py-0.5 rounded text-xs flex items-center gap-1">
              <Star size={10} fill="currentColor" /> {stars?.dataflow || 'Star'}
            </span>
          </a>

          <a
            href="https://github.com/OpenDCAI/Paper2Any"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg transition-all"
          >
            <Github size={16} className="text-white" />
            <span className="text-white text-sm font-medium">Paper2Any</span>
            <span className="bg-white/20 text-white px-2 py-0.5 rounded text-xs flex items-center gap-1">
              <Star size={10} fill="currentColor" /> {stars?.agent || 'Star'}
            </span>
          </a>

          <a
            href="https://github.com/OpenDCAI/DataFlex"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg transition-all"
          >
            <Github size={16} className="text-white" />
            <span className="text-white text-sm font-medium">DataFlex</span>
            <span className="bg-white/20 text-white px-2 py-0.5 rounded text-xs flex items-center gap-1">
              <Star size={10} fill="currentColor" /> {stars?.dataflex || 'Star'}
            </span>
          </a>
        </div>
      </div>

      {/* 重置按钮 */}
      <button
        onClick={handleReset}
        className="w-full py-4 px-6 bg-white/10 hover:bg-white/20 text-white font-semibold rounded-xl transition-all flex items-center justify-center gap-2"
      >
        <RotateCcw className="w-5 h-5" />
        {t('complete.reset', '生成新海报')}
      </button>
    </div>
  );
};

export default CompleteStep;
