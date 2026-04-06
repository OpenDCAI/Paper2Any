import React from 'react';
import {
  AlertCircle,
  CheckCircle2,
  Download,
  Loader2,
  RotateCcw,
  Sparkles,
} from 'lucide-react';
import { FrontendSlide } from './types';
import FrontendSlidePreview from './FrontendSlidePreview';

interface FrontendCompleteStepProps {
  slides: FrontendSlide[];
  downloadUrl: string | null;
  pdfPreviewUrl: string | null;
  isGeneratingFinal: boolean;
  taskMessage?: string;
  handleGenerateFinal: () => void;
  handleDownloadPptx: () => void;
  handleDownloadPdf: () => void;
  handleReset: () => void;
  error: string | null;
}

const FrontendCompleteStep: React.FC<FrontendCompleteStepProps> = ({
  slides,
  downloadUrl,
  pdfPreviewUrl,
  isGeneratingFinal,
  taskMessage,
  handleGenerateFinal,
  handleDownloadPptx,
  handleDownloadPdf,
  handleReset,
  error,
}) => {
  const doneCount = slides.filter((slide) => slide.status === 'done').length;

  return (
    <div className="max-w-5xl mx-auto">
      <div className="mb-8 text-center">
        <div className="w-20 h-20 rounded-full bg-gradient-to-br from-cyan-500 to-sky-500 flex items-center justify-center mx-auto mb-4">
          <CheckCircle2 size={40} className="text-white" />
        </div>
        <h2 className="text-2xl font-bold text-white mb-2">可编辑版 PPT 已生成</h2>
        <p className="text-gray-400">共处理 {slides.length} 页，当前可编辑页面 {doneCount} 页</p>
      </div>

      <div className="glass rounded-xl border border-white/10 p-6 mb-6">
        <h3 className="text-white font-semibold mb-4">最终导出前预览</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {slides.map((slide) => (
            <div key={slide.slideId} className="space-y-2">
              <FrontendSlidePreview slide={slide} />
              <p className="text-xs text-gray-400">
                第 {slide.pageNum} 页 · {slide.title}
              </p>
            </div>
          ))}
        </div>
      </div>

      {!(downloadUrl || pdfPreviewUrl) ? (
        <div className="text-center">
          <button
            onClick={handleGenerateFinal}
            disabled={isGeneratingFinal}
            className="px-8 py-3 rounded-lg bg-gradient-to-r from-cyan-500 to-sky-500 text-white font-semibold flex items-center justify-center gap-2 mx-auto transition-all"
          >
            {isGeneratingFinal ? (
              <>
                <Loader2 size={18} className="animate-spin" /> 正在截图并导出...
              </>
            ) : (
              <>
                <Sparkles size={18} /> 生成最终文件
              </>
            )}
          </button>
          <p className="text-xs text-gray-500 mt-3">
            导出会将每一页前端渲染结果截图，再打包成整页图片版 PPTX / PDF。
          </p>
        </div>
      ) : (
        <div className="space-y-4 text-center">
          <div className="flex gap-4 justify-center">
            {downloadUrl && (
              <button
                onClick={handleDownloadPptx}
                className="px-6 py-3 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-500 text-white font-semibold flex items-center gap-2 transition-all"
              >
                <Download size={18} /> 下载 PPTX
              </button>
            )}
            {pdfPreviewUrl && (
              <button
                onClick={handleDownloadPdf}
                className="px-6 py-3 rounded-lg bg-gradient-to-r from-emerald-500 to-teal-500 text-white font-semibold flex items-center gap-2 transition-all"
              >
                <Download size={18} /> 下载 PDF
              </button>
            )}
          </div>

          <button
            onClick={handleReset}
            className="text-sm text-gray-400 hover:text-white transition-colors"
          >
            <RotateCcw size={14} className="inline mr-1" /> 处理新的论文
          </button>
        </div>
      )}

      {isGeneratingFinal && taskMessage && (
        <div className="mt-4 text-sm text-cyan-200 bg-cyan-500/10 border border-cyan-500/30 rounded-lg px-4 py-3 text-center">
          {taskMessage}
        </div>
      )}

      {error && (
        <div className="mt-4 flex items-center gap-2 text-sm text-red-300 bg-red-500/10 border border-red-500/40 rounded-lg px-4 py-3 justify-center">
          <AlertCircle size={16} /> {error}
        </div>
      )}
    </div>
  );
};

export default FrontendCompleteStep;
