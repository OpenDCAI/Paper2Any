import React from 'react';
import {
  AlertCircle,
  CheckCircle2,
  Download,
  ExternalLink,
  FileCode2,
  FileText,
  Loader2,
  RotateCcw,
} from 'lucide-react';
import { CodeSlideArtifact, SlideOutline } from './types';

interface CodeCompleteStepProps {
  outlineData: SlideOutline[];
  downloadUrl: string | null;
  pdfPreviewUrl: string | null;
  irUrl: string | null;
  renderLogUrl: string | null;
  plannedIrUrl?: string | null;
  finalIrUrl?: string | null;
  materialsManifestUrl?: string | null;
  materialResolutionUrl?: string | null;
  slideArtifacts?: CodeSlideArtifact[];
  isGeneratingFinal: boolean;
  taskMessage?: string;
  handleGenerateFinal: () => void;
  handleDownloadPptx: () => void;
  handleDownloadPdf: () => void;
  handleReset: () => void;
  error: string | null;
}

const DebugRow: React.FC<{ label: string; url?: string | null }> = ({ label, url }) => (
  <div className="flex items-center justify-between gap-3 rounded-xl border border-white/10 bg-black/20 px-4 py-3">
    <span>{label}</span>
    {url ? (
      <a href={url} target="_blank" rel="noreferrer"
         className="inline-flex items-center gap-1 text-cyan-200 hover:text-white">
        <ExternalLink size={14} /> 打开
      </a>
    ) : (
      <span className="text-xs text-gray-500">未生成</span>
    )}
  </div>
);

const CodeCompleteStep: React.FC<CodeCompleteStepProps> = ({
  outlineData,
  downloadUrl,
  pdfPreviewUrl,
  irUrl,
  renderLogUrl,
  plannedIrUrl,
  finalIrUrl,
  materialsManifestUrl,
  materialResolutionUrl,
  slideArtifacts = [],
  isGeneratingFinal,
  taskMessage,
  handleGenerateFinal,
  handleDownloadPptx,
  handleDownloadPdf,
  handleReset,
  error,
}) => {
  return (
    <div className="max-w-5xl mx-auto">
      <div className="mb-8 text-center">
        <div className="w-20 h-20 rounded-full bg-gradient-to-br from-cyan-500 to-blue-500 flex items-center justify-center mx-auto mb-4">
          <CheckCircle2 size={40} className="text-white" />
        </div>
        <h2 className="text-2xl font-bold text-white mb-2">代码型可编辑 PPT 已准备就绪</h2>
        <p className="text-gray-400">共处理 {outlineData.length} 页</p>
      </div>

      <div className="text-center mb-6">
        <button
          onClick={handleGenerateFinal}
          disabled={isGeneratingFinal}
          className="px-8 py-3 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-500 text-white font-semibold flex items-center justify-center gap-2 mx-auto transition-all"
        >
          {isGeneratingFinal ? (
            <><Loader2 size={18} className="animate-spin" /> 正在组装完整 PPTX...</>
          ) : (
            <><Download size={18} /> 导出完整 PPTX</>
          )}
        </button>
        {isGeneratingFinal && taskMessage && (
          <div className="mt-3 text-sm text-cyan-200">{taskMessage}</div>
        )}
      </div>

      <div className="glass rounded-2xl border border-white/10 p-6 mb-6">
        <div className="flex items-center gap-2 text-white font-semibold mb-4">
          <FileCode2 size={18} className="text-cyan-300" />
          <span>主产物</span>
        </div>
        <div className="grid gap-4 md:grid-cols-2">
          <div className="flex items-center justify-between gap-3 rounded-xl border border-white/10 bg-black/20 px-4 py-3">
            <span className="text-cyan-100/90">完整可编辑 PPTX</span>
            {downloadUrl ? (
              <button onClick={handleDownloadPptx} className="inline-flex items-center gap-1 text-cyan-200 hover:text-white">
                <Download size={14} /> 下载
              </button>
            ) : (
              <span className="text-xs text-gray-500">未组装</span>
            )}
          </div>
          <div className="flex items-center justify-between gap-3 rounded-xl border border-white/10 bg-black/20 px-4 py-3">
            <span className="text-cyan-100/90">PDF 预览</span>
            {pdfPreviewUrl ? (
              <button onClick={handleDownloadPdf} className="inline-flex items-center gap-1 text-cyan-200 hover:text-white">
                <Download size={14} /> 下载
              </button>
            ) : (
              <span className="text-xs text-gray-500">未组装</span>
            )}
          </div>
        </div>
      </div>

      <div className="glass rounded-2xl border border-white/10 p-6 mb-6">
        <div className="flex items-center gap-2 text-white font-semibold mb-4">
          <FileText size={18} className="text-cyan-300" />
          <span>Agent 调试产物</span>
        </div>
        <div className="grid gap-3 md:grid-cols-2">
          <DebugRow label="Deck IR" url={irUrl} />
          <DebugRow label="Run Log" url={renderLogUrl} />
          <DebugRow label="Planned IR" url={plannedIrUrl} />
          <DebugRow label="Final IR" url={finalIrUrl} />
          <DebugRow label="Materials Manifest" url={materialsManifestUrl} />
          <DebugRow label="Material Resolution" url={materialResolutionUrl} />
        </div>
      </div>

      {slideArtifacts.length > 0 && (
        <details className="glass rounded-2xl border border-white/10 p-6 mb-6">
          <summary className="cursor-pointer text-white font-semibold">单页 PPTX 下载 ({slideArtifacts.length})</summary>
          <div className="mt-4 grid gap-2 md:grid-cols-2">
            {slideArtifacts.map((a) => (
              <div key={a.index}
                   className="flex items-center justify-between gap-3 rounded-xl border border-white/10 bg-black/20 px-4 py-2">
                <span className="text-sm text-gray-200 line-clamp-1">
                  第 {a.index + 1} 页 · {a.title || '-'}
                </span>
                {a.pptxUrl ? (
                  <a href={a.pptxUrl} download className="text-xs text-cyan-200 hover:text-white inline-flex items-center gap-1">
                    <Download size={12} /> 下载
                  </a>
                ) : (
                  <span className="text-xs text-gray-500">-</span>
                )}
              </div>
            ))}
          </div>
        </details>
      )}

      <div className="text-center">
        <button
          onClick={handleReset}
          className="text-sm text-gray-400 hover:text-white transition-colors"
        >
          <RotateCcw size={14} className="inline mr-1" /> 处理新的论文
        </button>
      </div>

      {error && (
        <div className="mt-4 flex items-center gap-2 text-sm text-red-300 bg-red-500/10 border border-red-500/40 rounded-lg px-4 py-3 justify-center">
          <AlertCircle size={16} /> {error}
        </div>
      )}
    </div>
  );
};

export default CodeCompleteStep;
