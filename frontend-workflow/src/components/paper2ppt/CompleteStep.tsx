import React from 'react';
import {
  CheckCircle2, Sparkles, Loader2, Download, RotateCcw,
  Star, MessageSquare, Copy, Github, AlertCircle
} from 'lucide-react';
import { SlideOutline, GenerateResult } from './types';

interface CompleteStepProps {
  outlineData: SlideOutline[];
  generateResults: GenerateResult[];
  downloadUrl: string | null;
  pdfPreviewUrl: string | null;
  isGeneratingFinal: boolean;
  handleGenerateFinal: () => void;
  handleDownloadPdf: () => void;
  handleReset: () => void;
  error: string | null;
  handleCopyShareText: () => void;
  copySuccess: string;
  stars: {
    dataflow: number | null;
    agent: number | null;
    dataflex: number | null;
  };
}

const CompleteStep: React.FC<CompleteStepProps> = ({
  outlineData,
  generateResults,
  downloadUrl,
  pdfPreviewUrl,
  isGeneratingFinal,
  handleGenerateFinal,
  handleDownloadPdf,
  handleReset,
  error,
  handleCopyShareText,
  copySuccess,
  stars
}) => {
  const doneCount = generateResults.filter(r => r.status === 'done').length;

  return (
    <div className="max-w-2xl mx-auto text-center">
      <div className="mb-8">
        <div className="mx-auto mb-4 flex h-20 w-20 items-center justify-center rounded-full bg-[linear-gradient(135deg,#8c1d40,#6c1634)] shadow-[0_18px_36px_rgba(140,29,64,0.24)]">
          <CheckCircle2 size={40} className="text-white" />
        </div>
        <h2 className="paper2ppt-title mb-2 text-3xl font-bold">生成完成</h2>
        <p className="paper2ppt-subtitle">共处理 {outlineData.length} 页，成功生成 {doneCount} 页</p>
      </div>

      <div className="paper2ppt-panel mb-6 rounded-[28px] p-6">
        <h3 className="mb-4 text-left text-lg font-semibold text-[#1d1c1a]">生成结果预览</h3>
        <div className="grid grid-cols-4 gap-2">
          {generateResults.map((result, index) => (
            <div key={result.slideId} className="paper2ppt-preview-frame aspect-[16/9] overflow-hidden rounded-xl">
              {result.afterImage ? (
                <img src={result.afterImage} alt={`Page ${index + 1}`} className="w-full h-full object-contain" />
              ) : (
                <div className="flex h-full w-full items-center justify-center text-xs text-[#675f58]">第 {index + 1} 页</div>
              )}
            </div>
          ))}
        </div>
      </div>

      {!(downloadUrl || pdfPreviewUrl) ? (
        <button onClick={handleGenerateFinal} disabled={isGeneratingFinal} className="paper2ppt-button-primary mx-auto flex items-center justify-center gap-2 rounded-xl px-8 py-3 font-semibold transition-all">
          {isGeneratingFinal ? (<><Loader2 size={18} className="animate-spin" /> 正在生成最终文件...</>) : (<><Sparkles size={18} /> 生成最终文件</>)}
        </button>
      ) : (
        <div className="space-y-4">
          <div className="flex gap-4 justify-center">
            {/* 已移除 PPTX 下载按钮 */}
            {pdfPreviewUrl && (
              <button onClick={handleDownloadPdf} className="paper2ppt-button-primary flex items-center gap-2 rounded-xl px-6 py-3 font-semibold transition-all">
                <Download size={18} /> 下载 PDF
              </button>
            )}
          </div>
          
          {/* 引导去 PDF2PPT */}
          <div className="rounded-2xl border border-[rgba(110,76,55,0.14)] bg-white/70 p-3 text-center text-sm text-[#675f58]">
            如果需要继续 PDF 转可编辑 PPTX，请前往 <a href="/pdf2ppt" className="paper2ppt-link font-medium">PDF2PPT 页面</a>
          </div>

          <div>
            <button onClick={handleReset} className="text-sm text-[#675f58] transition-colors hover:text-[#8c1d40]">
              <RotateCcw size={14} className="inline mr-1" /> 处理新的论文
            </button>
          </div>
        </div>
      )}

      {error && (
        <div className="paper2ppt-status-error mt-4 flex items-center justify-center gap-2 px-4 py-3 text-sm">
          <AlertCircle size={16} /> {error}
        </div>
      )}

      <div className="mt-8 grid grid-cols-1 gap-4 text-left md:grid-cols-2">
        <div className="paper2ppt-panel flex flex-col items-center rounded-[24px] p-5 text-center">
          <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-[rgba(140,29,64,0.08)] text-[#8c1d40]">
            <Star size={24} />
          </div>
          <h4 className="mb-2 font-semibold text-[#1d1c1a]">项目资源</h4>
          <p className="mb-4 text-xs leading-relaxed text-[#675f58]">
            平台内网部署版本已取消登录限制与次数限制。<br />
            如果需要了解项目背景或对外介绍，可直接查看或复制项目说明。
          </p>

          <div className="mb-5 flex w-full items-center justify-center gap-4">
            <button onClick={handleCopyShareText} className="group flex flex-col items-center gap-1">
              <div className="flex h-10 w-10 items-center justify-center rounded-full border border-[rgba(140,29,64,0.18)] bg-[rgba(140,29,64,0.08)] text-[#8c1d40] transition-transform group-hover:scale-110">
                <MessageSquare size={18} />
              </div>
              <span className="text-[10px] text-[#675f58]">说明</span>
            </button>
            <button onClick={handleCopyShareText} className="group flex flex-col items-center gap-1">
              <div className="flex h-10 w-10 items-center justify-center rounded-full border border-[rgba(197,155,91,0.28)] bg-[rgba(197,155,91,0.18)] text-[#8c1d40] transition-transform group-hover:scale-110">
                <Copy size={18} />
              </div>
              <span className="text-[10px] text-[#675f58]">复制</span>
            </button>
          </div>

          {copySuccess && (
            <div className="paper2ppt-status-success mb-4 px-3 py-1 text-xs animate-in fade-in zoom-in">
              {copySuccess}
            </div>
          )}

          <div className="w-full space-y-2">
             <a href="https://github.com/OpenDCAI/Paper2Any" target="_blank" rel="noopener noreferrer" className="block w-full rounded-2xl border border-[rgba(110,76,55,0.14)] bg-white/76 px-3 py-2 text-center text-xs text-[#8c1d40] transition-colors hover:bg-white">
               查看 Paper2Any 项目主页
             </a>
             <div className="flex gap-2">
               <a href="https://github.com/OpenDCAI/Paper2Any" target="_blank" rel="noopener noreferrer" className="flex-1 inline-flex items-center justify-center gap-1 rounded-full bg-white px-2 py-1.5 text-[10px] font-semibold text-[#1d1c1a] shadow-[0_10px_22px_rgba(87,48,46,0.1)] transition-all hover:-translate-y-0.5">
                 <Github size={10} />
                 <span>Agent</span>
                 <span className="flex items-center gap-0.5 rounded-full bg-[rgba(140,29,64,0.08)] px-1 py-0.5 text-[9px] text-[#6c1634]"><Star size={7} fill="currentColor" /> {stars.agent || 'Star'}</span>
               </a>
               <a href="https://github.com/OpenDCAI/DataFlow" target="_blank" rel="noopener noreferrer" className="flex-1 inline-flex items-center justify-center gap-1 rounded-full bg-white px-2 py-1.5 text-[10px] font-semibold text-[#1d1c1a] shadow-[0_10px_22px_rgba(87,48,46,0.1)] transition-all hover:-translate-y-0.5">
                 <Github size={10} />
                 <span>Core</span>
                 <span className="flex items-center gap-0.5 rounded-full bg-[rgba(140,29,64,0.08)] px-1 py-0.5 text-[9px] text-[#6c1634]"><Star size={7} fill="currentColor" /> {stars.dataflow || 'Star'}</span>
               </a>
             </div>
          </div>
        </div>

        <div className="paper2ppt-panel flex flex-col items-center rounded-[24px] p-5 text-center">
          <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-[rgba(35,132,90,0.1)] text-[#21704c]">
            <MessageSquare size={24} />
          </div>
          <h4 className="mb-2 font-semibold text-[#1d1c1a]">反馈与讨论</h4>
          <p className="mb-4 text-xs text-[#675f58]">
            效果满意？遇到问题？<br/>欢迎扫码加入交流群反馈与讨论
          </p>
          <div className="mb-2 h-32 w-32 rounded-xl bg-white p-1 shadow-[0_12px_24px_rgba(87,48,46,0.12)]">
            <img src="/wechat.png" alt="交流群二维码" className="w-full h-full object-contain" />
          </div>
          <p className="text-[10px] text-[#675f58]">扫码加入微信交流群</p>
        </div>
      </div>
    </div>
  );
};

export default CompleteStep;
