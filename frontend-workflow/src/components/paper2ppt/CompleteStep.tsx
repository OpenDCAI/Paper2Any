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
  handleDownloadPptx: () => void;
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
  handleDownloadPptx,
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
        <div className="w-20 h-20 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center mx-auto mb-4">
          <CheckCircle2 size={40} className="text-white" />
        </div>
        <h2 className="text-2xl font-bold text-white mb-2">生成完成！</h2>
        <p className="text-gray-400">共处理 {outlineData.length} 页，成功生成 {doneCount} 页</p>
      </div>

      <div className="glass rounded-xl border border-white/10 p-6 mb-6">
        <h3 className="text-white font-semibold mb-4">生成结果预览</h3>
        <div className="grid grid-cols-4 gap-2">
          {generateResults.map((result, index) => (
            <div key={result.slideId} className="aspect-[16/9] rounded-lg border border-white/20 overflow-hidden bg-white/5">
              {result.afterImage ? (
                <img src={result.afterImage} alt={`Page ${index + 1}`} className="w-full h-full object-contain" />
              ) : (
                <div className="w-full h-full flex items-center justify-center text-gray-500 text-xs">第 {index + 1} 页</div>
              )}
            </div>
          ))}
        </div>
      </div>

      {!(downloadUrl || pdfPreviewUrl) ? (
        <button onClick={handleGenerateFinal} disabled={isGeneratingFinal} className="px-8 py-3 rounded-lg bg-gradient-to-r from-purple-600 to-pink-600 text-white font-semibold flex items-center justify-center gap-2 mx-auto transition-all">
          {isGeneratingFinal ? (<><Loader2 size={18} className="animate-spin" /> 正在生成最终文件...</>) : (<><Sparkles size={18} /> 生成最终文件</>)}
        </button>
      ) : (
        <div className="space-y-4">
          <div className="flex gap-4 justify-center">
            {downloadUrl && (
              <button onClick={handleDownloadPptx} className="px-6 py-3 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-500 text-white font-semibold flex items-center gap-2 transition-all">
                <Download size={18} /> 下载 PPTX
              </button>
            )}
            {pdfPreviewUrl && (
              <button onClick={handleDownloadPdf} className="px-6 py-3 rounded-lg bg-gradient-to-r from-purple-500 to-pink-500 text-white font-semibold flex items-center gap-2 transition-all">
                <Download size={18} /> 下载 PDF
              </button>
            )}
          </div>

          <div>
            <button onClick={handleReset} className="text-sm text-gray-400 hover:text-white transition-colors">
              <RotateCcw size={14} className="inline mr-1" /> 处理新的论文
            </button>
          </div>
        </div>
      )}

      {error && (
        <div className="mt-4 flex items-center gap-2 text-sm text-red-300 bg-red-500/10 border border-red-500/40 rounded-lg px-4 py-3 justify-center">
          <AlertCircle size={16} /> {error}
        </div>
      )}

      {/* 分享与交流群区域 */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-8 text-left">
        {/* 获取免费 Key */}
        <div className="glass rounded-xl border border-white/10 p-5 flex flex-col items-center text-center hover:bg-white/5 transition-colors">
          <div className="w-12 h-12 rounded-full bg-yellow-500/20 text-yellow-300 flex items-center justify-center mb-3">
            <Star size={24} />
          </div>
          <h4 className="text-white font-semibold mb-2">获取免费 API Key</h4>
          <p className="text-xs text-gray-400 mb-4 leading-relaxed">
            点击下方平台图标复制推广文案<br/>
            分享至朋友圈/小红书/推特，截图联系微信群管理员领 Key！
          </p>
          
          {/* 分享按钮组 */}
          <div className="flex items-center justify-center gap-4 mb-5 w-full">
            <button onClick={handleCopyShareText} className="flex flex-col items-center gap-1 group">
              <div className="w-10 h-10 rounded-full bg-[#00C300]/20 text-[#00C300] flex items-center justify-center border border-[#00C300]/30 group-hover:scale-110 transition-transform">
                <MessageSquare size={18} />
              </div>
              <span className="text-[10px] text-gray-400">微信</span>
            </button>
            <button onClick={handleCopyShareText} className="flex flex-col items-center gap-1 group">
              <div className="w-10 h-10 rounded-full bg-[#FF2442]/20 text-[#FF2442] flex items-center justify-center border border-[#FF2442]/30 group-hover:scale-110 transition-transform">
                <span className="font-bold text-xs">小红书</span>
              </div>
              <span className="text-[10px] text-gray-400">小红书</span>
            </button>
            <button onClick={handleCopyShareText} className="flex flex-col items-center gap-1 group">
              <div className="w-10 h-10 rounded-full bg-white/10 text-white flex items-center justify-center border border-white/20 group-hover:scale-110 transition-transform">
                <span className="font-bold text-lg">𝕏</span>
              </div>
              <span className="text-[10px] text-gray-400">Twitter</span>
            </button>
            <button onClick={handleCopyShareText} className="flex flex-col items-center gap-1 group">
              <div className="w-10 h-10 rounded-full bg-purple-500/20 text-purple-300 flex items-center justify-center border border-purple-500/30 group-hover:scale-110 transition-transform">
                <Copy size={18} />
              </div>
              <span className="text-[10px] text-gray-400">复制</span>
            </button>
          </div>

          {copySuccess && (
            <div className="mb-4 px-3 py-1 bg-green-500/20 text-green-300 text-xs rounded-full animate-in fade-in zoom-in">
              ✨ {copySuccess}
            </div>
          )}

          <div className="w-full space-y-2">
             <a href="https://github.com/OpenDCAI/Paper2Any" target="_blank" rel="noopener noreferrer" className="block w-full py-1.5 px-3 rounded bg-white/5 hover:bg-white/10 text-xs text-purple-300 truncate transition-colors border border-white/5 text-center">
               ✨如果本项目对你有帮助，可以点个star嘛～
             </a>
             <div className="flex gap-2">
               <a href="https://github.com/OpenDCAI/Paper2Any" target="_blank" rel="noopener noreferrer" className="flex-1 inline-flex items-center justify-center gap-1 px-2 py-1.5 bg-white/95 hover:bg-white text-gray-900 rounded-full text-[10px] font-semibold transition-all hover:scale-105 shadow-lg">
                 <Github size={10} />
                 <span>Agent</span>
                 <span className="bg-gray-200 text-gray-800 px-1 py-0.5 rounded-full text-[9px] flex items-center gap-0.5"><Star size={7} fill="currentColor" /> {stars.agent || 'Star'}</span>
               </a>
               <a href="https://github.com/OpenDCAI/DataFlow" target="_blank" rel="noopener noreferrer" className="flex-1 inline-flex items-center justify-center gap-1 px-2 py-1.5 bg-white/95 hover:bg-white text-gray-900 rounded-full text-[10px] font-semibold transition-all hover:scale-105 shadow-lg">
                 <Github size={10} />
                 <span>Core</span>
                 <span className="bg-gray-200 text-gray-800 px-1 py-0.5 rounded-full text-[9px] flex items-center gap-0.5"><Star size={7} fill="currentColor" /> {stars.dataflow || 'Star'}</span>
               </a>
             </div>
          </div>
        </div>

        {/* 交流群 */}
        <div className="glass rounded-xl border border-white/10 p-5 flex flex-col items-center text-center hover:bg-white/5 transition-colors">
          <div className="w-12 h-12 rounded-full bg-green-500/20 text-green-300 flex items-center justify-center mb-3">
            <MessageSquare size={24} />
          </div>
          <h4 className="text-white font-semibold mb-2">加入交流群</h4>
          <p className="text-xs text-gray-400 mb-4">
            效果满意？遇到问题？<br/>欢迎扫码加入交流群反馈与讨论
          </p>
          <div className="w-32 h-32 bg-white p-1 rounded-lg mb-2">
            <img src="/wechat.png" alt="交流群二维码" className="w-full h-full object-contain" />
          </div>
          <p className="text-[10px] text-gray-500">扫码加入微信交流群</p>
        </div>
      </div>
    </div>
  );
};

export default CompleteStep;
