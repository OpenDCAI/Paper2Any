import React from 'react';
import {
  FileText, Sparkles, Loader2, MessageSquare, RefreshCw,
  ArrowLeft, CheckCircle2, AlertCircle
} from 'lucide-react';
import { SlideOutline, GenerateResult, Step } from './types';
import VersionHistory from './VersionHistory';

interface GenerateStepProps {
  outlineData: SlideOutline[];
  currentSlideIndex: number;
  setCurrentSlideIndex: (index: number) => void;
  generateResults: GenerateResult[];
  isGenerating: boolean;
  slidePrompt: string;
  setSlidePrompt: (prompt: string) => void;
  handleRegenerateSlide: () => void;
  handleConfirmSlide: () => void;
  setCurrentStep: (step: Step) => void;
  error: string | null;
  handleRevertToVersion: (versionNumber: number) => void;
}

const GenerateStep: React.FC<GenerateStepProps> = ({
  outlineData,
  currentSlideIndex,
  setCurrentSlideIndex,
  generateResults,
  isGenerating,
  slidePrompt,
  setSlidePrompt,
  handleRegenerateSlide,
  handleConfirmSlide,
  setCurrentStep,
  error,
  handleRevertToVersion
}) => {
  const currentSlide = outlineData[currentSlideIndex];
  const currentResult = generateResults[currentSlideIndex];

  return (
    <div className="max-w-6xl mx-auto">
      <div className="text-center mb-6">
        <h2 className="text-2xl font-bold text-white mb-2">逐页生成</h2>
        <p className="text-gray-400">第 {currentSlideIndex + 1} / {outlineData.length} 页：{currentSlide?.title}</p>
      </div>

      <div className="mb-6">
        <div className="flex gap-1">
          {generateResults.map((result, index) => (
            <div key={result.slideId} className={`flex-1 h-2 rounded-full transition-all ${
              result.status === 'done' ? 'bg-purple-400' : result.status === 'processing' ? 'bg-gradient-to-r from-purple-400 to-pink-400 animate-pulse' : index === currentSlideIndex ? 'bg-purple-400/50' : 'bg-white/10'
            }`} />
          ))}
        </div>
      </div>

      {currentSlide && (
        <div className="glass rounded-xl border border-white/10 p-4 mb-4">
          <div className="mb-3">
            <h4 className="text-sm text-gray-400 mb-2 flex items-center gap-2"><FileText size={14} className="text-purple-400" /> 布局描述</h4>
            <p className="text-xs text-purple-400/80 italic">{currentSlide.layout_description}</p>
          </div>
          <div className="pt-3 border-t border-white/10">
            <h4 className="text-sm text-gray-400 mb-2">要点内容</h4>
            <ul className="grid grid-cols-1 md:grid-cols-2 gap-1">
              {currentSlide.key_points.slice(0, 4).map((point, idx) => (
                <li key={idx} className="text-xs text-gray-400 flex items-start gap-1"><span className="text-purple-400">•</span><span className="line-clamp-1">{point}</span></li>
              ))}
              {currentSlide.key_points.length > 4 && (<li className="text-xs text-gray-500 italic">...还有 {currentSlide.key_points.length - 4} 条</li>)}
            </ul>
          </div>
        </div>
      )}

      <div className="glass rounded-xl border border-white/10 p-6 mb-6">
        <div className="max-w-3xl mx-auto">
          <h4 className="text-sm text-gray-400 mb-3 flex items-center justify-center gap-2"><Sparkles size={14} className="text-purple-400" /> AI 生成结果</h4>
          <div className="rounded-lg overflow-hidden border border-purple-500/30 aspect-[16/9] bg-gradient-to-br from-purple-500/10 to-pink-500/10 flex items-center justify-center">
            {isGenerating ? (
              <div className="text-center">
                <Loader2 size={40} className="text-purple-400 animate-spin mx-auto mb-3" />
                <p className="text-base text-purple-300">{generateResults.every(r => r.status === 'processing') ? '正在批量生成所有页面...' : '正在重新生成当前页...'}</p>
                <p className="text-xs text-gray-500 mt-1">{generateResults.every(r => r.status === 'processing') ? `共 ${outlineData.length} 页，请稍候` : 'AI 正在根据您的提示重新创建'}</p>
              </div>
            ) : currentResult?.afterImage ? (
              <img src={currentResult.afterImage} alt="Generated" className="w-full h-full object-contain" />
            ) : (
              <div className="text-center"><FileText size={32} className="text-gray-500 mx-auto mb-2" /><span className="text-gray-500">等待生成</span></div>
            )}
          </div>
        </div>
      </div>

      {currentResult?.versionHistory && currentResult.versionHistory.length > 0 && (
        <VersionHistory
          versions={currentResult.versionHistory}
          currentVersionIndex={currentResult.currentVersionIndex}
          onRevert={handleRevertToVersion}
          isGenerating={isGenerating}
        />
      )}

      <div className="glass rounded-xl border border-white/10 p-4 mb-6">
        <div className="flex items-center gap-3">
          <MessageSquare size={18} className="text-purple-400" />
          <input type="text" value={slidePrompt} onChange={e => setSlidePrompt(e.target.value)} placeholder="输入微调 Prompt，然后点击重新生成..." className="flex-1 bg-transparent outline-none text-white text-sm placeholder:text-gray-500" />
          <button onClick={handleRegenerateSlide} disabled={isGenerating || !slidePrompt.trim()} className="px-4 py-2 rounded-lg bg-white/10 hover:bg-white/20 text-gray-300 text-sm flex items-center gap-2 disabled:opacity-50">
            <RefreshCw size={14} /> 重新生成
          </button>
        </div>
      </div>

      <div className="flex justify-between">
        <button onClick={() => setCurrentStep('outline')} className="px-6 py-2.5 rounded-lg border border-white/20 text-gray-300 hover:bg-white/10 flex items-center gap-2">
          <ArrowLeft size={18} /> 返回大纲
        </button>
        <div className="flex gap-3">
          <button 
            onClick={() => {
              if (currentSlideIndex > 0) {
                setCurrentSlideIndex(currentSlideIndex - 1);
                setSlidePrompt('');
              }
            }}
            disabled={currentSlideIndex === 0 || isGenerating}
            className="px-6 py-2.5 rounded-lg border border-white/20 text-gray-300 hover:bg-white/10 flex items-center gap-2 disabled:opacity-30"
          >
            <ArrowLeft size={18} /> 上一页
          </button>
          <button onClick={handleConfirmSlide} disabled={isGenerating || currentResult?.status !== 'done'} className="px-6 py-2.5 rounded-lg bg-gradient-to-r from-purple-600 to-pink-600 text-white font-semibold flex items-center gap-2 disabled:opacity-50">
            <CheckCircle2 size={18} /> {currentSlideIndex < outlineData.length - 1 ? '确认并继续' : '完成生成'}
          </button>
        </div>
      </div>

      {error && (
        <div className="mt-4 flex items-center gap-2 text-sm text-red-300 bg-red-500/10 border border-red-500/40 rounded-lg px-4 py-3">
          <AlertCircle size={16} /> {error}
        </div>
      )}
    </div>
  );
};

export default GenerateStep;
