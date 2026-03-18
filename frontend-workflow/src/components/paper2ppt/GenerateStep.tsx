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
      <div className="mb-6 text-center">
        <h2 className="paper2ppt-title mb-2 text-3xl font-bold">逐页生成</h2>
        <p className="paper2ppt-subtitle">第 {currentSlideIndex + 1} / {outlineData.length} 页：{currentSlide?.title}</p>
      </div>

      <div className="mb-6">
        <div className="flex gap-1">
          {generateResults.map((result, index) => (
            <div key={result.slideId} className={`flex-1 h-2 rounded-full transition-all ${
              result.status === 'done' ? 'bg-[#8c1d40]' : result.status === 'processing' ? 'paper2ppt-progress-value animate-pulse' : index === currentSlideIndex ? 'bg-[rgba(140,29,64,0.32)]' : 'paper2ppt-progress-track'
            }`} />
          ))}
        </div>
      </div>

      {currentSlide && (
        <div className="paper2ppt-panel mb-4 rounded-[24px] p-4">
          <div className="mb-3">
            <h4 className="mb-2 flex items-center gap-2 text-sm text-[#675f58]"><FileText size={14} className="text-[#8c1d40]" /> 布局描述</h4>
            <p className="text-xs italic text-[#8c1d40]">{currentSlide.layout_description}</p>
          </div>
          <div className="border-t border-[rgba(110,76,55,0.14)] pt-3">
            <h4 className="mb-2 text-sm text-[#675f58]">要点内容</h4>
            <ul className="grid grid-cols-1 md:grid-cols-2 gap-1">
              {currentSlide.key_points.slice(0, 4).map((point, idx) => (
                <li key={idx} className="flex items-start gap-1 text-xs text-[#675f58]"><span className="text-[#8c1d40]">•</span><span className="line-clamp-1">{point}</span></li>
              ))}
              {currentSlide.key_points.length > 4 && (<li className="text-xs italic text-[#675f58]">...还有 {currentSlide.key_points.length - 4} 条</li>)}
            </ul>
          </div>
        </div>
      )}

      <div className="paper2ppt-panel mb-6 rounded-[28px] p-6">
        <div className="max-w-3xl mx-auto">
          <h4 className="mb-3 flex items-center justify-center gap-2 text-sm text-[#675f58]"><Sparkles size={14} className="text-[#8c1d40]" /> AI 生成结果</h4>
          <div className="paper2ppt-preview-frame flex aspect-[16/9] items-center justify-center overflow-hidden rounded-2xl">
            {isGenerating ? (
              <div className="text-center">
                <Loader2 size={40} className="mx-auto mb-3 animate-spin text-[#8c1d40]" />
                <p className="text-base font-medium text-[#8c1d40]">{generateResults.every(r => r.status === 'processing') ? '正在批量生成所有页面...' : '正在重新生成当前页...'}</p>
                <p className="mt-1 text-xs text-[#675f58]">{generateResults.every(r => r.status === 'processing') ? `共 ${outlineData.length} 页，请稍候` : 'AI 正在根据您的提示重新创建'}</p>
              </div>
            ) : currentResult?.afterImage ? (
              <img src={currentResult.afterImage} alt="Generated" className="w-full h-full object-contain" />
            ) : (
              <div className="text-center"><FileText size={32} className="mx-auto mb-2 text-[#8c1d40]" /><span className="text-[#675f58]">等待生成</span></div>
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

      <div className="paper2ppt-panel mb-6 rounded-[24px] p-4">
        <div className="flex items-center gap-3">
          <MessageSquare size={18} className="text-[#8c1d40]" />
          <input type="text" value={slidePrompt} onChange={e => setSlidePrompt(e.target.value)} placeholder="输入微调 Prompt，然后点击重新生成..." className="paper2ppt-input flex-1 rounded-xl px-3 py-2 text-sm" />
          <button onClick={handleRegenerateSlide} disabled={isGenerating || !slidePrompt.trim()} className="paper2ppt-button-secondary flex items-center gap-2 rounded-xl px-4 py-2 text-sm disabled:opacity-50">
            <RefreshCw size={14} /> 重新生成
          </button>
        </div>
      </div>

      <div className="flex justify-between">
        <button onClick={() => setCurrentStep('outline')} className="paper2ppt-button-secondary flex items-center gap-2 rounded-xl px-6 py-2.5">
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
            className="paper2ppt-button-secondary flex items-center gap-2 rounded-xl px-6 py-2.5 disabled:opacity-30"
          >
            <ArrowLeft size={18} /> 上一页
          </button>
          <button onClick={handleConfirmSlide} disabled={isGenerating || currentResult?.status !== 'done'} className="paper2ppt-button-primary flex items-center gap-2 rounded-xl px-6 py-2.5 font-semibold disabled:opacity-50">
            <CheckCircle2 size={18} /> {currentSlideIndex < outlineData.length - 1 ? '确认并继续' : '完成生成'}
          </button>
        </div>
      </div>

      {error && (
        <div className="paper2ppt-status-error mt-4 flex items-center gap-2 px-4 py-3 text-sm">
          <AlertCircle size={16} /> {error}
        </div>
      )}
    </div>
  );
};

export default GenerateStep;
