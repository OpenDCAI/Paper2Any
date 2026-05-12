import React, { useState } from 'react';
import {
  AlertCircle,
  ArrowLeft,
  CheckCircle2,
  Code2,
  Download,
  Edit3,
  FileText,
  Image,
  Loader2,
  MessageSquare,
  Type,
  X,
} from 'lucide-react';
import { CodeSlideArtifact, CodeStage, CodeTaskProgress, PatchFeedbackType, SlideOutline, Step } from './types';

interface PatchState {
  slideIndex: number;
  feedback: string;
  feedbackType: PatchFeedbackType;
  isSubmitting: boolean;
  error: string | null;
}

interface CodeGenerateStepProps {
  outlineData: SlideOutline[];
  isGenerating: boolean;
  taskMessage?: string;
  progress: CodeTaskProgress | null;
  slideArtifacts: CodeSlideArtifact[];
  error: string | null;
  resultPath: string | null;
  setCurrentStep: (step: Step) => void;
  handleConfirmSlide: () => void;
  onPatchSlide: (slideIndex: number, feedback: string, feedbackType: PatchFeedbackType) => Promise<void>;
  patchingSlideIndex: number | null;
}

const STAGE_ORDER: CodeStage[] = ['planning', 'final_ir', 'slide_rendering'];
const STAGE_LABEL: Record<string, string> = {
  planning: '规划并绑定素材',
  final_ir: '生成最终 IR',
  slide_rendering: '逐页渲染 PPTX',
  done: '完成',
};

const QUICK_FEEDBACK: Array<{ label: string; text: string; type: PatchFeedbackType; icon: React.ReactNode }> = [
  { label: '补充文本', text: '请补充更多详细的文本内容', type: 'text', icon: <Type size={12} /> },
  { label: '添加图片', text: '请为这一页添加合适的配图', type: 'image', icon: <Image size={12} /> },
  { label: '文本+图片', text: '请补充文本内容并添加配图', type: 'both', icon: <MessageSquare size={12} /> },
];

const stageReached = (progress: CodeTaskProgress | null, stage: CodeStage): boolean => {
  if (!progress) return false;
  const currentIdx = STAGE_ORDER.indexOf(progress.stage as CodeStage);
  const targetIdx = STAGE_ORDER.indexOf(stage);
  if (progress.stage === 'done') return true;
  return currentIdx > targetIdx;
};

const stageActive = (progress: CodeTaskProgress | null, stage: CodeStage): boolean => {
  return progress?.stage === stage;
};

const CodeGenerateStep: React.FC<CodeGenerateStepProps> = ({
  outlineData,
  isGenerating,
  taskMessage,
  progress,
  slideArtifacts,
  error,
  setCurrentStep,
  handleConfirmSlide,
  onPatchSlide,
  patchingSlideIndex,
}) => {
  const allRendered = !isGenerating && slideArtifacts.length > 0;
  const [openPatchIndex, setOpenPatchIndex] = useState<number | null>(null);
  const [patchState, setPatchState] = useState<PatchState>({
    slideIndex: -1,
    feedback: '',
    feedbackType: 'auto',
    isSubmitting: false,
    error: null,
  });

  const openPatch = (idx: number) => {
    setOpenPatchIndex(idx);
    setPatchState({ slideIndex: idx, feedback: '', feedbackType: 'auto', isSubmitting: false, error: null });
  };

  const closePatch = () => {
    setOpenPatchIndex(null);
    setPatchState({ slideIndex: -1, feedback: '', feedbackType: 'auto', isSubmitting: false, error: null });
  };

  const applyQuickFeedback = (text: string, type: PatchFeedbackType) => {
    setPatchState((prev) => ({ ...prev, feedback: text, feedbackType: type }));
  };

  const submitPatch = async (idx: number) => {
    if (!patchState.feedback.trim()) return;
    setPatchState((prev) => ({ ...prev, isSubmitting: true, error: null }));
    try {
      await onPatchSlide(idx, patchState.feedback.trim(), patchState.feedbackType);
      closePatch();
    } catch (err) {
      setPatchState((prev) => ({
        ...prev,
        isSubmitting: false,
        error: err instanceof Error ? err.message : '修改失败，请重试',
      }));
    }
  };

  return (
    <div className="max-w-6xl mx-auto">
      <div className="text-center mb-6">
        <p className="text-xs uppercase tracking-[0.22em] text-cyan-300 mb-2 font-semibold">
          Code Deck Runtime
        </p>
        <h2 className="text-2xl font-bold text-white mb-2">Agent 正在逐页生成可编辑 PPTX</h2>
      </div>

      {/* Stage progress bar */}
      <div className="glass rounded-2xl border border-white/10 p-6 mb-6">
        <div className="flex items-center justify-between gap-2">
          {STAGE_ORDER.map((stage, idx) => {
            const reached = stageReached(progress, stage);
            const active = stageActive(progress, stage);
            return (
              <React.Fragment key={stage}>
                <div className="flex flex-col items-center">
                  <div
                    className={`w-8 h-8 rounded-full border flex items-center justify-center text-xs ${
                      reached
                        ? 'bg-cyan-500 border-cyan-400 text-white'
                        : active
                        ? 'bg-cyan-500/30 border-cyan-400 text-cyan-100'
                        : 'border-white/20 text-gray-500'
                    }`}
                  >
                    {reached ? <CheckCircle2 size={14} /> : active ? <Loader2 size={14} className="animate-spin" /> : idx + 1}
                  </div>
                  <div className="mt-2 text-[11px] text-gray-300 text-center max-w-[120px]">
                    {STAGE_LABEL[stage]}
                  </div>
                </div>
                {idx < STAGE_ORDER.length - 1 && <div className="flex-1 h-[1px] bg-white/15 mx-2 mb-5" />}
              </React.Fragment>
            );
          })}
        </div>
        {(progress || taskMessage) && (
          <div className="mt-4 text-sm text-cyan-100/80 text-center">
            {progress?.message || taskMessage || '等待中...'}
            {progress?.stage === 'slide_rendering' && progress.slideTotal > 0 && (
              <span className="ml-2 text-xs text-cyan-200/70">
                （{progress.slideDone}/{progress.slideTotal}）
              </span>
            )}
          </div>
        )}
      </div>

      {/* Per-slide grid */}
      <div className="glass rounded-2xl border border-white/10 p-6 mb-6">
        <div className="flex items-center gap-2 text-white font-semibold mb-4">
          <FileText size={18} className="text-cyan-300" />
          <span>单页产物</span>
          {!isGenerating && slideArtifacts.length > 0 && (
            <span className="ml-auto text-xs text-cyan-300/70 font-normal">点击卡片底部「修改」可对单页提出修改意见</span>
          )}
        </div>
        <div className="grid gap-4 grid-cols-1 md:grid-cols-2 lg:grid-cols-3">
          {outlineData.map((slide, idx) => {
            const artifact = slideArtifacts.find((a) => a.index === idx);
            const isPatching = patchingSlideIndex === idx;
            const isPatchOpen = openPatchIndex === idx;
            const canPatch = !isGenerating && artifact != null && !isPatching;

            return (
              <div
                key={slide.id}
                className={`rounded-xl border overflow-hidden transition-all ${
                  isPatching
                    ? 'border-amber-400/50 bg-amber-500/5'
                    : isPatchOpen
                    ? 'border-cyan-400/50 bg-cyan-500/5'
                    : 'border-white/10 bg-white/5'
                }`}
              >
                {/* Preview area */}
                <div className="aspect-video bg-black/40 flex items-center justify-center relative">
                  {isPatching ? (
                    <div className="flex flex-col items-center gap-2">
                      <Loader2 size={20} className="text-amber-400 animate-spin" />
                      <span className="text-xs text-amber-300/80">正在修改...</span>
                    </div>
                  ) : artifact?.previewUrl ? (
                    <img
                      src={artifact.previewUrl}
                      alt={artifact.title || slide.title}
                      className="w-full h-full object-contain"
                    />
                  ) : artifact?.status === 'code_ready' ? (
                    <div className="flex flex-col items-center gap-1.5">
                      <Code2 size={18} className="text-cyan-400" />
                      <span className="text-xs text-cyan-300/80">代码已生成</span>
                    </div>
                  ) : artifact ? (
                    <span className="text-xs text-cyan-200/70">已生成（无缩略图）</span>
                  ) : (
                    <Loader2 size={20} className="text-cyan-200 animate-spin" />
                  )}
                </div>

                {/* Card info */}
                <div className="p-3">
                  <div className="text-sm text-white font-medium line-clamp-1 mb-1">
                    第 {idx + 1} 页 · {artifact?.title || slide.title}
                  </div>
                  <div className="flex items-center justify-between">
                    {artifact?.pptxUrl ? (
                      <a
                        href={artifact.pptxUrl}
                        download
                        className="inline-flex items-center gap-1 text-xs text-cyan-200 hover:text-white"
                      >
                        <Download size={12} /> 下载 PPTX
                      </a>
                    ) : artifact?.status === 'code_ready' ? (
                      <div className="text-xs text-cyan-400/70">等待渲染...</div>
                    ) : (
                      <div className="text-xs text-gray-500">等待渲染...</div>
                    )}

                    {canPatch && (
                      <button
                        onClick={() => isPatchOpen ? closePatch() : openPatch(idx)}
                        className={`inline-flex items-center gap-1 text-xs px-2 py-1 rounded-md transition-colors ${
                          isPatchOpen
                            ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-400/40'
                            : 'text-gray-400 hover:text-cyan-300 hover:bg-white/10'
                        }`}
                      >
                        {isPatchOpen ? <X size={11} /> : <Edit3 size={11} />}
                        {isPatchOpen ? '取消' : '修改'}
                      </button>
                    )}
                  </div>
                </div>

                {/* Patch panel */}
                {isPatchOpen && (
                  <div className="border-t border-white/10 p-3 bg-black/20">
                    {/* Quick feedback buttons */}
                    <div className="flex gap-1.5 mb-2 flex-wrap">
                      {QUICK_FEEDBACK.map((qf) => (
                        <button
                          key={qf.label}
                          onClick={() => applyQuickFeedback(qf.text, qf.type)}
                          className={`inline-flex items-center gap-1 text-[11px] px-2 py-1 rounded-md border transition-colors ${
                            patchState.feedback === qf.text
                              ? 'bg-cyan-500/20 border-cyan-400/50 text-cyan-200'
                              : 'border-white/15 text-gray-400 hover:border-cyan-400/40 hover:text-cyan-300'
                          }`}
                        >
                          {qf.icon} {qf.label}
                        </button>
                      ))}
                    </div>

                    {/* Feedback textarea */}
                    <textarea
                      value={patchState.feedback}
                      onChange={(e) => setPatchState((prev) => ({ ...prev, feedback: e.target.value }))}
                      placeholder="描述修改意见，例如：补充实验结果数据，或添加架构图..."
                      rows={2}
                      className="w-full bg-white/5 border border-white/15 rounded-lg px-3 py-2 text-xs text-white placeholder-gray-500 resize-none focus:outline-none focus:border-cyan-400/50"
                    />

                    {patchState.error && (
                      <div className="mt-1.5 flex items-center gap-1 text-xs text-red-300">
                        <AlertCircle size={11} /> {patchState.error}
                      </div>
                    )}

                    <div className="mt-2 flex justify-end">
                      <button
                        onClick={() => submitPatch(idx)}
                        disabled={!patchState.feedback.trim() || patchState.isSubmitting}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-cyan-500/20 border border-cyan-400/40 text-cyan-200 text-xs font-medium hover:bg-cyan-500/30 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                      >
                        {patchState.isSubmitting ? (
                          <Loader2 size={12} className="animate-spin" />
                        ) : (
                          <CheckCircle2 size={12} />
                        )}
                        提交修改
                      </button>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Bottom actions */}
      <div className="flex justify-between">
        <button
          onClick={() => setCurrentStep('outline')}
          disabled={isGenerating}
          className="px-6 py-2.5 rounded-lg border border-white/20 text-gray-300 hover:bg-white/10 flex items-center gap-2 disabled:opacity-50"
        >
          <ArrowLeft size={18} /> 返回大纲
        </button>
        <button
          onClick={handleConfirmSlide}
          disabled={isGenerating || !allRendered}
          className="px-6 py-2.5 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-500 text-white font-semibold flex items-center gap-2 disabled:opacity-50"
        >
          <CheckCircle2 size={18} /> 确认并继续
        </button>
      </div>

      {error && (
        <div className="mt-4 flex items-center gap-2 text-sm text-red-300 bg-red-500/10 border border-red-500/40 rounded-lg px-4 py-3">
          <AlertCircle size={16} /> {error}
        </div>
      )}
    </div>
  );
};

export default CodeGenerateStep;
