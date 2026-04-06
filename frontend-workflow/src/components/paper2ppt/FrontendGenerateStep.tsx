import React, { useEffect, useState } from 'react';
import {
  AlertCircle,
  ArrowLeft,
  CheckCircle2,
  Code2,
  FileText,
  Loader2,
  MonitorSmartphone,
  Plus,
  RefreshCw,
  RotateCcw,
  ScanSearch,
  ShieldCheck,
  Trash2,
} from 'lucide-react';
import { FrontendDeckTheme, FrontendSlide, SlideOutline, Step } from './types';
import FrontendSlidePreview from './FrontendSlidePreview';

interface FrontendGenerateStepProps {
  outlineData: SlideOutline[];
  frontendSlides: FrontendSlide[];
  deckTheme?: FrontendDeckTheme | null;
  currentSlideIndex: number;
  setCurrentSlideIndex: (index: number) => void;
  isGenerating: boolean;
  taskMessage?: string;
  slidePrompt: string;
  setSlidePrompt: (prompt: string) => void;
  handleRegenerateSlide: () => void;
  handleReviewSlide: () => void;
  applyCodeEdit: (htmlTemplate: string, cssCode: string) => boolean;
  handleDebugCodeEdit: (htmlTemplate: string, cssCode: string) => Promise<void>;
  handleConfirmSlide: () => void;
  setCurrentStep: (step: Step) => void;
  error: string | null;
  isReviewing: boolean;
  updateFieldValue: (slideIndex: number, fieldKey: string, value: string) => void;
  updateListItem: (slideIndex: number, fieldKey: string, itemIndex: number, value: string) => void;
  replaceListItems: (slideIndex: number, fieldKey: string, items: string[]) => void;
  addListItem: (slideIndex: number, fieldKey: string) => void;
  removeListItem: (slideIndex: number, fieldKey: string, itemIndex: number) => void;
  replaceVisualAsset: (slideIndex: number, imageKey: string, file: File) => Promise<void>;
}

const FrontendGenerateStep: React.FC<FrontendGenerateStepProps> = ({
  outlineData,
  frontendSlides,
  deckTheme = null,
  currentSlideIndex,
  setCurrentSlideIndex,
  isGenerating,
  taskMessage,
  slidePrompt,
  setSlidePrompt,
  handleRegenerateSlide,
  handleReviewSlide,
  applyCodeEdit,
  handleDebugCodeEdit,
  handleConfirmSlide,
  setCurrentStep,
  error,
  isReviewing,
  updateFieldValue,
  updateListItem,
  replaceListItems,
  addListItem,
  removeListItem,
  replaceVisualAsset,
}) => {
  const [panelMode, setPanelMode] = useState<'preview' | 'code'>('preview');
  const [draftHtml, setDraftHtml] = useState('');
  const [draftCss, setDraftCss] = useState('');
  const [codeStatus, setCodeStatus] = useState<string | null>(null);
  const currentSlide = frontendSlides[currentSlideIndex];
  const outlineSlide = outlineData[currentSlideIndex];
  const isCodeDirty = draftHtml !== (currentSlide?.htmlTemplate || '') || draftCss !== (currentSlide?.cssCode || '');
  const busyMessage = taskMessage || (currentSlide?.status === 'processing' ? '当前页仍在生成中，请稍候。' : '后台任务仍在处理中，请稍候。');
  const reviewStatusMessage = isReviewing
    ? taskMessage || '当前页正在进行视觉检查，确认并继续会在检查结束后解锁。'
    : taskMessage || '';
  const reviewDisabledReason = !currentSlide
    ? '当前页尚未生成'
    : isGenerating
      ? busyMessage
      : isReviewing
        ? '当前页正在进行视觉检查'
        : '';
  const confirmDisabledReason = !currentSlide
    ? '当前页尚未生成'
    : isGenerating
      ? busyMessage
      : isReviewing
        ? '当前页正在进行视觉检查，检查完成后才能确认并继续'
        : currentSlide.status !== 'done'
          ? '当前页尚未完成生成'
          : '';

  useEffect(() => {
    setDraftHtml(currentSlide?.htmlTemplate || '');
    setDraftCss(currentSlide?.cssCode || '');
    setCodeStatus(null);
  }, [currentSlide?.slideId, currentSlide?.htmlTemplate, currentSlide?.cssCode]);

  return (
    <div className="max-w-7xl mx-auto">
      <div className="text-center mb-6">
        <h2 className="text-2xl font-bold text-white mb-2">逐页生成可编辑版 PPT</h2>
        <p className="text-gray-400">
          第 {currentSlideIndex + 1} / {outlineData.length} 页：{outlineSlide?.title}
        </p>
      </div>

      <div className="mb-6">
        <div className="flex gap-1">
          {frontendSlides.map((slide, index) => (
            <div
              key={slide.slideId}
              className={`flex-1 h-2 rounded-full transition-all ${
                slide.status === 'done'
                  ? 'bg-cyan-400'
                  : slide.status === 'processing'
                    ? 'bg-gradient-to-r from-cyan-400 to-sky-400 animate-pulse'
                    : index === currentSlideIndex
                      ? 'bg-cyan-400/50'
                      : 'bg-white/10'
              }`}
            />
          ))}
        </div>
      </div>

      {taskMessage ? (
        <div className="mb-4 rounded-xl border border-cyan-400/20 bg-cyan-500/10 px-4 py-3 text-sm text-cyan-100/90">
          <div className="flex items-center gap-2 font-medium">
            {(isGenerating || isReviewing) ? <Loader2 size={14} className="animate-spin" /> : <AlertCircle size={14} />}
            <span>{taskMessage}</span>
          </div>
          {(isGenerating || isReviewing) ? (
            <div className="mt-1 text-xs leading-6 text-cyan-100/70">
              后台任务完成后，当前页的继续按钮会自动恢复可点击。
            </div>
          ) : null}
        </div>
      ) : null}

      <div className="grid grid-cols-1 xl:grid-cols-[1.2fr_0.8fr] gap-6">
        <div className="space-y-4">
          <div className="glass rounded-xl border border-white/10 p-4">
            <div className="flex items-center justify-between mb-3">
              <div>
                <div className="text-sm text-gray-400 flex items-center gap-2">
                  <FileText size={14} className="text-cyan-400" /> 结构说明
                </div>
                <p className="text-xs text-cyan-300/80 italic mt-1">
                  {outlineSlide?.layout_description || '模型将自动规划文本优先的前端布局'}
                </p>
              </div>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => setPanelMode('preview')}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium flex items-center gap-1 ${
                    panelMode === 'preview'
                      ? 'bg-cyan-500 text-white'
                      : 'bg-white/5 text-gray-300 hover:bg-white/10'
                  }`}
                >
                  <MonitorSmartphone size={14} /> 预览
                </button>
                <button
                  type="button"
                  onClick={() => setPanelMode('code')}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium flex items-center gap-1 ${
                    panelMode === 'code'
                      ? 'bg-cyan-500 text-white'
                      : 'bg-white/5 text-gray-300 hover:bg-white/10'
                  }`}
                >
                  <Code2 size={14} /> 代码
                </button>
              </div>
            </div>

            <div className="rounded-2xl overflow-hidden border border-cyan-500/20 bg-black/20">
              {panelMode === 'preview' ? (
                isReviewing ? (
                  <div className="aspect-[16/9] flex flex-col items-center justify-center text-center px-6">
                    <Loader2 size={40} className="text-cyan-400 animate-spin mb-3" />
                    <p className="text-base text-cyan-100">视觉检查正在进行中...</p>
                    <p className="text-xs text-gray-500 mt-1 max-w-lg">
                      {reviewStatusMessage}
                    </p>
                  </div>
                ) : currentSlide?.review?.status === 'repairing' ? (
                  <div className="aspect-[16/9] flex flex-col items-center justify-center text-center px-6">
                    <Loader2 size={40} className="text-cyan-400 animate-spin mb-3" />
                    <p className="text-base text-cyan-100">当前页正在自动修复...</p>
                    <p className="text-xs text-gray-500 mt-1 max-w-lg">
                      {currentSlide.review.summary || '请稍候，修复完成后会恢复可继续操作。'}
                    </p>
                  </div>
                ) : isGenerating && currentSlide?.status === 'processing' ? (
                  <div className="aspect-[16/9] flex flex-col items-center justify-center text-center">
                    <Loader2 size={40} className="text-cyan-400 animate-spin mb-3" />
                    <p className="text-base text-cyan-200">正在生成这一页的前端代码...</p>
                    <p className="text-xs text-gray-500 mt-1">
                      {taskMessage || '大模型正在编排 HTML/CSS 模板'}
                    </p>
                  </div>
                ) : currentSlide ? (
                  <FrontendSlidePreview
                    slide={currentSlide}
                    inlineEditEnabled
                    onInlineFieldChange={(fieldKey, value) =>
                      updateFieldValue(currentSlideIndex, fieldKey, value)
                    }
                    onInlineListItemChange={(fieldKey, itemIndex, value) =>
                      updateListItem(currentSlideIndex, fieldKey, itemIndex, value)
                    }
                    onInlineListReplace={(fieldKey, items) =>
                      replaceListItems(currentSlideIndex, fieldKey, items)
                    }
                    onReplaceImage={(imageKey, file) =>
                      replaceVisualAsset(currentSlideIndex, imageKey, file)
                    }
                  />
                ) : (
                  <div className="aspect-[16/9] flex items-center justify-center text-gray-500">
                    等待生成
                  </div>
                )
              ) : (
                <div className="grid grid-cols-1 gap-3 p-4">
                  <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-cyan-500/20 bg-cyan-500/5 px-3 py-2">
                    <div className="text-xs text-cyan-100/80">
                      {'允许直接编辑当前页 HTML/CSS。请保留 `{{field:key}}` / `{{list:key}}` 占位符。'}
                    </div>
                    <div className="flex flex-wrap gap-2">
                      <button
                        type="button"
                        onClick={() => {
                          const ok = applyCodeEdit(draftHtml, draftCss);
                          setCodeStatus(ok ? '代码已应用到当前页。' : null);
                        }}
                        disabled={isGenerating || isReviewing || !currentSlide}
                        title={reviewDisabledReason || undefined}
                        className="px-3 py-1.5 rounded-lg bg-cyan-500 text-white text-xs font-medium flex items-center gap-1 disabled:opacity-50"
                      >
                        <ShieldCheck size={14} /> 应用代码
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          setDraftHtml(currentSlide?.htmlTemplate || '');
                          setDraftCss(currentSlide?.cssCode || '');
                          setCodeStatus('已恢复到当前页已保存代码。');
                        }}
                        disabled={isGenerating || isReviewing || !isCodeDirty}
                        className="px-3 py-1.5 rounded-lg bg-white/10 text-gray-200 text-xs font-medium flex items-center gap-1 disabled:opacity-50"
                      >
                        <RotateCcw size={14} /> 恢复当前页
                      </button>
                      <button
                        type="button"
                        onClick={async () => {
                          setCodeStatus('正在调用 AI 调试当前代码...');
                          await handleDebugCodeEdit(draftHtml, draftCss);
                        }}
                        disabled={isGenerating || isReviewing || !currentSlide}
                        title={reviewDisabledReason || undefined}
                        className="px-3 py-1.5 rounded-lg border border-amber-400/30 bg-amber-500/10 text-amber-100 text-xs font-medium flex items-center gap-1 disabled:opacity-50"
                      >
                        {isGenerating ? <Loader2 size={14} className="animate-spin" /> : <Code2 size={14} />}
                        AI 调试代码
                      </button>
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-400 mb-1">HTML Template</div>
                    <textarea
                      value={draftHtml}
                      onChange={(e) => setDraftHtml(e.target.value)}
                      rows={14}
                      className="w-full rounded-xl border border-white/10 bg-black/40 px-3 py-2 text-xs text-cyan-100 outline-none resize-none font-mono focus:ring-2 focus:ring-cyan-500"
                    />
                  </div>
                  <div>
                    <div className="text-xs text-gray-400 mb-1">CSS</div>
                    <textarea
                      value={draftCss}
                      onChange={(e) => setDraftCss(e.target.value)}
                      rows={12}
                      className="w-full rounded-xl border border-white/10 bg-black/40 px-3 py-2 text-xs text-cyan-100 outline-none resize-none font-mono focus:ring-2 focus:ring-cyan-500"
                    />
                  </div>
                  {codeStatus && (
                    <div className="rounded-xl border border-cyan-500/20 bg-cyan-500/5 px-3 py-2 text-xs text-cyan-100/90">
                      {codeStatus}
                    </div>
                  )}
                </div>
              )}
            </div>

            {currentSlide?.generationNote && (
              <p className="mt-3 text-xs text-cyan-200/80">{currentSlide.generationNote}</p>
            )}

            <div className="mt-3 rounded-xl border border-cyan-500/20 bg-cyan-500/5 px-4 py-3 text-xs text-cyan-100/90">
              批量生成阶段会在后端并行生成页面代码，并复用同一套 deck theme，减少每页风格漂移。
            </div>

            {(isReviewing || currentSlide?.review?.status === 'repairing') && (
              <div className="mt-3 rounded-xl border border-cyan-400/25 bg-cyan-500/10 px-4 py-3 text-sm text-cyan-100/95">
                <div className="flex items-center gap-2 font-medium">
                  <Loader2 size={14} className="animate-spin" />
                  <span>{reviewStatusMessage || '当前页正在进行视觉检查，请稍候。'}</span>
                </div>
                <div className="mt-1 text-xs leading-6 text-cyan-100/75">
                  检查期间“视觉检查并修复”和“确认并继续”会暂时锁定，结束后会恢复可点击。
                </div>
              </div>
            )}

            {currentSlide?.review && currentSlide.review.status !== 'idle' && (
              <div
                className={`mt-3 rounded-xl border px-4 py-3 text-sm ${
                  currentSlide.review.status === 'passed'
                    ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-100'
                    : currentSlide.review.status === 'repairing'
                      ? 'border-cyan-500/30 bg-cyan-500/10 text-cyan-100'
                      : 'border-amber-500/30 bg-amber-500/10 text-amber-100'
                }`}
              >
                <div className="font-medium">{currentSlide.review.summary}</div>
                {currentSlide.review.issues.length > 0 && (
                  <div className="mt-2 space-y-1 text-xs text-current/90">
                    {currentSlide.review.issues.map((issue, index) => (
                      <div key={`${issue}-${index}`}>- {issue}</div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>

          <div className="glass rounded-xl border border-white/10 p-4">
            <div className="flex flex-wrap items-center gap-3">
              <RefreshCw size={18} className="text-cyan-400" />
              <input
                type="text"
                value={slidePrompt}
                onChange={(e) => setSlidePrompt(e.target.value)}
                placeholder="例如：改成 keynote 风、标题更克制、摘要更像学术报告..."
                className="flex-1 bg-transparent outline-none text-white text-sm placeholder:text-gray-500"
              />
              <button
                onClick={handleRegenerateSlide}
                disabled={isGenerating || !slidePrompt.trim()}
                className="px-4 py-2 rounded-lg bg-white/10 hover:bg-white/20 text-gray-300 text-sm flex items-center gap-2 disabled:opacity-50"
              >
                <RefreshCw size={14} /> 重新生成
              </button>
              <button
                onClick={handleReviewSlide}
                disabled={isGenerating || isReviewing || !currentSlide}
                title={reviewDisabledReason || undefined}
                className="px-4 py-2 rounded-lg border border-cyan-400/30 bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-100 text-sm flex items-center gap-2 disabled:opacity-50"
              >
                {isReviewing ? <Loader2 size={14} className="animate-spin" /> : <ScanSearch size={14} />}
                视觉检查并修复
              </button>
              {reviewDisabledReason ? (
                <span className="w-full text-xs text-cyan-100/70">{reviewDisabledReason}</span>
              ) : null}
            </div>
          </div>
        </div>

        <div className="glass rounded-xl border border-white/10 p-5">
          {deckTheme && (
            <div className="mb-4 rounded-2xl border border-cyan-400/20 bg-cyan-500/5 p-4">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <div className="text-[11px] uppercase tracking-[0.2em] text-cyan-200/75">Deck Theme Locked</div>
                  <div className="mt-1 text-sm font-semibold text-white">{deckTheme.themeName}</div>
                </div>
                <div className="rounded-full border border-cyan-400/20 bg-[#06101d]/80 px-3 py-1 text-[11px] text-cyan-100/80">
                  单页重生成继承整套主题
                </div>
              </div>
              {deckTheme.themeLock.componentSignature && (
                <p className="mt-3 text-xs leading-6 text-cyan-100/80">
                  {deckTheme.themeLock.componentSignature}
                </p>
              )}
              {deckTheme.themeLock.mustKeep.length > 0 && (
                <div className="mt-3 flex flex-wrap gap-2">
                  {deckTheme.themeLock.mustKeep.slice(0, 4).map((rule) => (
                    <span
                      key={rule}
                      className="rounded-full border border-cyan-400/15 bg-white/5 px-2.5 py-1 text-[11px] text-cyan-50/85"
                    >
                      {rule}
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}

          <h3 className="text-white font-semibold mb-4">可编辑文本字段</h3>
          {currentSlide?.visualAssets && currentSlide.visualAssets.length > 0 && (
            <div className="mb-4 rounded-xl border border-amber-400/20 bg-amber-500/5 p-3 text-xs text-amber-100/90">
              当前页已启用图片槽位。直接点击画布内图片即可替换为你自己的文件。
            </div>
          )}
          <div className="space-y-4 max-h-[760px] overflow-auto pr-1">
            {currentSlide?.editableFields?.map((field) => (
              <div key={field.key} className="rounded-xl border border-white/10 bg-white/5 p-3">
                <div className="text-xs text-gray-400 mb-2">{field.label}</div>
                {field.type === 'list' ? (
                  <div className="space-y-2">
                    {field.items.map((item, itemIndex) => (
                      <div key={`${field.key}-${itemIndex}`} className="flex gap-2">
                        <input
                          type="text"
                          value={item}
                          onChange={(e) =>
                            updateListItem(currentSlideIndex, field.key, itemIndex, e.target.value)
                          }
                          disabled={isGenerating}
                          className="flex-1 rounded-lg border border-white/10 bg-black/40 px-3 py-2 text-sm text-white outline-none focus:ring-2 focus:ring-cyan-500"
                        />
                        <button
                          type="button"
                          onClick={() => removeListItem(currentSlideIndex, field.key, itemIndex)}
                          disabled={isGenerating}
                          className="p-2 rounded-lg bg-white/5 text-gray-400 hover:text-red-300 disabled:opacity-50"
                        >
                          <Trash2 size={14} />
                        </button>
                      </div>
                    ))}
                    <button
                      type="button"
                      onClick={() => addListItem(currentSlideIndex, field.key)}
                      disabled={isGenerating}
                      className="w-full py-2 rounded-lg border border-dashed border-cyan-500/30 text-cyan-200 text-xs hover:bg-cyan-500/10 disabled:opacity-50 flex items-center justify-center gap-1"
                    >
                      <Plus size={14} /> 添加一条
                    </button>
                  </div>
                ) : field.type === 'textarea' ? (
                  <textarea
                    value={field.value}
                    onChange={(e) => updateFieldValue(currentSlideIndex, field.key, e.target.value)}
                    disabled={isGenerating}
                    rows={4}
                    className="w-full rounded-lg border border-white/10 bg-black/40 px-3 py-2 text-sm text-white outline-none resize-none focus:ring-2 focus:ring-cyan-500"
                  />
                ) : (
                  <input
                    type="text"
                    value={field.value}
                    onChange={(e) => updateFieldValue(currentSlideIndex, field.key, e.target.value)}
                    disabled={isGenerating}
                    className="w-full rounded-lg border border-white/10 bg-black/40 px-3 py-2 text-sm text-white outline-none focus:ring-2 focus:ring-cyan-500"
                  />
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="flex justify-between mt-6">
        <button
          onClick={() => setCurrentStep('outline')}
          className="px-6 py-2.5 rounded-lg border border-white/20 text-gray-300 hover:bg-white/10 flex items-center gap-2"
        >
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
          <button
            onClick={handleConfirmSlide}
            disabled={isGenerating || currentSlide?.status !== 'done'}
            title={confirmDisabledReason || undefined}
            className="px-6 py-2.5 rounded-lg bg-gradient-to-r from-cyan-500 to-sky-500 text-white font-semibold flex items-center gap-2 disabled:opacity-50"
          >
            <CheckCircle2 size={18} /> {currentSlideIndex < outlineData.length - 1 ? '确认并继续' : '完成生成'}
          </button>
        </div>
      </div>

      {confirmDisabledReason ? (
        <div className="mt-2 text-right text-xs text-cyan-100/60">
          {confirmDisabledReason}
        </div>
      ) : null}

      {error && (
        <div className="mt-4 flex items-center gap-2 text-sm text-red-300 bg-red-500/10 border border-red-500/40 rounded-lg px-4 py-3">
          <AlertCircle size={16} /> {error}
        </div>
      )}
    </div>
  );
};

export default FrontendGenerateStep;
