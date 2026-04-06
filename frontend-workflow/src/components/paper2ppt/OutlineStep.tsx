import React from 'react';
import {
  GripVertical, Check, Trash2, Edit3, ChevronUp, ChevronDown, Plus,
  ArrowLeft, ArrowRight, AlertCircle, Sparkles
} from 'lucide-react';
import { SlideOutline, Step } from './types';

interface OutlineStepProps {
  outlineData: SlideOutline[];
  editingId: string | null;
  editContent: {
    title: string;
    layout_description: string;
    key_points: string[];
  };
  setEditContent: React.Dispatch<React.SetStateAction<{
    title: string;
    layout_description: string;
    key_points: string[];
  }>>;
  handleEditStart: (slide: SlideOutline) => void;
  handleEditSave: () => void;
  handleEditCancel: () => void;
  handleKeyPointChange: (index: number, value: string) => void;
  handleAddKeyPoint: () => void;
  handleRemoveKeyPoint: (index: number) => void;
  handleDeleteSlide: (id: string) => void;
  handleAddSlide: (index: number) => void;
  handleMoveSlide: (index: number, direction: 'up' | 'down') => void;
  handleConfirmOutline: () => void;
  handleRefineOutline: () => void;
  setCurrentStep: (step: Step) => void;
  error: string | null;
  outlineFeedback: string;
  setOutlineFeedback: React.Dispatch<React.SetStateAction<string>>;
  isRefiningOutline: boolean;
  isGenerating: boolean;
}

const OutlineStep: React.FC<OutlineStepProps> = ({
  outlineData,
  editingId,
  editContent,
  setEditContent,
  handleEditStart,
  handleEditSave,
  handleEditCancel,
  handleKeyPointChange,
  handleAddKeyPoint,
  handleRemoveKeyPoint,
  handleDeleteSlide,
  handleAddSlide,
  handleMoveSlide,
  handleConfirmOutline,
  handleRefineOutline,
  setCurrentStep,
  error,
  outlineFeedback,
  setOutlineFeedback,
  isRefiningOutline,
  isGenerating,
}) => {
  const disabledClass = "disabled:opacity-50 disabled:cursor-not-allowed";
  const isBusy = isRefiningOutline || isGenerating;
  return (
    <div className="max-w-5xl mx-auto">
      <div className="text-center mb-8">
        <h2 className="text-2xl font-bold text-white mb-2">确认大纲</h2>
        <p className="text-gray-400">检查从论文提取的内容结构，可编辑、排序或删除</p>
      </div>

      <div className="glass rounded-xl border border-white/10 p-6 mb-6">
        <div className="space-y-3">
          {outlineData.map((slide, index) => (
            <div 
              key={slide.id} 
              className={`flex items-start gap-4 p-4 rounded-lg border transition-all ${
                editingId === slide.id 
                  ? 'bg-purple-500/10 border-purple-500/40' 
                  : 'bg-white/5 border-white/10 hover:border-white/20'
              }`}
            >
              <div className="flex items-center gap-2 pt-1">
                <GripVertical size={16} className="text-gray-500" />
                <span className="w-8 h-8 rounded-full bg-purple-500/20 text-purple-300 text-sm font-medium flex items-center justify-center">
                  {slide.pageNum}
                </span>
              </div>
              
              <div className="flex-1">
                {editingId === slide.id ? (
                  <div className="space-y-3">
                    <input type="text" value={editContent.title} onChange={e => setEditContent(p => ({ ...p, title: e.target.value }))} disabled={isBusy} className={`w-full px-3 py-2 rounded-lg bg-black/40 border border-white/20 text-white text-sm outline-none focus:ring-2 focus:ring-purple-500 ${disabledClass}`} placeholder="标题" />
                    <textarea value={editContent.layout_description} onChange={e => setEditContent(p => ({ ...p, layout_description: e.target.value }))} rows={2} disabled={isBusy} className={`w-full px-3 py-2 rounded-lg bg-black/40 border border-white/20 text-white text-sm outline-none focus:ring-2 focus:ring-purple-500 resize-none ${disabledClass}`} placeholder="布局描述" />
                    <div className="space-y-2">
                      {editContent.key_points.map((p, i) => (
                        <div key={i} className="flex gap-2">
                          <input type="text" value={p} onChange={e => handleKeyPointChange(i, e.target.value)} disabled={isBusy} className={`flex-1 px-3 py-2 rounded-lg bg-black/40 border border-white/20 text-white text-sm ${disabledClass}`} placeholder={`要点 ${i + 1}`} />
                          <button onClick={() => handleRemoveKeyPoint(i)} disabled={isBusy} className={`p-2 text-gray-400 hover:text-red-400 ${disabledClass}`}><Trash2 size={14} /></button>
                        </div>
                      ))}
                      <button onClick={handleAddKeyPoint} disabled={isBusy} className={`px-3 py-1.5 rounded-lg bg-white/5 border border-dashed border-white/20 text-gray-400 text-sm w-full hover:text-purple-400 hover:border-purple-400 ${disabledClass}`}>+ 添加要点</button>
                    </div>
                    <div className="flex gap-2 pt-2">
                      <button onClick={handleEditSave} disabled={isBusy} className={`px-3 py-1.5 rounded-lg bg-purple-500 text-white text-sm flex items-center gap-1 ${disabledClass}`}><Check size={14} /> 保存</button>
                      <button onClick={handleEditCancel} disabled={isBusy} className={`px-3 py-1.5 rounded-lg bg-white/10 text-gray-300 text-sm ${disabledClass}`}>取消</button>
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="mb-2"><h4 className="text-white font-medium">{slide.title}</h4></div>
                    <p className="text-xs text-purple-400/70 mb-2 italic">📐 {slide.layout_description}</p>
                    <ul className="space-y-1">
                      {slide.key_points.map((p, i) => (
                        <li key={i} className="text-sm text-gray-400 flex items-start gap-2">
                          <span className="text-purple-400 mt-0.5">•</span><span>{p}</span>
                        </li>
                      ))}
                    </ul>
                  </>
                )}
              </div>

              {editingId !== slide.id && (
                <div className="flex flex-col items-end gap-2 self-stretch justify-between py-1">
                  <div className="flex items-center gap-1">
                    <button onClick={() => handleMoveSlide(index, 'up')} disabled={isBusy || index === 0} className={`p-2 text-gray-400 hover:text-white disabled:opacity-30 ${disabledClass}`}><ChevronUp size={16} /></button>
                    <button onClick={() => handleMoveSlide(index, 'down')} disabled={isBusy || index === outlineData.length - 1} className={`p-2 text-gray-400 hover:text-white disabled:opacity-30 ${disabledClass}`}><ChevronDown size={16} /></button>
                    <button onClick={() => handleEditStart(slide)} disabled={isBusy} className={`p-2 text-gray-400 hover:text-purple-400 ${disabledClass}`}><Edit3 size={16} /></button>
                    <button onClick={() => handleDeleteSlide(slide.id)} disabled={isBusy} className={`p-2 text-gray-400 hover:text-red-400 ${disabledClass}`}><Trash2 size={16} /></button>
                  </div>
                  <button onClick={() => handleAddSlide(index)} disabled={isBusy} className={`p-2 text-gray-400 hover:text-green-400 hover:bg-green-500/10 rounded-lg transition-colors ${disabledClass}`} title="在此后添加新页面">
                    <Plus size={18} />
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      <div className="flex justify-between">
        <button onClick={() => setCurrentStep('upload')} disabled={isBusy} className={`px-6 py-2.5 rounded-lg border border-white/20 text-gray-300 hover:bg-white/10 flex items-center gap-2 ${disabledClass}`}>
          <ArrowLeft size={18} /> 返回上传
        </button>
        <button onClick={handleConfirmOutline} disabled={isBusy} className={`px-6 py-2.5 rounded-lg bg-gradient-to-r from-purple-600 to-pink-600 text-white font-semibold flex items-center gap-2 transition-all ${disabledClass}`}>
          {isGenerating ? '正在提交生成任务...' : '确认并开始生成'} <ArrowRight size={18} />
        </button>
      </div>

      <div className="mt-6 glass rounded-xl border border-white/10 p-4">
        <h3 className="text-sm font-semibold text-white mb-2 flex items-center gap-2">
          <Sparkles size={16} className="text-purple-400" /> AI 辅助修改
        </h3>
        <div className="flex gap-3">
          <textarea
            value={outlineFeedback}
            onChange={(e) => setOutlineFeedback(e.target.value)}
            placeholder="输入修改需求，例如：第3页更偏技术细节，突出方法贡献..."
            rows={2}
            disabled={isBusy}
            className={`flex-1 px-3 py-2 rounded-lg bg-black/40 border border-white/20 text-white text-sm outline-none focus:ring-2 focus:ring-purple-500 resize-none ${disabledClass}`}
          />
          <button
            onClick={handleRefineOutline}
            disabled={isBusy || !outlineFeedback.trim()}
            className={`px-4 py-2 rounded-lg bg-white/10 text-gray-200 text-sm flex items-center gap-2 hover:bg-white/20 ${disabledClass}`}
          >
            {isRefiningOutline ? 'AI 调整中...' : '开始调整'}
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

export default OutlineStep;
