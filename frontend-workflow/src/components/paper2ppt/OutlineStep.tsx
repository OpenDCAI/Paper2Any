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
  isRefiningOutline
}) => {
  const disabledClass = "disabled:opacity-50 disabled:cursor-not-allowed";
  return (
    <div className="max-w-5xl mx-auto">
      <div className="mb-8 text-center">
        <h2 className="paper2ppt-title mb-2 text-3xl font-bold">确认大纲</h2>
        <p className="paper2ppt-subtitle">检查从论文提取的内容结构，可编辑、排序或删除</p>
      </div>

      <div className="paper2ppt-panel mb-6 rounded-[28px] p-6">
        <div className="space-y-3">
          {outlineData.map((slide, index) => (
            <div 
              key={slide.id} 
              className={`flex items-start gap-4 rounded-2xl p-4 transition-all ${
                editingId === slide.id 
                  ? 'paper2ppt-outline-card paper2ppt-outline-card-active'
                  : 'paper2ppt-outline-card hover:border-[rgba(140,29,64,0.24)]'
              }`}
            >
              <div className="flex items-center gap-2 pt-1">
                <GripVertical size={16} className="text-[#8c1d40]" />
                <span className="paper2ppt-page-badge flex h-8 w-8 items-center justify-center rounded-full text-sm font-semibold">
                  {slide.pageNum}
                </span>
              </div>
              
              <div className="flex-1">
                {editingId === slide.id ? (
                  <div className="space-y-3">
                    <input type="text" value={editContent.title} onChange={e => setEditContent(p => ({ ...p, title: e.target.value }))} disabled={isRefiningOutline} className={`paper2ppt-input w-full rounded-xl px-3 py-2 text-sm ${disabledClass}`} placeholder="标题" />
                    <textarea value={editContent.layout_description} onChange={e => setEditContent(p => ({ ...p, layout_description: e.target.value }))} rows={2} disabled={isRefiningOutline} className={`paper2ppt-input w-full rounded-xl px-3 py-2 text-sm resize-none ${disabledClass}`} placeholder="布局描述" />
                    <div className="space-y-2">
                      {editContent.key_points.map((p, i) => (
                        <div key={i} className="flex gap-2">
                          <input type="text" value={p} onChange={e => handleKeyPointChange(i, e.target.value)} disabled={isRefiningOutline} className={`paper2ppt-input flex-1 rounded-xl px-3 py-2 text-sm ${disabledClass}`} placeholder={`要点 ${i + 1}`} />
                          <button onClick={() => handleRemoveKeyPoint(i)} disabled={isRefiningOutline} className={`paper2ppt-icon-button p-2 hover:text-[#b12d3e] ${disabledClass}`}><Trash2 size={14} /></button>
                        </div>
                      ))}
                      <button onClick={handleAddKeyPoint} disabled={isRefiningOutline} className={`paper2ppt-button-ghost w-full rounded-xl border border-dashed px-3 py-1.5 text-sm ${disabledClass}`}>+ 添加要点</button>
                    </div>
                    <div className="flex gap-2 pt-2">
                      <button onClick={handleEditSave} disabled={isRefiningOutline} className={`paper2ppt-button-primary flex items-center gap-1 rounded-xl px-3 py-1.5 text-sm ${disabledClass}`}><Check size={14} /> 保存</button>
                      <button onClick={handleEditCancel} disabled={isRefiningOutline} className={`paper2ppt-button-secondary rounded-xl px-3 py-1.5 text-sm ${disabledClass}`}>取消</button>
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="mb-2"><h4 className="text-[1.02rem] font-semibold text-[#1d1c1a]">{slide.title}</h4></div>
                    <p className="mb-2 text-xs italic text-[#8c1d40]">布局描述: {slide.layout_description}</p>
                    <ul className="space-y-1">
                      {slide.key_points.map((p, i) => (
                        <li key={i} className="flex items-start gap-2 text-sm text-[#675f58]">
                          <span className="mt-0.5 text-[#8c1d40]">•</span><span>{p}</span>
                        </li>
                      ))}
                    </ul>
                  </>
                )}
              </div>

              {editingId !== slide.id && (
                <div className="flex flex-col items-end gap-2 self-stretch justify-between py-1">
                  <div className="flex items-center gap-1">
                    <button onClick={() => handleMoveSlide(index, 'up')} disabled={isRefiningOutline || index === 0} className={`paper2ppt-icon-button p-2 disabled:opacity-30 ${disabledClass}`}><ChevronUp size={16} /></button>
                    <button onClick={() => handleMoveSlide(index, 'down')} disabled={isRefiningOutline || index === outlineData.length - 1} className={`paper2ppt-icon-button p-2 disabled:opacity-30 ${disabledClass}`}><ChevronDown size={16} /></button>
                    <button onClick={() => handleEditStart(slide)} disabled={isRefiningOutline} className={`paper2ppt-icon-button p-2 ${disabledClass}`}><Edit3 size={16} /></button>
                    <button onClick={() => handleDeleteSlide(slide.id)} disabled={isRefiningOutline} className={`paper2ppt-icon-button p-2 hover:text-[#b12d3e] ${disabledClass}`}><Trash2 size={16} /></button>
                  </div>
                  <button onClick={() => handleAddSlide(index)} disabled={isRefiningOutline} className={`paper2ppt-icon-button rounded-lg p-2 hover:text-[#21704c] ${disabledClass}`} title="在此后添加新页面">
                    <Plus size={18} />
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      <div className="flex justify-between">
        <button onClick={() => setCurrentStep('upload')} disabled={isRefiningOutline} className={`paper2ppt-button-secondary flex items-center gap-2 rounded-xl px-6 py-2.5 ${disabledClass}`}>
          <ArrowLeft size={18} /> 返回上传
        </button>
        <button onClick={handleConfirmOutline} disabled={isRefiningOutline} className={`paper2ppt-button-primary flex items-center gap-2 rounded-xl px-6 py-2.5 font-semibold ${disabledClass}`}>
          确认并开始生成 <ArrowRight size={18} />
        </button>
      </div>

      <div className="paper2ppt-panel mt-6 rounded-[24px] p-4">
        <h3 className="mb-2 flex items-center gap-2 text-sm font-semibold text-[#1d1c1a]">
          <Sparkles size={16} className="text-[#8c1d40]" /> AI 辅助修改
        </h3>
        <div className="flex gap-3">
          <textarea
            value={outlineFeedback}
            onChange={(e) => setOutlineFeedback(e.target.value)}
            placeholder="输入修改需求，例如：第3页更偏技术细节，突出方法贡献..."
            rows={2}
            disabled={isRefiningOutline}
            className={`paper2ppt-input flex-1 rounded-xl px-3 py-2 text-sm resize-none ${disabledClass}`}
          />
          <button
            onClick={handleRefineOutline}
            disabled={isRefiningOutline || !outlineFeedback.trim()}
            className={`paper2ppt-button-secondary flex items-center gap-2 rounded-xl px-4 py-2 text-sm ${disabledClass}`}
          >
            {isRefiningOutline ? 'AI 调整中...' : '开始调整'}
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

export default OutlineStep;
