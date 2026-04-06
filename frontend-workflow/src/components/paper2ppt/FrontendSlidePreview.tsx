import React, { useEffect, useRef, useState } from 'react';
import { Check, Pencil, X } from 'lucide-react';
import { FrontendSlide } from './types';
import { buildFrontendSlideMarkup } from './frontendSlideUtils';

interface FrontendSlidePreviewProps {
  slide: FrontendSlide;
  className?: string;
  mode?: 'responsive' | 'capture';
  captureRef?: (node: HTMLDivElement | null) => void;
  inlineEditEnabled?: boolean;
  onInlineFieldChange?: (fieldKey: string, value: string) => void;
  onInlineListItemChange?: (fieldKey: string, itemIndex: number, value: string) => void;
  onInlineListReplace?: (fieldKey: string, items: string[]) => void;
  onReplaceImage?: (imageKey: string, file: File) => void | Promise<void>;
}

interface InlineEditorState {
  fieldKey: string;
  fieldLabel: string;
  fieldType: 'text' | 'textarea' | 'list';
  itemIndex?: number;
  value: string;
  left: number;
  top: number;
  width: number;
  multiline: boolean;
}

const FrontendSlidePreview: React.FC<FrontendSlidePreviewProps> = ({
  slide,
  className = '',
  mode = 'responsive',
  captureRef,
  inlineEditEnabled = false,
  onInlineFieldChange,
  onInlineListItemChange,
  onInlineListReplace,
  onReplaceImage,
}) => {
  const DESIGN_WIDTH = 1600;
  const DESIGN_HEIGHT = 900;
  const containerRef = useRef<HTMLDivElement | null>(null);
  const imageInputRef = useRef<HTMLInputElement | null>(null);
  const [scale, setScale] = useState(1);
  const [inlineEditor, setInlineEditor] = useState<InlineEditorState | null>(null);
  const [pendingImageKey, setPendingImageKey] = useState<string | null>(null);

  useEffect(() => {
    if (mode !== 'responsive' || !containerRef.current) {
      return undefined;
    }

    const node = containerRef.current;
    const updateScale = () => {
      const rect = node.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      setScale(Math.min(rect.width / DESIGN_WIDTH, rect.height / DESIGN_HEIGHT));
    };

    updateScale();
    const observer = new ResizeObserver(() => updateScale());
    observer.observe(node);

    return () => {
      observer.disconnect();
    };
  }, [mode]);

  useEffect(() => {
    setInlineEditor(null);
  }, [slide.slideId, slide.htmlTemplate, slide.cssCode]);

  useEffect(() => {
    if (!inlineEditor) {
      return undefined;
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setInlineEditor(null);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [inlineEditor]);

  const persistInlineEdit = (editor: InlineEditorState | null) => {
    if (!editor) return;

    const nextValue = editor.value;
    if (editor.fieldType === 'list') {
      if (typeof editor.itemIndex === 'number') {
        onInlineListItemChange?.(editor.fieldKey, editor.itemIndex, nextValue);
      } else {
        const items = nextValue
          .split(/\n|•/g)
          .map((item) => item.trim())
          .filter(Boolean);
        onInlineListReplace?.(editor.fieldKey, items);
      }
    } else {
      onInlineFieldChange?.(editor.fieldKey, nextValue);
    }
  };

  const commitInlineEdit = () => {
    if (!inlineEditor) return;
    persistInlineEdit(inlineEditor);
    setInlineEditor(null);
  };

  const openImagePicker = (imageKey: string) => {
    if (!imageKey || !inlineEditEnabled || mode !== 'responsive') {
      return;
    }
    setPendingImageKey(imageKey);
    imageInputRef.current?.click();
  };

  const handleImageInputChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    const imageKey = pendingImageKey;
    event.target.value = '';
    setPendingImageKey(null);
    if (!file || !imageKey) {
      return;
    }
    await onReplaceImage?.(imageKey, file);
  };

  const handleEditableClick = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!inlineEditEnabled || mode !== 'responsive' || !containerRef.current) {
      return;
    }

    const target = event.target as HTMLElement;
    if (target.closest('[data-inline-editor="true"]')) {
      return;
    }

    const imageNode = target.closest('[data-image-key]') as HTMLElement | null;
    if (imageNode) {
      event.preventDefault();
      event.stopPropagation();
      if (inlineEditor) {
        persistInlineEdit(inlineEditor);
      }
      setInlineEditor(null);
      openImagePicker(imageNode.dataset.imageKey || '');
      return;
    }

    const editableNode = target.closest('[data-edit-key]') as HTMLElement | null;
    if (!editableNode) {
      if (inlineEditor) {
        persistInlineEdit(inlineEditor);
      }
      setInlineEditor(null);
      return;
    }

    event.preventDefault();
    event.stopPropagation();

    const fieldKey = editableNode.dataset.editKey || '';
    const fieldType = (editableNode.dataset.editType || 'text') as InlineEditorState['fieldType'];
    const itemIndexRaw = editableNode.dataset.editIndex;
    const itemIndex = itemIndexRaw !== undefined ? Number.parseInt(itemIndexRaw, 10) : undefined;
    const field = slide.editableFields.find((item) => item.key === fieldKey);
    if (!field) {
      return;
    }
    if (inlineEditor) {
      persistInlineEdit(inlineEditor);
    }

    const containerRect = containerRef.current.getBoundingClientRect();
    const targetRect = editableNode.getBoundingClientRect();
    const rawWidth = Math.max(targetRect.width + 28, 220);
    const width = Math.min(rawWidth, Math.max(260, containerRect.width - 24));
    const left = Math.min(
      Math.max(12, targetRect.left - containerRect.left - 8),
      Math.max(12, containerRect.width - width - 12),
    );
    const heightGuess = Math.max(targetRect.height + 18, field.type === 'textarea' || fieldType === 'list' ? 120 : 48);
    const top = Math.min(
      Math.max(12, targetRect.top - containerRect.top - 10),
      Math.max(12, containerRect.height - heightGuess - 12),
    );

    const value = field.type === 'list'
      ? typeof itemIndex === 'number'
        ? field.items[itemIndex] || ''
        : field.items.join('\n')
      : field.value || '';
    const multiline = field.type === 'textarea'
      || field.type === 'list'
      || editableNode.tagName === 'P'
      || editableNode.tagName === 'DIV'
      || value.includes('\n')
      || targetRect.height >= 44;

    setInlineEditor({
      fieldKey,
      fieldLabel: field.label || fieldKey,
      fieldType,
      itemIndex,
      value,
      left,
      top,
      width,
      multiline,
    });
  };

  if (mode === 'capture') {
    return (
      <div
        ref={captureRef}
        className={className}
        style={{
          width: `${DESIGN_WIDTH}px`,
          height: `${DESIGN_HEIGHT}px`,
          display: 'block',
          overflow: 'hidden',
          background: '#07101f',
        }}
      >
        <div
          style={{
            width: '100%',
            height: '100%',
            display: 'block',
            overflow: 'hidden',
            borderRadius: '28px',
            background: '#0b1020',
            boxShadow: '0 20px 60px rgba(0, 0, 0, 0.3)',
          }}
          dangerouslySetInnerHTML={{ __html: buildFrontendSlideMarkup(slide) }}
        />
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className={`relative w-full aspect-[16/9] overflow-hidden rounded-[28px] bg-[#07101f] ${className}`}
      onMouseDown={handleEditableClick}
    >
      <div
        className="absolute left-1/2 top-1/2"
        style={{
          width: `${DESIGN_WIDTH}px`,
          height: `${DESIGN_HEIGHT}px`,
          transform: `translate(-50%, -50%) scale(${scale})`,
          transformOrigin: 'center center',
        }}
      >
        <div
          className="w-full h-full overflow-hidden rounded-[28px] bg-[#0b1020] shadow-[0_20px_60px_rgba(0,0,0,0.3)]"
          dangerouslySetInnerHTML={{ __html: buildFrontendSlideMarkup(slide) }}
        />
      </div>

      {inlineEditEnabled && (
        <div className="pointer-events-none absolute inset-x-4 bottom-4 z-20">
          <div className="inline-flex items-center gap-2 rounded-full border border-cyan-400/20 bg-[#06101d]/85 px-3 py-1.5 text-[11px] text-cyan-100/85 shadow-[0_12px_28px_rgba(0,0,0,0.28)] backdrop-blur-xl">
            <Pencil size={12} />
            点击文字可直接编辑，点击图片可替换
          </div>
        </div>
      )}

      {inlineEditor && (
        <div
          data-inline-editor="true"
          className="absolute z-30 rounded-2xl border border-cyan-400/30 bg-[#07101d]/95 p-3 shadow-[0_18px_50px_rgba(0,0,0,0.4)] backdrop-blur-xl"
          style={{
            left: `${inlineEditor.left}px`,
            top: `${inlineEditor.top}px`,
            width: `${inlineEditor.width}px`,
          }}
          onMouseDown={(event) => {
            event.stopPropagation();
          }}
        >
          <div className="mb-2 flex items-center justify-between gap-2">
            <div className="text-[11px] font-medium uppercase tracking-[0.18em] text-cyan-200/80">
              {inlineEditor.fieldLabel}
            </div>
            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={commitInlineEdit}
                className="rounded-lg bg-cyan-500 px-2 py-1 text-[11px] font-medium text-white"
              >
                <span className="inline-flex items-center gap-1">
                  <Check size={12} /> 保存
                </span>
              </button>
              <button
                type="button"
                onClick={() => setInlineEditor(null)}
                className="rounded-lg bg-white/10 px-2 py-1 text-[11px] font-medium text-gray-200"
              >
                <span className="inline-flex items-center gap-1">
                  <X size={12} /> 取消
                </span>
              </button>
            </div>
          </div>
          {inlineEditor.multiline ? (
            <textarea
              autoFocus
              value={inlineEditor.value}
              onChange={(event) =>
                setInlineEditor((prev) => (prev ? { ...prev, value: event.target.value } : prev))
              }
              onKeyDown={(event) => {
                if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
                  event.preventDefault();
                  commitInlineEdit();
                }
              }}
              rows={inlineEditor.fieldType === 'list' && inlineEditor.itemIndex === undefined ? 5 : 4}
              className="w-full rounded-xl border border-white/10 bg-black/35 px-3 py-2 text-sm text-white outline-none resize-none focus:ring-2 focus:ring-cyan-500"
            />
          ) : (
            <input
              autoFocus
              type="text"
              value={inlineEditor.value}
              onChange={(event) =>
                setInlineEditor((prev) => (prev ? { ...prev, value: event.target.value } : prev))
              }
              onKeyDown={(event) => {
                if (event.key === 'Enter') {
                  event.preventDefault();
                  commitInlineEdit();
                }
              }}
              className="w-full rounded-xl border border-white/10 bg-black/35 px-3 py-2 text-sm text-white outline-none focus:ring-2 focus:ring-cyan-500"
            />
          )}
        </div>
      )}

      {inlineEditEnabled && (
        <input
          ref={imageInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={handleImageInputChange}
        />
      )}
    </div>
  );
};

export default FrontendSlidePreview;
