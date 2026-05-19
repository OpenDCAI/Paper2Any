import React, { useEffect, useRef, useState } from 'react';
import { Check, Pencil, X } from 'lucide-react';
import {
  buildFrontendInsertZoneTarget,
  FrontendDeckTheme,
  FrontendLayoutIR,
  FrontendSlide,
} from './types';
import { buildFrontendSlideMarkup } from './frontendSlideUtils';
import { DESIGN_HEIGHT, DESIGN_WIDTH } from './structuredSlideModel';

interface FrontendSlidePreviewProps {
  slide: FrontendSlide;
  deckTheme?: FrontendDeckTheme | null;
  className?: string;
  mode?: 'responsive' | 'capture';
  captureRef?: (node: HTMLDivElement | null) => void;
  inlineEditEnabled?: boolean;
  onInlineFieldChange?: (fieldKey: string, value: string) => void;
  onInlineListItemChange?: (fieldKey: string, itemIndex: number, value: string) => void;
  onInlineListReplace?: (fieldKey: string, items: string[]) => void;
  onReplaceImage?: (imageKey: string, file: File) => void | Promise<void>;
  onDeleteImage?: (imageKey: string) => void;
  selectedBlockId?: string | null;
  onSelectBlock?: (blockId: string | null) => void;
  onHoverBlock?: (blockId: string | null) => void;
  onLayoutIrChange?: (layoutIr: FrontendLayoutIR) => void;
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

interface BlockHoverState {
  blockId: string;
  role: string;
  kind: 'block' | 'zone';
  left: number;
  top: number;
  width: number;
  height: number;
}

const FrontendSlidePreview: React.FC<FrontendSlidePreviewProps> = ({
  slide,
  deckTheme = null,
  className = '',
  mode = 'responsive',
  captureRef,
  inlineEditEnabled = false,
  onInlineFieldChange,
  onInlineListItemChange,
  onInlineListReplace,
  onReplaceImage,
  onDeleteImage,
  selectedBlockId = null,
  onSelectBlock,
  onHoverBlock,
  onLayoutIrChange,
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const imageInputRef = useRef<HTMLInputElement | null>(null);
  const [scale, setScale] = useState(1);
  const [inlineEditor, setInlineEditor] = useState<InlineEditorState | null>(null);
  const [pendingImageKey, setPendingImageKey] = useState<string | null>(null);
  const [hoveredBlock, setHoveredBlock] = useState<BlockHoverState | null>(null);
  const lastHoverBlockIdRef = useRef<string | null>(null);
  const onLayoutIrChangeRef = useRef(onLayoutIrChange);

  useEffect(() => {
    onLayoutIrChangeRef.current = onLayoutIrChange;
  }, [onLayoutIrChange]);

  useEffect(() => {
    if (!containerRef.current) {
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
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    setInlineEditor(null);
    setHoveredBlock(null);
    lastHoverBlockIdRef.current = null;
    onHoverBlock?.(null);
  }, [slide.slideId]);

  useEffect(() => {
    setInlineEditor(null);
  }, [slide.htmlTemplate, slide.cssCode, slide.templateKey, slide.layoutMode, slide.blocks, slide.root, slide.visualSpec, slide.renderEngine]);

  useEffect(() => {
    if (slide.renderEngine !== 'canvas' || !containerRef.current) {
      return undefined;
    }

    let cancelled = false;
    const measure = () => {
      const container = containerRef.current;
      if (!container || cancelled) return;
      const canvasRoot = container.querySelector('.template-canvas-schema') as HTMLElement | null;
      if (!canvasRoot) {
        return;
      }

      const slideRoot = (canvasRoot.closest('.slide-root') as HTMLElement | null) || canvasRoot;
      const rootRect = slideRoot.getBoundingClientRect();
      const nodes = Array.from(canvasRoot.querySelectorAll('[data-canvas-node-id]')) as HTMLElement[];
      const overflowIssues: string[] = [];
      const irNodes = nodes.map((node) => {
        const rect = node.getBoundingClientRect();
        const style = window.getComputedStyle(node);
        const nodeId = node.dataset.canvasNodeId || node.dataset.blockId || '';
        const left = (rect.left - rootRect.left) / Math.max(scale, 0.0001);
        const top = (rect.top - rootRect.top) / Math.max(scale, 0.0001);
        const width = rect.width / Math.max(scale, 0.0001);
        const height = rect.height / Math.max(scale, 0.0001);
        const overflow = node.scrollWidth > node.clientWidth + 1
          || node.scrollHeight > node.clientHeight + 1
          || left < -1
          || top < -1
          || left + width > DESIGN_WIDTH + 1
          || top + height > DESIGN_HEIGHT + 1;
        if (overflow) {
          overflowIssues.push(`${nodeId || 'unknown'} overflow`);
        }
        return {
          nodeId,
          type: node.dataset.blockRole === 'container' ? 'container' as const : 'component' as const,
          component: node.dataset.blockRole === 'container' ? undefined : node.dataset.blockRole as FrontendLayoutIR['nodes'][number]['component'],
          box: {
            x: Math.round(left),
            y: Math.round(top),
            w: Math.round(width),
            h: Math.round(height),
          },
          computedStyle: {
            fontFamily: style.fontFamily,
            fontSize: style.fontSize,
            fontWeight: style.fontWeight,
            fontStyle: style.fontStyle,
            lineHeight: style.lineHeight,
            color: style.color,
            backgroundColor: style.backgroundColor,
            borderColor: style.borderColor,
            borderTopColor: style.borderTopColor,
            borderRightColor: style.borderRightColor,
            borderBottomColor: style.borderBottomColor,
            borderLeftColor: style.borderLeftColor,
            borderTopWidth: style.borderTopWidth,
            borderRightWidth: style.borderRightWidth,
            borderBottomWidth: style.borderBottomWidth,
            borderLeftWidth: style.borderLeftWidth,
            paddingTop: style.paddingTop,
            paddingRight: style.paddingRight,
            paddingBottom: style.paddingBottom,
            paddingLeft: style.paddingLeft,
            textAlign: style.textAlign,
            verticalAlign: style.verticalAlign,
            display: style.display,
            alignItems: style.alignItems,
            justifyContent: style.justifyContent,
          },
          overflow,
        };
      });

      onLayoutIrChangeRef.current?.({
        schemaVersion: 'ppt_layout_ir_v1',
        slideId: slide.slideId,
        viewport: {
          width: DESIGN_WIDTH,
          height: DESIGN_HEIGHT,
          scale,
        },
        nodes: irNodes,
        overflowIssues,
      });
    };

    const frame = window.requestAnimationFrame(() => {
      measure();
      window.setTimeout(measure, 120);
    });
    const observer = new ResizeObserver(() => measure());
    observer.observe(containerRef.current);

    return () => {
      cancelled = true;
      window.cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, [mode, slide.renderEngine, slide.slideId, slide.root, slide.visualSpec, slide.blocks, slide.editableFields, slide.visualAssets, scale]);

  useEffect(() => {
    if (!inlineEditEnabled || mode !== 'responsive') {
      return undefined;
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setInlineEditor(null);
        return;
      }
      if ((event.key === 'Delete' || event.key === 'Backspace') && selectedBlockId) {
        const target = event.target as HTMLElement | null;
        if (target?.closest('input, textarea, [contenteditable="true"]')) {
          return;
        }
        const asset = slide.visualAssets.find((item) => item.key === selectedBlockId);
        if (!asset) {
          return;
        }
        event.preventDefault();
        onDeleteImage?.(asset.key);
        setInlineEditor(null);
        onSelectBlock?.(null);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [inlineEditEnabled, mode, onDeleteImage, onSelectBlock, selectedBlockId, slide.visualAssets]);

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
    if (!imageKey || !inlineEditEnabled) return;
    setPendingImageKey(imageKey);
    imageInputRef.current?.click();
  };

  const resolveBlockNode = (target: HTMLElement) => {
    if (target.closest('[data-inline-editor="true"]')) {
      return null;
    }
    return target.closest('[data-block-id]') as HTMLElement | null;
  };

  const resolveInsertZoneNode = (target: HTMLElement) => {
    if (target.closest('[data-inline-editor="true"]')) {
      return null;
    }
    return target.closest('[data-insert-zone]') as HTMLElement | null;
  };

  const notifyHoverBlock = (blockId: string | null) => {
    if (lastHoverBlockIdRef.current === blockId) {
      return;
    }
    lastHoverBlockIdRef.current = blockId;
    onHoverBlock?.(blockId);
  };

  const clearHoveredBlock = () => {
    setHoveredBlock(null);
    notifyHoverBlock(null);
  };

  const handleBlockMouseMove = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!inlineEditEnabled || mode !== 'responsive' || !containerRef.current) {
      return;
    }

    const target = event.target as HTMLElement;
    const blockNode = resolveBlockNode(target);
    const zoneNode = resolveInsertZoneNode(target);
    const blockId = blockNode?.dataset.blockId || '';
    const zone = zoneNode?.dataset.insertZone || '';
    const hoverTargetId = blockId || (zone ? buildFrontendInsertZoneTarget(zone) : '');
    const hoverNode = blockNode || zoneNode;
    if (!hoverNode || !hoverTargetId) {
      clearHoveredBlock();
      return;
    }

    const containerRect = containerRef.current.getBoundingClientRect();
    const blockRect = hoverNode.getBoundingClientRect();
    const nextHoveredBlock: BlockHoverState = {
      blockId: hoverTargetId,
      role: blockNode?.dataset.blockRole || zoneNode?.dataset.insertZoneLabel || zone,
      kind: blockId ? 'block' : 'zone',
      left: Math.max(0, blockRect.left - containerRect.left),
      top: Math.max(0, blockRect.top - containerRect.top),
      width: Math.max(0, blockRect.width),
      height: Math.max(0, blockRect.height),
    };

    setHoveredBlock((prev) => {
      if (
        prev
        && prev.blockId === nextHoveredBlock.blockId
        && Math.abs(prev.left - nextHoveredBlock.left) < 1
        && Math.abs(prev.top - nextHoveredBlock.top) < 1
        && Math.abs(prev.width - nextHoveredBlock.width) < 1
        && Math.abs(prev.height - nextHoveredBlock.height) < 1
      ) {
        return prev;
      }
      return nextHoveredBlock;
    });
    notifyHoverBlock(hoverTargetId);
  };

  const handleImageInputChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    const imageKey = pendingImageKey;
    event.target.value = '';
    setPendingImageKey(null);
    if (!file || !imageKey) return;
    await onReplaceImage?.(imageKey, file);
  };

  const handleEditableClick = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!inlineEditEnabled || !containerRef.current) {
      return;
    }

    const target = event.target as HTMLElement;
    if (target.closest('[data-inline-editor="true"]')) {
      return;
    }

    const blockNode = resolveBlockNode(target);
    if (blockNode?.dataset.blockId) {
      onSelectBlock?.(blockNode.dataset.blockId);
    }
    const zoneNode = resolveInsertZoneNode(target);
    const zone = zoneNode?.dataset.insertZone || '';
    if (!blockNode && zone) {
      onSelectBlock?.(buildFrontendInsertZoneTarget(zone));
    }

    const imageNode = target.closest('[data-image-key]') as HTMLElement | null;
    if (imageNode) {
      event.preventDefault();
      event.stopPropagation();
      if (inlineEditor) persistInlineEdit(inlineEditor);
      setInlineEditor(null);
      const imageKey = imageNode.dataset.imageKey || '';
      onSelectBlock?.(imageKey);
      openImagePicker(imageKey);
      return;
    }

    const editableNode = target.closest('[data-edit-key]') as HTMLElement | null;
    if (!editableNode) {
      if (inlineEditor) persistInlineEdit(inlineEditor);
      setInlineEditor(null);
      if (!blockNode && !zoneNode) {
        onSelectBlock?.(null);
      }
      return;
    }

    event.preventDefault();
    event.stopPropagation();

    const fieldKey = editableNode.dataset.editKey || '';
    const fieldType = (editableNode.dataset.editType || 'text') as InlineEditorState['fieldType'];
    const itemIndexRaw = editableNode.dataset.editIndex;
    const itemIndex = itemIndexRaw !== undefined ? Number.parseInt(itemIndexRaw, 10) : undefined;
    const field = slide.editableFields.find((item) => item.key === fieldKey);
    if (!field) return;
    if (inlineEditor) persistInlineEdit(inlineEditor);

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
        ref={(node) => {
          containerRef.current = node;
          captureRef?.(node);
        }}
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
          dangerouslySetInnerHTML={{ __html: buildFrontendSlideMarkup(slide, deckTheme) }}
        />
      </div>
    );
  }

  return (
    <div className={`w-full ${className}`}>
      <div
        ref={containerRef}
        className="relative w-full aspect-[16/9] overflow-hidden rounded-[28px] bg-[#07101f]"
        onMouseMove={handleBlockMouseMove}
        onMouseLeave={clearHoveredBlock}
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
          dangerouslySetInnerHTML={{ __html: buildFrontendSlideMarkup(slide, deckTheme) }}
        />
      </div>

        {hoveredBlock && (
          <div
            className="pointer-events-none absolute z-20 rounded-[18px] border border-white/85 bg-white/5 shadow-[0_0_0_1px_rgba(255,255,255,0.35),0_18px_46px_rgba(15,23,42,0.26)]"
            style={{
              left: `${hoveredBlock.left}px`,
              top: `${hoveredBlock.top}px`,
              width: `${hoveredBlock.width}px`,
              height: `${hoveredBlock.height}px`,
            }}
          />
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

      {inlineEditEnabled && (
        <div className="mt-3 flex flex-wrap items-center gap-2 text-[11px] text-cyan-100/80">
          <span className="inline-flex items-center gap-1 rounded-full border border-cyan-400/15 bg-cyan-500/10 px-3 py-1.5">
            <Pencil size={12} />
            {hoveredBlock
              ? hoveredBlock.kind === 'zone'
                ? slide.renderEngine === 'canvas'
                  ? `鼠标所在空白区域：${hoveredBlock.role}，点击后可新增同级节点`
                  : `鼠标所在空白区域：${hoveredBlock.role}，点击后可新增同级 block`
                : `鼠标所在区域：${hoveredBlock.blockId}`
              : selectedBlockId
                ? `当前选择区域：${selectedBlockId}`
                : slide.renderEngine === 'canvas'
                  ? '鼠标移到 Canvas 节点或空白区域上查看目标，点击固定插入位置'
                  : '鼠标移到 block 或空白区域上查看目标，点击固定插入位置'}
          </span>
          {selectedBlockId && (
            <span className="inline-flex rounded-full border border-emerald-400/20 bg-emerald-500/10 px-3 py-1.5 text-emerald-100/85">
              已选择：{selectedBlockId}
            </span>
          )}
          {selectedBlockId && slide.visualAssets.some((asset) => asset.key === selectedBlockId) && (
            <span className="inline-flex rounded-full border border-rose-400/20 bg-rose-500/10 px-3 py-1.5 text-rose-100/85">
              按 Delete 删除图片
            </span>
          )}
        </div>
      )}
    </div>
  );
};

export default FrontendSlidePreview;
