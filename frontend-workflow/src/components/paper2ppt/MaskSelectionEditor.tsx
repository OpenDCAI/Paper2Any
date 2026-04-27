import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Circle, Crop, Eraser } from 'lucide-react';
import type { MaskSelectionShape, MaskSelectionSpec } from './types';

interface MaskSelectionEditorProps {
  imageUrl: string;
  alt: string;
  disabled?: boolean;
  value: MaskSelectionSpec | null;
  onChange: (nextValue: MaskSelectionSpec | null) => void;
}

interface DragSelection {
  x: number;
  y: number;
  width: number;
  height: number;
}

const MIN_NORMALIZED_SIZE = 0.01;

const clamp01 = (value: number) => Math.min(1, Math.max(0, value));

const normalizeSelection = (
  startX: number,
  startY: number,
  endX: number,
  endY: number,
): DragSelection => {
  const x = Math.min(startX, endX);
  const y = Math.min(startY, endY);
  const width = Math.abs(endX - startX);
  const height = Math.abs(endY - startY);
  return {
    x: clamp01(x),
    y: clamp01(y),
    width: clamp01(width),
    height: clamp01(height),
  };
};

const MaskSelectionEditor: React.FC<MaskSelectionEditorProps> = ({
  imageUrl,
  alt,
  disabled = false,
  value,
  onChange,
}) => {
  const overlayRef = useRef<HTMLDivElement | null>(null);
  const dragStartRef = useRef<{ x: number; y: number } | null>(null);
  const [activeShape, setActiveShape] = useState<MaskSelectionShape>(value?.shape || 'rect');
  const [draftSelection, setDraftSelection] = useState<DragSelection | null>(null);

  useEffect(() => {
    if (value?.shape) {
      setActiveShape(value.shape);
    }
  }, [value?.shape]);

  useEffect(() => {
    setDraftSelection(null);
    dragStartRef.current = null;
  }, [imageUrl]);

  const selection = useMemo<MaskSelectionSpec | null>(() => {
    if (!draftSelection) return value;
    return {
      shape: activeShape,
      ...draftSelection,
    };
  }, [activeShape, draftSelection, value]);

  const updateShape = (shape: MaskSelectionShape) => {
    setActiveShape(shape);
    if (value) {
      onChange({ ...value, shape });
    }
  };

  const getRelativePoint = (clientX: number, clientY: number) => {
    const rect = overlayRef.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0 || rect.height <= 0) return null;

    const x = clamp01((clientX - rect.left) / rect.width);
    const y = clamp01((clientY - rect.top) / rect.height);
    return { x, y };
  };

  const handlePointerDown = (event: React.MouseEvent<HTMLDivElement>) => {
    if (disabled) return;
    const point = getRelativePoint(event.clientX, event.clientY);
    if (!point) return;
    dragStartRef.current = point;
    setDraftSelection({ x: point.x, y: point.y, width: 0, height: 0 });
  };

  const handlePointerMove = (event: React.MouseEvent<HTMLDivElement>) => {
    if (disabled) return;
    const start = dragStartRef.current;
    if (!start) return;
    const point = getRelativePoint(event.clientX, event.clientY);
    if (!point) return;
    setDraftSelection(normalizeSelection(start.x, start.y, point.x, point.y));
  };

  const finalizeSelection = () => {
    const start = dragStartRef.current;
    const current = draftSelection;
    dragStartRef.current = null;
    if (!start || !current) {
      setDraftSelection(null);
      return;
    }
    if (current.width < MIN_NORMALIZED_SIZE || current.height < MIN_NORMALIZED_SIZE) {
      setDraftSelection(null);
      onChange(null);
      return;
    }
    onChange({
      shape: activeShape,
      x: current.x,
      y: current.y,
      width: current.width,
      height: current.height,
    });
    setDraftSelection(null);
  };

  const selectionStyle = selection
    ? {
        left: `${selection.x * 100}%`,
        top: `${selection.y * 100}%`,
        width: `${selection.width * 100}%`,
        height: `${selection.height * 100}%`,
        borderRadius: selection.shape === 'circle' ? '9999px' : '0.5rem',
      }
    : null;

  return (
    <div className="w-full min-h-0 flex flex-col gap-3">
      <div className="shrink-0 flex items-center justify-between gap-3 text-xs text-gray-300">
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => updateShape('rect')}
            disabled={disabled}
            className={`px-3 py-1.5 rounded-lg border flex items-center gap-1.5 transition-colors ${
              activeShape === 'rect'
                ? 'border-cyan-400 bg-cyan-500/20 text-cyan-200'
                : 'border-white/15 bg-black/20 hover:bg-white/10'
            } disabled:opacity-40`}
          >
            <Crop size={13} />
            矩形
          </button>
          <button
            type="button"
            onClick={() => updateShape('circle')}
            disabled={disabled}
            className={`px-3 py-1.5 rounded-lg border flex items-center gap-1.5 transition-colors ${
              activeShape === 'circle'
                ? 'border-cyan-400 bg-cyan-500/20 text-cyan-200'
                : 'border-white/15 bg-black/20 hover:bg-white/10'
            } disabled:opacity-40`}
          >
            <Circle size={13} />
            圆形
          </button>
          <button
            type="button"
            onClick={() => onChange(null)}
            disabled={disabled || !value}
            className="px-3 py-1.5 rounded-lg border border-white/15 bg-black/20 hover:bg-white/10 flex items-center gap-1.5 disabled:opacity-40"
          >
            <Eraser size={13} />
            清除选区
          </button>
        </div>
        <span className="text-gray-400">
          在图片上拖拽，标出要局部修改的区域
        </span>
      </div>

      <div className="w-full min-h-0 flex items-center justify-center">
        <div className="w-full aspect-[16/9] max-h-full min-h-0 flex items-center justify-center overflow-hidden rounded-lg bg-black/10">
          <div className="relative inline-block max-w-full max-h-full">
            <img
              src={imageUrl}
              alt={alt}
              className="block max-w-full max-h-full object-contain select-none pointer-events-none"
              draggable={false}
            />
            <div
              ref={overlayRef}
              className={`absolute inset-0 ${disabled ? 'cursor-not-allowed' : 'cursor-crosshair'}`}
              onMouseDown={handlePointerDown}
              onMouseMove={handlePointerMove}
              onMouseUp={finalizeSelection}
              onMouseLeave={finalizeSelection}
            >
              <div className="absolute inset-0 bg-black/10" />
              {selectionStyle ? (
                <div
                  className="absolute border-2 border-cyan-300 bg-cyan-400/25 shadow-[0_0_0_9999px_rgba(0,0,0,0.22)]"
                  style={selectionStyle}
                />
              ) : null}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MaskSelectionEditor;
