import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import mermaid from 'mermaid';
import { Download, Eye, Code, Maximize2, X } from 'lucide-react';

interface MermaidPreviewProps {
  mermaidCode: string;
  title?: string;
}

export const MermaidPreview = ({ mermaidCode, title = "思维导图预览" }: MermaidPreviewProps) => {
  const mermaidRef = useRef<HTMLDivElement>(null);
  const [showCode, setShowCode] = useState(false);
  const [renderError, setRenderError] = useState<string | null>(null);
  const [renderedSvg, setRenderedSvg] = useState('');
  const [showModal, setShowModal] = useState(false);
  const [modalSvg, setModalSvg] = useState('');
  const [zoom, setZoom] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [dragOrigin, setDragOrigin] = useState({ x: 0, y: 0 });

  useEffect(() => {
    // Initialize mermaid with dark theme
    mermaid.initialize({
      startOnLoad: false,
      theme: 'dark',
      themeVariables: {
        primaryColor: '#0ea5e9',
        primaryTextColor: '#fff',
        primaryBorderColor: '#0284c7',
        lineColor: '#06b6d4',
        secondaryColor: '#0891b2',
        tertiaryColor: '#164e63',
      },
      fontFamily: 'ui-sans-serif, system-ui, sans-serif',
    });
  }, []);

  useEffect(() => {
    const renderMermaid = async () => {
      if (!mermaidCode || !mermaidRef.current) return;

      try {
        setRenderError(null);
        setRenderedSvg('');

        // Clear previous content
        mermaidRef.current.innerHTML = '';

        // Generate unique ID for this diagram
        const id = `mermaid-${Date.now()}`;

        // Render mermaid diagram
        const { svg } = await mermaid.render(id, mermaidCode);
        setRenderedSvg(svg);

        // Insert rendered SVG
        if (mermaidRef.current) {
          mermaidRef.current.innerHTML = svg;
        }
      } catch (error: any) {
        console.error('Mermaid render error:', error);
        setRenderError(error.message || 'Failed to render diagram');
        setRenderedSvg('');
      }
    };

    renderMermaid();
  }, [mermaidCode]);

  const renderSvgForExport = async () => {
    if (renderedSvg) return renderedSvg;
    const id = `mermaid-export-${Date.now()}`;
    const { svg } = await mermaid.render(id, mermaidCode);
    return svg;
  };

  const normalizeSvg = (svg: string) => {
    return svg.replace(/<svg([^>]*?)>/i, (match, attrs) => {
      let next = attrs
        .replace(/\swidth="[^"]*"/i, '')
        .replace(/\sheight="[^"]*"/i, '');

      if (!/preserveAspectRatio=/i.test(next)) {
        next += ' preserveAspectRatio="xMidYMid meet"';
      }

      if (/style="/i.test(next)) {
        next = next.replace(/style="([^"]*)"/i, (_: string, style: string) => {
          const merged = `${style}; width:100%; height:100%;`;
          return `style="${merged}"`;
        });
      } else {
        next += ' style="width:100%; height:100%;"';
      }

      return `<svg${next}>`;
    });
  };

  const handleDownloadSVG = async () => {
    if (!mermaidCode) return;
    try {
      const svgData = await renderSvgForExport();
      const blob = new Blob([svgData], { type: 'image/svg+xml' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `mindmap_${Date.now()}.svg`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Download SVG failed:', error);
    }
  };

  const handleDownloadCode = () => {
    const blob = new Blob([mermaidCode], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.href = url;
    link.download = `mindmap_${Date.now()}.mmd`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleExpand = async () => {
    if (!mermaidCode) return;
    try {
      const svgData = await renderSvgForExport();
      setModalSvg(normalizeSvg(svgData));
      setZoom(1);
      setOffset({ x: 0, y: 0 });
      setShowModal(true);
    } catch (error) {
      console.error('Expand preview failed:', error);
    }
  };

  const clampZoom = (value: number) => Math.min(5, Math.max(0.2, value));

  const handleWheel = (e: React.WheelEvent<HTMLDivElement>) => {
    const direction = e.deltaY > 0 ? 0.9 : 1.1;
    setZoom(prev => clampZoom(prev * direction));
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX, y: e.clientY });
    setDragOrigin(offset);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!isDragging) return;
    const dx = e.clientX - dragStart.x;
    const dy = e.clientY - dragStart.y;
    setOffset({ x: dragOrigin.x + dx, y: dragOrigin.y + dy });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  return (
    <div className="border-t border-white/10 pt-6">
      <div className="flex items-center justify-between mb-4">
        <h4 className="text-sm font-medium text-gray-300">{title}</h4>
        <div className="flex items-center gap-2">
          <button
            onClick={handleExpand}
            className="px-3 py-1.5 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-xs text-gray-300 flex items-center gap-1.5 transition-colors"
          >
            <Maximize2 size={14} />
            放大
          </button>
          <button
            onClick={() => setShowCode(!showCode)}
            className="px-3 py-1.5 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-xs text-gray-300 flex items-center gap-1.5 transition-colors"
          >
            {showCode ? <Eye size={14} /> : <Code size={14} />}
            {showCode ? '查看图形' : '查看代码'}
          </button>
          <button
            onClick={handleDownloadSVG}
            className="px-3 py-1.5 bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-500/30 rounded-lg text-xs text-cyan-300 flex items-center gap-1.5 transition-colors"
          >
            <Download size={14} />
            下载 SVG
          </button>
          <button
            onClick={handleDownloadCode}
            className="px-3 py-1.5 bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-500/30 rounded-lg text-xs text-cyan-300 flex items-center gap-1.5 transition-colors"
          >
            <Download size={14} />
            下载代码
          </button>
        </div>
      </div>

      {showCode ? (
        <div className="bg-white/5 border border-white/10 rounded-lg p-4">
          <div className="text-xs text-gray-400 mb-2">Mermaid 代码:</div>
          <pre className="text-xs text-gray-300 bg-black/40 p-3 rounded overflow-x-auto max-h-96">
            {mermaidCode}
          </pre>
        </div>
      ) : (
        <div className="bg-white/5 border border-white/10 rounded-lg p-6">
          {renderError ? (
            <div className="text-center py-8">
              <div className="text-red-400 text-sm mb-2">渲染失败</div>
              <div className="text-xs text-gray-500">{renderError}</div>
              <button
                onClick={() => setShowCode(true)}
                className="mt-4 text-xs text-cyan-400 hover:text-cyan-300"
              >
                查看原始代码
              </button>
            </div>
          ) : (
            <div
              ref={mermaidRef}
              className="flex items-center justify-center overflow-x-auto"
              style={{ minHeight: '200px' }}
            />
          )}
        </div>
      )}

      <div className="mt-3 text-xs text-gray-500">
        提示: 您可以切换查看图形或代码，也可以下载 SVG 文件或 Mermaid 代码文件
      </div>

      {showModal && createPortal(
        <div
          className="fixed inset-0 z-[200] bg-black/60 backdrop-blur-md flex items-center justify-center p-6"
          onClick={() => setShowModal(false)}
        >
          <div
            className="w-[92vw] h-[90vh] max-w-none bg-[#0b0b1a] border border-white/10 rounded-2xl shadow-2xl overflow-hidden flex flex-col"
            onClick={e => e.stopPropagation()}
          >
            <div className="flex items-center justify-between px-4 py-3 border-b border-white/10">
              <div className="text-sm text-gray-300">思维导图放大预览</div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setZoom(prev => clampZoom(prev * 0.9))}
                  className="px-2 py-1 text-xs rounded-lg bg-white/5 hover:bg-white/10 text-gray-300"
                >
                  缩小
                </button>
                <button
                  onClick={() => setZoom(prev => clampZoom(prev * 1.1))}
                  className="px-2 py-1 text-xs rounded-lg bg-white/5 hover:bg-white/10 text-gray-300"
                >
                  放大
                </button>
                <button
                  onClick={() => {
                    setZoom(1);
                    setOffset({ x: 0, y: 0 });
                  }}
                  className="px-2 py-1 text-xs rounded-lg bg-white/5 hover:bg-white/10 text-gray-300"
                >
                  复位
                </button>
                <button
                  onClick={() => setShowModal(false)}
                  className="p-2 rounded-lg text-gray-400 hover:text-white hover:bg-white/10 transition-colors"
                >
                  <X size={18} />
                </button>
              </div>
            </div>
            <div
              className="flex-1 overflow-hidden p-6"
              onWheel={handleWheel}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
              style={{ cursor: isDragging ? 'grabbing' : 'grab' }}
            >
              {modalSvg ? (
                <div className="w-full h-full flex items-center justify-center">
                  <div
                    style={{
                      transform: `translate(${offset.x}px, ${offset.y}px) scale(${zoom})`,
                      transformOrigin: 'center center'
                    }}
                    dangerouslySetInnerHTML={{ __html: modalSvg }}
                  />
                </div>
              ) : (
                <div className="text-sm text-gray-500">暂无可预览内容</div>
              )}
            </div>
          </div>
        </div>,
        document.body
      )}
    </div>
  );
};
