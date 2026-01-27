import { useEffect, useRef, useState } from 'react';
import mermaid from 'mermaid';
import { Download, Eye, Code } from 'lucide-react';

interface MermaidPreviewProps {
  mermaidCode: string;
  title?: string;
}

export const MermaidPreview = ({ mermaidCode, title = "思维导图预览" }: MermaidPreviewProps) => {
  const mermaidRef = useRef<HTMLDivElement>(null);
  const [showCode, setShowCode] = useState(false);
  const [renderError, setRenderError] = useState<string | null>(null);

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

        // Clear previous content
        mermaidRef.current.innerHTML = '';

        // Generate unique ID for this diagram
        const id = `mermaid-${Date.now()}`;

        // Render mermaid diagram
        const { svg } = await mermaid.render(id, mermaidCode);

        // Insert rendered SVG
        if (mermaidRef.current) {
          mermaidRef.current.innerHTML = svg;
        }
      } catch (error: any) {
        console.error('Mermaid render error:', error);
        setRenderError(error.message || 'Failed to render diagram');
      }
    };

    renderMermaid();
  }, [mermaidCode]);

  const handleDownloadSVG = () => {
    if (!mermaidRef.current) return;

    const svgElement = mermaidRef.current.querySelector('svg');
    if (!svgElement) return;

    // Get SVG content
    const svgData = new XMLSerializer().serializeToString(svgElement);
    const blob = new Blob([svgData], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);

    // Create download link
    const link = document.createElement('a');
    link.href = url;
    link.download = `mindmap_${Date.now()}.svg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
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

  return (
    <div className="border-t border-white/10 pt-6">
      <div className="flex items-center justify-between mb-4">
        <h4 className="text-sm font-medium text-gray-300">{title}</h4>
        <div className="flex items-center gap-2">
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
    </div>
  );
};
