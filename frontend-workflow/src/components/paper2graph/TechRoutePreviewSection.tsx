import React from 'react';
import { ImageIcon, Download } from 'lucide-react';
import { GraphType } from './types';

interface TechRoutePreviewSectionProps {
  graphType: GraphType;
  techRouteStep: 'input' | 'preview' | 'done';
  svgPreviewUrl: string | null;
  svgBwPath: string | null;
  svgColorPath: string | null;
}

const TechRoutePreviewSection: React.FC<TechRoutePreviewSectionProps> = ({
  graphType,
  techRouteStep,
  svgPreviewUrl,
  svgBwPath,
  svgColorPath,
}) => {
  const [imgError, setImgError] = React.useState(false);

  React.useEffect(() => {
    setImgError(false);
  }, [svgPreviewUrl]);

  // Only show for tech_route when in preview or done step
  if (graphType !== 'tech_route' || techRouteStep === 'input' || !svgPreviewUrl) {
    return null;
  }

  const handleDownloadSvg = async (path: string, filename: string) => {
    try {
      const response = await fetch(path);
      const svgText = await response.text();
      // 创建带有正确 MIME 类型的 blob
      const blob = new Blob([svgText], { type: 'image/svg+xml' });
      const blobUrl = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = blobUrl;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(blobUrl);
    } catch (error) {
      console.error('SVG download failed:', error);
      // 如果下载失败，尝试直接下载而不是打开新标签页
      const a = document.createElement('a');
      a.href = path;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
    }
  };

  return (
    <div className="mb-8 portal-panel-dark rounded-xl border border-white/10 p-6 animate-fade-in relative overflow-hidden">
      <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-primary-500/70 via-primary-400/45 to-amber-500/55" />

      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold text-[#fff4ee] flex items-center gap-2">
          <ImageIcon size={20} className="text-primary-400" />
          Technical Route Preview
        </h3>

        <div className="flex gap-2">
          {svgBwPath && (
            <button
              type="button"
              onClick={() => handleDownloadSvg(svgBwPath, 'tech_route_bw.svg')}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-primary-500/14 hover:bg-primary-500/24 text-xs text-[#f4ddd6] border border-primary-400/35 transition-colors"
            >
              <Download size={14} />
              BW SVG
            </button>
          )}
          {svgColorPath && (
            <button
              type="button"
              onClick={() => handleDownloadSvg(svgColorPath, 'tech_route_color.svg')}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-amber-500/12 hover:bg-amber-500/22 text-xs text-[#fde2b5] border border-amber-400/35 transition-colors"
            >
              <Download size={14} />
              Color SVG
            </button>
          )}
        </div>
      </div>

      {/* SVG Preview */}
      <div className="w-full bg-black/30 rounded-xl border border-white/10 flex items-center justify-center overflow-hidden p-4 min-h-[300px]">
        {imgError ? (
          <div className="flex flex-col items-center justify-center text-[#ccb5af] p-4">
            <ImageIcon size={48} className="mb-4 opacity-50" />
            <p className="mb-2 font-medium">Image load failed</p>
            <a
              href={svgPreviewUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="mt-4 px-4 py-2 bg-white/10 hover:bg-white/16 rounded-lg text-sm transition-colors border border-white/10"
            >
              Open in new tab
            </a>
          </div>
        ) : (
          <img
            src={svgPreviewUrl}
            alt="Technical Route Preview"
            className="max-w-full h-auto object-contain max-h-[600px] rounded-lg shadow-2xl"
            onError={() => setImgError(true)}
          />
        )}
      </div>
    </div>
  );
};

export default TechRoutePreviewSection;
