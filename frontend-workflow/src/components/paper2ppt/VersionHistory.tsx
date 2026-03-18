import React, { useState } from 'react';
import { History, RotateCcw, Clock, ImageOff } from 'lucide-react';
import { ImageVersion } from './types';

interface VersionHistoryProps {
  versions: ImageVersion[];
  currentVersionIndex: number;
  onRevert: (versionNumber: number) => void;
  isGenerating: boolean;
}

const VersionHistory: React.FC<VersionHistoryProps> = ({
  versions,
  currentVersionIndex,
  onRevert,
  isGenerating
}) => {
  // 使用 URL 作为键，这样 URL 变化时会自动重试
  const [imageErrors, setImageErrors] = useState<Record<string, boolean>>({});

  if (versions.length === 0) {
    return null;
  }

  const formatTimestamp = (timestamp: number) => {
    const date = new Date(timestamp * 1000);
    return date.toLocaleString('zh-CN', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const handleImageError = (versionNumber: number, imageUrl: string) => {
    console.error(`[VersionHistory] 图片加载失败 - 版本${versionNumber}:`, imageUrl);
    setImageErrors(prev => ({ ...prev, [imageUrl]: true }));
  };

  return (
    <div className="paper2ppt-panel rounded-3xl p-4 mb-4">
      <div className="flex items-center gap-2 mb-3">
        <History size={16} className="text-[#8c1d40]" />
        <h4 className="text-sm font-semibold text-[#1d1c1a]">版本历史</h4>
        <span className="text-xs text-[#675f58]">
          ({versions.length} 个版本)
        </span>
      </div>

      <div className="flex gap-2 overflow-x-auto pb-2">
        {versions.map((version, index) => {
          const isCurrent = index === currentVersionIndex;

          return (
            <div
              key={version.versionNumber}
              className={`flex-shrink-0 w-32 rounded-lg border transition-all ${
                isCurrent
                  ? 'border-[rgba(140,29,64,0.28)] bg-[rgba(140,29,64,0.08)]'
                  : 'border-[rgba(110,76,55,0.14)] bg-white/70 hover:border-[rgba(140,29,64,0.28)]'
              }`}
            >
              <div className="relative aspect-video overflow-hidden rounded-t-lg bg-[rgba(255,255,255,0.82)]">
                {imageErrors[version.imageUrl] ? (
                  <div className="flex h-full w-full flex-col items-center justify-center text-[#675f58]">
                    <ImageOff size={20} className="mb-1" />
                    <span className="text-xs">加载失败</span>
                  </div>
                ) : (
                  <img
                    src={version.imageUrl}
                    alt={`版本 ${version.versionNumber}`}
                    className="w-full h-full object-cover"
                    onError={() => handleImageError(version.versionNumber, version.imageUrl)}
                    onLoad={() => console.log(`[VersionHistory] 图片加载成功 - 版本${version.versionNumber}:`, version.imageUrl)}
                    loading="lazy"
                    title={version.imageUrl}
                  />
                )}
                {isCurrent && (
                  <div className="absolute top-1 right-1 rounded bg-[#8c1d40] px-1.5 py-0.5 text-xs text-white">
                    当前
                  </div>
                )}
              </div>

              <div className="p-2">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs text-[#675f58]">
                    v{version.versionNumber}
                  </span>
                  <Clock size={10} className="text-[#8c1d40]" />
                </div>

                <p className="mb-2 line-clamp-2 text-xs text-[#675f58]">
                  {version.prompt || '初始生成'}
                </p>

                <p className="mb-2 text-xs text-[#4d4742]">
                  {formatTimestamp(version.timestamp)}
                </p>

                {!isCurrent && (
                  <button
                    onClick={() => onRevert(version.versionNumber)}
                    disabled={isGenerating}
                    className="paper2ppt-button-secondary flex w-full items-center justify-center gap-1 rounded-lg px-2 py-1 text-xs disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    <RotateCcw size={10} />
                    恢复
                  </button>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default VersionHistory;
