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
    <div className="glass rounded-xl border border-white/10 p-4 mb-4">
      <div className="flex items-center gap-2 mb-3">
        <History size={16} className="text-purple-400" />
        <h4 className="text-sm text-gray-300 font-medium">版本历史</h4>
        <span className="text-xs text-gray-500">
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
                  ? 'border-purple-500 bg-purple-500/10'
                  : 'border-white/10 bg-white/5 hover:border-purple-400/50'
              }`}
            >
              <div className="relative aspect-video rounded-t-lg overflow-hidden bg-white/5">
                {imageErrors[version.imageUrl] ? (
                  <div className="w-full h-full flex flex-col items-center justify-center text-gray-500">
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
                  <div className="absolute top-1 right-1 bg-purple-500 text-white text-xs px-1.5 py-0.5 rounded">
                    当前
                  </div>
                )}
              </div>

              <div className="p-2">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs text-gray-400">
                    v{version.versionNumber}
                  </span>
                  <Clock size={10} className="text-gray-500" />
                </div>

                <p className="text-xs text-gray-500 mb-2 line-clamp-2">
                  {version.prompt || '初始生成'}
                </p>

                <p className="text-xs text-gray-600 mb-2">
                  {formatTimestamp(version.timestamp)}
                </p>

                {!isCurrent && (
                  <button
                    onClick={() => onRevert(version.versionNumber)}
                    disabled={isGenerating}
                    className="w-full px-2 py-1 text-xs rounded bg-white/10 hover:bg-white/20 text-gray-300 flex items-center justify-center gap-1 disabled:opacity-50 disabled:cursor-not-allowed"
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
