import React, { ChangeEvent } from 'react';
import { FileText, Type, UploadCloud, Network, GitBranch, BarChart3 } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { UploadMode, GraphType, FileKind } from './types';
import { IMAGE_EXTENSIONS } from './constants';

interface UploadCardProps {
  graphType: GraphType;
  setGraphType: (type: GraphType) => void;
  allowedGraphTypes?: GraphType[];
  uploadMode: UploadMode;
  setUploadMode: (mode: UploadMode) => void;
  selectedFile: File | null;
  fileKind: FileKind;
  isDragOver: boolean;
  handleDragOver: (e: React.DragEvent<HTMLDivElement>) => void;
  handleDragLeave: (e: React.DragEvent<HTMLDivElement>) => void;
  handleDrop: (e: React.DragEvent<HTMLDivElement>) => void;
  handleFileChange: (e: ChangeEvent<HTMLInputElement>) => void;
  textContent: string;
  setTextContent: (text: string) => void;
}

const UploadCard: React.FC<UploadCardProps> = ({
  graphType,
  setGraphType,
  allowedGraphTypes,
  uploadMode,
  setUploadMode,
  selectedFile,
  fileKind,
  isDragOver,
  handleDragOver,
  handleDragLeave,
  handleDrop,
  handleFileChange,
  textContent,
  setTextContent,
}) => {
  const { t } = useTranslation('paper2graph');

  const showFileHint = () => {
    if (!selectedFile) return t('upload.fileHint');
    if (fileKind === 'pdf') return `PDF：${selectedFile.name}`;
    if (fileKind === 'image') return `Image：${selectedFile.name}`;
    return `Unknown file type: ${selectedFile.name}`;
  };

  const graphTypeOptions: { value: GraphType; label: string; icon: React.ReactNode }[] = [
    { value: 'model_arch', label: t('graphType.model_arch'), icon: <Network size={20} /> },
    { value: 'tech_route', label: t('graphType.tech_route'), icon: <GitBranch size={20} /> },
    { value: 'exp_data', label: t('graphType.exp_data'), icon: <BarChart3 size={20} /> },
  ];
  const visibleGraphTypeOptions = allowedGraphTypes?.length
    ? graphTypeOptions.filter(option => allowedGraphTypes.includes(option.value))
    : graphTypeOptions;
  const gridColsClass =
    visibleGraphTypeOptions.length === 1
      ? 'md:grid-cols-1'
      : visibleGraphTypeOptions.length === 2
        ? 'md:grid-cols-2'
        : 'md:grid-cols-3';

  const getAcceptTypes = () => {
    if (graphType === 'exp_data') {
      return '.pdf,' + IMAGE_EXTENSIONS.map(ext => '.' + ext).join(',');
    }
    return '.pdf';
  };

  return (
    <div className="glass rounded-xl border border-primary-100 p-6 lg:p-8 relative overflow-hidden flex flex-col">
      {/* 装饰背景光 */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-2/3 h-1 bg-gradient-to-r from-transparent via-primary-500 to-transparent opacity-50 blur-sm"></div>

      <div className="relative">
        {/* 绘图类型选择 (Dynamic Cards) */}
        <div className="mb-6">
          <label className="mb-2 block text-xs font-medium text-[var(--text-secondary)]">{t('graphType.label')}</label>
          <div className={`grid grid-cols-1 ${gridColsClass} gap-3`}>
            {visibleGraphTypeOptions.map((option) => (
              <button
                key={option.value}
                type="button"
                onClick={() => {
                  setGraphType(option.value);
                }}
                className={`relative group flex flex-col items-center justify-center p-3 rounded-xl transition-all duration-300 overflow-hidden border ${
                  graphType === option.value
                    ? 'bg-gradient-to-br from-primary-600 to-primary-400 text-white shadow-lg shadow-primary-500/30 border-white/20 scale-[1.02]'
                    : 'bg-white/78 text-[var(--text-secondary)] border-primary-100 hover:bg-white hover:text-primary-800 hover:border-primary-200'
                }`}
              >
                {graphType === option.value && (
                  <div className="absolute inset-0 w-full h-full bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full animate-shimmer-fast"></div>
                )}
                <div className={`mb-2 transition-colors ${graphType === option.value ? 'text-white' : 'text-[#8b726b] group-hover:text-primary-600'}`}>
                  {option.icon}
                </div>
                <span className="text-xs font-bold tracking-wide text-center leading-tight">
                  {option.label}
                </span>
              </button>
            ))}
          </div>
        </div>

        {/* 上传模式 Tab (炫酷卡片式 - 蓝色系) */}
        <div className="mb-6 grid grid-cols-2 gap-3 rounded-2xl border border-primary-100 bg-white/70 p-1.5">
          <button
            type="button"
            onClick={() => setUploadMode('file')}
            className={`relative group flex flex-col items-center justify-center py-3 rounded-xl transition-all duration-300 overflow-hidden ${
              uploadMode === 'file'
                ? 'bg-gradient-to-br from-primary-600 to-primary-500 text-white shadow-lg shadow-primary-500/30 scale-[1.02] ring-1 ring-white/20'
                : 'bg-transparent text-[var(--text-secondary)] hover:bg-white hover:text-primary-800 hover:scale-[1.02]'
            }`}
          >
             {uploadMode === 'file' && (
                <div className="absolute inset-0 w-full h-full bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full animate-shimmer-fast"></div>
             )}
             <FileText size={22} className={`mb-1.5 transition-colors ${uploadMode === 'file' ? 'text-white' : 'text-[#8b726b] group-hover:text-primary-600'}`} />
             <span className={`text-sm font-bold tracking-wide ${uploadMode === 'file' ? 'text-white' : 'text-[var(--text-primary)]'}`}>{t('uploadTabs.file')}</span>
             <span className={`text-[10px] uppercase tracking-wider font-medium ${uploadMode === 'file' ? 'text-primary-100' : 'text-[#8b726b]'}`}>{t('uploadTabs.fileSub')}</span>
          </button>

          <button
            type="button"
            onClick={() => setUploadMode('text')}
            className={`relative group flex flex-col items-center justify-center py-3 rounded-xl transition-all duration-300 overflow-hidden ${
              uploadMode === 'text'
                ? 'bg-gradient-to-br from-primary-600 to-primary-500 text-white shadow-lg shadow-primary-500/30 scale-[1.02] ring-1 ring-white/20'
                : 'bg-transparent text-[var(--text-secondary)] hover:bg-white hover:text-primary-800 hover:scale-[1.02]'
            }`}
          >
             {uploadMode === 'text' && (
                <div className="absolute inset-0 w-full h-full bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full animate-shimmer-fast"></div>
             )}
             <Type size={22} className={`mb-1.5 transition-colors ${uploadMode === 'text' ? 'text-white' : 'text-[#8b726b] group-hover:text-primary-600'}`} />
             <span className={`text-sm font-bold tracking-wide ${uploadMode === 'text' ? 'text-white' : 'text-[var(--text-primary)]'}`}>{t('uploadTabs.text')}</span>
             <span className={`text-[10px] uppercase tracking-wider font-medium ${uploadMode === 'text' ? 'text-primary-100' : 'text-[#8b726b]'}`}>{t('uploadTabs.textSub')}</span>
          </button>
        </div>

        {/* 不同模式内容区域 */}
        {uploadMode === 'file' && (
          <div
            className={`border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center text-center gap-4 transition-all h-[300px] ${
              isDragOver ? 'border-primary-500 bg-primary-500/10' : 'border-primary-100 hover:border-primary-300 bg-white/72'
            }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <div className="w-16 h-16 rounded-full bg-gradient-to-br from-primary-500/20 to-primary-400/20 flex items-center justify-center">
              <UploadCloud size={32} className="text-primary-400" />
            </div>
            <div>
              <p className="mb-1 font-medium text-[var(--text-primary)]">
                {t('upload.fileDragTitleFile')}
              </p>
              <p className="mb-1 text-xs text-primary-700/80">
                 {graphType === 'exp_data' ? '(支持 PDF 或 图片)' : '(仅支持 PDF)'}
              </p>
              <p className="text-sm text-[var(--text-secondary)]">
                {showFileHint()}
              </p>
            </div>
            <label className="px-6 py-2.5 rounded-full bg-gradient-to-r from-primary-600 to-primary-500 text-white text-sm font-medium cursor-pointer hover:from-primary-700 hover:to-primary-600 transition-all shadow-lg shadow-primary-500/20">
              {t('upload.selectFile')}
              <input
                type="file"
                accept={
                  uploadMode === 'file'
                    ? getAcceptTypes()
                    : undefined
                }
                className="hidden"
                onChange={handleFileChange}
              />
            </label>
            {selectedFile && (
                <div className="px-4 py-2 bg-primary-500/20 border border-primary-500/40 rounded-lg animate-fade-in">
                  <p className="text-sm font-medium text-primary-800">✓ {selectedFile.name}</p>
                </div>
            )}
          </div>
        )}

        {uploadMode === 'text' && (
          <div className="space-y-3 h-[300px] flex flex-col">
            <label className="block text-xs font-medium text-[var(--text-secondary)]">
              {t('upload.textLabel')}
            </label>
            <textarea
              value={textContent}
              onChange={e => setTextContent(e.target.value)}
              placeholder={t('upload.textPlaceholder')}
              className="portal-input-soft flex-1 w-full rounded-xl px-4 py-3 text-sm outline-none focus:ring-2 focus:ring-primary-500 resize-none"
            />
            <p className="text-right text-[11px] text-[#8b726b]">
              {t('upload.textTip')}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default UploadCard;
