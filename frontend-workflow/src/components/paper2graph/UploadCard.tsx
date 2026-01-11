import React, { ChangeEvent } from 'react';
import { FileText, Type, UploadCloud } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { UploadMode, GraphType, FileKind } from './types';

interface UploadCardProps {
  graphType: GraphType;
  setGraphType: (type: GraphType) => void;
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

  return (
    <div className="glass rounded-xl border border-white/10 p-6 lg:p-8 relative overflow-hidden flex flex-col">
      {/* 装饰背景光 */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-2/3 h-1 bg-gradient-to-r from-transparent via-blue-500 to-transparent opacity-50 blur-sm"></div>

      <div className="relative">
        {/* 绘图类型选择 */}
        <div className="mb-6">
          <label className="block text-xs font-medium text-gray-400 mb-2">{t('graphType.label')}</label>
          <select
            value={graphType}
            onChange={e => {
              const newType = e.target.value as GraphType;
              setGraphType(newType);
              // 实验数据图强制使用 file 模式
              if (newType === 'exp_data') {
                setUploadMode('file');
              }
            }}
            className="w-full rounded-xl border border-white/10 bg-black/40 px-4 py-3 text-sm text-gray-200 outline-none focus:ring-2 focus:ring-blue-500 transition-all"
          >
            <option value="model_arch">{t('graphType.model_arch')}</option>
            <option value="tech_route">{t('graphType.tech_route')}</option>
            <option value="exp_data">{t('graphType.exp_data')}</option>
          </select>
        </div>

        {/* 上传模式 Tab (炫酷卡片式 - 蓝色系) */}
        <div className="grid grid-cols-2 gap-3 mb-6 p-1.5 bg-black/40 rounded-2xl border border-white/5">
          <button
            type="button"
            onClick={() => setUploadMode('file')}
            className={`relative group flex flex-col items-center justify-center py-3 rounded-xl transition-all duration-300 overflow-hidden ${
              uploadMode === 'file'
                ? 'bg-gradient-to-br from-blue-600 to-cyan-500 text-white shadow-lg shadow-blue-500/30 scale-[1.02] ring-1 ring-white/20'
                : 'bg-white/5 text-gray-400 hover:bg-white/10 hover:text-gray-200 hover:scale-[1.02]'
            }`}
          >
             {uploadMode === 'file' && (
                <div className="absolute inset-0 w-full h-full bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full animate-shimmer-fast"></div>
             )}
             <FileText size={22} className={`mb-1.5 transition-colors ${uploadMode === 'file' ? 'text-white' : 'text-gray-500 group-hover:text-blue-400'}`} />
             <span className={`text-sm font-bold tracking-wide ${uploadMode === 'file' ? 'text-white' : 'text-gray-300'}`}>{t('uploadTabs.file')}</span>
             <span className={`text-[10px] uppercase tracking-wider font-medium ${uploadMode === 'file' ? 'text-blue-100' : 'text-gray-600'}`}>{t('uploadTabs.fileSub')}</span>
          </button>

          <button
            type="button"
            onClick={() => setUploadMode('text')}
            className={`relative group flex flex-col items-center justify-center py-3 rounded-xl transition-all duration-300 overflow-hidden ${
              uploadMode === 'text'
                ? 'bg-gradient-to-br from-blue-600 to-cyan-500 text-white shadow-lg shadow-blue-500/30 scale-[1.02] ring-1 ring-white/20'
                : 'bg-white/5 text-gray-400 hover:bg-white/10 hover:text-gray-200 hover:scale-[1.02]'
            }`}
          >
             {uploadMode === 'text' && (
                <div className="absolute inset-0 w-full h-full bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full animate-shimmer-fast"></div>
             )}
             <Type size={22} className={`mb-1.5 transition-colors ${uploadMode === 'text' ? 'text-white' : 'text-gray-500 group-hover:text-blue-400'}`} />
             <span className={`text-sm font-bold tracking-wide ${uploadMode === 'text' ? 'text-white' : 'text-gray-300'}`}>{t('uploadTabs.text')}</span>
             <span className={`text-[10px] uppercase tracking-wider font-medium ${uploadMode === 'text' ? 'text-blue-100' : 'text-gray-600'}`}>{t('uploadTabs.textSub')}</span>
          </button>
        </div>

        {/* 不同模式内容区域 */}
        {uploadMode === 'file' && (
          <div
            className={`border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center text-center gap-4 transition-all h-[300px] ${
              isDragOver ? 'border-blue-500 bg-blue-500/10' : 'border-white/20 hover:border-blue-400 bg-black/20'
            }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <div className="w-16 h-16 rounded-full bg-gradient-to-br from-blue-500/20 to-cyan-500/20 flex items-center justify-center">
              <UploadCloud size={32} className="text-blue-400" />
            </div>
            <div>
              <p className="text-white font-medium mb-1">
                {uploadMode === 'file' ? t('upload.fileDragTitleFile') : t('upload.fileDragTitleImage')}
              </p>
              <p className="text-sm text-gray-400">
                {showFileHint()}
              </p>
            </div>
            <label className="px-6 py-2.5 rounded-full bg-gradient-to-r from-blue-600 to-cyan-600 text-white text-sm font-medium cursor-pointer hover:from-blue-700 hover:to-cyan-700 transition-all shadow-lg shadow-blue-500/20">
              {t('upload.selectFile')}
              <input
                type="file"
                accept={
                  uploadMode === 'file'
                    ? '.pdf'
                    : undefined
                }
                className="hidden"
                onChange={handleFileChange}
              />
            </label>
            {selectedFile && (
                <div className="px-4 py-2 bg-blue-500/20 border border-blue-500/40 rounded-lg animate-fade-in">
                  <p className="text-sm text-blue-300 font-medium">✓ {selectedFile.name}</p>
                </div>
            )}
          </div>
        )}

        {uploadMode === 'text' && (
          <div className="space-y-3 h-[300px] flex flex-col">
            <label className="block text-xs font-medium text-gray-400">
              {t('upload.textLabel')}
            </label>
            <textarea
              value={textContent}
              onChange={e => setTextContent(e.target.value)}
              placeholder={t('upload.textPlaceholder')}
              className="flex-1 w-full rounded-xl border border-white/20 bg-black/40 px-4 py-3 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-blue-500 resize-none placeholder:text-gray-600"
            />
            <p className="text-[11px] text-gray-500 text-right">
              {t('upload.textTip')}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default UploadCard;
