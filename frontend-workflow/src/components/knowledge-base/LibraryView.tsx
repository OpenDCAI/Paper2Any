import { Grid, CheckCircle2, ImageIcon, FileText, Video, Link as LinkIcon } from 'lucide-react';
import { KnowledgeFile } from './types';

interface LibraryViewProps {
  files: KnowledgeFile[];
  selectedIds: Set<string>;
  onToggleSelect: (id: string) => void;
  onGoToUpload: () => void;
}

export const LibraryView = ({ files, selectedIds, onToggleSelect, onGoToUpload }: LibraryViewProps) => {
  if (files.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-[60vh] text-gray-500">
        <div className="w-20 h-20 rounded-full bg-white/5 flex items-center justify-center mb-6">
          <Grid size={32} className="opacity-40" />
        </div>
        <h3 className="text-lg font-medium text-gray-300 mb-2">暂无文件</h3>
        <p className="text-sm text-gray-500 max-w-xs text-center mb-6">
          您的知识库目前是空的。开始上传文档、图片或链接来构建您的个人知识中心。
        </p>
        <button 
          onClick={onGoToUpload}
          className="px-6 py-2.5 rounded-xl bg-primary-600 hover:bg-primary-500 text-white text-sm font-medium transition-colors"
        >
          去上传
        </button>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-6">
      {files.map(file => (
        <div 
          key={file.id} 
          onClick={() => onToggleSelect(file.id)}
          className={`group relative aspect-[4/5] rounded-2xl border transition-all duration-300 cursor-pointer flex flex-col p-4 overflow-hidden ${
            selectedIds.has(file.id) 
              ? 'bg-primary-500/10 border-primary-500 shadow-[0_0_20px_rgba(59,130,246,0.15)] scale-[1.02]' 
              : 'bg-white/[0.03] border-white/5 hover:bg-white/[0.06] hover:border-white/20 hover:-translate-y-1 hover:shadow-xl'
          }`}
        >
          {/* Checkbox */}
          <div className={`absolute top-3 right-3 w-6 h-6 rounded-lg border flex items-center justify-center transition-all z-10 ${
            selectedIds.has(file.id)
              ? 'bg-primary-500 border-primary-500'
              : 'border-white/20 bg-black/40 opacity-0 group-hover:opacity-100'
          }`}>
            {selectedIds.has(file.id) && <CheckCircle2 size={14} className="text-white" />}
          </div>

          {/* Preview / Icon */}
          <div className="flex-1 flex items-center justify-center mb-4 relative">
            <div className={`w-16 h-16 rounded-2xl flex items-center justify-center ${
              file.type === 'image' ? 'bg-purple-500/20 text-purple-400' :
              file.type === 'doc' ? 'bg-blue-500/20 text-blue-400' :
              file.type === 'video' ? 'bg-pink-500/20 text-pink-400' :
              'bg-green-500/20 text-green-400'
            }`}>
              {file.type === 'image' && <ImageIcon size={32} />}
              {file.type === 'doc' && <FileText size={32} />}
              {file.type === 'video' && <Video size={32} />}
              {file.type === 'link' && <LinkIcon size={32} />}
            </div>
          </div>

          {/* Metadata */}
          <div className="space-y-1 relative z-10">
            <h4 className="text-sm font-medium text-gray-200 line-clamp-2 leading-snug break-all">{file.name}</h4>
            <div className="flex items-center justify-between text-[10px] text-gray-500">
              <span>{file.size}</span>
              <span>{file.type.toUpperCase()}</span>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};
