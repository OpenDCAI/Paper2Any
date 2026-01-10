import { Presentation, Download, Trash2 } from 'lucide-react';
import { KnowledgeFile, ToolType } from './types';

interface OutputViewProps {
  files: KnowledgeFile[];
  onGoToTool: (tool: ToolType) => void;
}

export const OutputView = ({ files, onGoToTool }: OutputViewProps) => {
  if (files.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-[50vh] text-gray-500">
        <Presentation size={48} className="mb-4 opacity-20" />
        <p className="text-sm">暂无生成内容</p>
        <button 
          onClick={() => onGoToTool('ppt')}
          className="mt-4 text-primary-400 hover:text-primary-300 text-sm"
        >
          在右侧面板尝试生成 PPT
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {files.map(file => (
        <div key={file.id} className="flex items-center gap-4 p-4 rounded-xl bg-white/[0.03] border border-white/5 hover:bg-white/[0.06] transition-colors group">
          <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-orange-500/20 to-red-500/20 flex items-center justify-center text-orange-400">
            <Presentation size={24} />
          </div>
          <div className="flex-1">
            <h4 className="text-sm font-medium text-gray-200">{file.name}</h4>
            <p className="text-xs text-gray-500 mt-0.5">{file.desc} • {file.uploadTime}</p>
          </div>
          <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
            <button className="p-2 text-gray-400 hover:text-white hover:bg-white/10 rounded-lg">
              <Download size={18} />
            </button>
            <button className="p-2 text-gray-400 hover:text-white hover:bg-white/10 rounded-lg">
              <Trash2 size={18} />
            </button>
          </div>
        </div>
      ))}
    </div>
  );
};
