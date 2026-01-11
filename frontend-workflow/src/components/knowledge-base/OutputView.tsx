import { KnowledgeFile, ToolType } from './types';
import { FileText, Download, ExternalLink, Clock } from 'lucide-react';

interface OutputViewProps {
  files: KnowledgeFile[];
  onGoToTool: (tool: ToolType) => void;
}

export const OutputView = ({ files, onGoToTool }: OutputViewProps) => {
  if (files.length === 0) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-center">
        <div className="w-16 h-16 bg-white/5 rounded-full flex items-center justify-center mb-4">
          <FileText className="text-gray-600" size={32} />
        </div>
        <h3 className="text-lg font-medium text-white mb-2">No Outputs Yet</h3>
        <p className="text-gray-500 text-sm max-w-xs mb-6">
          Generate content from your knowledge base using the tools in the right panel.
        </p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {files.map(file => (
        <div key={file.id} className="bg-white/5 border border-white/10 rounded-xl p-4 hover:border-purple-500/30 transition-all">
          <div className="flex items-start justify-between mb-4">
            <div className="p-2 bg-purple-500/10 rounded-lg">
              <FileText className="text-purple-400" size={24} />
            </div>
            <span className="text-xs text-purple-300 bg-purple-500/10 px-2 py-1 rounded-full">
              Generated
            </span>
          </div>
          
          <h3 className="text-white font-medium mb-1">{file.name}</h3>
          <p className="text-gray-500 text-xs mb-4 line-clamp-2">{file.desc}</p>
          
          <div className="flex items-center justify-between pt-4 border-t border-white/5">
            <div className="flex items-center gap-2 text-xs text-gray-500">
              <Clock size={12} />
              <span>{file.uploadTime}</span>
            </div>
            <div className="flex gap-2">
              <a 
                href={file.url} 
                className="p-2 text-gray-400 hover:text-white hover:bg-white/10 rounded-lg transition-colors"
                title="Download"
              >
                <Download size={16} />
              </a>
              <button 
                className="p-2 text-gray-400 hover:text-white hover:bg-white/10 rounded-lg transition-colors"
                title="View"
              >
                <ExternalLink size={16} />
              </button>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};
