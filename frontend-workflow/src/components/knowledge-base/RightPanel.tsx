import { useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { ToolType, KnowledgeFile } from './types';
import { ToolSelector } from './ToolSelector';
import { ChatTool } from './tools/ChatTool';
import { SearchTool } from './tools/SearchTool';
import { PptTool } from './tools/PptTool';
import { MindMapTool } from './tools/MindMapTool';
import { PodcastTool } from './tools/PodcastTool';
import { VideoTool } from './tools/VideoTool';

interface RightPanelProps {
  activeTool: ToolType;
  onToolChange: (tool: ToolType) => void;
  files: KnowledgeFile[];
  selectedIds: Set<string>;
  onGenerateSuccess: (file: KnowledgeFile) => void;
}

export const RightPanel = ({ activeTool, onToolChange, files, selectedIds, onGenerateSuccess }: RightPanelProps) => {
  const [isToolsCollapsed, setIsToolsCollapsed] = useState(false);
  const toolLabels: Record<ToolType, string> = {
    chat: '智能问答',
    search: '语义检索',
    ppt: 'PPT 生成',
    mindmap: '思维导图',
    podcast: '知识播客',
    video: '视频讲解'
  };

  return (
    <div className="w-[400px] flex-shrink-0 flex flex-col border-l border-white/5 bg-[#0a0a1a] relative z-20 shadow-2xl">
      {/* Tools Header */}
      <div className="flex items-center justify-between px-4 py-2 bg-[#050512] border-b border-white/5">
        <div className="text-xs text-gray-400">
          当前工具：<span className="text-gray-200">{toolLabels[activeTool]}</span>
        </div>
        <button
          onClick={() => setIsToolsCollapsed(prev => !prev)}
          className="flex items-center gap-1 text-xs text-purple-300 hover:text-purple-200 transition-colors"
        >
          {isToolsCollapsed ? '展开功能卡片' : '收起功能卡片'}
          {isToolsCollapsed ? <ChevronDown size={14} /> : <ChevronUp size={14} />}
        </button>
      </div>

      {/* Top Grid Selector */}
      <div
        className={`transition-all duration-300 ease-in-out overflow-hidden ${
          isToolsCollapsed ? 'max-h-0 opacity-0 pointer-events-none' : 'max-h-[240px] opacity-100'
        }`}
      >
        <ToolSelector activeTool={activeTool} onToolChange={onToolChange} />
      </div>

      {/* Content Area with Slide Animation */}
      <div className="flex-1 overflow-hidden relative">
        <div className={`absolute inset-0 transition-transform duration-300 ease-in-out ${activeTool === 'chat' ? 'translate-x-0' : 'translate-x-full opacity-0 pointer-events-none'}`}>
          <ChatTool files={files} selectedIds={selectedIds} />
        </div>
        <div className={`absolute inset-0 transition-transform duration-300 ease-in-out ${activeTool === 'search' ? 'translate-x-0' : 'translate-x-full opacity-0 pointer-events-none'}`}>
          <SearchTool files={files} selectedIds={selectedIds} />
        </div>
        <div className={`absolute inset-0 transition-transform duration-300 ease-in-out ${activeTool === 'ppt' ? 'translate-x-0' : 'translate-x-full opacity-0 pointer-events-none'}`}>
          <PptTool files={files} selectedIds={selectedIds} onGenerateSuccess={onGenerateSuccess} />
        </div>
        <div className={`absolute inset-0 transition-transform duration-300 ease-in-out ${activeTool === 'mindmap' ? 'translate-x-0' : 'translate-x-full opacity-0 pointer-events-none'}`}>
          <MindMapTool files={files} selectedIds={selectedIds} onGenerateSuccess={onGenerateSuccess} />
        </div>
        <div className={`absolute inset-0 transition-transform duration-300 ease-in-out ${activeTool === 'podcast' ? 'translate-x-0' : 'translate-x-full opacity-0 pointer-events-none'}`}>
          <PodcastTool files={files} selectedIds={selectedIds} onGenerateSuccess={onGenerateSuccess} />
        </div>
        <div className={`absolute inset-0 transition-transform duration-300 ease-in-out ${activeTool === 'video' ? 'translate-x-0' : 'translate-x-full opacity-0 pointer-events-none'}`}>
          <VideoTool />
        </div>
      </div>
    </div>
  );
};
