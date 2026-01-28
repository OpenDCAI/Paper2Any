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
  return (
    <div className="w-[400px] flex-shrink-0 flex flex-col border-l border-white/5 bg-[#0a0a1a] relative z-20 shadow-2xl">
      {/* Top Grid Selector */}
      <ToolSelector activeTool={activeTool} onToolChange={onToolChange} />

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
