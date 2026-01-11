import { MessageSquare, Presentation, BrainCircuit, Headphones, PlayCircle } from 'lucide-react';
import { ToolType } from './types';

interface ToolSelectorProps {
  activeTool: ToolType;
  onToolChange: (tool: ToolType) => void;
}

export const ToolSelector = ({ activeTool, onToolChange }: ToolSelectorProps) => {
  const tools = [
    { id: 'chat', icon: MessageSquare, label: '智能问答' },
    { id: 'ppt', icon: Presentation, label: 'PPT 生成' },
    { id: 'mindmap', icon: BrainCircuit, label: '思维导图' },
    { id: 'podcast', icon: Headphones, label: '知识播客' },
    { id: 'video', icon: PlayCircle, label: '视频讲解' },
  ];

  return (
    <div className="p-4 grid grid-cols-2 gap-2 bg-[#050512] border-b border-white/5">
      {tools.map((tool) => (
        <button
          key={tool.id}
          onClick={() => onToolChange(tool.id as ToolType)}
          className={`flex items-center justify-center gap-2 py-3 px-2 rounded-xl text-sm font-medium transition-all ${
            activeTool === tool.id
              ? 'bg-purple-500/20 text-purple-400 border border-purple-500/30'
              : 'bg-white/5 text-gray-400 hover:bg-white/10 hover:text-gray-200 border border-transparent'
          }`}
        >
          <tool.icon size={16} />
          <span>{tool.label}</span>
        </button>
      ))}
    </div>
  );
};
