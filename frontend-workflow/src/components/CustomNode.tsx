import { memo } from 'react';
import { Handle, Position } from 'reactflow';
import * as Icons from 'lucide-react';
import { AgentType } from '../types';

interface CustomNodeProps {
  data: {
    label: string;
    agentType: AgentType;
  };
  selected: boolean;
}

const CustomNode = memo(({ data, selected }: CustomNodeProps) => {
  const { agentType } = data;
  const IconComponent = Icons[agentType.icon as keyof typeof Icons] as any;

  const getCategoryClass = (category: string) => {
    const classes = {
      data: 'node-data',
      code: 'node-code',
      pipeline: 'node-pipeline',
      image: 'node-image',
      utility: 'node-utility',
    };
    return classes[category as keyof typeof classes] || 'node-data';
  };

  return (
    <div
      className={`
        relative px-6 py-4 rounded-xl glass transition-all duration-300
        ${getCategoryClass(agentType.category)}
        ${selected ? 'ring-2 ring-primary-400 glow' : ''}
        hover:scale-105 hover:glow
        min-w-[200px]
      `}
      style={{
        borderColor: agentType.color,
        borderWidth: '2px',
      }}
    >
      {/* 输入端口 */}
      {agentType.inputs > 0 && (
        <Handle
          type="target"
          position={Position.Left}
          className="!w-3 !h-3 !border-2"
          style={{ background: agentType.color }}
        />
      )}

      {/* 节点内容 */}
      <div className="flex items-center gap-3">
        <div
          className="p-2 rounded-lg"
          style={{
            background: `${agentType.color}20`,
            color: agentType.color,
          }}
        >
          {IconComponent && <IconComponent size={24} />}
        </div>
        <div className="flex-1">
          <div className="font-semibold text-white text-sm">
            {agentType.displayName}
          </div>
          <div className="text-xs text-gray-400 mt-1">
            {agentType.name}
          </div>
        </div>
      </div>

      {/* 状态指示器 */}
      {selected && (
        <div className="absolute -top-1 -right-1">
          <div className="w-3 h-3 rounded-full bg-primary-500 animate-pulse" />
          <div className="pulse-ring" />
        </div>
      )}

      {/* 输出端口 */}
      {agentType.outputs > 0 && (
        <Handle
          type="source"
          position={Position.Right}
          className="!w-3 !h-3 !border-2"
          style={{ background: agentType.color }}
        />
      )}
    </div>
  );
});

CustomNode.displayName = 'CustomNode';

export default CustomNode;
