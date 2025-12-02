import { useState } from 'react';
import { Search, ChevronDown, ChevronRight } from 'lucide-react';
import * as Icons from 'lucide-react';
import { agentTypes, categoryNames, getAgentsByCategory } from '../data/agentTypes';
import { AgentType } from '../types';

interface AgentNodePanelProps {
  onDragStart: (event: React.DragEvent, agentType: AgentType) => void;
}

const AgentNodePanel = ({ onDragStart }: AgentNodePanelProps) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(
    new Set(['custom', 'data', 'code', 'pipeline', 'image', 'utility'])
  );

  const toggleCategory = (category: string) => {
    const newExpanded = new Set(expandedCategories);
    if (newExpanded.has(category)) {
      newExpanded.delete(category);
    } else {
      newExpanded.add(category);
    }
    setExpandedCategories(newExpanded);
  };

  const filteredAgents = agentTypes.filter(agent =>
    agent.displayName.toLowerCase().includes(searchTerm.toLowerCase()) ||
    agent.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    agent.description.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const categories = ['custom', 'data', 'code', 'pipeline', 'image', 'utility'];

  return (
    <div className="w-80 h-full glass-dark border-r border-white/10 flex flex-col">
      {/* 标题 */}
      <div className="p-6 border-b border-white/10">
        <h2 className="text-xl font-bold text-white glow-text mb-2">
          Agent 节点
        </h2>
        <p className="text-sm text-gray-400">
          拖拽节点到画布创建工作流
        </p>
      </div>

      {/* 搜索框 */}
      <div className="p-4 border-b border-white/10">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={18} />
          <input
            type="text"
            placeholder="搜索 Agent..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-white/5 border border-white/10 rounded-lg
                     text-white placeholder-gray-500 focus:outline-none focus:border-primary-500
                     transition-colors"
          />
        </div>
      </div>

      {/* Agent 列表 */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {categories.map(category => {
          const categoryAgents = getAgentsByCategory(category).filter(agent =>
            filteredAgents.includes(agent)
          );

          if (categoryAgents.length === 0) return null;

          const isExpanded = expandedCategories.has(category);

          return (
            <div key={category} className="animate-fade-in">
              {/* 分类标题 */}
              <button
                onClick={() => toggleCategory(category)}
                className="w-full flex items-center justify-between p-3 rounded-lg
                         bg-white/5 hover:bg-white/10 transition-colors mb-2"
              >
                <span className="font-semibold text-white">
                  {categoryNames[category]}
                </span>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-gray-400">
                    {categoryAgents.length}
                  </span>
                  {isExpanded ? (
                    <ChevronDown size={18} className="text-gray-400" />
                  ) : (
                    <ChevronRight size={18} className="text-gray-400" />
                  )}
                </div>
              </button>

              {/* Agent 卡片 */}
              {isExpanded && (
                <div className="space-y-2 animate-slide-in">
                  {categoryAgents.map(agent => {
                    const IconComponent = Icons[agent.icon as keyof typeof Icons] as any;
                    
                    return (
                      <div
                        key={agent.id}
                        draggable
                        onDragStart={(e) => onDragStart(e, agent)}
                        className="p-3 rounded-lg glass cursor-move hover:scale-105
                                 transition-all duration-200 group"
                        style={{
                          borderLeft: `3px solid ${agent.color}`,
                        }}
                      >
                        <div className="flex items-start gap-3">
                          <div
                            className="p-2 rounded-lg flex-shrink-0"
                            style={{
                              background: `${agent.color}20`,
                              color: agent.color,
                            }}
                          >
                            {IconComponent && <IconComponent size={20} />}
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="font-medium text-white text-sm mb-1">
                              {agent.displayName}
                            </div>
                            <div className="text-xs text-gray-400 mb-2">
                              {agent.name}
                            </div>
                            <div className="text-xs text-gray-500 line-clamp-2">
                              {agent.description}
                            </div>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}

        {filteredAgents.length === 0 && (
          <div className="text-center py-12 text-gray-500">
            <Search size={48} className="mx-auto mb-4 opacity-50" />
            <p>未找到匹配的 Agent</p>
          </div>
        )}
      </div>

      {/* 底部统计 */}
      <div className="p-4 border-t border-white/10">
        <div className="text-xs text-gray-400 text-center">
          共 {agentTypes.length} 个 Agent 可用
        </div>
      </div>
    </div>
  );
};

export default AgentNodePanel;
