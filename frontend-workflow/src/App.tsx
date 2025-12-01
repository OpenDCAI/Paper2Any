import { useState, useCallback } from 'react';
import { ReactFlowProvider, Node, Edge } from 'reactflow';
import ParticleBackground from './components/ParticleBackground';
import AgentNodePanel from './components/AgentNodePanel';
import WorkflowCanvas from './components/WorkflowCanvas';
import NodeConfigPanel from './components/NodeConfigPanel';
import Paper2GraphPage from './components/Paper2GraphPage';
import { AgentType, WorkflowNode, AgentConfig } from './types';
import { Workflow, Zap, Save, FolderOpen, Trash2 } from 'lucide-react';

function App() {
  const [selectedNode, setSelectedNode] = useState<WorkflowNode | null>(null);
  const [nodeCount, setNodeCount] = useState(0);
  const [edgeCount, setEdgeCount] = useState(0);
  const [activePage, setActivePage] = useState<'workflow' | 'paper2graph'>('workflow');

  const handleDragStart = useCallback(
    (event: React.DragEvent, agentType: AgentType) => {
      event.dataTransfer.setData('application/reactflow', JSON.stringify(agentType));
      event.dataTransfer.effectAllowed = 'move';
    },
    []
  );

  const handleNodeSelect = useCallback((node: WorkflowNode | null) => {
    setSelectedNode(node);
  }, []);

  const handleNodesChange = useCallback((nodes: Node[]) => {
    setNodeCount(nodes.length);
    // 如果当前选中的节点被删除，更新选中状态
    if (selectedNode && !nodes.find(n => n.id === selectedNode.id)) {
      setSelectedNode(null);
    }
  }, [selectedNode]);

  const handleEdgesChange = useCallback((edges: Edge[]) => {
    setEdgeCount(edges.length);
  }, []);

  const handleUpdateNode = useCallback((nodeId: string, config: AgentConfig) => {
    // 通过全局API更新节点配置
    const api = (window as any).workflowCanvasAPI;
    if (api && api.updateNode) {
      api.updateNode(nodeId, { config });
      
      // 同时更新选中节点的状态
      if (selectedNode && selectedNode.id === nodeId) {
        setSelectedNode({
          ...selectedNode,
          data: {
            ...selectedNode.data,
            config,
          },
        });
      }
    }
  }, [selectedNode]);

  const handleDeleteNode = useCallback((nodeId: string) => {
    // 通过全局API删除节点
    const api = (window as any).workflowCanvasAPI;
    if (api && api.deleteNode) {
      api.deleteNode(nodeId);
    }
  }, []);

  const handleClearWorkflow = useCallback(() => {
    if (window.confirm('确定要清空整个工作流吗？此操作无法撤销。')) {
      // 重新加载页面来清空工作流
      window.location.reload();
    }
  }, []);

  return (
    <div className="w-screen h-screen bg-[#0a0a1a] overflow-hidden relative">
      {/* 粒子背景 */}
      <ParticleBackground />

      {/* 顶部导航栏 */}
      <header className="absolute top-0 left-0 right-0 h-16 glass-dark border-b border-white/10 z-10">
        <div className="h-full px-6 flex items-center justify-between">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-primary-500/20">
              <Workflow className="text-primary-400" size={24} />
            </div>
            <div>
              <h1 className="text-lg font-bold text-white glow-text">
                DataFlow Agent
              </h1>
              <p className="text-xs text-gray-400">Workflow Editor</p>
            </div>
          </div>

          {/* 工具栏 */}
          <div className="flex items-center gap-4">
            {/* 页面切换 Tab */}
            <div className="flex items-center gap-2">
              <button
                onClick={() => setActivePage('workflow')}
                className={`px-3 py-1.5 rounded-full text-sm ${
                  activePage === 'workflow'
                    ? 'bg-primary-500 text-white shadow'
                    : 'glass text-gray-300 hover:bg-white/10'
                }`}
              >
                工作流编辑器
              </button>
              <button
                onClick={() => setActivePage('paper2graph')}
                className={`px-3 py-1.5 rounded-full text-sm ${
                  activePage === 'paper2graph'
                    ? 'bg-primary-500 text-white shadow'
                    : 'glass text-gray-300 hover:bg-white/10'
                }`}
              >
                Paper2Graph 文档解析
              </button>
            </div>

            {/* 右侧操作按钮 */}
            <div className="flex items-center gap-2">
              <button className="flex items-center gap-2 px-4 py-2 rounded-lg glass hover:bg-white/10 transition-colors">
                <FolderOpen size={18} className="text-gray-400" />
                <span className="text-sm text-white">打开</span>
              </button>
              <button className="flex items-center gap-2 px-4 py-2 rounded-lg glass hover:bg-white/10 transition-colors">
                <Save size={18} className="text-gray-400" />
                <span className="text-sm text-white">保存</span>
              </button>
              <button
                onClick={handleClearWorkflow}
                className="flex items-center gap-2 px-4 py-2 rounded-lg glass hover:bg-white/10 transition-colors"
              >
                <Trash2 size={18} className="text-gray-400" />
                <span className="text-sm text-white">清空</span>
              </button>
              <div className="w-px h-8 bg-white/10 mx-2" />
              <button className="flex items-center gap-2 px-4 py-2 rounded-lg bg-primary-500 hover:bg-primary-600 transition-colors glow">
                <Zap size={18} className="text-white" />
                <span className="text-sm text-white font-medium">
                  {activePage === 'workflow' ? '运行工作流' : '生成 PPTX'}
                </span>
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* 主内容区 */}
      <main className="absolute top-16 bottom-0 left-0 right-0 flex">
        {activePage === 'workflow' ? (
          <ReactFlowProvider>
            {/* 左侧面板 - Agent节点列表 */}
            <AgentNodePanel onDragStart={handleDragStart} />

            {/* 中央画布 */}
            <div className="flex-1 relative">
              <WorkflowCanvas
                onNodeSelect={handleNodeSelect}
                onNodesChange={handleNodesChange}
                onEdgesChange={handleEdgesChange}
              />

              {/* 画布提示 */}
              <div className="absolute bottom-6 left-1/2 -translate-x-1/2 pointer-events-none">
                <div className="px-4 py-2 rounded-full glass text-sm text-gray-400">
                  拖拽左侧 Agent 到画布 · 连接节点创建工作流 · 滚轮缩放
                </div>
              </div>
            </div>

            {/* 右侧面板 - 节点配置 */}
            {selectedNode && (
              <NodeConfigPanel
                node={selectedNode}
                onClose={() => setSelectedNode(null)}
                onDelete={handleDeleteNode}
                onUpdate={handleUpdateNode}
              />
            )}
          </ReactFlowProvider>
        ) : (
          <div className="flex-1">
            <Paper2GraphPage />
          </div>
        )}
      </main>

      {/* 底部状态栏 */}
      <footer className="absolute bottom-0 left-0 right-0 h-8 glass-dark border-t border-white/10 z-10">
        <div className="h-full px-4 flex items-center justify-between text-xs text-gray-500">
          <div className="flex items-center gap-4">
            <span>节点: {nodeCount}</span>
            <span>连接: {edgeCount}</span>
          </div>
          <div className="flex items-center gap-4">
            <span>DataFlow Agent v1.0.0</span>
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
              <span>就绪</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
