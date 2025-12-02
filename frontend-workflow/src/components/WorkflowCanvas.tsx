import { useCallback, useRef, useState } from 'react';
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  addEdge,
  Connection,
  Edge,
  Node,
  useNodesState,
  useEdgesState,
  BackgroundVariant,
} from 'reactflow';
import 'reactflow/dist/style.css';
import CustomNode from './CustomNode';
import { AgentType, WorkflowNode } from '../types';

const nodeTypes = {
  custom: CustomNode,
};

interface WorkflowCanvasProps {
  onNodeSelect: (node: WorkflowNode | null) => void;
  onNodesChange?: (nodes: Node[]) => void;
  onEdgesChange?: (edges: Edge[]) => void;
}

const WorkflowCanvas = ({ onNodeSelect, onNodesChange: onNodesChangeCallback, onEdgesChange: onEdgesChangeCallback }: WorkflowCanvasProps) => {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [reactFlowInstance, setReactFlowInstance] = useState<any>(null);

  // 通知父组件节点变化
  const handleNodesChange = useCallback((changes: any) => {
    onNodesChange(changes);
    if (onNodesChangeCallback) {
      // 延迟调用以确保状态已更新
      setTimeout(() => {
        setNodes((nds) => {
          onNodesChangeCallback(nds);
          return nds;
        });
      }, 0);
    }
  }, [onNodesChange, onNodesChangeCallback, setNodes]);

  // 通知父组件边变化
  const handleEdgesChange = useCallback((changes: any) => {
    onEdgesChange(changes);
    if (onEdgesChangeCallback) {
      setTimeout(() => {
        setEdges((eds) => {
          onEdgesChangeCallback(eds);
          return eds;
        });
      }, 0);
    }
  }, [onEdgesChange, onEdgesChangeCallback, setEdges]);

  // 暴露更新节点的方法
  const updateNode = useCallback((nodeId: string, data: any) => {
    setNodes((nds) =>
      nds.map((node) =>
        node.id === nodeId
          ? { ...node, data: { ...node.data, ...data } }
          : node
      )
    );
  }, [setNodes]);

  // 暴露删除节点的方法
  const deleteNode = useCallback((nodeId: string) => {
    setNodes((nds) => nds.filter((node) => node.id !== nodeId));
    setEdges((eds) => eds.filter((edge) => edge.source !== nodeId && edge.target !== nodeId));
    onNodeSelect(null);
  }, [setNodes, setEdges, onNodeSelect]);

  // 将方法暴露给父组件
  (window as any).workflowCanvasAPI = {
    updateNode,
    deleteNode,
  };

  const onConnect = useCallback(
    (params: Connection | Edge) => {
      setEdges((eds) =>
        addEdge(
          {
            ...params,
            animated: true,
            style: { stroke: '#0070f3', strokeWidth: 2 },
          },
          eds
        )
      );
    },
    [setEdges]
  );

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      if (!reactFlowWrapper.current || !reactFlowInstance) return;

      const reactFlowBounds = reactFlowWrapper.current.getBoundingClientRect();
      const agentTypeData = event.dataTransfer.getData('application/reactflow');

      if (!agentTypeData) return;

      const agentType: AgentType = JSON.parse(agentTypeData);
      const position = reactFlowInstance.project({
        x: event.clientX - reactFlowBounds.left,
        y: event.clientY - reactFlowBounds.top,
      });

      const newNode: WorkflowNode = {
        id: `${agentType.id}-${Date.now()}`,
        type: 'custom',
        position,
        data: {
          label: agentType.displayName,
          agentType,
        },
      };

      setNodes((nds) => nds.concat(newNode));
    },
    [reactFlowInstance, setNodes]
  );

  const onNodeClick = useCallback(
    (_event: React.MouseEvent, node: Node) => {
      onNodeSelect(node as WorkflowNode);
    },
    [onNodeSelect]
  );

  const onPaneClick = useCallback(() => {
    onNodeSelect(null);
  }, [onNodeSelect]);

  return (
    <div ref={reactFlowWrapper} className="w-full h-full">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={handleNodesChange}
        onEdgesChange={handleEdgesChange}
        onConnect={onConnect}
        onInit={setReactFlowInstance}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        nodeTypes={nodeTypes}
        fitView
        attributionPosition="bottom-left"
      >
        <Background
          variant={BackgroundVariant.Dots}
          gap={20}
          size={1}
          color="rgba(0, 112, 243, 0.2)"
        />
        <Controls className="!bg-white/5 !border-white/10" />
        <MiniMap
          nodeColor={(node) => {
            const agentType = (node.data as any).agentType;
            return agentType?.color || '#0070f3';
          }}
          className="!bg-black/50 !border-white/10"
        />
      </ReactFlow>
    </div>
  );
};

export default WorkflowCanvas;
