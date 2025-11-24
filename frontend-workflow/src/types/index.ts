import { Node, Edge } from 'reactflow';

export type AgentCategory = 'data' | 'code' | 'pipeline' | 'image' | 'utility' | 'custom';

export type AgentMode = 'simple' | 'react' | 'graph' | 'vlm' | 'parallel';

export interface AgentConfig {
  // 基础参数
  model_name?: string;
  chat_api_url?: string;
  temperature?: number;
  max_tokens?: number;
  tool_mode?: 'auto' | 'none' | 'required';
  parser_type?: 'json' | 'xml' | 'text';
  
  // 提示词
  system_prompt?: string;
  task_prompt?: string;
  
  // JSON Schema配置
  response_schema?: Record<string, string>;
  response_schema_description?: string;
  response_example?: string;
  required_fields?: string[];
  
  // 模式特定参数
  mode?: AgentMode;
  max_retries?: number; // for react mode
  concurrency_limit?: number; // for parallel mode
  vlm_mode?: 'understanding' | 'generation' | 'edit'; // for vlm mode
  image_detail?: 'low' | 'high' | 'auto'; // for vlm mode
}

export interface AgentType {
  id: string;
  name: string;
  displayName: string;
  category: AgentCategory;
  description: string;
  icon: string;
  color: string;
  inputs: number;
  outputs: number;
  mode?: AgentMode;
  defaultConfig?: AgentConfig;
}

export interface WorkflowNode extends Node {
  data: {
    label: string;
    agentType: AgentType;
    config?: AgentConfig;
  };
}

export interface WorkflowEdge extends Edge {
  animated?: boolean;
}

export interface WorkflowState {
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  selectedNode: WorkflowNode | null;
  addNode: (node: WorkflowNode) => void;
  updateNode: (id: string, data: Partial<WorkflowNode['data']>) => void;
  deleteNode: (id: string) => void;
  addEdge: (edge: WorkflowEdge) => void;
  deleteEdge: (id: string) => void;
  setSelectedNode: (node: WorkflowNode | null) => void;
  clearWorkflow: () => void;
}
