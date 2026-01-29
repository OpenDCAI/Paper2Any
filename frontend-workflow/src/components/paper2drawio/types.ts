// Paper2Drawio 类型定义

export type UploadMode = 'file' | 'text';
export type DiagramType = 'flowchart' | 'architecture' | 'sequence' | 'mindmap' | 'er' | 'auto';
export type DiagramStyle = 'minimal' | 'sketch' | 'default';

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface GenerateRequest {
  input_type: string;
  text_content?: string;
  diagram_type: DiagramType;
  diagram_style: DiagramStyle;
  language: string;
  chat_api_url: string;
  api_key: string;
  model: string;
}

export interface GenerateResponse {
  success: boolean;
  xml_content: string;
  file_path: string;
  error?: string;
}

export interface ChatRequest {
  current_xml: string;
  message: string;
  chat_history: ChatMessage[];
  chat_api_url: string;
  api_key: string;
  model: string;
}

export interface ChatResponse {
  success: boolean;
  xml_content: string;
  message: string;
  error?: string;
}
