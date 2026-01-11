export type MaterialType = 'image' | 'doc' | 'video' | 'link';

export interface KnowledgeFile {
  id: string;
  name: string;
  type: MaterialType;
  url?: string;
  file?: File;
  desc?: string;
  size?: string;
  uploadTime: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  time: string;
}

export type SectionType = 'library' | 'upload' | 'output';
export type ToolType = 'chat' | 'ppt' | 'mindmap' | 'podcast' | 'video';
