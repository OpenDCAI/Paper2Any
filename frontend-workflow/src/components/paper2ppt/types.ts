export type Step = 'upload' | 'outline' | 'generate' | 'complete';

export interface SlideOutline {
  id: string;
  pageNum: number;
  title: string;
  layout_description: string;
  key_points: string[];
  asset_ref: string | null;
  generated_img_path?: string;
}

export interface ImageVersion {
  versionNumber: number;
  imageUrl: string;
  prompt: string;
  timestamp: number;
  isCurrentVersion: boolean;
}

export interface GenerateResult {
  slideId: string;
  beforeImage: string;
  afterImage: string;
  status: 'pending' | 'processing' | 'done';
  userPrompt?: string;
  versionHistory: ImageVersion[];
  currentVersionIndex: number;
}

export type UploadMode = 'file' | 'text' | 'topic';
export type StyleMode = 'prompt' | 'reference';
export type StylePreset = 'modern' | 'business' | 'academic' | 'creative';
