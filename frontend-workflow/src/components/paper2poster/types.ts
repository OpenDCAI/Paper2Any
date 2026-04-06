export type Step = 'upload' | 'generate' | 'complete';

export interface PosterConfig {
  text_model: string;
  vision_model: string;
  poster_width: number;
  poster_height: number;
}

export interface GenerateResult {
  status: 'pending' | 'processing' | 'done';
  pptxUrl?: string;
  pngUrl?: string;
  progress?: number;
}

export type UploadMode = 'file';
