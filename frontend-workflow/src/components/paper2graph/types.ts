export type UploadMode = 'file' | 'text';
export type FileKind = 'pdf' | 'image' | null;
export type GraphType = 'model_arch' | 'tech_route' | 'exp_data';
export type Language = 'zh' | 'en';
export type StyleType = 'cartoon' | 'realistic' | 'Low Poly 3D' | 'blocky LEGO aesthetic';
export type FigureComplex = 'easy' | 'mid' | 'hard';

export type GenerationStage = {
  id: number;
  message: string;
  duration: number; // Duration of this stage in seconds
};

export type QuotaInfo = {
  remaining: number;
  isAuthenticated: boolean;
};
