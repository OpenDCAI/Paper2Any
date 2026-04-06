export type Step = 'upload' | 'outline' | 'generate' | 'complete';
export type PptGenerationMode = 'image' | 'frontend';

export interface SlideOutline {
  id: string;
  pageNum: number;
  title: string;
  layout_description: string;
  key_points: string[];
  asset_ref: string | null;
  asset_ref_preview_path?: string;
  generated_img_path?: string;
  generated_img_preview_path?: string;
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
  beforeImagePreview?: string;
  afterImage: string;
  afterImagePreview?: string;
  status: 'pending' | 'processing' | 'done';
  userPrompt?: string;
  versionHistory: ImageVersion[];
  currentVersionIndex: number;
}

export type FrontendFieldType = 'text' | 'textarea' | 'list';

export interface FrontendSlideReview {
  status: 'idle' | 'passed' | 'needs_repair' | 'repairing';
  summary: string;
  issues: string[];
}

export interface FrontendEditableField {
  key: string;
  label: string;
  type: FrontendFieldType;
  value: string;
  items: string[];
}

export type FrontendVisualAssetSource = 'generated' | 'paper_asset' | 'upload';

export interface FrontendVisualAsset {
  key: string;
  label: string;
  src: string;
  previewSrc?: string;
  originalSrc?: string;
  alt: string;
  sourceType: FrontendVisualAssetSource;
  storagePath?: string;
  previewStoragePath?: string;
  prompt?: string;
  style?: string;
}

export interface FrontendSlide {
  slideId: string;
  pageNum: number;
  title: string;
  htmlTemplate: string;
  cssCode: string;
  editableFields: FrontendEditableField[];
  visualAssets: FrontendVisualAsset[];
  generationNote?: string;
  status: 'pending' | 'processing' | 'done';
  review?: FrontendSlideReview;
}

export interface FrontendThemeLock {
  mustKeep: string[];
  preferredLayoutPatterns: string[];
  componentSignature: string;
  avoid: string[];
}

export interface FrontendDeckTheme {
  themeName: string;
  visualMood: string;
  footerText: string;
  sectionLabelTemplate: string;
  themeLock: FrontendThemeLock;
}

export type Paper2PPTTaskStatus = 'queued' | 'running' | 'done' | 'failed';

export interface Paper2PPTTaskResponse {
  success: boolean;
  task_id: string;
  task_type: string;
  status: Paper2PPTTaskStatus;
  message: string;
  error?: string | null;
  result?: {
    success: boolean;
    ppt_pdf_path?: string;
    ppt_pptx_path?: string;
    pagecontent?: Array<Record<string, unknown>>;
    result_path?: string;
    all_output_files?: string[];
  } | null;
}

export type UploadMode = 'file' | 'text' | 'topic';
export type StyleMode = 'prompt' | 'reference';
export type StylePreset = 'modern' | 'business' | 'academic' | 'creative';
