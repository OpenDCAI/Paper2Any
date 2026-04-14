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

export type StructuredSlideLayoutType =
  | 'cover'
  | 'section'
  | 'bullets'
  | 'two_column'
  | 'cards_2x2'
  | 'image_focus'
  | 'comparison'
  | 'timeline';

interface BaseLayoutData {
  eyebrowKey?: string;
  titleKey: string;
  footerKey?: string;
  summaryKey?: string;
}

export interface CoverLayoutData extends BaseLayoutData {
  type: 'cover';
  subtitleKey: string;
  presenterKey?: string;
}

export interface SectionLayoutData extends BaseLayoutData {
  type: 'section';
  quoteKey?: string;
}

export interface BulletsLayoutData extends BaseLayoutData {
  type: 'bullets';
  bulletsKey: string;
  takeawayKey?: string;
}

export interface TwoColumnLayoutData extends BaseLayoutData {
  type: 'two_column';
  leftHeadingKey: string;
  leftBodyKey: string;
  leftPointsKey?: string;
  rightHeadingKey: string;
  rightBodyKey: string;
  rightPointsKey?: string;
}

export interface CardRef {
  titleKey: string;
  bodyKey: string;
}

export interface Cards2x2LayoutData extends BaseLayoutData {
  type: 'cards_2x2';
  cards: CardRef[];
}

export interface ImageFocusLayoutData extends BaseLayoutData {
  type: 'image_focus';
  bulletsKey?: string;
  visualKey: string;
  visualCaptionKey?: string;
}

export interface ComparisonLayoutData extends BaseLayoutData {
  type: 'comparison';
  leftTitleKey: string;
  leftPointsKey: string;
  rightTitleKey: string;
  rightPointsKey: string;
}

export interface TimelineItemRef {
  labelKey: string;
  bodyKey: string;
}

export interface TimelineLayoutData extends BaseLayoutData {
  type: 'timeline';
  timeline: TimelineItemRef[];
}

export type FrontendSlideLayoutData =
  | CoverLayoutData
  | SectionLayoutData
  | BulletsLayoutData
  | TwoColumnLayoutData
  | Cards2x2LayoutData
  | ImageFocusLayoutData
  | ComparisonLayoutData
  | TimelineLayoutData;

export interface FrontendSlide {
  slideId: string;
  pageNum: number;
  title: string;
  layoutType: StructuredSlideLayoutType;
  layoutData: FrontendSlideLayoutData;
  htmlTemplate?: string;
  cssCode?: string;
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

export interface FrontendDeckPalette {
  bg: string;
  panel: string;
  primary: string;
  secondary: string;
  accent: string;
  text: string;
  muted: string;
}

export interface FrontendDeckTypography {
  titleFontStack: string;
  bodyFontStack: string;
  eyebrowSize: number;
  titleSize: number;
  summarySize: number;
  bodySize: number;
}

export type FrontendDeckStyleFamily = 'modern' | 'business' | 'academic' | 'creative';

export interface FrontendDeckTheme {
  themeName: string;
  visualMood: string;
  styleFamily: FrontendDeckStyleFamily;
  footerText: string;
  sectionLabelTemplate: string;
  palette: FrontendDeckPalette;
  typography: FrontendDeckTypography;
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
