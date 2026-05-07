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

export type FrontendLayoutZone =
  | 'header'
  | 'main'
  | 'aside'
  | 'footer'
  | 'full'
  | 'left'
  | 'right';

export type FrontendLayoutWidthHint = 'full' | 'wide' | 'half' | 'third' | 'narrow' | 'auto';
export type FrontendLayoutSideHint = 'left' | 'right' | 'center' | 'auto';
export type FrontendLayoutEmphasis = 'high' | 'medium' | 'low';
export type FrontendLayoutMode = 'fluid' | 'hybrid' | 'fixed';
export type FrontendSlideBlockType = 'text' | 'list' | 'image' | 'quote' | 'stat' | 'callout' | 'table';
export type FrontendRenderEngine = 'blocks' | 'canvas';
export type FrontendCanvasNodeType = 'container' | 'component';
export type FrontendCanvasDirection = 'row' | 'column' | 'grid';
export type FrontendCanvasComponentType =
  | 'heading'
  | 'text'
  | 'bullets'
  | 'quote'
  | 'stat'
  | 'callout'
  | 'figure'
  | 'table'
  | 'placeholder';

export const FRONTEND_INSERT_ZONE_TARGET_PREFIX = '__insert_zone__:';

export const buildFrontendInsertZoneTarget = (zone: FrontendLayoutZone | string) =>
  `${FRONTEND_INSERT_ZONE_TARGET_PREFIX}${zone}`;

export const parseFrontendInsertZoneTarget = (target?: string | null): FrontendLayoutZone | null => {
  if (!target?.startsWith(FRONTEND_INSERT_ZONE_TARGET_PREFIX)) {
    return null;
  }
  const zone = target.slice(FRONTEND_INSERT_ZONE_TARGET_PREFIX.length);
  return ['header', 'main', 'aside', 'footer', 'full', 'left', 'right'].includes(zone)
    ? zone as FrontendLayoutZone
    : null;
};

export interface FrontendBlockLayout {
  zone: FrontendLayoutZone;
  span: number;
  order: number;
  preferredWidth: FrontendLayoutWidthHint;
  preferredSide: FrontendLayoutSideHint;
  emphasis: FrontendLayoutEmphasis;
}

export interface FrontendTableData {
  headers: string[];
  rows: string[][];
}

export interface FrontendBlockChild {
  id: string;
  type: FrontendSlideBlockType;
  role: string;
  content: string;
  items: string[];
  assetKey?: string;
  tableData?: FrontendTableData;
}

export interface FrontendSlideBlock {
  id: string;
  type: FrontendSlideBlockType;
  role: string;
  content: string;
  items: string[];
  assetKey?: string;
  tableData?: FrontendTableData;
  children?: FrontendBlockChild[];
  layout: FrontendBlockLayout;
}

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

export interface FrontendCanvasNodeStyle {
  direction?: FrontendCanvasDirection;
  gap?: number;
  padding?: number;
  weight?: number;
  basis?: string | number;
  align?: 'start' | 'center' | 'end' | 'stretch';
  justify?: 'start' | 'center' | 'end' | 'between' | 'around';
  wrap?: boolean;
  columns?: number;
  minWidth?: number;
  maxWidth?: number;
  variant?: string;
  emphasis?: FrontendLayoutEmphasis;
}

export interface FrontendCanvasNode {
  type: FrontendCanvasNodeType;
  id: string;
  style?: FrontendCanvasNodeStyle;
  component?: FrontendCanvasComponentType;
  props?: Record<string, unknown>;
  children?: FrontendCanvasNode[];
}

export interface FrontendCanvasContentAsset {
  type: 'image';
  src: string;
  previewSrc?: string;
  originalSrc?: string;
  alt?: string;
  assetKey?: string;
}

export interface FrontendCanvasValidationIssue {
  severity: 'info' | 'repairable' | 'warning' | 'error';
  code: string;
  nodeId?: string;
  ref?: string;
  suggestedRef?: string;
  message: string;
}

export interface FrontendCanvasValidation {
  ok: boolean;
  usedRefs: string[];
  definedContentKeys: string[];
  missingRefs: string[];
  orphanContentKeys: string[];
  emptyComponents: string[];
  issues: FrontendCanvasValidationIssue[];
}

export interface FrontendLayoutIRNode {
  nodeId: string;
  type: FrontendCanvasNodeType;
  component?: FrontendCanvasComponentType;
  box: {
    x: number;
    y: number;
    w: number;
    h: number;
  };
  computedStyle?: Record<string, unknown>;
  overflow?: boolean;
}

export interface FrontendLayoutIR {
  schemaVersion: string;
  slideId: string;
  viewport: {
    width: number;
    height: number;
    scale: number;
  };
  nodes: FrontendLayoutIRNode[];
  overflowIssues?: string[];
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

export interface FrontendSlide {
  slideId: string;
  pageNum: number;
  title: string;
  schemaVersion?: string;
  renderEngine?: FrontendRenderEngine;
  templateKey?: string;
  layoutMode?: FrontendLayoutMode;
  blocks: FrontendSlideBlock[];
  layoutFamily?: string;
  root?: FrontendCanvasNode;
  content?: Record<string, unknown>;
  constraints?: Record<string, unknown>;
  editableMap?: Record<string, string>;
  canvasValidation?: FrontendCanvasValidation;
  layoutIr?: FrontendLayoutIR;
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
  palette?: FrontendDeckPalette;
  typography?: FrontendDeckTypography;
  layoutRules?: string[];
  componentRules?: string[];
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
