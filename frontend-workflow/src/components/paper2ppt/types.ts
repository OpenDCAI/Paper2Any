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
  status: 'pending' | 'processing' | 'done' | 'failed';
  userPrompt?: string;
  versionHistory: ImageVersion[];
  currentVersionIndex: number;
}

export type MaskSelectionShape = 'rect' | 'circle';

export interface MaskSelectionSpec {
  shape: MaskSelectionShape;
  x: number;
  y: number;
  width: number;
  height: number;
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
  autoDeleteOnEmpty?: boolean;
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

export interface FrontendCanvasVisualStyle {
  fill?: string;
  color?: string;
  borderColor?: string;
  borderWidth?: number;
  radius?: number;
  padding?: number;
  fontFamily?: string;
  fontSize?: number;
  fontWeight?: number | string;
  fontStyle?: 'normal' | 'italic';
  lineHeight?: number;
  textAlign?: 'left' | 'center' | 'right' | 'justify';
  opacity?: number;
  imageFit?: 'contain' | 'cover' | 'fill';
  emphasis?: FrontendLayoutEmphasis;
}

export interface FrontendCanvasVisualSpec {
  palette?: Partial<FrontendDeckPalette>;
  typography?: Partial<FrontendDeckTypography>;
  surface?: {
    background?: string;
    panel?: string;
    primary?: string;
    secondary?: string;
    accent?: string;
    text?: string;
    muted?: string;
    cardRadius?: number;
    cardPadding?: number;
    sectionGap?: number;
  };
  layout?: {
    safeMargin?: number;
    sectionGap?: number;
    contentGap?: number;
    maxColumns?: number;
  };
  nodeStyles?: Record<string, FrontendCanvasVisualStyle>;
  componentStyles?: Partial<Record<FrontendCanvasComponentType, FrontendCanvasVisualStyle>>;
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

export interface FrontendCanvasVisualStyle {
  fill?: string;
  color?: string;
  borderColor?: string;
  borderWidth?: number;
  radius?: number;
  padding?: number;
  fontFamily?: string;
  fontSize?: number;
  fontWeight?: number | string;
  fontStyle?: 'normal' | 'italic';
  lineHeight?: number;
  textAlign?: 'left' | 'center' | 'right' | 'justify';
  opacity?: number;
  imageFit?: 'contain' | 'cover' | 'fill';
  emphasis?: FrontendLayoutEmphasis;
}

export interface FrontendCanvasVisualSpec {
  palette?: Partial<FrontendDeckPalette>;
  typography?: Partial<FrontendDeckTypography>;
  surface?: {
    background?: string;
    panel?: string;
    primary?: string;
    secondary?: string;
    accent?: string;
    text?: string;
    muted?: string;
    cardRadius?: number;
    cardPadding?: number;
    sectionGap?: number;
  };
  layout?: {
    safeMargin?: number;
    sectionGap?: number;
    contentGap?: number;
    maxColumns?: number;
  };
  nodeStyles?: Record<string, FrontendCanvasVisualStyle>;
  componentStyles?: Partial<Record<FrontendCanvasComponentType, FrontendCanvasVisualStyle>>;
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
  layoutType?: StructuredSlideLayoutType;
  layoutData?: FrontendSlideLayoutData;
  schemaVersion?: string;
  renderEngine?: FrontendRenderEngine;
  templateKey?: string;
  layoutMode?: FrontendLayoutMode;
  blocks: FrontendSlideBlock[];
  layoutFamily?: string;
  root?: FrontendCanvasNode;
  content?: Record<string, unknown>;
  visualSpec?: FrontendCanvasVisualSpec;
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
  stylePrompt?: string;
  visualMood: string;
  styleFamily: FrontendDeckStyleFamily;
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
