import PptxGenJS from 'pptxgenjs';
import {
  FrontendCanvasNode,
  FrontendCanvasVisualSpec,
  FrontendCanvasVisualStyle,
  FrontendDeckTheme,
  FrontendLayoutIRNode,
  FrontendSlide,
  FrontendTableData,
  FrontendVisualAsset,
} from './types';

const DESIGN_WIDTH = 1600;
const DESIGN_HEIGHT = 900;
const SLIDE_WIDTH_IN = 13.333;
const SLIDE_HEIGHT_IN = 7.5;
const POINTS_PER_CSS_PX = 0.55;

const DEFAULT_PALETTE = {
  bg: '#0b1020',
  panel: 'rgba(15, 23, 42, 0.92)',
  primary: '#7dd3fc',
  secondary: '#38bdf8',
  accent: '#f59e0b',
  text: '#e2e8f0',
  muted: '#94a3b8',
};

const DEFAULT_TYPOGRAPHY = {
  titleFontStack: 'Georgia, "Times New Roman", serif',
  bodyFontStack: '"Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif',
  eyebrowSize: 18,
  titleSize: 56,
  summarySize: 26,
  bodySize: 24,
};

type PptBox = {
  x: number;
  y: number;
  w: number;
  h: number;
};

type ResolvedTheme = {
  palette: typeof DEFAULT_PALETTE;
  typography: typeof DEFAULT_TYPOGRAPHY;
};

type ComputedStyleMap = Record<string, unknown>;

const SYSTEM_PAGE_LABEL_RE = /^(?:slide\s*)?\d{1,3}\s*\/\s*\d{1,3}$|^第\s*\d{1,3}\s*\/\s*\d{1,3}\s*页$/i;

type ResolvedCanvasVisualTheme = ResolvedTheme & {
  surface: {
    cardRadius: number;
    cardPadding: number;
    sectionGap: number;
  };
  layout: {
    safeMargin: number;
    sectionGap: number;
    contentGap: number;
    maxColumns: number;
  };
};

const resolveTheme = (theme?: FrontendDeckTheme | null): ResolvedTheme => ({
  palette: {
    ...DEFAULT_PALETTE,
    ...(theme?.palette || {}),
  },
  typography: {
    ...DEFAULT_TYPOGRAPHY,
    ...(theme?.typography || {}),
  },
});

const resolveCanvasVisualTheme = (
  theme?: FrontendDeckTheme | null,
  visualSpec?: FrontendCanvasVisualSpec | null,
): ResolvedCanvasVisualTheme => {
  const baseTheme = resolveTheme(theme);
  const surface = visualSpec?.surface || {};
  return {
    palette: {
      ...baseTheme.palette,
      ...(visualSpec?.palette || {}),
      bg: surface.background || visualSpec?.palette?.bg || baseTheme.palette.bg,
      panel: surface.panel || visualSpec?.palette?.panel || baseTheme.palette.panel,
      primary: surface.primary || visualSpec?.palette?.primary || baseTheme.palette.primary,
      secondary: surface.secondary || visualSpec?.palette?.secondary || baseTheme.palette.secondary,
      accent: surface.accent || visualSpec?.palette?.accent || baseTheme.palette.accent,
      text: surface.text || visualSpec?.palette?.text || baseTheme.palette.text,
      muted: surface.muted || visualSpec?.palette?.muted || baseTheme.palette.muted,
    },
    typography: {
      ...baseTheme.typography,
      ...(visualSpec?.typography || {}),
    },
    surface: {
      cardRadius: surface.cardRadius ?? 28,
      cardPadding: surface.cardPadding ?? 24,
      sectionGap: surface.sectionGap ?? 22,
    },
    layout: {
      safeMargin: visualSpec?.layout?.safeMargin ?? 62,
      sectionGap: visualSpec?.layout?.sectionGap ?? 22,
      contentGap: visualSpec?.layout?.contentGap ?? 18,
      maxColumns: Math.max(1, Math.min(4, visualSpec?.layout?.maxColumns ?? 2)),
    },
  };
};

const stripHash = (value: string) => value.trim().replace(/^#/, '');

const toHexColor = (value: unknown, fallback: string) => {
  const raw = String(value || '').trim();
  if (!raw) return stripHash(fallback);
  const hex = raw.match(/^#?([0-9a-f]{3}|[0-9a-f]{6})$/i)?.[1];
  if (hex) {
    if (hex.length === 3) {
      return hex.split('').map((char) => `${char}${char}`).join('').toUpperCase();
    }
    return hex.toUpperCase();
  }
  const rgba = raw.match(/rgba?\(([^)]+)\)/i);
  if (rgba) {
    const [r, g, b] = rgba[1].split(',').map((part) => Number.parseFloat(part.trim()));
    if ([r, g, b].every((channel) => Number.isFinite(channel))) {
      return [r, g, b]
        .map((channel) => Math.max(0, Math.min(255, Math.round(channel))).toString(16).padStart(2, '0'))
        .join('')
        .toUpperCase();
    }
  }
  return stripHash(fallback);
};

const firstFont = (fontStack: string, fallback: string) => {
  const first = fontStack
    .split(',')
    .map((item) => item.trim().replace(/^["']|["']$/g, ''))
    .find(Boolean);
  return first || fallback;
};

const pxToIn = (value: number, axis: 'x' | 'y') =>
  (value / (axis === 'x' ? DESIGN_WIDTH : DESIGN_HEIGHT)) *
  (axis === 'x' ? SLIDE_WIDTH_IN : SLIDE_HEIGHT_IN);

const pxToPt = (value: number) => value * POINTS_PER_CSS_PX;

const parseCssPx = (value: unknown, fallback = 0) => {
  const parsed = Number.parseFloat(String(value || ''));
  return Number.isFinite(parsed) ? parsed : fallback;
};

const styleValue = (layoutNode: FrontendLayoutIRNode, key: string, fallback = '') =>
  String((layoutNode.computedStyle as ComputedStyleMap | undefined)?.[key] || fallback);

const fontSizeFromStyle = (layoutNode: FrontendLayoutIRNode, fallbackPx: number) =>
  Math.max(6, Math.min(60, pxToPt(parseCssPx(styleValue(layoutNode, 'fontSize'), fallbackPx))));

const lineSpacingFromStyle = (layoutNode: FrontendLayoutIRNode, fontSizePt: number) => {
  const lineHeight = styleValue(layoutNode, 'lineHeight');
  if (!lineHeight || lineHeight === 'normal') {
    return Math.round(fontSizePt * 1.22);
  }
  const lineHeightPx = parseCssPx(lineHeight, 0);
  return lineHeightPx > 0 ? Math.max(fontSizePt, Math.round(pxToPt(lineHeightPx))) : Math.round(fontSizePt * 1.22);
};

const marginFromPadding = (layoutNode: FrontendLayoutIRNode): [number, number, number, number] => {
  const paddingTop = pxToPt(parseCssPx(styleValue(layoutNode, 'paddingTop'), 0));
  const paddingRight = pxToPt(parseCssPx(styleValue(layoutNode, 'paddingRight'), 0));
  const paddingBottom = pxToPt(parseCssPx(styleValue(layoutNode, 'paddingBottom'), 0));
  const paddingLeft = pxToPt(parseCssPx(styleValue(layoutNode, 'paddingLeft'), 0));
  return [
    Math.max(0, Math.min(36, paddingTop)),
    Math.max(0, Math.min(36, paddingRight)),
    Math.max(0, Math.min(36, paddingBottom)),
    Math.max(0, Math.min(36, paddingLeft)),
  ];
};

const colorFromStyle = (layoutNode: FrontendLayoutIRNode, key: string, fallback: string) =>
  toHexColor(styleValue(layoutNode, key), fallback);

const fontFaceFromStyle = (layoutNode: FrontendLayoutIRNode, fallback: string) =>
  firstFont(styleValue(layoutNode, 'fontFamily'), fallback);

const isBoldStyle = (layoutNode: FrontendLayoutIRNode, fallback = false) => {
  const raw = styleValue(layoutNode, 'fontWeight');
  const numeric = Number.parseInt(raw, 10);
  if (Number.isFinite(numeric)) return numeric >= 600;
  return raw === 'bold' || raw === 'bolder' || fallback;
};

const isItalicStyle = (layoutNode: FrontendLayoutIRNode, fallback = false) =>
  styleValue(layoutNode, 'fontStyle') === 'italic' || fallback;

const alignFromStyle = (layoutNode: FrontendLayoutIRNode): PptxGenJS.HAlign => {
  const align = styleValue(layoutNode, 'textAlign').toLowerCase();
  if (align === 'center') return 'center';
  if (align === 'right' || align === 'end') return 'right';
  if (align === 'justify') return 'justify';
  return 'left';
};

const verticalAlignFromStyle = (layoutNode: FrontendLayoutIRNode): PptxGenJS.VAlign => {
  const alignItems = styleValue(layoutNode, 'alignItems').toLowerCase();
  const justify = styleValue(layoutNode, 'justifyContent').toLowerCase();
  const vertical = styleValue(layoutNode, 'verticalAlign').toLowerCase();
  if (alignItems === 'center' || justify === 'center' || vertical === 'middle') return 'middle';
  if (alignItems === 'flex-end' || justify === 'flex-end' || vertical === 'bottom') return 'bottom';
  return 'top';
};

const toPptBox = (node: FrontendLayoutIRNode): PptBox => ({
  x: pxToIn(node.box.x, 'x'),
  y: pxToIn(node.box.y, 'y'),
  w: Math.max(0.08, pxToIn(node.box.w, 'x')),
  h: Math.max(0.08, pxToIn(node.box.h, 'y')),
});

const insetBoxForText = (box: PptBox) => ({
  ...box,
  x: box.x + 0.01,
  y: box.y + 0.01,
  w: Math.max(0.08, box.w - 0.02),
  h: Math.max(0.08, box.h - 0.02),
});

const normalizeComponent = (value?: unknown) => {
  const raw = String(value || '').trim().toLowerCase();
  const aliases: Record<string, string> = {
    table_card: 'table',
    bullet_list: 'bullets',
    visual: 'figure',
    image: 'figure',
  };
  return aliases[raw] || raw;
};

const walkCanvasNodes = (node: FrontendCanvasNode | undefined, output: Map<string, FrontendCanvasNode>) => {
  if (!node?.id) return;
  output.set(node.id, node);
  (node.children || []).forEach((child) => walkCanvasNodes(child, output));
};

const resolveContentPath = (content: Record<string, unknown> | undefined, path: unknown): unknown => {
  const parts = String(path || '').split('.').filter(Boolean);
  let current: unknown = content || {};
  for (const part of parts) {
    if (!current || typeof current !== 'object' || !(part in current)) {
      return undefined;
    }
    current = (current as Record<string, unknown>)[part];
  }
  return current;
};

const resolveTextRef = (
  slide: FrontendSlide,
  ref: unknown,
  fallback = '',
) => {
  const value = ref ? resolveContentPath(slide.content, ref) : undefined;
  if (Array.isArray(value)) {
    return value.map((item) => String(item || '').trim()).filter(Boolean).join('\n');
  }
  if (value !== undefined && value !== null && typeof value !== 'object') {
    return String(value);
  }
  return fallback;
};

const isSystemPageLabel = (value: unknown) =>
  SYSTEM_PAGE_LABEL_RE.test(String(value || '').trim());

const getSystemPageLabel = (slide: FrontendSlide) => {
  const value = resolveTextRef(slide, 'eyebrow', '');
  return isSystemPageLabel(value) ? value.trim() : '';
};

const resolveListRef = (slide: FrontendSlide, ref: unknown, fallback: string[] = []) => {
  const value = ref ? resolveContentPath(slide.content, ref) : undefined;
  if (Array.isArray(value)) {
    return value.map((item) => String(item || '').trim()).filter(Boolean);
  }
  if (typeof value === 'string' && value.trim()) {
    return value.split(/\n+/).map((item) => item.trim()).filter(Boolean);
  }
  return fallback;
};

const normalizeTableData = (value: unknown): FrontendTableData | null => {
  if (!value || typeof value !== 'object') return null;
  const source = value as Record<string, unknown>;
  const headers = Array.isArray(source.headers)
    ? source.headers.map((item) => String(item || '').trim())
    : [];
  const rows = Array.isArray(source.rows)
    ? source.rows.map((row) =>
        Array.isArray(row)
          ? row.map((cell) => String(cell || '').trim())
          : [String(row || '').trim()],
      )
    : [];
  if (headers.length === 0 && rows.length === 0) return null;
  return { headers, rows };
};

const resolveTableData = (slide: FrontendSlide, node: FrontendCanvasNode) => {
  const props = node.props || {};
  const ref = props.table_ref || props.tableRef || props.data_ref || props.dataRef || props.ref;
  return normalizeTableData(ref ? resolveContentPath(slide.content, ref) : undefined)
    || normalizeTableData(props.table_data || props.tableData || props.data);
};

const buildAssetMap = (slide: FrontendSlide) => {
  const assets = new Map<string, FrontendVisualAsset>();
  slide.visualAssets.forEach((asset) => {
    if (asset.key) assets.set(asset.key, asset);
  });
  const contentAssets = slide.content?.assets;
  if (contentAssets && typeof contentAssets === 'object') {
    Object.entries(contentAssets as Record<string, Record<string, unknown>>).forEach(([key, raw]) => {
      if (!raw || typeof raw !== 'object') return;
      const assetKey = String(raw.asset_key || raw.assetKey || key);
      const existing = assets.get(assetKey);
      assets.set(assetKey, {
        key: assetKey,
        label: existing?.label || String(raw.label || assetKey),
        src: existing?.src || String(raw.src || ''),
        previewSrc: existing?.previewSrc || (raw.preview_src ? String(raw.preview_src) : raw.previewSrc ? String(raw.previewSrc) : undefined),
        originalSrc: existing?.originalSrc || (raw.original_src ? String(raw.original_src) : raw.originalSrc ? String(raw.originalSrc) : undefined),
        alt: existing?.alt || String(raw.alt || raw.label || assetKey),
        sourceType: existing?.sourceType || (raw.source_type === 'paper_asset' || raw.sourceType === 'paper_asset' ? 'paper_asset' : raw.source_type === 'generated' || raw.sourceType === 'generated' ? 'generated' : 'upload'),
      });
    });
  }
  return assets;
};

const toFiniteNumber = (value: unknown, fallback: number) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const clampNumber = (value: number, min: number, max: number) =>
  Math.max(min, Math.min(max, value));

const getCanvasNodeVisualStyle = (
  slide: FrontendSlide,
  node: FrontendCanvasNode,
  component: string,
): FrontendCanvasVisualStyle => {
  const visualSpec = slide.visualSpec || {};
  const nodeStyles = visualSpec.nodeStyles || {};
  const componentStyles = visualSpec.componentStyles || {};
  const rawNodeStyle = nodeStyles[node.id] || {};
  const rawComponentStyle = componentStyles[component as keyof typeof componentStyles] || {};
  const propsStyle = (node.props as Record<string, unknown> | undefined)?.visual_style
    || (node.props as Record<string, unknown> | undefined)?.visualStyle
    || {};
  return {
    ...rawComponentStyle,
    ...rawNodeStyle,
    ...(typeof propsStyle === 'object' ? propsStyle as FrontendCanvasVisualStyle : {}),
  };
};

const getComponentFontSize = (
  component: string,
  theme: ResolvedCanvasVisualTheme,
  visualStyle: FrontendCanvasVisualStyle,
) => {
  if (typeof visualStyle.fontSize === 'number' && visualStyle.fontSize > 0) {
    return visualStyle.fontSize;
  }
  if (component === 'heading') return theme.typography.titleSize;
  if (component === 'quote') return Math.max(20, Math.round(theme.typography.titleSize * 0.72));
  if (component === 'stat') return Math.max(20, Math.round(theme.typography.titleSize * 0.62));
  return theme.typography.bodySize;
};

const getComponentPadding = (
  component: string,
  theme: ResolvedCanvasVisualTheme,
  visualStyle: FrontendCanvasVisualStyle,
) => {
  if (typeof visualStyle.padding === 'number' && visualStyle.padding >= 0) {
    return visualStyle.padding;
  }
  if (component === 'figure') return 14;
  if (component === 'stat') return Math.max(16, theme.surface.cardPadding - 2);
  return theme.surface.cardPadding;
};

const estimateComponentHeight = (
  slide: FrontendSlide,
  node: FrontendCanvasNode,
  component: string,
  width: number,
  theme: ResolvedCanvasVisualTheme,
  visualStyle: FrontendCanvasVisualStyle,
) => {
  const fontSize = getComponentFontSize(component, theme, visualStyle);
  const padding = getComponentPadding(component, theme, visualStyle);
  const innerWidth = Math.max(140, width - padding * 2);
  const charsPerLine = Math.max(12, Math.floor(innerWidth / Math.max(8, fontSize * 0.56)));
  const ref = node.props || {};

  if (component === 'heading') {
    const text = resolveTextRef(slide, ref.text_ref || ref.textRef || ref.ref, String(ref.text || slide.title || ''));
    const lines = Math.max(1, Math.ceil(Math.max(1, text.length) / charsPerLine));
    return clampNumber(lines * fontSize * 1.28 + padding * 2.2, 72, 280);
  }

  if (component === 'bullets') {
    const items = resolveListRef(slide, ref.items_ref || ref.itemsRef || ref.ref, Array.isArray(ref.items) ? ref.items.map(String) : []);
    const lines = Math.max(1, items.reduce((sum, item) => sum + Math.max(1, Math.ceil(item.length / charsPerLine)), 0));
    return clampNumber(lines * fontSize * 1.28 + padding * 2.2 + Math.max(0, items.length - 1) * 6, 96, 360);
  }

  if (component === 'quote' || component === 'callout' || component === 'text') {
    const text = resolveTextRef(slide, ref.text_ref || ref.textRef || ref.ref, String(ref.text || ref.content || ''));
    const lines = Math.max(1, Math.ceil(Math.max(1, text.length) / charsPerLine));
    return clampNumber(lines * fontSize * 1.36 + padding * 2.2, 84, 320);
  }

  if (component === 'stat') {
    return clampNumber(fontSize * 2.4 + padding * 2, 112, 220);
  }

  if (component === 'figure') {
    const assetRef = String(ref.asset_ref || ref.assetRef || ref.asset_key || ref.assetKey || ref.ref || '').trim();
    const asset = buildAssetMap(slide).get(assetRef) || buildAssetMap(slide).get(String(ref.asset_key || ref.assetKey || ''));
    if (asset && asset.previewSrc) {
      return clampNumber(width * 0.66, 200, 420);
    }
    return clampNumber(width * 0.54, 180, 360);
  }

  if (component === 'table') {
    const tableData = resolveTableData(slide, node);
    const rowCount = tableData ? tableData.rows.length : 2;
    const headerCount = tableData ? tableData.headers.length : 2;
    const lineHeight = fontSize * 1.15;
    return clampNumber((rowCount + 1) * lineHeight + Math.max(1, headerCount) * 18 + padding * 2, 120, 420);
  }

  return clampNumber(fontSize * 2 + padding * 2, 72, 280);
};

const estimateCanvasNodeHeight = (
  slide: FrontendSlide,
  node: FrontendCanvasNode,
  width: number,
  theme: ResolvedCanvasVisualTheme,
): number => {
  if (node.type === 'component') {
    const component = normalizeComponent(node.component || node.props?.component || node.props?.kind);
    const visualStyle = getCanvasNodeVisualStyle(slide, node, component);
    return estimateComponentHeight(slide, node, component, width, theme, visualStyle);
  }

  const children = (node.children || []).filter(Boolean);
  if (children.length === 0) {
    return theme.surface.cardPadding * 2 + 48;
  }

  const style = node.style || {};
  const visualStyle = getCanvasNodeVisualStyle(slide, node, 'container');
  const direction = style.direction === 'row' || style.direction === 'grid' ? style.direction : 'column';
  const gap = clampNumber(toFiniteNumber(style.gap, theme.layout.contentGap), 0, 72);
  const padding = clampNumber(
    toFiniteNumber(visualStyle.padding, toFiniteNumber(style.padding, node.id === 'root' ? theme.layout.safeMargin : theme.surface.cardPadding)),
    0,
    96,
  );
  const innerWidth = Math.max(140, width - padding * 2);

  if (direction === 'row') {
    const weights = children.map((child) => Math.max(0.25, toFiniteNumber(child.style?.weight, 1)));
    const totalWeight = weights.reduce((sum, weight) => sum + weight, 0) || children.length;
    const availableWidth = Math.max(0, innerWidth - gap * Math.max(0, children.length - 1));
    const heights = children.map((child, index) => {
      const childWidth = Math.max(120, availableWidth * (weights[index] / totalWeight));
      return estimateCanvasNodeHeight(slide, child, childWidth, theme);
    });
    return Math.max(64, Math.max(...heights, 0)) + padding * 2;
  }

  if (direction === 'grid') {
    const columns = clampNumber(
      Math.round(toFiniteNumber(style.columns, theme.layout.maxColumns)),
      1,
      4,
    );
    const cellWidth = (innerWidth - gap * (columns - 1)) / columns;
    const rows = Math.max(1, Math.ceil(children.length / columns));
    let totalHeight = padding * 2;
    for (let rowIndex = 0; rowIndex < rows; rowIndex += 1) {
      const rowChildren = children.slice(rowIndex * columns, rowIndex * columns + columns);
      const rowHeights = rowChildren.map((child) => estimateCanvasNodeHeight(slide, child, cellWidth, theme));
      totalHeight += Math.max(...rowHeights, 0);
      if (rowIndex < rows - 1) totalHeight += gap;
    }
    return totalHeight;
  }

  const childHeights = children.map((child) => estimateCanvasNodeHeight(slide, child, innerWidth, theme));
  return childHeights.reduce((sum, height) => sum + height, 0) + gap * Math.max(0, children.length - 1) + padding * 2;
};

const buildComputedStyleForComponent = (
  component: string,
  theme: ResolvedCanvasVisualTheme,
  visualStyle: FrontendCanvasVisualStyle,
): ComputedStyleMap => {
  const fontSize = getComponentFontSize(component, theme, visualStyle);
  const padding = getComponentPadding(component, theme, visualStyle);
  const color = visualStyle.color
    || (component === 'heading' ? theme.palette.text : component === 'stat' ? theme.palette.accent : theme.palette.text);
  const backgroundColor = visualStyle.fill
    || (component === 'figure' || component === 'quote' || component === 'callout' || component === 'stat'
      ? theme.palette.panel
      : 'transparent');
  const borderColor = visualStyle.borderColor
    || (component === 'figure' || component === 'quote' || component === 'callout' || component === 'stat'
      ? theme.palette.primary
      : 'transparent');
  const borderWidth = typeof visualStyle.borderWidth === 'number' ? visualStyle.borderWidth : (component === 'figure' ? 1 : 0);
  const lineHeight = visualStyle.lineHeight || Math.round(fontSize * (component === 'heading' ? 1.12 : component === 'stat' ? 1.05 : 1.28));
  const fontWeight = visualStyle.fontWeight || (component === 'heading' ? 700 : component === 'stat' ? 800 : 400);
  const fontStyle = visualStyle.fontStyle || (component === 'quote' ? 'italic' : 'normal');
  const textAlign = visualStyle.textAlign || (component === 'stat' ? 'center' : 'left');
  const alignItems = textAlign === 'center' ? 'center' : 'flex-start';
  const justifyContent = component === 'stat' ? 'center' : 'flex-start';
  const verticalAlign = component === 'stat' ? 'middle' : 'top';
  const normalizedImageFit = visualStyle.imageFit === 'contain' || visualStyle.imageFit === 'fill'
    ? visualStyle.imageFit
    : 'cover';
  return {
    fontFamily: visualStyle.fontFamily
      || (component === 'heading' ? theme.typography.titleFontStack : theme.typography.bodyFontStack),
    fontSize: `${fontSize}px`,
    fontWeight: String(fontWeight),
    fontStyle,
    lineHeight: `${lineHeight}px`,
    color,
    backgroundColor,
    borderColor,
    borderTopColor: borderColor,
    borderRightColor: borderColor,
    borderBottomColor: borderColor,
    borderLeftColor: borderColor,
    borderTopWidth: `${borderWidth}px`,
    borderRightWidth: `${borderWidth}px`,
    borderBottomWidth: `${borderWidth}px`,
    borderLeftWidth: `${borderWidth}px`,
    paddingTop: `${padding}px`,
    paddingRight: `${padding}px`,
    paddingBottom: `${padding}px`,
    paddingLeft: `${padding}px`,
    textAlign,
    verticalAlign,
    display: 'flex',
    alignItems,
    justifyContent,
    opacity: visualStyle.opacity !== undefined ? visualStyle.opacity : 1,
    imageFit: normalizedImageFit,
  };
};

const buildComputedStyleForContainer = (
  node: FrontendCanvasNode,
  theme: ResolvedCanvasVisualTheme,
  visualStyle: FrontendCanvasVisualStyle,
): ComputedStyleMap => {
  const nodeStyle = node.style || {};
  const padding = typeof visualStyle.padding === 'number'
    ? visualStyle.padding
    : clampNumber(toFiniteNumber(nodeStyle.padding, node.id === 'root' ? theme.layout.safeMargin : 0), 0, 96);
  const borderColor = visualStyle.borderColor || 'transparent';
  const borderWidth = typeof visualStyle.borderWidth === 'number' ? visualStyle.borderWidth : 0;
  const textAlign = visualStyle.textAlign || 'left';
  const alignItems = nodeStyle.align === 'center'
    ? 'center'
    : nodeStyle.align === 'end'
      ? 'flex-end'
      : nodeStyle.align === 'stretch'
        ? 'stretch'
        : 'flex-start';
  const justifyContent = nodeStyle.justify === 'center'
    ? 'center'
    : nodeStyle.justify === 'end'
      ? 'flex-end'
      : nodeStyle.justify === 'between'
        ? 'space-between'
        : nodeStyle.justify === 'around'
          ? 'space-around'
          : 'flex-start';
  return {
    fontFamily: visualStyle.fontFamily || theme.typography.bodyFontStack,
    fontSize: `${visualStyle.fontSize || theme.typography.bodySize}px`,
    fontWeight: String(visualStyle.fontWeight || 400),
    fontStyle: visualStyle.fontStyle || 'normal',
    lineHeight: `${visualStyle.lineHeight || Math.round((visualStyle.fontSize || theme.typography.bodySize) * 1.28)}px`,
    color: visualStyle.color || theme.palette.text,
    backgroundColor: visualStyle.fill || 'transparent',
    borderColor,
    borderTopColor: borderColor,
    borderRightColor: borderColor,
    borderBottomColor: borderColor,
    borderLeftColor: borderColor,
    borderTopWidth: `${borderWidth}px`,
    borderRightWidth: `${borderWidth}px`,
    borderBottomWidth: `${borderWidth}px`,
    borderLeftWidth: `${borderWidth}px`,
    paddingTop: `${padding}px`,
    paddingRight: `${padding}px`,
    paddingBottom: `${padding}px`,
    paddingLeft: `${padding}px`,
    textAlign,
    verticalAlign: justifyContent === 'center' ? 'middle' : justifyContent === 'flex-end' ? 'bottom' : 'top',
    display: nodeStyle.direction === 'grid' ? 'grid' : 'flex',
    alignItems,
    justifyContent,
    opacity: visualStyle.opacity !== undefined ? visualStyle.opacity : 1,
  };
};

const buildFallbackLayoutNodes = (
  slide: FrontendSlide,
  theme: ResolvedCanvasVisualTheme,
): FrontendLayoutIRNode[] => {
  const root = slide.root;
  if (!root) {
    return [];
  }
  const layoutNodes: FrontendLayoutIRNode[] = [];

  const visit = (node: FrontendCanvasNode, box: PptBox): void => {
    if (!node?.id) return;
    if (node.type === 'component') {
      const component = normalizeComponent(node.component || node.props?.component || node.props?.kind);
      const visualStyle = getCanvasNodeVisualStyle(slide, node, component);
      layoutNodes.push({
        nodeId: node.id,
        type: 'component',
        component: component as FrontendLayoutIRNode['component'],
        box: {
          x: Math.round(box.x),
          y: Math.round(box.y),
          w: Math.round(box.w),
          h: Math.round(box.h),
        },
        computedStyle: buildComputedStyleForComponent(component, theme, visualStyle),
        overflow: false,
      });
      return;
    }

    const children = (node.children || []).filter(Boolean);
    const style = node.style || {};
    const visualStyle = getCanvasNodeVisualStyle(slide, node, 'container');
    layoutNodes.push({
      nodeId: node.id,
      type: 'container',
      box: {
        x: Math.round(box.x),
        y: Math.round(box.y),
        w: Math.round(box.w),
        h: Math.round(box.h),
      },
      computedStyle: buildComputedStyleForContainer(node, theme, visualStyle),
      overflow: false,
    });
    const direction = style.direction === 'row' || style.direction === 'grid' ? style.direction : 'column';
    const gap = clampNumber(toFiniteNumber(style.gap, theme.layout.contentGap), 0, 72);
    const padding = clampNumber(
      toFiniteNumber(visualStyle.padding, toFiniteNumber(style.padding, node.id === 'root' ? theme.layout.safeMargin : theme.surface.cardPadding)),
      0,
      96,
    );
    const inner = {
      x: box.x + padding,
      y: box.y + padding,
      w: Math.max(0, box.w - padding * 2),
      h: Math.max(0, box.h - padding * 2),
    };

    if (children.length === 0) {
      return;
    }

    if (direction === 'row') {
      const weights = children.map((child) => Math.max(0.25, toFiniteNumber(child.style?.weight, 1)));
      const totalWeight = weights.reduce((sum, weight) => sum + weight, 0) || children.length;
      const availableWidth = Math.max(0, inner.w - gap * Math.max(0, children.length - 1));
      let cursorX = inner.x;
      children.forEach((child, index) => {
        const childWidth = index === children.length - 1
          ? Math.max(0, inner.x + inner.w - cursorX)
          : Math.max(120, availableWidth * (weights[index] / totalWeight));
        visit(child, {
          x: cursorX,
          y: inner.y,
          w: childWidth,
          h: inner.h,
        });
        cursorX += childWidth + gap;
      });
      return;
    }

    if (direction === 'grid') {
      const columns = clampNumber(
        Math.round(toFiniteNumber(style.columns, theme.layout.maxColumns)),
        1,
        4,
      );
      const rows = Math.max(1, Math.ceil(children.length / columns));
      const cellWidth = (inner.w - gap * (columns - 1)) / columns;
      const rowHeights = Array.from({ length: rows }, (_, rowIndex) => {
        const rowChildren = children.slice(rowIndex * columns, rowIndex * columns + columns);
        return Math.max(...rowChildren.map((child) => estimateCanvasNodeHeight(slide, child, cellWidth, theme)), 0);
      });
      let cursorY = inner.y;
      for (let rowIndex = 0; rowIndex < rows; rowIndex += 1) {
        const rowChildren = children.slice(rowIndex * columns, rowIndex * columns + columns);
        let cursorX = inner.x;
        const rowHeight = rowHeights[rowIndex] || 0;
        rowChildren.forEach((child) => {
          visit(child, {
            x: cursorX,
            y: cursorY,
            w: cellWidth,
            h: rowHeight,
          });
          cursorX += cellWidth + gap;
        });
        cursorY += rowHeight + gap;
      }
      return;
    }

    const estimatedHeights = children.map((child) => estimateCanvasNodeHeight(slide, child, inner.w, theme));
    const totalEstimated = estimatedHeights.reduce((sum, item) => sum + item, 0) || children.length;
    const availableHeight = Math.max(0, inner.h - gap * Math.max(0, children.length - 1));
    let cursorY = inner.y;
    children.forEach((child, index) => {
      const childHeight = index === children.length - 1
        ? Math.max(0, inner.y + inner.h - cursorY)
        : Math.max(64, availableHeight * (estimatedHeights[index] / totalEstimated));
      visit(child, {
        x: inner.x,
        y: cursorY,
        w: inner.w,
        h: childHeight,
      });
      cursorY += childHeight + gap;
    });
  };

  visit(root, {
    x: 0,
    y: 0,
    w: DESIGN_WIDTH,
    h: DESIGN_HEIGHT,
  });

  return layoutNodes;
};

const blobToDataUrl = (blob: Blob) =>
  new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ''));
    reader.onerror = () => reject(reader.error || new Error('Failed to read image data'));
    reader.readAsDataURL(blob);
  });

const resolveImageData = async (asset: FrontendVisualAsset | undefined) => {
  const src = (asset?.originalSrc || asset?.src || asset?.previewSrc || '').trim();
  if (!src) return '';
  if (src.startsWith('data:')) return src;
  const response = await fetch(src);
  if (!response.ok) {
    throw new Error(`图片资源读取失败：${asset?.label || asset?.key || src}`);
  }
  return blobToDataUrl(await response.blob());
};

const getImageDimensions = (dataUrl: string) =>
  new Promise<{ width: number; height: number }>((resolve) => {
    if (typeof Image === 'undefined' || !dataUrl) {
      resolve({ width: 0, height: 0 });
      return;
    }
    const image = new Image();
    image.onload = () => resolve({ width: image.naturalWidth || image.width || 0, height: image.naturalHeight || image.height || 0 });
    image.onerror = () => resolve({ width: 0, height: 0 });
    image.src = dataUrl;
  });

const fitImageInsideBox = async (dataUrl: string, box: PptBox): Promise<PptBox> => {
  const dimensions = await getImageDimensions(dataUrl);
  if (!dimensions.width || !dimensions.height || !box.w || !box.h) {
    return box;
  }
  const imageRatio = dimensions.width / dimensions.height;
  const boxRatio = box.w / box.h;
  if (!Number.isFinite(imageRatio) || imageRatio <= 0 || !Number.isFinite(boxRatio) || boxRatio <= 0) {
    return box;
  }
  if (imageRatio > boxRatio) {
    const fittedHeight = box.w / imageRatio;
    return { ...box, y: box.y + (box.h - fittedHeight) / 2, h: fittedHeight };
  }
  const fittedWidth = box.h * imageRatio;
  return { ...box, x: box.x + (box.w - fittedWidth) / 2, w: fittedWidth };
};

const addPanelShape = (
  slide: PptxGenJS.Slide,
  pptx: PptxGenJS,
  box: PptBox,
  fillColor: string,
  lineColor: string,
  transparency = 8,
) => {
  slide.addShape(pptx.ShapeType.roundRect, {
    ...box,
    rectRadius: 0.08,
    fill: { color: fillColor, transparency },
    line: { color: lineColor, transparency: 45, width: 0.8 },
  });
};

const renderCanvasContainerFrame = (
  pptSlide: PptxGenJS.Slide,
  pptx: PptxGenJS,
  layoutNode: FrontendLayoutIRNode,
  theme: ResolvedTheme,
) => {
  const box = toPptBox(layoutNode);
  const background = styleValue(layoutNode, 'backgroundColor');
  const borderWidth = Math.max(
    parseCssPx(styleValue(layoutNode, 'borderTopWidth'), 0),
    parseCssPx(styleValue(layoutNode, 'borderRightWidth'), 0),
    parseCssPx(styleValue(layoutNode, 'borderBottomWidth'), 0),
    parseCssPx(styleValue(layoutNode, 'borderLeftWidth'), 0),
  );
  const hasBackground = Boolean(background && background !== 'transparent' && background !== 'rgba(0, 0, 0, 0)');
  const hasBorder = borderWidth > 0;
  if (!hasBackground && !hasBorder) return;
  pptSlide.addShape(pptx.ShapeType.roundRect, {
    ...box,
    rectRadius: 0.04,
    fill: {
      color: hasBackground ? toHexColor(background, theme.palette.panel) : toHexColor(theme.palette.panel, '#0F172A'),
      transparency: hasBackground ? 8 : 100,
    },
    line: {
      color: hasBorder ? colorFromStyle(layoutNode, 'borderColor', theme.palette.primary) : toHexColor(theme.palette.primary, '#7DD3FC'),
      transparency: hasBorder ? 35 : 100,
      width: Math.max(0.2, Math.min(3, pxToPt(borderWidth))),
    },
  });
};

const renderFixedPageLabel = (
  pptSlide: PptxGenJS.Slide,
  label: string,
  theme: ResolvedTheme,
) => {
  if (!label.trim()) return;
  pptSlide.addText(label.trim(), {
    x: SLIDE_WIDTH_IN - 1.32,
    y: SLIDE_HEIGHT_IN - 0.42,
    w: 1.06,
    h: 0.18,
    margin: [0, 2, 0, 2],
    fontFace: firstFont(theme.typography.bodyFontStack, 'Arial'),
    fontSize: 7.5,
    bold: true,
    color: toHexColor(theme.palette.muted, '#94A3B8'),
    align: 'right',
    valign: 'middle',
    breakLine: false,
    fit: 'shrink',
  });
};

const addText = (
  slide: PptxGenJS.Slide,
  text: string,
  box: PptBox,
  options: PptxGenJS.TextPropsOptions,
) => {
  const normalized = text.trim();
  if (!normalized) return;
  slide.addText(normalized, {
    ...box,
    margin: [2, 4, 2, 4],
    breakLine: false,
    fit: 'shrink',
    wrap: true,
    ...options,
  });
};

const addBullets = (
  slide: PptxGenJS.Slide,
  items: string[],
  box: PptBox,
  theme: ResolvedTheme,
  layoutNode: FrontendLayoutIRNode,
) => {
  const bodyFont = fontFaceFromStyle(layoutNode, firstFont(theme.typography.bodyFontStack, 'Arial'));
  const fontSize = fontSizeFromStyle(layoutNode, theme.typography.bodySize - 1);
  const textColor = colorFromStyle(layoutNode, 'color', theme.palette.text);
  const runs: PptxGenJS.TextProps[] = items.map((item, index) => ({
    text: item,
    options: {
      bullet: { type: 'bullet', indent: Math.max(10, fontSize * 0.85) },
      breakLine: index < items.length - 1,
      hanging: Math.max(3, fontSize * 0.22),
    },
  }));
  slide.addText(runs, {
    ...box,
    margin: marginFromPadding(layoutNode),
    fontFace: bodyFont,
    fontSize,
    color: textColor,
    breakLine: false,
    fit: 'shrink',
    lineSpacing: lineSpacingFromStyle(layoutNode, fontSize),
    paraSpaceAfter: Math.max(2, fontSize * 0.25),
    valign: verticalAlignFromStyle(layoutNode),
    wrap: true,
  });
};

const addTable = (
  slide: PptxGenJS.Slide,
  tableData: FrontendTableData,
  box: PptBox,
  theme: ResolvedTheme,
) => {
  const primary = toHexColor(theme.palette.primary, '#7DD3FC');
  const text = toHexColor(theme.palette.text, '#E2E8F0');
  const panel = toHexColor(theme.palette.panel, '#0F172A');
  const bodyFont = firstFont(theme.typography.bodyFontStack, 'Arial');
  const rows = [
    ...(tableData.headers.length > 0 ? [tableData.headers] : []),
    ...tableData.rows,
  ];
  if (rows.length === 0) return;
  const tableRows: PptxGenJS.TableRow[] = rows.map((row, rowIndex) =>
    row.map((cell) => ({
      text: String(cell || ''),
      options: {
        fontFace: bodyFont,
        fontSize: Math.max(8, Math.min(15, theme.typography.bodySize - 8)),
        color: rowIndex === 0 && tableData.headers.length > 0 ? primary : text,
        bold: rowIndex === 0 && tableData.headers.length > 0,
        fill: { color: panel, transparency: rowIndex === 0 && tableData.headers.length > 0 ? 10 : 28 },
        border: { type: 'solid', color: primary, pt: 0.6 },
        margin: 0.06,
        valign: 'middle',
      },
    })),
  );
  const columnCount = Math.max(...rows.map((row) => row.length), 1);
  slide.addTable(tableRows, {
    ...box,
    colW: Array.from({ length: columnCount }, () => box.w / columnCount),
    border: { type: 'solid', color: primary, pt: 0.6 },
    margin: 0.04,
    autoPage: false,
  });
};

const renderCanvasComponent = async (
  pptSlide: PptxGenJS.Slide,
  pptx: PptxGenJS,
  sourceSlide: FrontendSlide,
  node: FrontendCanvasNode,
  layoutNode: FrontendLayoutIRNode,
  theme: ResolvedTheme,
  assets: Map<string, FrontendVisualAsset>,
) => {
  const component = normalizeComponent(node.component || node.props?.component || node.props?.kind || layoutNode.component);
  const props = node.props || {};
  const visualStyle = getCanvasNodeVisualStyle(sourceSlide, node, component);
  const box = toPptBox(layoutNode);
  const titleFont = firstFont(theme.typography.titleFontStack, 'Georgia');
  const bodyFont = firstFont(theme.typography.bodyFontStack, 'Arial');
  const textColor = toHexColor(theme.palette.text, '#E2E8F0');
  const mutedColor = toHexColor(theme.palette.muted, '#94A3B8');
  const primaryColor = toHexColor(theme.palette.primary, '#7DD3FC');
  const accentColor = toHexColor(theme.palette.accent, '#F59E0B');
  const panelColor = toHexColor(theme.palette.panel, '#0F172A');
  const rawTextRef = props.text_ref || props.textRef || props.ref;
  if (String(rawTextRef || '').trim() === 'eyebrow' && isSystemPageLabel(resolveTextRef(sourceSlide, rawTextRef, String(props.text || props.content || '')))) {
    return;
  }

  if (component === 'heading') {
    const fontSize = fontSizeFromStyle(layoutNode, theme.typography.titleSize);
    addText(
      pptSlide,
      resolveTextRef(sourceSlide, rawTextRef, String(props.text || sourceSlide.title || 'Untitled')),
      insetBoxForText(box),
      {
        fontFace: fontFaceFromStyle(layoutNode, titleFont),
        fontSize,
        bold: isBoldStyle(layoutNode, true),
        color: colorFromStyle(layoutNode, 'color', textColor),
        lineSpacing: lineSpacingFromStyle(layoutNode, fontSize),
        margin: marginFromPadding(layoutNode),
        align: alignFromStyle(layoutNode),
        valign: verticalAlignFromStyle(layoutNode),
      },
    );
    return;
  }

  if (component === 'bullets') {
    addBullets(
      pptSlide,
      resolveListRef(sourceSlide, props.items_ref || props.itemsRef || props.ref, Array.isArray(props.items) ? props.items.map(String) : []),
      insetBoxForText(box),
      theme,
      layoutNode,
    );
    return;
  }

  if (component === 'figure') {
    const assetRef = String(props.asset_ref || props.assetRef || props.asset_key || props.assetKey || props.ref || '').trim();
    const asset = assets.get(assetRef) || assets.get(String(props.asset_key || props.assetKey || ''));
    const isPaperAsset = asset?.sourceType === 'paper_asset';
    const isGeneratedAsset = asset?.sourceType === 'generated';
    const useTransparentImage = isPaperAsset || isGeneratedAsset;
    const imageFit = useTransparentImage
      ? 'contain'
      : visualStyle.imageFit === 'contain' || visualStyle.imageFit === 'fill'
        ? visualStyle.imageFit
        : 'cover';
    if (!useTransparentImage) {
      addPanelShape(pptSlide, pptx, box, panelColor, primaryColor, 18);
    }
    try {
      const data = await resolveImageData(asset);
      if (data) {
        if (useTransparentImage || imageFit === 'contain') {
          const fittedBox = await fitImageInsideBox(data, box);
          pptSlide.addImage({
            data,
            ...fittedBox,
            altText: asset?.alt || asset?.label || assetRef || 'Slide image',
          });
        } else {
          pptSlide.addImage({
            data,
            ...box,
            sizing: {
              type: 'cover',
              x: box.x,
              y: box.y,
              w: box.w,
              h: box.h,
            },
            altText: asset?.alt || asset?.label || assetRef || 'Slide image',
          });
        }
      }
    } catch {
      const fontSize = fontSizeFromStyle(layoutNode, 18);
      addText(pptSlide, asset?.label || 'Image unavailable', box, {
        fontFace: fontFaceFromStyle(layoutNode, bodyFont),
        fontSize,
        color: mutedColor,
        align: 'center',
        valign: 'middle',
      });
    }
    return;
  }

  if (component === 'table') {
    const tableData = resolveTableData(sourceSlide, node);
    if (tableData) {
      addTable(pptSlide, tableData, box, theme);
    }
    return;
  }

  if (component === 'stat') {
    addPanelShape(pptSlide, pptx, box, panelColor, accentColor, 12);
    const value = resolveTextRef(sourceSlide, props.value_ref || props.valueRef || props.ref, String(props.value || ''));
    const label = resolveTextRef(sourceSlide, props.label_ref || props.labelRef, String(props.label || ''));
    const valueBox = { ...box, h: Math.max(0.2, box.h * 0.58) };
    const labelBox = { ...box, y: box.y + box.h * 0.58, h: Math.max(0.2, box.h * 0.34) };
    const statFontSize = fontSizeFromStyle(layoutNode, theme.typography.titleSize * 0.72);
    addText(pptSlide, value, valueBox, {
      fontFace: fontFaceFromStyle(layoutNode, titleFont),
      fontSize: statFontSize,
      bold: isBoldStyle(layoutNode, true),
      color: accentColor,
      valign: 'middle',
      margin: marginFromPadding(layoutNode),
    });
    addText(pptSlide, label, labelBox, {
      fontFace: bodyFont,
      fontSize: Math.max(8, statFontSize * 0.38),
      color: mutedColor,
      valign: 'top',
      margin: marginFromPadding(layoutNode),
    });
    return;
  }

  const value = resolveTextRef(sourceSlide, rawTextRef, String(props.text || props.content || ''));
  if (component === 'quote' || component === 'callout') {
    addPanelShape(pptSlide, pptx, box, panelColor, component === 'quote' ? primaryColor : accentColor, 14);
  }
  const fontSize = fontSizeFromStyle(
    layoutNode,
    component === 'quote' ? theme.typography.titleSize * 0.72 : theme.typography.bodySize,
  );
  addText(pptSlide, value, box, {
    fontFace: fontFaceFromStyle(layoutNode, component === 'quote' ? titleFont : bodyFont),
    fontSize,
    bold: isBoldStyle(layoutNode),
    italic: isItalicStyle(layoutNode, component === 'quote'),
    color: colorFromStyle(layoutNode, 'color', textColor),
    lineSpacing: lineSpacingFromStyle(layoutNode, fontSize),
    margin: marginFromPadding(layoutNode),
    align: alignFromStyle(layoutNode),
    valign: verticalAlignFromStyle(layoutNode),
  });
};

export const buildCanvasSlidesPptxBlob = async (
  slides: FrontendSlide[],
  deckTheme?: FrontendDeckTheme | null,
) => {
  const pptx = new PptxGenJS();
  pptx.defineLayout({ name: 'PAPER2PPT_CANVAS_WIDE', width: SLIDE_WIDTH_IN, height: SLIDE_HEIGHT_IN });
  pptx.layout = 'PAPER2PPT_CANVAS_WIDE';
  pptx.author = 'Paper2Any';
  pptx.company = 'Paper2Any';
  pptx.subject = 'Editable Canvas PPT export';
  pptx.title = 'paper2ppt_editable';

  const theme = resolveCanvasVisualTheme(deckTheme, slides[0]?.visualSpec || null);
  pptx.theme = {
    headFontFace: firstFont(theme.typography.titleFontStack, 'Georgia'),
    bodyFontFace: firstFont(theme.typography.bodyFontStack, 'Arial'),
  };

  for (const sourceSlide of slides) {
    const slideTheme = resolveCanvasVisualTheme(deckTheme, sourceSlide.visualSpec || null);
    const pptSlide = pptx.addSlide();
    pptSlide.background = { color: toHexColor(slideTheme.palette.bg, '#0B1020') };
    const nodeMap = new Map<string, FrontendCanvasNode>();
    walkCanvasNodes(sourceSlide.root, nodeMap);
    const measuredLayoutNodes = (sourceSlide.layoutIr?.nodes || [])
      .filter((item) => nodeMap.has(item.nodeId));
    const measuredIds = new Set(measuredLayoutNodes.map((item) => item.nodeId));
    const fallbackLayoutNodes = buildFallbackLayoutNodes(sourceSlide, slideTheme)
      .filter((item) => nodeMap.has(item.nodeId) && !measuredIds.has(item.nodeId));
    const layoutNodes = [...measuredLayoutNodes, ...fallbackLayoutNodes];
    const assets = buildAssetMap(sourceSlide);

    for (const layoutNode of layoutNodes.filter((item) => item.type === 'container')) {
      renderCanvasContainerFrame(pptSlide, pptx, layoutNode, slideTheme);
    }

    for (const layoutNode of layoutNodes) {
      if (layoutNode.type !== 'component') continue;
      const node = nodeMap.get(layoutNode.nodeId);
      if (!node) continue;
      await renderCanvasComponent(pptSlide, pptx, sourceSlide, node, layoutNode, slideTheme, assets);
    }
    renderFixedPageLabel(pptSlide, getSystemPageLabel(sourceSlide), slideTheme);
  }

  const output = await pptx.write({ outputType: 'blob', compression: true });
  return output instanceof Blob
    ? output
    : new Blob([output as BlobPart], {
        type: 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
      });
};

export const writeCanvasSlidesToPptx = async (
  slides: FrontendSlide[],
  deckTheme?: FrontendDeckTheme | null,
  fileName = 'paper2ppt_editable.pptx',
) => {
  const blob = await buildCanvasSlidesPptxBlob(slides, deckTheme);
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = fileName;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
  return blob;
};

export const canExportCanvasSlidesToPptx = (slides: FrontendSlide[]) =>
  slides.length > 0 &&
  slides.every((slide) =>
    slide.renderEngine === 'canvas' &&
    Boolean(slide.root),
  );
