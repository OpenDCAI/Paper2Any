import PptxGenJS from 'pptxgenjs';
import {
  FrontendCanvasNode,
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

const toPptBox = (node: FrontendLayoutIRNode): PptBox => ({
  x: pxToIn(node.box.x, 'x'),
  y: pxToIn(node.box.y, 'y'),
  w: Math.max(0.08, pxToIn(node.box.w, 'x')),
  h: Math.max(0.08, pxToIn(node.box.h, 'y')),
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
      assets.set(assetKey, {
        key: assetKey,
        label: String(raw.label || assetKey),
        src: String(raw.src || ''),
        previewSrc: raw.preview_src ? String(raw.preview_src) : raw.previewSrc ? String(raw.previewSrc) : undefined,
        originalSrc: raw.original_src ? String(raw.original_src) : raw.originalSrc ? String(raw.originalSrc) : undefined,
        alt: String(raw.alt || raw.label || assetKey),
        sourceType: 'upload',
      });
    });
  }
  return assets;
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
) => {
  const bodyFont = firstFont(theme.typography.bodyFontStack, 'Arial');
  const textColor = toHexColor(theme.palette.text, '#E2E8F0');
  const runs: PptxGenJS.TextProps[] = items.map((item, index) => ({
    text: item,
    options: {
      bullet: { type: 'bullet', indent: 16 },
      breakLine: index < items.length - 1,
      hanging: 4,
    },
  }));
  slide.addText(runs, {
    ...box,
    margin: [4, 8, 4, 10],
    fontFace: bodyFont,
    fontSize: Math.max(12, Math.min(24, theme.typography.bodySize - 3)),
    color: textColor,
    breakLine: false,
    fit: 'shrink',
    paraSpaceAfter: 7,
    valign: 'top',
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
  const box = toPptBox(layoutNode);
  const titleFont = firstFont(theme.typography.titleFontStack, 'Georgia');
  const bodyFont = firstFont(theme.typography.bodyFontStack, 'Arial');
  const textColor = toHexColor(theme.palette.text, '#E2E8F0');
  const mutedColor = toHexColor(theme.palette.muted, '#94A3B8');
  const primaryColor = toHexColor(theme.palette.primary, '#7DD3FC');
  const accentColor = toHexColor(theme.palette.accent, '#F59E0B');
  const panelColor = toHexColor(theme.palette.panel, '#0F172A');

  if (component === 'heading') {
    addText(
      pptSlide,
      resolveTextRef(sourceSlide, props.text_ref || props.textRef || props.ref, String(props.text || sourceSlide.title || 'Untitled')),
      box,
      {
        fontFace: titleFont,
        fontSize: Math.max(24, Math.min(48, theme.typography.titleSize * Math.min(1, box.h / 0.85))),
        bold: true,
        color: textColor,
        valign: 'middle',
      },
    );
    return;
  }

  if (component === 'bullets') {
    addBullets(
      pptSlide,
      resolveListRef(sourceSlide, props.items_ref || props.itemsRef || props.ref, Array.isArray(props.items) ? props.items.map(String) : []),
      box,
      theme,
    );
    return;
  }

  if (component === 'figure') {
    const assetRef = String(props.asset_ref || props.assetRef || props.asset_key || props.assetKey || props.ref || '').trim();
    const asset = assets.get(assetRef) || assets.get(String(props.asset_key || props.assetKey || ''));
    addPanelShape(pptSlide, pptx, box, panelColor, primaryColor, 18);
    try {
      const data = await resolveImageData(asset);
      if (data) {
        pptSlide.addImage({
          data,
          ...box,
          sizing: {
            type: 'cover',
            w: box.w,
            h: box.h,
          },
          altText: asset?.alt || asset?.label || assetRef || 'Slide image',
        });
      }
    } catch {
      addText(pptSlide, asset?.label || 'Image unavailable', box, {
        fontFace: bodyFont,
        fontSize: 14,
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
    addText(pptSlide, value, valueBox, {
      fontFace: titleFont,
      fontSize: Math.max(20, Math.min(42, theme.typography.titleSize * 0.7)),
      bold: true,
      color: accentColor,
      valign: 'middle',
    });
    addText(pptSlide, label, labelBox, {
      fontFace: bodyFont,
      fontSize: Math.max(10, Math.min(18, theme.typography.bodySize - 5)),
      color: mutedColor,
      valign: 'top',
    });
    return;
  }

  const value = resolveTextRef(sourceSlide, props.text_ref || props.textRef || props.ref, String(props.text || props.content || ''));
  if (component === 'quote' || component === 'callout') {
    addPanelShape(pptSlide, pptx, box, panelColor, component === 'quote' ? primaryColor : accentColor, 14);
  }
  addText(pptSlide, value, box, {
    fontFace: component === 'quote' ? titleFont : bodyFont,
    fontSize: component === 'quote'
      ? Math.max(16, Math.min(34, theme.typography.titleSize * 0.55))
      : Math.max(12, Math.min(24, theme.typography.bodySize)),
    italic: component === 'quote',
    color: textColor,
    valign: 'top',
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

  const theme = resolveTheme(deckTheme);
  pptx.theme = {
    headFontFace: firstFont(theme.typography.titleFontStack, 'Georgia'),
    bodyFontFace: firstFont(theme.typography.bodyFontStack, 'Arial'),
  };

  for (const sourceSlide of slides) {
    const pptSlide = pptx.addSlide();
    pptSlide.background = { color: toHexColor(theme.palette.bg, '#0B1020') };
    const nodeMap = new Map<string, FrontendCanvasNode>();
    walkCanvasNodes(sourceSlide.root, nodeMap);
    const layoutNodes = (sourceSlide.layoutIr?.nodes || [])
      .filter((item) => item.type === 'component' && nodeMap.has(item.nodeId));
    const assets = buildAssetMap(sourceSlide);

    for (const layoutNode of layoutNodes) {
      const node = nodeMap.get(layoutNode.nodeId);
      if (!node) continue;
      await renderCanvasComponent(pptSlide, pptx, sourceSlide, node, layoutNode, theme, assets);
    }
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
    slide.root &&
    slide.layoutIr?.nodes?.some((node) => node.type === 'component'),
  );
