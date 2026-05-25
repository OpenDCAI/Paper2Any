import {
  FrontendDeckPalette,
  FrontendDeckTheme,
  FrontendDeckTypography,
  FrontendEditableField,
  FrontendBlockChild,
  FrontendCanvasNode,
  FrontendCanvasVisualSpec,
  FrontendCanvasVisualStyle,
  FrontendSlide,
  FrontendSlideBlock,
  FrontendVisualAsset,
} from './types';

export const SUPPORTED_SCHEMA_TEMPLATES = [
  'title_cover',
  'section_divider',
  'text_focus',
  'hero_visual',
  'split_media',
  'visual_compare',
  'insight_grid',
  'metrics_dashboard',
  'timeline_overview',
  'stacked_cards',
  'quote_focus',
  'dual_list',
] as const;

const DEFAULT_PALETTE: FrontendDeckPalette = {
  bg: '#0b1020',
  panel: 'rgba(15, 23, 42, 0.92)',
  primary: '#7dd3fc',
  secondary: '#38bdf8',
  accent: '#f59e0b',
  text: '#e2e8f0',
  muted: '#94a3b8',
};

const DEFAULT_TYPOGRAPHY: FrontendDeckTypography = {
  titleFontStack: 'Georgia, "Times New Roman", serif',
  bodyFontStack: '"Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif',
  eyebrowSize: 18,
  titleSize: 56,
  summarySize: 26,
  bodySize: 24,
};

const PREFERRED_FIELD_KEYS: Record<string, string> = {
  title: 'title',
  summary: 'summary',
  key_points: 'key_points',
  takeaway: 'takeaway',
  footer: 'footer',
  eyebrow: 'eyebrow',
};

const SYSTEM_PAGE_LABEL_RE = /^(?:slide\s*)?\d{1,3}\s*\/\s*\d{1,3}$|^第\s*\d{1,3}\s*\/\s*\d{1,3}\s*页$/i;

const escapeHtml = (value: string) =>
  value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');

const formatTextValue = (value: string) => escapeHtml(value).replace(/\n/g, '<br />');

const slugify = (value: string) =>
  value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');

const buildFieldMap = (slide: FrontendSlide) =>
  new Map(slide.editableFields.map((field) => [field.key, field]));

const buildAssetMap = (slide: FrontendSlide) =>
  new Map(slide.visualAssets.map((asset) => [asset.key, asset]));

const resolveContentPath = (content: Record<string, unknown> | undefined, rawPath: unknown): unknown => {
  const path = String(rawPath || '').trim();
  if (!path || !content) return undefined;
  return path.split('.').reduce<unknown>((current, part) => {
    if (!current || typeof current !== 'object') return undefined;
    return (current as Record<string, unknown>)[part];
  }, content);
};

const resolveTextRef = (slide: FrontendSlide, rawRef: unknown, fallback = '') => {
  const ref = String(rawRef || '').trim();
  const field = ref ? buildFieldMap(slide).get(ref) : undefined;
  if (field) {
    if (field.type === 'list') return field.items.filter((item) => item.trim()).join(' • ');
    return field.value || fallback;
  }
  const value = resolveContentPath(slide.content, ref);
  if (Array.isArray(value)) return value.map((item) => String(item || '').trim()).filter(Boolean).join(' • ');
  if (typeof value === 'string' || typeof value === 'number') return String(value);
  return fallback;
};

const isSystemPageLabel = (value: unknown) =>
  SYSTEM_PAGE_LABEL_RE.test(String(value || '').trim());

const getSystemPageLabel = (slide: FrontendSlide) => {
  const value = resolveTextRef(slide, 'eyebrow', '');
  return isSystemPageLabel(value) ? value.trim() : '';
};

const renderFixedPageLabel = (slide: FrontendSlide) => {
  const label = getSystemPageLabel(slide);
  return label ? `<div class="schema-fixed-page-label">${formatTextValue(label)}</div>` : '';
};

const appendFixedPageLabel = (markup: string, slide: FrontendSlide) => {
  const label = renderFixedPageLabel(slide);
  return label ? markup.replace(/<\/div>\s*$/, `${label}</div>`) : markup;
};

const resolveListRef = (slide: FrontendSlide, rawRef: unknown, fallback: string[] = []) => {
  const ref = String(rawRef || '').trim();
  const field = ref ? buildFieldMap(slide).get(ref) : undefined;
  if (field) {
    if (field.type === 'list') return field.items.filter((item) => item.trim());
    return field.value ? [field.value] : fallback;
  }
  const value = resolveContentPath(slide.content, ref);
  if (Array.isArray(value)) return value.map((item) => String(item || '').trim()).filter(Boolean);
  if (typeof value === 'string' && value.trim()) {
    return value
      .split(/\n|•/g)
      .map((item) => item.replace(/^[\s-]+/, '').trim())
      .filter(Boolean);
  }
  return fallback;
};

const toFiniteCssNumber = (value: unknown, fallback?: number) => {
  const num = Number(value);
  return Number.isFinite(num) ? num : fallback;
};

const buildCanvasStyle = (node: FrontendCanvasNode) => {
  const style = node.style || {};
  const rules: string[] = [];
  if (node.type === 'container') {
    if (style.direction === 'grid') {
      const columns = Math.max(1, Math.min(4, Math.round(toFiniteCssNumber(style.columns, 2) || 2)));
      rules.push('display:grid', `grid-template-columns:repeat(${columns},minmax(0,1fr))`);
    } else {
      rules.push('display:flex', `flex-direction:${style.direction === 'row' ? 'row' : 'column'}`);
      if (style.wrap) rules.push('flex-wrap:wrap');
    }
  }
  const gap = toFiniteCssNumber(style.gap);
  if (typeof gap === 'number') rules.push(`gap:${Math.max(0, Math.min(72, gap))}px`);
  const padding = toFiniteCssNumber(style.padding);
  if (typeof padding === 'number') rules.push(`padding:${Math.max(0, Math.min(96, padding))}px`);
  const weight = toFiniteCssNumber(style.weight);
  if (typeof weight === 'number') rules.push(`flex:${Math.max(0, Math.min(8, weight))} 1 0`);
  if (typeof style.basis === 'number') rules.push(`flex-basis:${Math.max(0, Math.min(1600, style.basis))}px`);
  if (typeof style.basis === 'string' && /^(auto|\d+(?:px|%)?)$/.test(style.basis)) rules.push(`flex-basis:${style.basis}`);
  if (style.align) rules.push(`align-items:${style.align === 'start' ? 'flex-start' : style.align === 'end' ? 'flex-end' : style.align}`);
  if (style.justify) {
    const justify = style.justify === 'between' ? 'space-between' : style.justify === 'around' ? 'space-around' : style.justify === 'start' ? 'flex-start' : style.justify === 'end' ? 'flex-end' : style.justify;
    rules.push(`justify-content:${justify}`);
  }
  const minWidth = toFiniteCssNumber(style.minWidth);
  if (typeof minWidth === 'number') rules.push(`min-width:${Math.max(0, Math.min(1000, minWidth))}px`);
  const maxWidth = toFiniteCssNumber(style.maxWidth);
  if (typeof maxWidth === 'number') rules.push(`max-width:${Math.max(0, Math.min(1600, maxWidth))}px`);
  return rules.join(';');
};

const renderCanvasPlaceholder = (node: FrontendCanvasNode, label: string) => `
<div class="schema-canvas-placeholder" data-block-id="${escapeHtml(node.id)}" data-block-role="placeholder" data-canvas-node-id="${escapeHtml(node.id)}">
  ${escapeHtml(label)}
</div>
`.trim();

const normalizeCanvasComponent = (rawComponent: unknown) => {
  const component = slugify(String(rawComponent || 'placeholder'));
  const aliases: Record<string, string> = {
    h1: 'heading',
    h2: 'heading',
    title: 'heading',
    subtitle: 'text',
    paragraph: 'text',
    body: 'text',
    body_text: 'text',
    bullet_list: 'bullets',
    bullet_points: 'bullets',
    key_points: 'bullets',
    list: 'bullets',
    points: 'bullets',
    image: 'figure',
    visual: 'figure',
    chart: 'figure',
    diagram: 'figure',
    table_card: 'table',
    data_table: 'table',
    metric: 'stat',
    number: 'stat',
    kpi: 'stat',
    card: 'callout',
    note: 'callout',
    insight: 'callout',
    timeline: 'bullets',
    timeline_item: 'text',
  };
  const normalized = aliases[component] || component;
  return ['heading', 'text', 'bullets', 'quote', 'stat', 'callout', 'figure', 'table', 'placeholder'].includes(normalized)
    ? normalized
    : 'text';
};

const normalizeCanvasTableData = (value: unknown) => {
  const source = value && typeof value === 'object' ? value as Record<string, unknown> : {};
  const headers = Array.isArray(source.headers)
    ? source.headers.map((item) => String(item || '').trim()).filter(Boolean)
    : Array.isArray(source.columns)
      ? source.columns.map((item) => String(item || '').trim()).filter(Boolean)
      : [];
  const rows = Array.isArray(source.rows)
    ? source.rows
        .filter((row): row is unknown[] => Array.isArray(row))
        .map((row) => row.map((cell) => String(cell ?? '').trim()))
        .filter((row) => row.length > 0)
    : [];
  const maxCols = Math.max(headers.length, ...rows.map((row) => row.length), 0);
  if (maxCols <= 0) return undefined;
  return {
    headers: Array.from({ length: maxCols }, (_, index) => headers[index] || `列 ${index + 1}`),
    rows: rows.length > 0
      ? rows.map((row) => Array.from({ length: maxCols }, (_, index) => row[index] || ''))
      : [Array.from({ length: maxCols }, () => '')],
  };
};

const renderCanvasTableMarkup = (slide: FrontendSlide, node: FrontendCanvasNode, attrs: string) => {
  const props = node.props || {};
  const ref = props.table_ref || props.tableRef || props.ref || node.id;
  const ownerId = String(ref || node.id);
  const fieldMap = buildFieldMap(slide);
  const tableData = normalizeCanvasTableData(
    props.table_data
    || props.tableData
    || props.table
    || resolveContentPath(slide.content, ownerId),
  );
  if (!tableData) return renderCanvasPlaceholder(node, 'Missing table data');
  return `
<div class="schema-canvas-component schema-table-card" ${attrs}>
  <div class="schema-table-scroll">
    <table class="schema-table">
      <thead>
        <tr>${tableData.headers.map((header, colIndex) => {
          const field = fieldMap.get(getTableCellFieldKey(ownerId, 'h', colIndex));
          return `<th>${wrapEditableText(field, field?.value || header)}</th>`;
        }).join('')}</tr>
      </thead>
      <tbody>
        ${tableData.rows.map((row, rowIndex) => `<tr>${row.map((cell, colIndex) => {
          const field = fieldMap.get(getTableCellFieldKey(ownerId, rowIndex, colIndex));
          return `<td>${wrapEditableText(field, field?.value || cell)}</td>`;
        }).join('')}</tr>`).join('')}
      </tbody>
    </table>
  </div>
</div>
`.trim();
};

const resolveTheme = (theme?: FrontendDeckTheme | null) => ({
  palette: {
    ...DEFAULT_PALETTE,
    ...(theme?.palette || {}),
  },
  typography: {
    ...DEFAULT_TYPOGRAPHY,
    ...(theme?.typography || {}),
  },
  footerText: theme?.footerText || 'Paper2Any Frontend PPT',
});

const resolveCanvasVisualTheme = (
  theme?: FrontendDeckTheme | null,
  visualSpec?: FrontendCanvasVisualSpec | null,
) => {
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
      maxColumns: visualSpec?.layout?.maxColumns ?? 2,
    },
    footerText: theme?.footerText || 'Paper2Any Frontend PPT',
  };
};

const getCanvasNodeVisualStyle = (
  slide: FrontendSlide,
  node: FrontendCanvasNode,
  component: string,
) => {
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
    ...(typeof propsStyle === 'object' ? propsStyle as Record<string, unknown> : {}),
  };
};

const buildCanvasStyleAttr = (style: Record<string, unknown>) => {
  const rules: string[] = [];
  const push = (key: string, cssKey = key, unit: string | null = null) => {
    const value = style[key];
    if (value === undefined || value === null) return;
    const text = String(value).trim();
    if (!text) return;
    if (unit) {
      const numeric = Number(text);
      if (Number.isFinite(numeric)) {
        rules.push(`${cssKey}:${Math.max(0, numeric)}${unit}`);
        return;
      }
    }
    rules.push(`${cssKey}:${text}`);
  };
  push('fill', 'background-color');
  push('color');
  push('borderColor', 'border-color');
  push('borderWidth', 'border-width', 'px');
  if (style.borderColor || style.borderWidth) {
    if (style.borderColor && style.borderWidth === undefined) {
      rules.push('border-width:1px');
    }
    rules.push('border-style:solid');
  }
  push('radius', 'border-radius', 'px');
  push('padding', 'padding', 'px');
  push('fontFamily', 'font-family');
  push('fontSize', 'font-size', 'px');
  push('fontWeight', 'font-weight');
  push('fontStyle', 'font-style');
  push('lineHeight', 'line-height', 'px');
  push('textAlign', 'text-align');
  push('opacity');
  return rules.join(';');
};

const getFieldKeyForBlock = (slide: FrontendSlide, block: FrontendSlideBlock) => {
  const candidates = [
    PREFERRED_FIELD_KEYS[block.role] || '',
    block.role || '',
    block.id || '',
    slugify(block.role || ''),
    slugify(block.id || ''),
  ].filter(Boolean);
  const fieldKeys = new Set(slide.editableFields.map((field) => field.key));
  return candidates.find((candidate) => fieldKeys.has(candidate));
};

const getFieldKeyForChild = (slide: FrontendSlide, child: FrontendBlockChild) => {
  const candidates = [
    PREFERRED_FIELD_KEYS[child.role] || '',
    child.role || '',
    child.id || '',
    slugify(child.role || ''),
    slugify(child.id || ''),
  ].filter(Boolean);
  const fieldKeys = new Set(slide.editableFields.map((field) => field.key));
  return candidates.find((candidate) => fieldKeys.has(candidate));
};

const wrapEditableText = (
  field: FrontendEditableField | undefined,
  fallbackValue: string,
  itemIndex?: number,
) => {
  const fieldKey = field?.key;
  const fieldType = field?.type || (typeof itemIndex === 'number' ? 'list' : 'text');
  const value = field
    ? typeof itemIndex === 'number'
      ? field.items[itemIndex] || fallbackValue
      : field.type === 'list'
        ? field.items.filter((item) => item.trim()).join(' • ')
        : field.value || fallbackValue
    : fallbackValue;
  const itemAttr = typeof itemIndex === 'number' ? ` data-edit-index="${itemIndex}"` : '';
  const attrs = fieldKey
    ? ` class="ppt-inline-editable${typeof itemIndex === 'number' ? ' ppt-inline-editable-list' : ''}" data-edit-key="${escapeHtml(fieldKey)}" data-edit-type="${escapeHtml(fieldType)}"${itemAttr}`
    : '';
  return `<span${attrs}>${formatTextValue(value)}</span>`;
};

const renderVisualAsset = (
  asset: FrontendVisualAsset | undefined,
  assetKey: string,
  label: string,
  imageFit?: FrontendCanvasVisualStyle['imageFit'],
) => {
  const previewSrc = (asset?.previewSrc || asset?.src || '').trim();
  const originalSrc = (asset?.originalSrc || previewSrc || '').trim();
  const sourceLabel = asset?.sourceType === 'paper_asset' ? '论文图表' : asset?.sourceType === 'upload' ? '用户上传' : 'AI 配图';
  const isPaperAsset = asset?.sourceType === 'paper_asset';
  const isGeneratedAsset = asset?.sourceType === 'generated';
  const useTransparentFrame = isPaperAsset || isGeneratedAsset;
  const safeAssetKey = escapeHtml(assetKey || asset?.key || 'main_visual');
  const safeLabel = escapeHtml(asset?.label || label || assetKey || 'Image');
  const safeAlt = escapeHtml(asset?.alt || safeLabel || 'Slide image');
  const resolvedImageFit = useTransparentFrame
    ? 'contain'
    : imageFit === 'contain' || imageFit === 'fill'
      ? imageFit
      : 'cover';
  const sourceClass = useTransparentFrame ? ' ppt-managed-image-transparent' : '';

  if (!previewSrc) {
    return `
<div class="ppt-managed-image${sourceClass}" data-image-key="${safeAssetKey}" data-image-label="${safeLabel}">
  <div class="ppt-managed-image-frame ppt-managed-image-frame-empty">
    <div class="ppt-managed-image-empty-text">点击上传图片</div>
  </div>
  <div class="ppt-managed-image-badge">${escapeHtml(sourceLabel)}</div>
</div>
`.trim();
  }

  return `
<div class="ppt-managed-image${sourceClass}" data-image-key="${safeAssetKey}" data-image-label="${safeLabel}">
  <div class="ppt-managed-image-frame">
    <img src="${escapeHtml(previewSrc)}" data-preview-src="${escapeHtml(previewSrc)}" data-original-src="${escapeHtml(originalSrc)}" alt="${safeAlt}" class="ppt-managed-image-el" style="object-fit:${escapeHtml(resolvedImageFit)};object-position:center;" />
  </div>
  <div class="ppt-managed-image-badge">${escapeHtml(sourceLabel)}</div>
</div>
`.trim();
};

const sortBlocks = (blocks: FrontendSlideBlock[]) =>
  [...blocks].sort((a, b) => {
    const orderA = Number(a.layout?.order || 0);
    const orderB = Number(b.layout?.order || 0);
    if (orderA !== orderB) return orderA - orderB;
    return a.id.localeCompare(b.id);
  });

const withManualBlockClass = (_block: FrontendSlideBlock, className: string) =>
  className;

const isRightColumnBlock = (block: FrontendSlideBlock) =>
  block.layout?.zone === 'right'
  || block.layout?.zone === 'aside'
  || block.layout?.preferredSide === 'right';

const isLeftColumnBlock = (block: FrontendSlideBlock) =>
  block.layout?.zone === 'left'
  || block.layout?.preferredSide === 'left';

const renderTextBlock = (
  slide: FrontendSlide,
  block: FrontendSlideBlock,
  className: string,
  tagName: 'p' | 'div' | 'h1' | 'h2' = 'p',
) => {
  const field = buildFieldMap(slide).get(getFieldKeyForBlock(slide, block) || '');
  const value = field?.value || block.content || '';
  const blockClassName = withManualBlockClass(block, className);
  if (block.children && block.children.length > 0) {
    const shouldRenderPrimary = Boolean(value.trim()) && !hasLegacyChildForBlock(block);
    return `
<div class="${blockClassName} schema-text-block-with-children" data-block-id="${escapeHtml(block.id)}" data-block-role="${escapeHtml(block.role)}">
  ${shouldRenderPrimary ? `<span class="schema-text-primary">${wrapEditableText(field, value)}</span>` : ''}
  ${renderBlockChildren(slide, block)}
</div>
`.trim();
  }
  return `<${tagName} class="${blockClassName}" data-block-id="${escapeHtml(block.id)}" data-block-role="${escapeHtml(block.role)}">${wrapEditableText(field, value)}</${tagName}>`;
};

const renderListBlock = (slide: FrontendSlide, block: FrontendSlideBlock, className: string) => {
  const field = buildFieldMap(slide).get(getFieldKeyForBlock(slide, block) || '');
  const items = (field?.items?.length ? field.items : block.items || []).filter((item) => item.trim());
  if (items.length === 0 && (!block.children || block.children.length === 0)) return '';
  if (block.children && block.children.length > 0) {
    const shouldRenderPrimary = items.length > 0 && !hasLegacyChildForBlock(block);
    return `
<div class="schema-list-block-with-children" data-block-id="${escapeHtml(block.id)}" data-block-role="${escapeHtml(block.role)}">
  ${shouldRenderPrimary ? `
    <ul class="${className}">
      ${items.map((item, index) => `<li>${wrapEditableText(field, item, index)}</li>`).join('')}
    </ul>
  ` : ''}
  ${renderBlockChildren(slide, block)}
</div>
`.trim();
  }
  return `
<ul class="${className}" data-block-id="${escapeHtml(block.id)}" data-block-role="${escapeHtml(block.role)}">
  ${items.map((item, index) => `<li>${wrapEditableText(field, item, index)}</li>`).join('')}
</ul>
`.trim();
};

const renderImageBlock = (slide: FrontendSlide, block: FrontendSlideBlock, className: string) => {
  const blockClassName = withManualBlockClass(block, className);
  if (block.children && block.children.length > 0) {
    return `<div class="${blockClassName}" data-block-id="${escapeHtml(block.id)}" data-block-role="${escapeHtml(block.role)}">${renderBlockChildren(slide, block)}</div>`;
  }
  const assetKey = block.assetKey || block.id || 'main_visual';
  const asset = buildAssetMap(slide).get(assetKey);
  return `<div class="${blockClassName}" data-block-id="${escapeHtml(block.id)}" data-block-role="${escapeHtml(block.role)}">${renderVisualAsset(asset, assetKey, block.role || assetKey)}</div>`;
};

const hasLegacyChildForBlock = (block: FrontendSlideBlock) =>
  (block.children || []).some((child) => child.id === `${block.id}_content`);

const getTableCellFieldKey = (ownerId: string, rowIndex: number | 'h', colIndex: number) =>
  `${ownerId}_cell_${rowIndex}_${colIndex}`;

const normalizeTableMatrix = (headers: string[], rows: string[][]) => {
  const maxCols = Math.max(
    headers.length,
    ...rows.map((row) => row.length),
    1,
  );
  return {
    headers: Array.from({ length: maxCols }, (_, index) => headers[index] || `列 ${index + 1}`),
    rows: rows.length > 0
      ? rows.map((row) => Array.from({ length: maxCols }, (_, index) => row[index] || ''))
      : [Array.from({ length: maxCols }, () => '')],
  };
};

const renderTableMarkup = (
  slide: FrontendSlide,
  owner: FrontendSlideBlock | FrontendBlockChild,
  className: string,
  includeBlockAttrs = true,
) => {
  const fieldMap = buildFieldMap(slide);
  const tableData = normalizeTableMatrix(owner.tableData?.headers || [], owner.tableData?.rows || []);
  const blockAttrs = includeBlockAttrs
    ? ` data-block-id="${escapeHtml(owner.id)}" data-block-role="${escapeHtml(owner.role)}"`
    : '';
  return `
<div class="${className}"${blockAttrs}>
  <div class="schema-table-scroll">
    <table class="schema-table">
      <thead>
        <tr>
          ${tableData.headers.map((header, colIndex) => {
            const field = fieldMap.get(getTableCellFieldKey(owner.id, 'h', colIndex));
            return `<th>${wrapEditableText(field, field?.value || header)}</th>`;
          }).join('')}
        </tr>
      </thead>
      <tbody>
        ${tableData.rows.map((row, rowIndex) => `
          <tr>
            ${row.map((cell, colIndex) => {
              const field = fieldMap.get(getTableCellFieldKey(owner.id, rowIndex, colIndex));
              return `<td>${wrapEditableText(field, field?.value || cell)}</td>`;
            }).join('')}
          </tr>
        `).join('')}
      </tbody>
    </table>
  </div>
</div>
`.trim();
};

const renderTableBlock = (slide: FrontendSlide, block: FrontendSlideBlock, className: string) =>
  renderTableMarkup(slide, block, withManualBlockClass(block, className));

const renderChildItem = (slide: FrontendSlide, child: FrontendBlockChild) => {
  const field = buildFieldMap(slide).get(getFieldKeyForChild(slide, child) || '');
  if (child.type === 'table') {
    return renderTableMarkup(slide, child, 'schema-child-item schema-table-card', false);
  }
  if (child.type === 'image') {
    const assetKey = child.assetKey || child.id || 'main_visual';
    const asset = buildAssetMap(slide).get(assetKey);
    return `<div class="schema-child-item schema-child-image">${renderVisualAsset(asset, assetKey, child.role || assetKey)}</div>`;
  }
  if (child.type === 'list') {
    const items = (field?.items?.length ? field.items : child.items || []).filter((item) => item.trim());
    if (items.length === 0) return '';
    return `
<div class="schema-child-item schema-child-list">
  <div class="schema-card-title">${escapeHtml(child.role.replace(/_/g, ' '))}</div>
  <ul class="schema-bullets tight">
    ${items.map((item, index) => `<li>${wrapEditableText(field, item, index)}</li>`).join('')}
  </ul>
</div>
`.trim();
  }
  const value = field?.value || child.content || '';
  if (child.type === 'callout') {
    return `<div class="schema-child-item schema-callout">${wrapEditableText(field, value)}</div>`;
  }
  if (child.type === 'quote') {
    return `<blockquote class="schema-child-item schema-quote">${wrapEditableText(field, value)}</blockquote>`;
  }
  if (child.type === 'stat') {
    return `
<div class="schema-child-item schema-stat-card">
  <div class="schema-stat-value">${wrapEditableText(field, value)}</div>
  <div class="schema-stat-label">${escapeHtml(child.role.replace(/_/g, ' ') || 'Highlight')}</div>
</div>
`.trim();
  }
  return `<div class="schema-child-item schema-card-text">${wrapEditableText(field, value)}</div>`;
};

const renderBlockChildren = (slide: FrontendSlide, block: FrontendSlideBlock) => {
  const children = block.children || [];
  if (children.length === 0) {
    return '';
  }
  return `<div class="schema-block-children">${children.map((child) => renderChildItem(slide, child)).join('')}</div>`;
};

const renderStatBlock = (slide: FrontendSlide, block: FrontendSlideBlock, className: string) => {
  const blockClassName = withManualBlockClass(block, className);
  if (block.children && block.children.length > 0) {
    return `<div class="${blockClassName}" data-block-id="${escapeHtml(block.id)}" data-block-role="${escapeHtml(block.role)}">${renderBlockChildren(slide, block)}</div>`;
  }
  const field = buildFieldMap(slide).get(getFieldKeyForBlock(slide, block) || '');
  const value = field?.value || block.content || '';
  return `
<div class="${blockClassName}" data-block-id="${escapeHtml(block.id)}" data-block-role="${escapeHtml(block.role)}">
  <div class="schema-stat-value">${wrapEditableText(field, value)}</div>
  <div class="schema-stat-label">${escapeHtml(block.role.replace(/_/g, ' ') || 'Highlight')}</div>
</div>
`.trim();
};

const renderCalloutBlock = (slide: FrontendSlide, block: FrontendSlideBlock, className: string) => {
  const blockClassName = withManualBlockClass(block, className);
  if (block.children && block.children.length > 0) {
    return `<div class="${blockClassName}" data-block-id="${escapeHtml(block.id)}" data-block-role="${escapeHtml(block.role)}">${renderBlockChildren(slide, block)}</div>`;
  }
  const field = buildFieldMap(slide).get(getFieldKeyForBlock(slide, block) || '');
  const value = field?.value || block.content || '';
  return `<div class="${blockClassName}" data-block-id="${escapeHtml(block.id)}" data-block-role="${escapeHtml(block.role)}">${wrapEditableText(field, value)}</div>`;
};

const renderQuoteBlock = (slide: FrontendSlide, block: FrontendSlideBlock, className: string) => {
  const blockClassName = withManualBlockClass(block, className);
  if (block.children && block.children.length > 0) {
    return `<blockquote class="${blockClassName}" data-block-id="${escapeHtml(block.id)}" data-block-role="${escapeHtml(block.role)}">${renderBlockChildren(slide, block)}</blockquote>`;
  }
  const field = buildFieldMap(slide).get(getFieldKeyForBlock(slide, block) || '');
  const value = field?.value || block.content || '';
  return `<blockquote class="${blockClassName}" data-block-id="${escapeHtml(block.id)}" data-block-role="${escapeHtml(block.role)}">${wrapEditableText(field, value)}</blockquote>`;
};

const renderGenericCard = (slide: FrontendSlide, block: FrontendSlideBlock, className = 'schema-card') => {
  const blockClassName = withManualBlockClass(block, className);
  if (block.children && block.children.length > 0) {
    return `
<div class="${blockClassName}" data-block-id="${escapeHtml(block.id)}" data-block-role="${escapeHtml(block.role)}">
  <div class="schema-card-title">${escapeHtml(block.role.replace(/_/g, ' '))}</div>
  ${renderBlockChildren(slide, block)}
</div>
`.trim();
  }
  if (block.type === 'image') return renderImageBlock(slide, block, `${className} schema-image-card`);
  if (block.type === 'table') return renderTableBlock(slide, block, `${className} schema-table-card`);
  if (block.type === 'list') {
    return `
<div class="${blockClassName}" data-block-id="${escapeHtml(block.id)}" data-block-role="${escapeHtml(block.role)}">
  <div class="schema-card-title">${escapeHtml(block.role.replace(/_/g, ' '))}</div>
  ${renderListBlock(slide, block, 'schema-bullets')}
</div>
`.trim();
  }
  if (block.type === 'quote') return renderQuoteBlock(slide, block, `${className} schema-quote`);
  if (block.type === 'stat') return renderStatBlock(slide, block, `${className} schema-stat-card`);
  if (block.type === 'callout') return renderCalloutBlock(slide, block, `${className} schema-callout`);
  return `
<div class="${blockClassName}" data-block-id="${escapeHtml(block.id)}" data-block-role="${escapeHtml(block.role)}">
  <div class="schema-card-title">${escapeHtml(block.role.replace(/_/g, ' '))}</div>
  ${renderTextBlock(slide, block, 'schema-card-text')}
</div>
`.trim();
};

const renderColumnBlock = (slide: FrontendSlide, block: FrontendSlideBlock) => {
  if (block.type === 'image') return renderImageBlock(slide, block, 'schema-visual-card');
  if (block.type === 'table') return renderTableBlock(slide, block, 'schema-table-card');
  if (block.type === 'callout') return renderCalloutBlock(slide, block, 'schema-callout');
  if (block.type === 'stat') return renderStatBlock(slide, block, 'schema-stat-card');
  if (block.type === 'quote') return renderQuoteBlock(slide, block, 'schema-quote');
  if (block.type === 'list') return renderGenericCard(slide, block);
  return renderGenericCard(slide, block);
};

const renderColumnBlocks = (
  slide: FrontendSlide,
  blocks: FrontendSlideBlock[],
  options: { limit?: number; bareLists?: boolean; tallFirstImage?: boolean } = {},
) => {
  const limit = options.limit ?? Number.POSITIVE_INFINITY;
  let imageCount = 0;
  return sortBlocks(blocks)
    .map((block) => {
      if (block.type === 'image') {
        const isFirstImage = imageCount === 0;
        imageCount += 1;
        return renderImageBlock(
          slide,
          block,
          options.tallFirstImage && isFirstImage ? 'schema-visual-card tall' : 'schema-visual-card',
        );
      }
      if (options.bareLists && block.type === 'list') {
        return renderListBlock(slide, block, 'schema-bullets');
      }
      return renderColumnBlock(slide, block);
    })
    .filter(Boolean)
    .slice(0, limit);
};

const pickSchemaTemplateKey = (slide: FrontendSlide) => {
  const requested = slide.templateKey || '';
  if (SUPPORTED_SCHEMA_TEMPLATES.includes(requested as (typeof SUPPORTED_SCHEMA_TEMPLATES)[number])) {
    return requested as (typeof SUPPORTED_SCHEMA_TEMPLATES)[number];
  }

  const blocks = sortBlocks(slide.blocks);
  const imageCount = blocks.filter((block) => block.type === 'image').length;
  const listCount = blocks.filter((block) => block.type === 'list').length;
  const statCount = blocks.filter((block) => block.type === 'stat').length;
  const quoteCount = blocks.filter((block) => block.type === 'quote').length;

  if (quoteCount > 0) return 'quote_focus';
  if (imageCount >= 2) return 'visual_compare';
  if (statCount >= 2) return 'metrics_dashboard';
  if (listCount >= 2) return 'dual_list';
  if (imageCount === 1 && listCount > 0) return 'split_media';
  if (imageCount === 1) return 'hero_visual';
  if (blocks.length <= 3) return 'section_divider';
  if (blocks.length >= 6) return 'insight_grid';
  return 'text_focus';
};

const buildSchemaBaseCss = (theme?: FrontendDeckTheme | null, visualSpec?: FrontendCanvasVisualSpec | null) => {
  const resolved = resolveCanvasVisualTheme(theme, visualSpec);
  return `
.slide-root.schema-root {
  position: relative;
  --schema-bg: ${resolved.palette.bg};
  --schema-panel: ${resolved.palette.panel};
  --schema-primary: ${resolved.palette.primary};
  --schema-secondary: ${resolved.palette.secondary};
  --schema-accent: ${resolved.palette.accent};
  --schema-text: ${resolved.palette.text};
  --schema-muted: ${resolved.palette.muted};
  --schema-title-font: ${resolved.typography.titleFontStack};
  --schema-body-font: ${resolved.typography.bodyFontStack};
  --schema-title-size: ${resolved.typography.titleSize}px;
  --schema-summary-size: ${resolved.typography.summarySize}px;
  --schema-body-size: ${resolved.typography.bodySize}px;
  --schema-eyebrow-size: ${resolved.typography.eyebrowSize}px;
  --schema-card-radius: ${resolved.surface.cardRadius}px;
  --schema-card-padding: ${resolved.surface.cardPadding}px;
  --schema-shell-padding-x: ${resolved.layout.safeMargin}px;
  --schema-shell-padding-top: ${resolved.layout.safeMargin}px;
  --schema-shell-padding-bottom: ${Math.max(40, Math.round(resolved.layout.safeMargin * 0.9))}px;
  --schema-shell-gap: ${resolved.layout.sectionGap}px;
  width: 100%;
  height: 100%;
  background:
    radial-gradient(circle at top right, rgba(148, 163, 184, 0.16) 0%, transparent 28%),
    radial-gradient(circle at bottom left, rgba(194, 165, 106, 0.14) 0%, transparent 34%),
    var(--schema-bg);
  color: var(--schema-text);
  overflow: hidden;
}
.slide-root.schema-root * {
  box-sizing: border-box;
}
.schema-shell {
  position: relative;
  width: 100%;
  height: 100%;
  padding: var(--schema-shell-padding-top) var(--schema-shell-padding-x) var(--schema-shell-padding-bottom);
  display: flex;
  flex-direction: column;
  gap: var(--schema-shell-gap);
  background:
    linear-gradient(90deg, rgba(148, 163, 184, 0.08) 1px, transparent 1px),
    linear-gradient(rgba(148, 163, 184, 0.08) 1px, transparent 1px);
  background-size: 52px 52px;
}
.schema-fixed-page-label {
  position: absolute;
  right: 44px;
  bottom: 30px;
  z-index: 20;
  padding: 6px 10px;
  border-radius: 999px;
  color: var(--schema-muted);
  background: rgba(255, 255, 255, 0.68);
  border: 1px solid rgba(148, 163, 184, 0.22);
  font-family: var(--schema-body-font);
  font-size: 13px;
  line-height: 1;
  font-weight: 700;
  letter-spacing: 0.04em;
  pointer-events: none;
}
.schema-header {
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.schema-eyebrow {
  align-self: flex-start;
  padding: 8px 14px;
  border-radius: 999px;
  background: rgba(148, 163, 184, 0.14);
  border: 1px solid rgba(148, 163, 184, 0.28);
  color: var(--schema-primary);
  font-size: var(--schema-eyebrow-size);
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.schema-title {
  margin: 0;
  font-family: var(--schema-title-font);
  font-size: var(--schema-title-size);
  line-height: 1.02;
  letter-spacing: 0;
}
.schema-summary {
  margin: 0;
  max-width: 860px;
  font-family: var(--schema-body-font);
  font-size: var(--schema-summary-size);
  line-height: 1.42;
  color: var(--schema-muted);
}
.schema-main {
  flex: 1 1 auto;
  min-height: 0;
}
.schema-footer {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 18px;
}
.schema-card,
.schema-callout,
.schema-stat-card,
.schema-visual-card,
.schema-image-card,
.schema-table-card {
  background: var(--schema-panel);
  border: 1px solid rgba(148, 163, 184, 0.24);
  box-shadow: 0 24px 60px rgba(0, 0, 0, 0.28);
  border-radius: var(--schema-card-radius);
  backdrop-filter: blur(10px);
}
.schema-card,
.schema-stat-card,
.schema-callout,
.schema-table-card {
  padding: var(--schema-card-padding) calc(var(--schema-card-padding) + 2px);
}
.schema-card-title,
.schema-label {
  font-size: calc(var(--schema-eyebrow-size) - 2px);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-weight: 700;
  color: var(--schema-primary);
  margin-bottom: 12px;
}
.schema-card-text,
.schema-body,
.schema-takeaway,
.schema-callout {
  font-family: var(--schema-body-font);
  font-size: var(--schema-body-size);
  line-height: 1.45;
  color: var(--schema-text);
}
.schema-block-children {
  display: grid;
  gap: 14px;
}
.schema-text-block-with-children,
.schema-list-block-with-children {
  display: grid;
  gap: 14px;
}
.schema-text-block-with-children .schema-block-children {
  font-family: var(--schema-body-font);
  font-size: var(--schema-body-size);
  line-height: 1.45;
  letter-spacing: 0;
  text-transform: none;
  font-weight: 400;
  color: var(--schema-text);
}
.schema-child-item {
  min-width: 0;
}
.schema-child-image {
  min-height: 220px;
}
.schema-bullets {
  margin: 0;
  padding-left: 24px;
  display: grid;
  gap: 12px;
  font-family: var(--schema-body-font);
  font-size: calc(var(--schema-body-size) - 1px);
  line-height: 1.35;
}
.schema-bullets.tight {
  gap: 8px;
  font-size: calc(var(--schema-body-size) - 2px);
}
.schema-bullets li {
  color: var(--schema-text);
}
.schema-table-card {
  min-width: 0;
}
.schema-table-scroll {
  width: 100%;
  overflow: auto;
}
.schema-table {
  width: 100%;
  border-collapse: collapse;
  font-family: var(--schema-body-font);
  font-size: calc(var(--schema-body-size) - 5px);
  line-height: 1.25;
}
.schema-table th,
.schema-table td {
  border: 1px solid rgba(148, 163, 184, 0.24);
  padding: 10px 12px;
  text-align: left;
  vertical-align: top;
  color: var(--schema-text);
}
.schema-table th {
  background: rgba(148, 163, 184, 0.14);
  color: var(--schema-primary);
  font-weight: 800;
}
.schema-table td {
  background: rgba(255, 255, 255, 0.03);
}
.schema-canvas-root {
  display: flex;
  flex-direction: column;
  width: 100%;
  height: 100%;
  min-height: 0;
}
.schema-canvas-container {
  min-width: 0;
  min-height: 0;
  width: 100%;
}
.schema-canvas-container[data-canvas-node-id="root"],
.schema-canvas-container[data-canvas-node-id="main"] {
  flex: 1 1 auto;
  height: 100%;
}
.schema-canvas-component {
  min-width: 0;
  overflow-wrap: anywhere;
}
.schema-canvas-heading {
  margin: 0;
  font-family: var(--schema-title-font);
  font-size: var(--schema-title-size);
  line-height: 1.05;
  letter-spacing: 0;
  color: var(--schema-text);
}
.schema-canvas-text {
  margin: 0;
  font-family: var(--schema-body-font);
  font-size: var(--schema-body-size);
  line-height: 1.42;
  color: var(--schema-text);
}
.schema-canvas-placeholder {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 120px;
  padding: 18px;
  border: 1px dashed rgba(148, 163, 184, 0.34);
  border-radius: 18px;
  color: var(--schema-muted);
  background: var(--schema-panel);
  font-family: var(--schema-body-font);
  font-size: 18px;
}
.template-canvas-schema .schema-shell {
  padding: 58px 64px 52px;
}
.template-canvas-schema .schema-main {
  display: flex;
  min-height: 0;
}
.template-canvas-schema .schema-visual-card {
  min-height: 220px;
  height: 100%;
}
.template-canvas-schema .schema-bullets {
  max-height: 100%;
  overflow: hidden;
}
.schema-quote {
  margin: 0;
  padding: 28px 30px;
  font-family: var(--schema-title-font);
  font-size: calc(var(--schema-title-size) * 0.72);
  line-height: 1.16;
  letter-spacing: 0;
  border-radius: 32px;
  background: rgba(148, 163, 184, 0.14);
  border: 1px solid rgba(148, 163, 184, 0.26);
  color: var(--schema-text);
}
.schema-stat-value {
  font-family: var(--schema-title-font);
  font-size: calc(var(--schema-title-size) * 0.72);
  line-height: 1;
  letter-spacing: 0;
}
.schema-stat-label {
  margin-top: 10px;
  font-size: 14px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--schema-muted);
}
.schema-tag {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 220px;
  padding: 13px 18px;
  border-radius: 999px;
  border: 1px solid rgba(194, 165, 106, 0.36);
  color: var(--schema-accent);
  background: rgba(7, 16, 29, 0.68);
  font-size: calc(var(--schema-eyebrow-size) - 2px);
  font-weight: 700;
  letter-spacing: 0.05em;
}
.schema-grid-2 {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 22px;
}
.schema-grid-3 {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 18px;
}
.schema-stack {
  display: flex;
  flex-direction: column;
  gap: 18px;
}
.schema-visual-card,
.schema-image-card {
  padding: 16px;
  min-height: 260px;
}
.schema-visual-card.tall {
  min-height: 380px;
}
.schema-timeline {
  display: grid;
  gap: 14px;
}
.schema-timeline-item {
  display: grid;
  grid-template-columns: 34px 1fr;
  gap: 14px;
  align-items: start;
}
.schema-timeline-dot {
  width: 18px;
  height: 18px;
  margin-top: 6px;
  border-radius: 999px;
  background: linear-gradient(135deg, var(--schema-primary), var(--schema-accent));
  box-shadow: 0 0 0 6px rgba(148, 163, 184, 0.18);
}
.schema-timeline-copy {
  padding: 14px 16px;
  border-radius: 20px;
  background: var(--schema-panel);
  border: 1px solid rgba(148, 163, 184, 0.18);
  font-family: var(--schema-body-font);
  font-size: calc(var(--schema-body-size) - 2px);
  line-height: 1.35;
}
.schema-columns {
  display: grid;
  grid-template-columns: 1.06fr 0.94fr;
  gap: 24px;
  height: 100%;
}
.schema-columns.equal {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}
.schema-main-copy {
  display: flex;
  flex-direction: column;
  gap: 16px;
}
[data-insert-zone] {
  position: relative;
  min-height: 96px;
}
.schema-surface {
  padding: 20px 22px;
  border-radius: 24px;
  background: var(--schema-panel);
  border: 1px solid rgba(148, 163, 184, 0.18);
}
.slide-root .ppt-inline-editable {
  cursor: text;
  transition: box-shadow 0.18s ease, background-color 0.18s ease;
}
.slide-root .ppt-inline-editable:hover {
  background: rgba(125, 211, 252, 0.08);
  box-shadow: 0 0 0 2px rgba(125, 211, 252, 0.16);
  border-radius: 0.2em;
}
.slide-root .ppt-managed-image {
  position: relative;
  width: 100%;
  height: 100%;
  min-height: 180px;
  cursor: pointer;
}
.slide-root .ppt-managed-image-frame {
  position: relative;
  width: 100%;
  height: 100%;
  overflow: hidden;
  border-radius: 20px;
  border: 1px solid rgba(148, 163, 184, 0.22);
  background:
    radial-gradient(circle at top right, rgba(125, 211, 252, 0.18), transparent 28%),
    linear-gradient(135deg, rgba(15, 23, 42, 0.08), rgba(15, 23, 42, 0.2));
}
.slide-root .ppt-managed-image-frame-empty {
  display: flex;
  align-items: center;
  justify-content: center;
  border-style: dashed;
}
.slide-root .ppt-managed-image-empty-text {
  padding: 10px 14px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.72);
  color: rgba(15, 23, 42, 0.76);
  font-size: 14px;
  font-weight: 600;
  letter-spacing: 0.04em;
}
.slide-root .ppt-managed-image-el {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: cover;
}
.schema-transparent-figure-card {
  background: transparent;
  border-color: transparent;
  box-shadow: none;
  backdrop-filter: none;
}
.slide-root .ppt-managed-image-transparent .ppt-managed-image-frame {
  display: flex;
  align-items: center;
  justify-content: center;
  background: transparent;
  border-color: transparent;
}
.slide-root .ppt-managed-image-transparent .ppt-managed-image-el {
  width: auto;
  height: auto;
  max-width: 100%;
  max-height: 100%;
  object-fit: contain !important;
}

.slide-root .ppt-managed-image-badge {
  position: absolute;
  left: 12px;
  bottom: 12px;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(6, 16, 29, 0.68);
  color: rgba(255, 255, 255, 0.92);
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.04em;
  backdrop-filter: blur(8px);
  opacity: 0;
  transform: translateY(6px);
  transition: opacity 0.18s ease, transform 0.18s ease;
}
.slide-root .ppt-managed-image:hover .ppt-managed-image-frame {
  box-shadow: 0 0 0 2px rgba(125, 211, 252, 0.2), 0 18px 36px rgba(15, 23, 42, 0.18);
}
.slide-root .ppt-managed-image:hover .ppt-managed-image-badge {
  opacity: 1;
  transform: translateY(0);
}
`.trim();
};

const buildSchemaSlideContext = (slide: FrontendSlide) => {
  const blocks = sortBlocks(slide.blocks || []);
  const findByRole = (...roles: string[]) => blocks.find((block) => roles.includes(block.role));
  const images = blocks.filter((block) => block.type === 'image');
  const lists = blocks.filter((block) => block.type === 'list');
  const stats = blocks.filter((block) => block.type === 'stat');
  const quotes = blocks.filter((block) => block.type === 'quote');
  const callouts = blocks.filter((block) => block.type === 'callout');
  const tables = blocks.filter((block) => block.type === 'table');
  const bodyTexts = blocks.filter(
    (block) =>
      block.type === 'text'
      && !['title', 'summary', 'eyebrow', 'footer', 'takeaway'].includes(block.role),
  );
  return {
    blocks,
    eyebrow: findByRole('eyebrow'),
    title: findByRole('title'),
    summary: findByRole('summary'),
    takeaway: findByRole('takeaway'),
    footer: findByRole('footer'),
    images,
    lists,
    stats,
    quotes,
    callouts,
    tables,
    bodyTexts,
    remaining: blocks.filter(
      (block) =>
        !['eyebrow', 'title', 'summary', 'takeaway', 'footer'].includes(block.role),
    ),
  };
};

const renderHeader = (slide: FrontendSlide, context: ReturnType<typeof buildSchemaSlideContext>) => `
  <div class="schema-header">
    ${context.eyebrow && !isSystemPageLabel(context.eyebrow.content) ? renderTextBlock(slide, context.eyebrow, 'schema-eyebrow', 'div') : ''}
    ${context.title ? renderTextBlock(slide, context.title, 'schema-title', 'h1') : `<h1 class="schema-title">${escapeHtml(slide.title)}</h1>`}
    ${context.summary ? renderTextBlock(slide, context.summary, 'schema-summary') : ''}
  </div>
`;

const renderFooter = (slide: FrontendSlide, context: ReturnType<typeof buildSchemaSlideContext>, theme?: FrontendDeckTheme | null) => {
  const resolved = resolveTheme(theme);
  const footerBlock = context.footer;
  const footerAttrs = footerBlock
    ? ` data-block-id="${escapeHtml(footerBlock.id)}" data-block-role="${escapeHtml(footerBlock.role)}"`
    : '';
  const footerField = footerBlock
    ? buildFieldMap(slide).get(getFieldKeyForBlock(slide, footerBlock) || '')
    : undefined;
  const shouldRenderFooterPrimary = footerBlock
    ? !hasLegacyChildForBlock(footerBlock)
    : true;
  return `
  <div class="schema-footer">
    <div class="schema-surface schema-takeaway">
      ${context.takeaway ? renderTextBlock(slide, context.takeaway, 'schema-takeaway') : ''}
    </div>
    <div class="schema-tag"${footerAttrs}>
      ${footerBlock
        ? shouldRenderFooterPrimary
          ? wrapEditableText(footerField, footerBlock.content)
          : ''
        : escapeHtml(resolved.footerText)}
      ${footerBlock?.children?.length ? renderBlockChildren(slide, footerBlock) : ''}
    </div>
  </div>
  `;
};

const renderCanvasComponent = (slide: FrontendSlide, node: FrontendCanvasNode) => {
  const props = node.props || {};
  const component = normalizeCanvasComponent(node.component || props.component || props.kind);
  const attrs = `data-block-id="${escapeHtml(node.id)}" data-block-role="${escapeHtml(component)}" data-canvas-node-id="${escapeHtml(node.id)}"`;
  const visualStyle = getCanvasNodeVisualStyle(slide, node, component);
  const styleAttr = buildCanvasStyleAttr(visualStyle);
  const style = styleAttr ? ` style="${escapeHtml(styleAttr)}"` : '';

  if (component === 'heading') {
    const ref = props.text_ref || props.textRef || props.ref;
    const value = resolveTextRef(slide, ref, String(props.text || slide.title || 'Untitled'));
    const field = ref ? buildFieldMap(slide).get(String(ref)) : undefined;
    return `<h1 class="schema-canvas-component schema-canvas-heading" ${attrs}${style}>${wrapEditableText(field, value || 'Untitled')}</h1>`;
  }

  if (component === 'text' || component === 'quote' || component === 'callout') {
    const ref = props.text_ref || props.textRef || props.ref;
    const value = resolveTextRef(slide, ref, String(props.text || ''));
    if (String(ref || '').trim() === 'eyebrow' && isSystemPageLabel(value)) return '';
    const field = ref ? buildFieldMap(slide).get(String(ref)) : undefined;
    const className = component === 'quote' ? 'schema-quote' : component === 'callout' ? 'schema-callout' : 'schema-canvas-text';
    if (!value.trim()) {
      return field
        ? `<div class="schema-canvas-component ${className}" ${attrs}${style}>${wrapEditableText(field, '点击编辑文本')}</div>`
        : '';
    }
    return `<div class="schema-canvas-component ${className}" ${attrs}${style}>${wrapEditableText(field, value)}</div>`;
  }

  if (component === 'bullets') {
    const ref = props.items_ref || props.itemsRef || props.ref;
    const items = resolveListRef(slide, ref, Array.isArray(props.items) ? props.items.map((item) => String(item || '').trim()).filter(Boolean) : []);
    const field = ref ? buildFieldMap(slide).get(String(ref)) : undefined;
    if (items.length === 0) {
      return field
        ? `
<ul class="schema-canvas-component schema-bullets" ${attrs}${style}>
  <li>${wrapEditableText(field, '点击添加要点', 0)}</li>
</ul>
`.trim()
        : '';
    }
    return `
<ul class="schema-canvas-component schema-bullets" ${attrs}${style}>
  ${items.map((item, index) => `<li>${wrapEditableText(field, item, index)}</li>`).join('')}
</ul>
`.trim();
  }

  if (component === 'figure') {
    const ref = String(props.asset_ref || props.assetRef || props.ref || '').trim();
    const assetFromContent = resolveContentPath(slide.content, `assets.${ref}`) as Record<string, unknown> | undefined;
    const assetKey = String(
      props.asset_key
      || props.assetKey
      || assetFromContent?.assetKey
      || assetFromContent?.asset_key
      || ref
      || node.id,
    );
    const asset = buildAssetMap(slide).get(assetKey);
    const cardClass = `schema-canvas-component schema-visual-card${asset?.sourceType === 'paper_asset' || asset?.sourceType === 'generated' ? ' schema-transparent-figure-card' : ''}`;
    return `<div class="${cardClass}" ${attrs}${style}>${renderVisualAsset(asset, assetKey, String(props.label || 'Figure'), visualStyle.imageFit)}</div>`;
  }

  if (component === 'stat') {
    const valueRef = props.value_ref || props.valueRef || props.ref || props.text_ref || props.textRef;
    const labelRef = props.label_ref || props.labelRef;
    const value = resolveTextRef(slide, valueRef, String(props.value || props.text || ''));
    const label = resolveTextRef(slide, labelRef, String(props.label || ''));
    const valueField = valueRef ? buildFieldMap(slide).get(String(valueRef)) : undefined;
    const labelField = labelRef ? buildFieldMap(slide).get(String(labelRef)) : undefined;
    if (!value.trim() && !label.trim() && !valueField && !labelField) return '';
    return `
<div class="schema-canvas-component schema-stat-card" ${attrs}${style}>
  <div class="schema-stat-value">${wrapEditableText(valueField, value || '点击编辑数值')}</div>
  ${label || labelField ? `<div class="schema-stat-label">${labelField ? wrapEditableText(labelField, label || '点击编辑标签') : formatTextValue(label)}</div>` : ''}
</div>
`.trim();
  }

  if (component === 'table') {
    return renderCanvasTableMarkup(slide, node, attrs);
  }

  const fallbackText = resolveTextRef(slide, props.text_ref || props.textRef || props.ref, String(props.text || props.content || node.id || ''));
  if (fallbackText.trim()) {
    return `<div class="schema-canvas-component schema-canvas-text" ${attrs}${style}>${formatTextValue(fallbackText)}</div>`;
  }
  return '';
};

const renderCanvasNode = (slide: FrontendSlide, node: FrontendCanvasNode): string => {
  if (!node || !node.id) return '';
  if (node.type === 'component') {
    return renderCanvasComponent(slide, node);
  }
  const children = (node.children || []).map((child) => renderCanvasNode(slide, child)).filter(Boolean).join('');
  if (!children) return '';
  const layoutStyle = buildCanvasStyle(node);
  const visualStyle = buildCanvasStyleAttr(getCanvasNodeVisualStyle(slide, node, 'container'));
  const style = [layoutStyle, visualStyle].filter(Boolean).join(';');
  const styleAttr = style ? ` style="${escapeHtml(style)}"` : '';
  return `
<div class="schema-canvas-container" data-block-id="${escapeHtml(node.id)}" data-block-role="container" data-canvas-node-id="${escapeHtml(node.id)}"${styleAttr}>
  ${children}
</div>
`.trim();
};

const renderCanvasSlide = (slide: FrontendSlide) => `
<div class="slide-root schema-root template-canvas-schema" data-layout-family="${escapeHtml(slide.layoutFamily || 'custom')}">
  <div class="schema-shell">
    <div class="schema-main schema-canvas-root">
      ${slide.root ? renderCanvasNode(slide, slide.root) : ''}
    </div>
    ${!slide.root ? renderCanvasPlaceholder({ type: 'component', id: 'missing_root', component: 'placeholder' }, 'Missing canvas root') : ''}
    ${renderFixedPageLabel(slide)}
  </div>
</div>
`.trim();

const TEMPLATE_RENDERERS: Record<string, (slide: FrontendSlide, theme?: FrontendDeckTheme | null) => string> = {
  title_cover: (slide, theme) => {
    const context = buildSchemaSlideContext(slide);
    const coverBlocks = [
      ...context.images,
      ...context.callouts,
      ...context.lists,
      ...context.tables,
      ...context.bodyTexts,
      ...context.stats,
    ];
    const shouldRenderOnTitleRight = (block: FrontendSlideBlock) =>
      block.type === 'image' ? !isLeftColumnBlock(block) : isRightColumnBlock(block);
    const leftBlocks = renderColumnBlocks(
      slide,
      coverBlocks.filter((block) => !shouldRenderOnTitleRight(block)),
      { limit: 4, bareLists: true },
    );
    const rightBlocks = renderColumnBlocks(
      slide,
      coverBlocks.filter(shouldRenderOnTitleRight),
      { limit: 4, tallFirstImage: true },
    );
    return `
<div class="slide-root schema-root template-title-cover">
  <div class="schema-shell">
    <div class="schema-main schema-stack" style="justify-content:center;">
      ${renderHeader(slide, context)}
      <div class="schema-columns">
        <div class="schema-stack" data-insert-zone="left" data-insert-zone-label="左侧空白">
          ${leftBlocks.join('')}
        </div>
        <div class="schema-stack" data-insert-zone="right" data-insert-zone-label="右侧区域">
          ${rightBlocks.length > 0 ? rightBlocks.join('') : `<div class="schema-visual-card tall">${renderVisualAsset(undefined, 'main_visual', 'Main Visual')}</div>`}
        </div>
      </div>
    </div>
    ${renderFooter(slide, context, theme)}
  </div>
</div>
`.trim();
  },
  section_divider: (slide, theme) => {
    const context = buildSchemaSlideContext(slide);
    return `
<div class="slide-root schema-root template-section-divider">
  <div class="schema-shell" style="justify-content:center;">
    <div class="schema-surface" style="padding:36px 40px;">
      ${renderHeader(slide, context)}
      ${context.callouts[0] ? renderCalloutBlock(slide, context.callouts[0], 'schema-callout') : ''}
    </div>
    ${renderFooter(slide, context, theme)}
  </div>
</div>
`.trim();
  },
  text_focus: (slide, theme) => {
    const context = buildSchemaSlideContext(slide);
    const leftBlocks = [
      ...context.lists.filter((block) => !isRightColumnBlock(block)).map((block) => renderListBlock(slide, block, 'schema-bullets')),
      ...context.bodyTexts.filter(isLeftColumnBlock).map((block) => renderGenericCard(slide, block)),
      ...context.tables.filter(isLeftColumnBlock).map((block) => renderTableBlock(slide, block, 'schema-table-card')),
      ...context.callouts.filter(isLeftColumnBlock).map((block) => renderCalloutBlock(slide, block, 'schema-callout')),
      ...context.stats.filter(isLeftColumnBlock).map((block) => renderStatBlock(slide, block, 'schema-stat-card')),
    ].filter(Boolean).slice(0, 5);
    const rightBlocks = [
      ...context.lists.filter(isRightColumnBlock).map((block) => renderGenericCard(slide, block)),
      ...context.bodyTexts.filter((block) => !isLeftColumnBlock(block)).map((block) => renderGenericCard(slide, block)),
      ...context.tables.filter((block) => !isLeftColumnBlock(block)).map((block) => renderTableBlock(slide, block, 'schema-table-card')),
      ...context.callouts.filter((block) => !isLeftColumnBlock(block)).map((block) => renderCalloutBlock(slide, block, 'schema-callout')),
      ...context.stats.filter((block) => !isLeftColumnBlock(block)).map((block) => renderStatBlock(slide, block, 'schema-stat-card')),
    ].slice(0, 5);
    return `
<div class="slide-root schema-root template-text-focus">
  <div class="schema-shell">
    ${renderHeader(slide, context)}
    <div class="schema-main schema-columns equal">
      <div class="schema-main-copy" data-insert-zone="left" data-insert-zone-label="左侧区域">
        ${leftBlocks.join('')}
      </div>
      <div class="schema-stack" data-insert-zone="right" data-insert-zone-label="右侧区域">
        ${rightBlocks.join('')}
      </div>
    </div>
    ${renderFooter(slide, context, theme)}
  </div>
</div>
`.trim();
  },
  hero_visual: (slide, theme) => {
    const context = buildSchemaSlideContext(slide);
    const contentBlocks = [
      ...context.lists,
      ...context.bodyTexts,
      ...context.tables,
      ...context.callouts,
      ...context.stats,
    ];
    const leftBlocks = renderColumnBlocks(
      slide,
      contentBlocks.filter((block) => !isRightColumnBlock(block)),
      { limit: 4, bareLists: true },
    );
    const rightBlocks = renderColumnBlocks(
      slide,
      [
        ...context.images,
        ...contentBlocks.filter(isRightColumnBlock),
      ],
      { limit: 5, tallFirstImage: true },
    );
    return `
<div class="slide-root schema-root template-hero-visual">
  <div class="schema-shell">
    ${renderHeader(slide, context)}
    <div class="schema-main schema-columns">
      <div class="schema-main-copy" data-insert-zone="left" data-insert-zone-label="左侧区域">
        ${leftBlocks.join('')}
      </div>
      <div class="schema-stack" data-insert-zone="right" data-insert-zone-label="右侧区域">
        ${rightBlocks.join('')}
      </div>
    </div>
    ${renderFooter(slide, context, theme)}
  </div>
</div>
`.trim();
  },
  split_media: (slide, theme) => {
    const context = buildSchemaSlideContext(slide);
    const contentBlocks = [
      ...context.lists,
      ...context.bodyTexts,
      ...context.tables,
      ...context.callouts,
      ...context.stats,
    ];
    const leftBlocks = renderColumnBlocks(
      slide,
      contentBlocks.filter((block) => !isRightColumnBlock(block)),
      { limit: 5, bareLists: true },
    );
    const rightBlocks = renderColumnBlocks(
      slide,
      [
        ...context.images,
        ...contentBlocks.filter(isRightColumnBlock),
      ],
      { limit: 5, tallFirstImage: true },
    );
    return `
<div class="slide-root schema-root template-split-media">
  <div class="schema-shell">
    ${renderHeader(slide, context)}
    <div class="schema-main schema-columns">
      <div class="schema-stack" data-insert-zone="left" data-insert-zone-label="左侧区域">
        ${leftBlocks.join('')}
      </div>
      <div class="schema-stack" data-insert-zone="right" data-insert-zone-label="右侧区域">
        ${rightBlocks.join('')}
      </div>
    </div>
    ${renderFooter(slide, context, theme)}
  </div>
</div>
`.trim();
  },
  visual_compare: (slide, theme) => {
    const context = buildSchemaSlideContext(slide);
    const cards = [
      ...context.lists.map((block) => renderGenericCard(slide, block)),
      ...context.tables.map((block) => renderTableBlock(slide, block, 'schema-table-card')),
      ...context.bodyTexts.map((block) => renderGenericCard(slide, block)),
      ...context.callouts.map((block) => renderCalloutBlock(slide, block, 'schema-callout')),
      ...context.stats.map((block) => renderStatBlock(slide, block, 'schema-stat-card')),
    ].slice(0, 4);
    return `
<div class="slide-root schema-root template-visual-compare">
  <div class="schema-shell">
    ${renderHeader(slide, context)}
    <div class="schema-main schema-stack">
      <div class="schema-grid-2">
        ${context.images.slice(0, 2).map((block) => renderImageBlock(slide, block, 'schema-visual-card tall')).join('')}
      </div>
      <div class="schema-grid-2">
        ${cards.join('')}
      </div>
    </div>
    ${renderFooter(slide, context, theme)}
  </div>
</div>
`.trim();
  },
  insight_grid: (slide, theme) => {
    const context = buildSchemaSlideContext(slide);
    const cards = context.remaining.filter((block) => block.type !== 'image').slice(0, 6);
    return `
<div class="slide-root schema-root template-insight-grid">
  <div class="schema-shell">
    ${renderHeader(slide, context)}
    <div class="schema-main schema-grid-3">
      ${cards.map((block) => renderGenericCard(slide, block)).join('')}
    </div>
    ${renderFooter(slide, context, theme)}
  </div>
</div>
`.trim();
  },
  metrics_dashboard: (slide, theme) => {
    const context = buildSchemaSlideContext(slide);
    return `
<div class="slide-root schema-root template-metrics-dashboard">
  <div class="schema-shell">
    ${renderHeader(slide, context)}
    <div class="schema-main schema-stack">
      <div class="schema-grid-3">
        ${context.stats.slice(0, 3).map((block) => renderStatBlock(slide, block, 'schema-stat-card')).join('')}
      </div>
      <div class="schema-columns">
        <div class="schema-stack">
          ${context.lists[0] ? renderListBlock(slide, context.lists[0], 'schema-bullets') : ''}
          ${context.callouts[0] ? renderCalloutBlock(slide, context.callouts[0], 'schema-callout') : ''}
        </div>
        ${context.images[0] ? renderImageBlock(slide, context.images[0], 'schema-visual-card') : context.bodyTexts[0] ? renderGenericCard(slide, context.bodyTexts[0]) : ''}
      </div>
    </div>
    ${renderFooter(slide, context, theme)}
  </div>
</div>
`.trim();
  },
  timeline_overview: (slide, theme) => {
    const context = buildSchemaSlideContext(slide);
    const timelineBlock = context.lists[0];
    const timelineField = buildFieldMap(slide).get(getFieldKeyForBlock(slide, timelineBlock || ({} as FrontendSlideBlock)) || '');
    const timelineItems = (timelineField?.items || timelineBlock?.items || []).filter((item) => item.trim());
    const shouldRenderTimelineItems = timelineItems.length > 0 && (!timelineBlock || !hasLegacyChildForBlock(timelineBlock));
    return `
<div class="slide-root schema-root template-timeline-overview">
  <div class="schema-shell">
    ${renderHeader(slide, context)}
    <div class="schema-main schema-columns">
      <div class="schema-timeline"${timelineBlock ? ` data-block-id="${escapeHtml(timelineBlock.id)}" data-block-role="${escapeHtml(timelineBlock.role)}"` : ''}>
        ${shouldRenderTimelineItems ? timelineItems.map((item, index) => `
          <div class="schema-timeline-item">
            <div class="schema-timeline-dot"></div>
            <div class="schema-timeline-copy">${wrapEditableText(timelineField, item, index)}</div>
          </div>
        `).join('') : ''}
        ${timelineBlock?.children?.length ? renderBlockChildren(slide, timelineBlock) : ''}
      </div>
      <div class="schema-stack">
        ${context.images[0] ? renderImageBlock(slide, context.images[0], 'schema-visual-card tall') : ''}
        ${context.callouts[0] ? renderCalloutBlock(slide, context.callouts[0], 'schema-callout') : ''}
      </div>
    </div>
    ${renderFooter(slide, context, theme)}
  </div>
</div>
`.trim();
  },
  stacked_cards: (slide, theme) => {
    const context = buildSchemaSlideContext(slide);
    const cards = [
      ...context.images.map((block) => renderImageBlock(slide, block, 'schema-visual-card')),
      ...context.remaining.filter((block) => block.type !== 'image').map((block) => renderGenericCard(slide, block)),
    ].slice(0, 6);
    return `
<div class="slide-root schema-root template-stacked-cards">
  <div class="schema-shell">
    ${renderHeader(slide, context)}
    <div class="schema-main schema-grid-2">
      ${cards.join('')}
    </div>
    ${renderFooter(slide, context, theme)}
  </div>
</div>
`.trim();
  },
  quote_focus: (slide, theme) => {
    const context = buildSchemaSlideContext(slide);
    const quoteBlock = context.quotes[0] || context.callouts[0] || context.bodyTexts[0];
    return `
<div class="slide-root schema-root template-quote-focus">
  <div class="schema-shell">
    ${renderHeader(slide, context)}
    <div class="schema-main schema-stack" style="justify-content:center;">
      ${quoteBlock ? (quoteBlock.type === 'quote' ? renderQuoteBlock(slide, quoteBlock, 'schema-quote') : renderCalloutBlock(slide, quoteBlock, 'schema-quote')) : ''}
      <div class="schema-grid-2">
        ${context.lists[0] ? renderGenericCard(slide, context.lists[0]) : ''}
        ${context.images[0] ? renderImageBlock(slide, context.images[0], 'schema-visual-card') : context.callouts.slice(1, 2).map((block) => renderCalloutBlock(slide, block, 'schema-callout')).join('')}
      </div>
    </div>
    ${renderFooter(slide, context, theme)}
  </div>
</div>
`.trim();
  },
  dual_list: (slide, theme) => {
    const context = buildSchemaSlideContext(slide);
    return `
<div class="slide-root schema-root template-dual-list">
  <div class="schema-shell">
    ${renderHeader(slide, context)}
    <div class="schema-main schema-grid-2">
      ${context.lists.slice(0, 2).map((block) => renderGenericCard(slide, block)).join('')}
      ${context.bodyTexts[0] ? renderGenericCard(slide, context.bodyTexts[0]) : context.callouts[0] ? renderCalloutBlock(slide, context.callouts[0], 'schema-callout') : ''}
      ${context.images[0] ? renderImageBlock(slide, context.images[0], 'schema-visual-card') : ''}
    </div>
    ${renderFooter(slide, context, theme)}
  </div>
</div>
`.trim();
  },
};

export const isSchemaDrivenSlide = (slide: FrontendSlide) =>
  Boolean(slide.root || slide.schemaVersion || (Array.isArray(slide.blocks) && slide.blocks.length > 0));

export const buildSchemaSlideMarkup = (slide: FrontendSlide, theme?: FrontendDeckTheme | null) => {
  if (slide.renderEngine === 'canvas' && slide.root) {
    return `<style>${buildSchemaBaseCss(theme, slide.visualSpec || null)}</style>${renderCanvasSlide(slide)}`;
  }
  const templateKey = pickSchemaTemplateKey(slide);
  const renderer = TEMPLATE_RENDERERS[templateKey] || TEMPLATE_RENDERERS.text_focus;
  return `<style>${buildSchemaBaseCss(theme)}</style>${appendFixedPageLabel(renderer(slide, theme), slide)}`;
};
