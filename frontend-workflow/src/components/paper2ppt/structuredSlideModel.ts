import {
  FrontendDeckTheme,
  FrontendEditableField,
  FrontendSlide,
  FrontendDeckStyleFamily,
  FrontendVisualAsset,
  StructuredSlideLayoutType,
} from './types';

export const DESIGN_WIDTH = 1600;
export const DESIGN_HEIGHT = 900;

export interface StructuredSlideValidationResult {
  ok: boolean;
  issues: string[];
}

const fallbackTheme: FrontendDeckTheme = {
  themeName: 'paper2ppt_structured',
  visualMood: 'calm academic dark theme',
  styleFamily: 'modern',
  footerText: 'Paper2Any Structured PPT',
  sectionLabelTemplate: 'Slide {page_num:02d}/{slide_count:02d}',
  palette: {
    bg: '#0b1020',
    panel: 'rgba(15, 23, 42, 0.92)',
    primary: '#7dd3fc',
    secondary: '#38bdf8',
    accent: '#f59e0b',
    text: '#e2e8f0',
    muted: '#94a3b8',
  },
  typography: {
    titleFontStack: 'Georgia, "Times New Roman", serif',
    bodyFontStack: '"Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif',
    eyebrowSize: 18,
    titleSize: 56,
    summarySize: 26,
    bodySize: 24,
  },
  themeLock: {
    mustKeep: [],
    preferredLayoutPatterns: [],
    componentSignature: '',
    avoid: [],
  },
};

export const ensureDeckTheme = (theme?: FrontendDeckTheme | null): FrontendDeckTheme => ({
  ...fallbackTheme,
  ...theme,
  palette: {
    ...fallbackTheme.palette,
    ...(theme?.palette || {}),
  },
  typography: {
    ...fallbackTheme.typography,
    ...(theme?.typography || {}),
  },
  themeLock: {
    ...fallbackTheme.themeLock,
    ...(theme?.themeLock || {}),
  },
});

export const getDeckStyleFamily = (theme?: FrontendDeckTheme | null): FrontendDeckStyleFamily => {
  const candidate = String(theme?.styleFamily || '').trim().toLowerCase();
  if (candidate === 'business' || candidate === 'academic' || candidate === 'creative') {
    return candidate;
  }
  return 'modern';
};

export const getFieldMap = (slide: FrontendSlide) =>
  new Map(slide.editableFields.map((field) => [field.key, field]));

export const getField = (slide: FrontendSlide, key?: string): FrontendEditableField | undefined => {
  if (!key) return undefined;
  return getFieldMap(slide).get(key);
};

export const getTextValue = (slide: FrontendSlide, key?: string) => {
  const field = getField(slide, key);
  if (!field) return '';
  if (field.type === 'list') {
    return field.items.filter((item) => item.trim()).join(' • ');
  }
  return field.value || '';
};

export const getListValue = (slide: FrontendSlide, key?: string) => {
  const field = getField(slide, key);
  if (!field) return [];
  if (field.type === 'list') {
    return field.items.filter((item) => item.trim());
  }
  return field.value ? [field.value] : [];
};

export const getVisualAsset = (
  slide: FrontendSlide,
  key?: string,
  useOriginal = false,
): FrontendVisualAsset | undefined => {
  if (!key) return undefined;
  const asset = slide.visualAssets.find((item) => item.key === key);
  if (!asset) return undefined;
  if (useOriginal && asset.originalSrc) {
    return { ...asset, src: asset.originalSrc };
  }
  if (asset.previewSrc && !useOriginal) {
    return { ...asset, src: asset.previewSrc };
  }
  return asset;
};

const countChars = (values: string[]) =>
  values.reduce((total, value) => total + value.trim().length, 0);

const getListCharLimit = (slide: FrontendSlide) => {
  switch (slide.layoutType) {
    case 'bullets':
      return 900;
    case 'image_focus':
      return 720;
    case 'comparison':
    case 'two_column':
      return 520;
    case 'timeline':
      return 360;
    default:
      return 600;
  }
};

export const validateStructuredSlide = (slide: FrontendSlide): StructuredSlideValidationResult => {
  const issues: string[] = [];
  const title = slide.title.trim();
  if (!title) {
    issues.push('缺少标题。');
  }
  if (title.length > 110) {
    issues.push('标题过长，请压缩到 110 字以内。');
  }

  const allFields = slide.editableFields;
  const allText = allFields.flatMap((field) =>
    field.type === 'list' ? field.items : [field.value],
  );
  if (countChars(allText) > 2200) {
    issues.push('当前页文本总量过大，建议精简后再导出。');
  }

  const listFields = allFields.filter((field) => field.type === 'list');
  const listCharLimit = getListCharLimit(slide);
  for (const field of listFields) {
    if (field.items.length > 6) {
      issues.push(`列表「${field.label}」项数过多，请控制在 6 条以内。`);
    }
    if (countChars(field.items) > listCharLimit) {
      issues.push(`列表「${field.label}」内容过长，请适当精简。`);
    }
  }

  const imageRequiredLayouts: StructuredSlideLayoutType[] = ['image_focus'];
  if (imageRequiredLayouts.includes(slide.layoutType) && slide.visualAssets.length === 0) {
    issues.push('当前页需要图片槽位，但没有可用图片。');
  }

  if (slide.layoutType === 'cards_2x2') {
    const cardCount = slide.layoutData.type === 'cards_2x2' ? slide.layoutData.cards.length : 0;
    if (cardCount !== 4) {
      issues.push('卡片页必须恰好包含 4 张卡片。');
    }
  }

  if (slide.layoutType === 'timeline') {
    const count = slide.layoutData.type === 'timeline' ? slide.layoutData.timeline.length : 0;
    if (count < 3 || count > 5) {
      issues.push('时间线页必须包含 3 到 5 个节点。');
    }
  }

  if (slide.layoutType === 'comparison') {
    if (
      slide.layoutData.type === 'comparison'
      && getListValue(slide, slide.layoutData.leftPointsKey).length === 0
      && getListValue(slide, slide.layoutData.rightPointsKey).length === 0
    ) {
      issues.push('对比页至少需要一侧包含要点列表。');
    }
  }

  return {
    ok: issues.length === 0,
    issues,
  };
};

export const buildStructuredSlideRepairPrompt = (
  slide: FrontendSlide,
  validation: StructuredSlideValidationResult,
) => {
  const summary = validation.issues.join('；') || '请优化当前页的结构和信息密度。';
  return [
    '请保持这一页的主题、配色、页型和主要信息不变，只做结构修正。',
    `当前页型：${slide.layoutType}。`,
    `修正目标：${summary}`,
    '要求：',
    '1. 保持可编辑结构，不要改成自由 HTML/CSS。',
    '2. 压缩过长文本，避免信息过密。',
    '3. 保持标题层级清晰，卡片/列表数量不要超限。',
    '4. 如果当前页是 image_focus，保留图片槽位并围绕图片重排文本。',
  ].join('\n');
};
