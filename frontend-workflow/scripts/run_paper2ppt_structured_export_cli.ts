#!/usr/bin/env node

import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { exportStructuredSlidesToPptx } from '../src/components/paper2ppt/exportStructuredSlides.ts';
import type { FrontendDeckTheme, FrontendSlide } from '../src/components/paper2ppt/types.ts';

interface CliArgs {
  slidesJson: string;
  themeJson?: string;
  output: string;
  assetBaseUrl?: string;
}

const normalizeLayoutData = (layoutData: any) => {
  if (!layoutData || typeof layoutData !== 'object') {
    return { type: 'bullets', titleKey: 'title' };
  }
  return {
    ...layoutData,
    eyebrowKey: layoutData.eyebrow_key || layoutData.eyebrowKey,
    titleKey: layoutData.title_key || layoutData.titleKey,
    footerKey: layoutData.footer_key || layoutData.footerKey,
    summaryKey: layoutData.summary_key || layoutData.summaryKey,
    subtitleKey: layoutData.subtitle_key || layoutData.subtitleKey,
    presenterKey: layoutData.presenter_key || layoutData.presenterKey,
    quoteKey: layoutData.quote_key || layoutData.quoteKey,
    bulletsKey: layoutData.bullets_key || layoutData.bulletsKey,
    takeawayKey: layoutData.takeaway_key || layoutData.takeawayKey,
    leftHeadingKey: layoutData.left_heading_key || layoutData.leftHeadingKey,
    leftBodyKey: layoutData.left_body_key || layoutData.leftBodyKey,
    leftPointsKey: layoutData.left_points_key || layoutData.leftPointsKey,
    rightHeadingKey: layoutData.right_heading_key || layoutData.rightHeadingKey,
    rightBodyKey: layoutData.right_body_key || layoutData.rightBodyKey,
    rightPointsKey: layoutData.right_points_key || layoutData.rightPointsKey,
    visualKey: layoutData.visual_key || layoutData.visualKey,
    visualCaptionKey: layoutData.visual_caption_key || layoutData.visualCaptionKey,
    leftTitleKey: layoutData.left_title_key || layoutData.leftTitleKey,
    rightTitleKey: layoutData.right_title_key || layoutData.rightTitleKey,
    cards: Array.isArray(layoutData.cards)
      ? layoutData.cards.map((card: any) => ({
          titleKey: card.title_key || card.titleKey,
          bodyKey: card.body_key || card.bodyKey,
        }))
      : [],
    timeline: Array.isArray(layoutData.timeline)
      ? layoutData.timeline.map((item: any) => ({
          labelKey: item.label_key || item.labelKey,
          bodyKey: item.body_key || item.bodyKey,
        }))
      : [],
  };
};

const normalizeThemeLock = (themeLock: any) => ({
  mustKeep: Array.isArray(themeLock?.must_keep || themeLock?.mustKeep)
    ? (themeLock.must_keep || themeLock.mustKeep).map((item: unknown) => String(item || '')).filter(Boolean)
    : [],
  preferredLayoutPatterns: Array.isArray(themeLock?.preferred_layout_patterns || themeLock?.preferredLayoutPatterns)
    ? (themeLock.preferred_layout_patterns || themeLock.preferredLayoutPatterns)
        .map((item: unknown) => String(item || ''))
        .filter(Boolean)
    : [],
  componentSignature: String(themeLock?.component_signature || themeLock?.componentSignature || ''),
  avoid: Array.isArray(themeLock?.avoid)
    ? themeLock.avoid.map((item: unknown) => String(item || '')).filter(Boolean)
    : [],
});

const normalizeTypography = (typography: any) => ({
  titleFontStack: String(typography?.title_font_stack || typography?.titleFontStack || ''),
  bodyFontStack: String(typography?.body_font_stack || typography?.bodyFontStack || ''),
  eyebrowSize: Number(typography?.eyebrow_size || typography?.eyebrowSize || 18),
  titleSize: Number(typography?.title_size || typography?.titleSize || 56),
  summarySize: Number(typography?.summary_size || typography?.summarySize || 26),
  bodySize: Number(typography?.body_size || typography?.bodySize || 24),
});

const normalizeFrontendSlides = (slides: any[]): FrontendSlide[] =>
  slides.map((slide: any, index: number) => ({
    slideId: String(slide.slide_id || slide.slideId || index + 1),
    pageNum: Number(slide.page_num || slide.pageNum || index + 1),
    title: slide.title || `Slide ${index + 1}`,
    layoutType: slide.layout_type || slide.layoutType || 'bullets',
    layoutData: normalizeLayoutData(slide.layout_data || slide.layoutData || {}),
    editableFields: Array.isArray(slide.editable_fields || slide.editableFields)
      ? (slide.editable_fields || slide.editableFields).map((field: any) => ({
          key: String(field.key || ''),
          label: String(field.label || field.key || ''),
          type: field.type === 'list' || field.type === 'textarea' ? field.type : 'text',
          value: String(field.value || ''),
          items: Array.isArray(field.items) ? field.items.map((item: any) => String(item || '')) : [],
        }))
      : [],
    visualAssets: Array.isArray(slide.visual_assets || slide.visualAssets)
      ? (slide.visual_assets || slide.visualAssets).map((asset: any, assetIndex: number) => ({
          key: String(asset.key || `main_visual_${assetIndex + 1}`),
          label: String(asset.label || asset.key || `Image ${assetIndex + 1}`),
          src: String(asset.src || ''),
          previewSrc: String(asset.preview_src || asset.previewSrc || asset.src || ''),
          originalSrc: String(asset.original_src || asset.originalSrc || asset.storage_path || asset.storagePath || asset.src || ''),
          alt: String(asset.alt || asset.label || asset.key || ''),
          sourceType: asset.source_type === 'paper_asset' || asset.sourceType === 'paper_asset'
            ? 'paper_asset'
            : asset.source_type === 'upload' || asset.sourceType === 'upload'
              ? 'upload'
              : 'generated',
          storagePath: asset.storage_path || asset.storagePath || undefined,
          previewStoragePath: asset.preview_storage_path || asset.previewStoragePath || undefined,
          prompt: asset.prompt || undefined,
          style: asset.style || undefined,
        }))
      : [],
    generationNote: slide.generation_note || slide.generationNote || '',
    status: slide.status === 'processing' || slide.status === 'pending' ? slide.status : 'done',
    review: {
      status: 'idle',
      summary: '',
      issues: [],
    },
  }));

const normalizeFrontendDeckTheme = (theme: any): FrontendDeckTheme | undefined => {
  if (!theme || typeof theme !== 'object') return undefined;
  const themeLock = theme.theme_lock || theme.themeLock || {};
  return {
    themeName: String(theme.theme_name || theme.themeName || 'locked_deck_theme'),
    visualMood: String(theme.visual_mood || theme.visualMood || ''),
    styleFamily: String(theme.style_family || theme.styleFamily || 'modern') as FrontendDeckTheme['styleFamily'],
    footerText: String(theme.footer_text || theme.footerText || ''),
    sectionLabelTemplate: String(theme.section_label_template || theme.sectionLabelTemplate || ''),
    palette: {
      bg: String(theme.palette?.bg || '#0b1020'),
      panel: String(theme.palette?.panel || 'rgba(15, 23, 42, 0.92)'),
      primary: String(theme.palette?.primary || '#7dd3fc'),
      secondary: String(theme.palette?.secondary || '#38bdf8'),
      accent: String(theme.palette?.accent || '#f59e0b'),
      text: String(theme.palette?.text || '#e2e8f0'),
      muted: String(theme.palette?.muted || '#94a3b8'),
    },
    typography: normalizeTypography(theme.typography || {}),
    themeLock: normalizeThemeLock(themeLock),
  };
};

const usage = () => {
  console.log(`Usage:
  npm run export:structured-ppt -- --slides-json /path/to/frontend_slides.json --theme-json /path/to/frontend_theme.json --output /path/to/out.pptx [--asset-base-url http://127.0.0.1:8000]
`);
};

const parseArgs = (): CliArgs => {
  const args = process.argv.slice(2);
  const read = (name: string) => {
    const index = args.indexOf(name);
    if (index === -1 || index + 1 >= args.length) return '';
    return args[index + 1];
  };

  const slidesJson = read('--slides-json');
  const output = read('--output');
  const themeJson = read('--theme-json') || undefined;
  const assetBaseUrl = read('--asset-base-url') || undefined;

  if (!slidesJson || !output) {
    usage();
    throw new Error('Missing required --slides-json or --output');
  }

  return { slidesJson, themeJson, output, assetBaseUrl };
};

const main = async () => {
  const args = parseArgs();
  const slides = normalizeFrontendSlides(JSON.parse(await readFile(args.slidesJson, 'utf-8')));
  const theme = args.themeJson
    ? normalizeFrontendDeckTheme(JSON.parse(await readFile(args.themeJson, 'utf-8')))
    : undefined;

  const result = await exportStructuredSlidesToPptx({
    slides,
    deckTheme: theme,
    fileName: path.basename(args.output),
    assetBaseUrl: args.assetBaseUrl,
    outputType: 'nodebuffer',
  });

  if (!('buffer' in result)) {
    throw new Error('Expected nodebuffer export result');
  }

  await mkdir(path.dirname(args.output), { recursive: true });
  await writeFile(args.output, Buffer.from(result.buffer));
  console.log(JSON.stringify({
    success: true,
    output: path.resolve(args.output),
    slide_count: slides.length,
    asset_base_url: args.assetBaseUrl || '',
  }));
};

main().catch((error) => {
  console.error(String(error?.stack || error?.message || error));
  process.exit(1);
});
