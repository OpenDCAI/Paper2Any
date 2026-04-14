import pptxgen from 'pptxgenjs';
import {
  FrontendDeckTheme,
  FrontendSlide,
} from './types';
import { ensureDeckTheme, getDeckStyleFamily, getListValue, getTextValue, getVisualAsset } from './structuredSlideModel';

const PX_PER_IN = 120;
const SHAPE_ROUND_RECT = 'roundRect' as const;
const SHAPE_ELLIPSE = 'ellipse' as const;
const SHAPE_LINE = 'line' as const;

const px = (value: number) => Number((value / PX_PER_IN).toFixed(3));

const stripHash = (value: string) => value.replace(/^#/, '');

const resolveHex = (value: string, fallback: string) => {
  const match = value.trim().match(/^#?([0-9a-fA-F]{6})$/);
  if (match) return match[1].toUpperCase();
  return stripHash(fallback).toUpperCase();
};

const normalizeFontFace = (value: string, fallback: string) => {
  const source = (value || fallback || '').trim();
  if (!source) return 'Arial';
  const primary = source
    .split(',')
    .map((item) => item.trim().replace(/^['"]+|['"]+$/g, ''))
    .find((item) => item.length > 0 && !/^(serif|sans-serif|monospace|cursive|fantasy|system-ui)$/i.test(item));
  return primary || fallback || 'Arial';
};

const toBulletText = (items: string[]) => items.filter(Boolean).map((item) => `• ${item}`).join('\n');

const blobToDataUrl = async (blob: Blob) => {
  if (typeof FileReader !== 'undefined') {
    return new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result || ''));
      reader.onerror = () => reject(reader.error || new Error('failed to read blob'));
      reader.readAsDataURL(blob);
    });
  }
  const bufferCtor = (globalThis as { Buffer?: { from(data: ArrayBuffer | Uint8Array): { toString(enc: string): string } } }).Buffer;
  if (!bufferCtor) {
    throw new Error('Buffer is not available in this runtime');
  }
  const bytes = bufferCtor.from(await blob.arrayBuffer());
  const mimeType = blob.type || 'application/octet-stream';
  return `data:${mimeType};base64,${bytes.toString('base64')}`;
};

const imageCache = new Map<string, Promise<string>>();

const fetchImageData = async (url: string, assetBaseUrl?: string) => {
  if (!imageCache.has(url)) {
    imageCache.set(
      url,
      (async () => {
        const resolvedUrl = url.startsWith('/') && assetBaseUrl ? new URL(url, assetBaseUrl).toString() : url;
        const res = await fetch(resolvedUrl);
        if (!res.ok) {
          throw new Error(`failed to fetch image: ${res.status}`);
        }
        return blobToDataUrl(await res.blob());
      })(),
    );
  }
  return imageCache.get(url)!;
};

const addTextBox = (
  slide: pptxgen.Slide,
  text: string,
  opts: {
    x: number;
    y: number;
    w: number;
    h: number;
    fontSize: number;
    color: string;
    fontFace: string;
    bold?: boolean;
    align?: 'left' | 'center' | 'right';
  },
) => {
  slide.addText(text, {
    x: px(opts.x),
    y: px(opts.y),
    w: px(opts.w),
    h: px(opts.h),
    fontFace: normalizeFontFace(opts.fontFace, 'Arial'),
    fontSize: opts.fontSize,
    color: opts.color,
    bold: opts.bold || false,
    align: opts.align || 'left',
    valign: 'top',
    margin: 0,
    breakLine: false,
    fit: 'shrink',
  });
};

const addPanel = (
  slide: pptxgen.Slide,
  theme: FrontendDeckTheme,
  box: { x: number; y: number; w: number; h: number },
) => {
  const family = getDeckStyleFamily(theme);
  slide.addShape(SHAPE_ROUND_RECT, {
    x: px(box.x),
    y: px(box.y),
    w: px(box.w),
    h: px(box.h),
    rectRadius: family === 'academic' ? 0.06 : family === 'business' ? 0.08 : 0.12,
    fill: { color: resolveHex(theme.palette.panel, '#0F172A'), transparency: family === 'academic' ? 2 : 8 },
    line: { color: resolveHex(theme.palette.primary, '#7dd3fc'), transparency: family === 'academic' ? 82 : 70, width: family === 'academic' ? 0.8 : 1 },
  });
};

const addFooterPill = (slide: pptxgen.Slide, theme: FrontendDeckTheme, text: string) => {
  const family = getDeckStyleFamily(theme);
  slide.addShape(SHAPE_ROUND_RECT, {
    x: px(1290),
    y: px(780),
    w: px(230),
    h: px(56),
    rectRadius: family === 'academic' ? 0.06 : 0.15,
    fill: { color: resolveHex(family === 'academic' ? theme.palette.panel : theme.palette.bg, '#0b1020'), transparency: family === 'academic' ? 0 : 20 },
    line: { color: resolveHex(theme.palette.accent, '#f59e0b'), transparency: family === 'academic' ? 72 : 55, width: 1 },
  });
  addTextBox(slide, text, {
    x: 1312,
    y: 795,
    w: 188,
    h: 24,
    fontSize: theme.typography.eyebrowSize,
    color: resolveHex(theme.palette.accent, '#f59e0b'),
    fontFace: theme.typography.bodyFontStack,
    bold: true,
    align: 'center',
  });
};

const addImageBox = async (
  slide: pptxgen.Slide,
  imageUrl: string,
  box: { x: number; y: number; w: number; h: number },
  assetBaseUrl?: string,
) => {
  const data = await fetchImageData(imageUrl, assetBaseUrl);
  slide.addImage({
    data,
    x: px(box.x),
    y: px(box.y),
    w: px(box.w),
    h: px(box.h),
    sizing: {
      type: 'cover',
      x: px(box.x),
      y: px(box.y),
      w: px(box.w),
      h: px(box.h),
    },
    rounding: true,
  });
};

export const exportStructuredSlidesToPptx = async ({
  slides,
  deckTheme,
  fileName,
  assetBaseUrl,
  outputType = 'blob',
}: {
  slides: FrontendSlide[];
  deckTheme?: FrontendDeckTheme | null;
  fileName?: string;
  assetBaseUrl?: string;
  outputType?: 'blob' | 'nodebuffer';
}) => {
  const theme = ensureDeckTheme(deckTheme);
  const family = getDeckStyleFamily(theme);
  const PptxGenCtor = ((pptxgen as unknown as { default?: unknown }).default ?? pptxgen) as new () => {
    layout: string;
    author: string;
    company: string;
    subject: string;
    title: string;
    addSlide(): pptxgen.Slide;
    write(options?: Record<string, unknown>): Promise<Blob | Uint8Array>;
  };
  const pres = new PptxGenCtor();
  pres.layout = 'LAYOUT_WIDE';
  pres.author = 'Paper2Any';
  pres.company = 'Paper2Any';
  pres.subject = 'Structured editable PPT';
  pres.title = fileName || 'paper2ppt_structured_editable';

  for (const structuredSlide of slides) {
    const slide = pres.addSlide();
    slide.background = { color: resolveHex(theme.palette.bg, '#0b1020') };
    if (family === 'business') {
      slide.addShape(SHAPE_ROUND_RECT, {
        x: 0,
        y: 0,
        w: px(1600),
        h: px(32),
        rectRadius: 0,
        fill: { color: resolveHex(theme.palette.accent, '#f59e0b') },
        line: { color: resolveHex(theme.palette.accent, '#f59e0b'), transparency: 100 },
      });
    } else if (family === 'creative') {
      slide.addShape(SHAPE_ELLIPSE, {
        x: px(1160),
        y: px(-40),
        w: px(360),
        h: px(280),
        fill: { color: resolveHex(theme.palette.secondary, '#38bdf8'), transparency: 72 },
        line: { color: resolveHex(theme.palette.secondary, '#38bdf8'), transparency: 100 },
      });
      slide.addShape(SHAPE_ELLIPSE, {
        x: px(-90),
        y: px(640),
        w: px(300),
        h: px(230),
        fill: { color: resolveHex(theme.palette.accent, '#f59e0b'), transparency: 78 },
        line: { color: resolveHex(theme.palette.accent, '#f59e0b'), transparency: 100 },
      });
    } else if (family === 'academic') {
      slide.addShape(SHAPE_LINE, {
        x: px(72),
        y: px(120),
        w: px(1456),
        h: 0,
        line: { color: resolveHex(theme.palette.primary, '#7dd3fc'), transparency: 82, width: 0.8 },
      });
    }
    const layout: any = structuredSlide.layoutData;

    const titleColor = resolveHex(theme.palette.text, '#e2e8f0');
    const primaryColor = resolveHex(theme.palette.primary, '#7dd3fc');
    const mutedColor = resolveHex(theme.palette.muted, '#94a3b8');
    const bodyFont = theme.typography.bodyFontStack;
    const titleFont = theme.typography.titleFontStack;
    const footerText = getTextValue(structuredSlide, layout.footerKey) || theme.footerText;
    const imageAsset = layout.type === 'image_focus'
      ? getVisualAsset(structuredSlide, layout.visualKey, true)
      : undefined;

    if (layout.eyebrowKey) {
      addTextBox(slide, getTextValue(structuredSlide, layout.eyebrowKey), {
        x: 88,
        y: 78,
        w: 340,
        h: 34,
        fontSize: theme.typography.eyebrowSize,
        color: primaryColor,
        fontFace: bodyFont,
        bold: true,
      });
    }

    switch (structuredSlide.layoutType) {
      case 'cover':
        addTextBox(slide, getTextValue(structuredSlide, layout.titleKey), {
          x: 250,
          y: 250,
          w: 1100,
          h: 140,
          fontSize: 30,
          color: titleColor,
          fontFace: titleFont,
          bold: true,
          align: 'center',
        });
        addTextBox(slide, getTextValue(structuredSlide, layout.subtitleKey), {
          x: 260,
          y: 398,
          w: 1080,
          h: 90,
          fontSize: 18,
          color: mutedColor,
          fontFace: bodyFont,
          align: 'center',
        });
        if (layout.presenterKey) {
          addTextBox(slide, getTextValue(structuredSlide, layout.presenterKey), {
            x: 420,
            y: 522,
            w: 760,
            h: 40,
            fontSize: 15,
            color: titleColor,
            fontFace: bodyFont,
            align: 'center',
          });
        }
        break;
      case 'section':
        addTextBox(slide, getTextValue(structuredSlide, layout.titleKey), {
          x: 88,
          y: 180,
          w: 760,
          h: 170,
          fontSize: 28,
          color: titleColor,
          fontFace: titleFont,
          bold: true,
        });
        addTextBox(slide, getTextValue(structuredSlide, layout.summaryKey), {
          x: 92,
          y: 352,
          w: 700,
          h: 160,
          fontSize: 18,
          color: mutedColor,
          fontFace: bodyFont,
        });
        addPanel(slide, theme, { x: 930, y: 220, w: 520, h: 280 });
        addTextBox(slide, getTextValue(structuredSlide, layout.quoteKey), {
          x: 972,
          y: 280,
          w: 430,
          h: 160,
          fontSize: 20,
          color: titleColor,
          fontFace: bodyFont,
        });
        break;
      case 'bullets':
        addTextBox(slide, getTextValue(structuredSlide, layout.titleKey), {
          x: 88,
          y: 148,
          w: 820,
          h: 110,
          fontSize: 28,
          color: titleColor,
          fontFace: titleFont,
          bold: true,
        });
        addTextBox(slide, getTextValue(structuredSlide, layout.summaryKey), {
          x: 92,
          y: 264,
          w: 760,
          h: 88,
          fontSize: 18,
          color: mutedColor,
          fontFace: bodyFont,
        });
        addTextBox(slide, toBulletText(getListValue(structuredSlide, layout.bulletsKey)), {
          x: 100,
          y: 370,
          w: 700,
          h: 250,
          fontSize: 17,
          color: titleColor,
          fontFace: bodyFont,
        });
        addPanel(slide, theme, { x: 960, y: 250, w: 470, h: 290 });
        addTextBox(slide, getTextValue(structuredSlide, layout.takeawayKey), {
          x: 995,
          y: 330,
          w: 390,
          h: 160,
          fontSize: 18,
          color: titleColor,
          fontFace: bodyFont,
        });
        break;
      case 'two_column':
        addTextBox(slide, getTextValue(structuredSlide, layout.titleKey), {
          x: 88,
          y: 138,
          w: 1000,
          h: 100,
          fontSize: 28,
          color: titleColor,
          fontFace: titleFont,
          bold: true,
        });
        addTextBox(slide, getTextValue(structuredSlide, layout.summaryKey), {
          x: 92,
          y: 248,
          w: 920,
          h: 70,
          fontSize: 17,
          color: mutedColor,
          fontFace: bodyFont,
        });
        addPanel(slide, theme, { x: 92, y: 336, w: 664, h: 356 });
        addPanel(slide, theme, { x: 796, y: 336, w: 664, h: 356 });
        addTextBox(slide, getTextValue(structuredSlide, layout.leftHeadingKey), {
          x: 126,
          y: 372,
          w: 580,
          h: 40,
          fontSize: 20,
          color: titleColor,
          fontFace: bodyFont,
          bold: true,
        });
        addTextBox(slide, getTextValue(structuredSlide, layout.leftBodyKey), {
          x: 126,
          y: 420,
          w: 580,
          h: 90,
          fontSize: 16,
          color: mutedColor,
          fontFace: bodyFont,
        });
        addTextBox(slide, toBulletText(getListValue(structuredSlide, layout.leftPointsKey)), {
          x: 132,
          y: 516,
          w: 560,
          h: 140,
          fontSize: 15,
          color: titleColor,
          fontFace: bodyFont,
        });
        addTextBox(slide, getTextValue(structuredSlide, layout.rightHeadingKey), {
          x: 830,
          y: 372,
          w: 580,
          h: 40,
          fontSize: 20,
          color: titleColor,
          fontFace: bodyFont,
          bold: true,
        });
        addTextBox(slide, getTextValue(structuredSlide, layout.rightBodyKey), {
          x: 830,
          y: 420,
          w: 580,
          h: 90,
          fontSize: 16,
          color: mutedColor,
          fontFace: bodyFont,
        });
        addTextBox(slide, toBulletText(getListValue(structuredSlide, layout.rightPointsKey)), {
          x: 836,
          y: 516,
          w: 560,
          h: 140,
          fontSize: 15,
          color: titleColor,
          fontFace: bodyFont,
        });
        break;
      case 'cards_2x2':
        addTextBox(slide, getTextValue(structuredSlide, layout.titleKey), {
          x: 88,
          y: 138,
          w: 1000,
          h: 100,
          fontSize: 28,
          color: titleColor,
          fontFace: titleFont,
          bold: true,
        });
        addTextBox(slide, getTextValue(structuredSlide, layout.summaryKey), {
          x: 92,
          y: 248,
          w: 920,
          h: 70,
          fontSize: 17,
          color: mutedColor,
          fontFace: bodyFont,
        });
        layout.cards.forEach((card: any, index: number) => {
          const col = index % 2;
          const row = Math.floor(index / 2);
          const x = col === 0 ? 92 : 786;
          const y = row === 0 ? 340 : 530;
          addPanel(slide, theme, { x, y, w: 640, h: 160 });
          addTextBox(slide, getTextValue(structuredSlide, card.titleKey), {
            x: x + 28,
            y: y + 22,
            w: 580,
            h: 36,
            fontSize: 18,
            color: titleColor,
            fontFace: bodyFont,
            bold: true,
          });
          addTextBox(slide, getTextValue(structuredSlide, card.bodyKey), {
            x: x + 28,
            y: y + 62,
            w: 570,
            h: 70,
            fontSize: 15,
            color: mutedColor,
            fontFace: bodyFont,
          });
        });
        break;
      case 'image_focus':
        addTextBox(slide, getTextValue(structuredSlide, layout.titleKey), {
          x: 88,
          y: 148,
          w: 700,
          h: 110,
          fontSize: 28,
          color: titleColor,
          fontFace: titleFont,
          bold: true,
        });
        addTextBox(slide, getTextValue(structuredSlide, layout.summaryKey), {
          x: 92,
          y: 264,
          w: 650,
          h: 88,
          fontSize: 18,
          color: mutedColor,
          fontFace: bodyFont,
        });
        addTextBox(slide, toBulletText(getListValue(structuredSlide, layout.bulletsKey)), {
          x: 100,
          y: 380,
          w: 610,
          h: 220,
          fontSize: 16,
          color: titleColor,
          fontFace: bodyFont,
        });
        addPanel(slide, theme, { x: 850, y: 220, w: 600, h: 430 });
        if (imageAsset?.src) {
          await addImageBox(slide, imageAsset.src, { x: 874, y: 244, w: 552, h: 324 }, assetBaseUrl);
        }
        addTextBox(slide, getTextValue(structuredSlide, layout.visualCaptionKey), {
          x: 882,
          y: 580,
          w: 530,
          h: 48,
          fontSize: 14,
          color: mutedColor,
          fontFace: bodyFont,
        });
        break;
      case 'comparison':
        addTextBox(slide, getTextValue(structuredSlide, layout.titleKey), {
          x: 88,
          y: 138,
          w: 1000,
          h: 100,
          fontSize: 28,
          color: titleColor,
          fontFace: titleFont,
          bold: true,
        });
        addTextBox(slide, getTextValue(structuredSlide, layout.summaryKey), {
          x: 92,
          y: 248,
          w: 920,
          h: 70,
          fontSize: 17,
          color: mutedColor,
          fontFace: bodyFont,
        });
        addPanel(slide, theme, { x: 92, y: 340, w: 640, h: 330 });
        addPanel(slide, theme, { x: 786, y: 340, w: 640, h: 330 });
        addTextBox(slide, getTextValue(structuredSlide, layout.leftTitleKey), {
          x: 126,
          y: 376,
          w: 560,
          h: 38,
          fontSize: 20,
          color: titleColor,
          fontFace: bodyFont,
          bold: true,
        });
        addTextBox(slide, toBulletText(getListValue(structuredSlide, layout.leftPointsKey)), {
          x: 132,
          y: 426,
          w: 540,
          h: 200,
          fontSize: 16,
          color: titleColor,
          fontFace: bodyFont,
        });
        addTextBox(slide, getTextValue(structuredSlide, layout.rightTitleKey), {
          x: 820,
          y: 376,
          w: 560,
          h: 38,
          fontSize: 20,
          color: titleColor,
          fontFace: bodyFont,
          bold: true,
        });
        addTextBox(slide, toBulletText(getListValue(structuredSlide, layout.rightPointsKey)), {
          x: 826,
          y: 426,
          w: 540,
          h: 200,
          fontSize: 16,
          color: titleColor,
          fontFace: bodyFont,
        });
        break;
      case 'timeline':
        addTextBox(slide, getTextValue(structuredSlide, layout.titleKey), {
          x: 88,
          y: 138,
          w: 1000,
          h: 100,
          fontSize: 28,
          color: titleColor,
          fontFace: titleFont,
          bold: true,
        });
        addTextBox(slide, getTextValue(structuredSlide, layout.summaryKey), {
          x: 92,
          y: 248,
          w: 920,
          h: 70,
          fontSize: 17,
          color: mutedColor,
          fontFace: bodyFont,
        });
        layout.timeline.forEach((item: any, index: number) => {
          const total = layout.timeline.length;
          const width = Math.min(320, Math.floor(1220 / total));
          const x = 110 + index * (width + 22);
          addPanel(slide, theme, { x, y: 360, w: width, h: 250 });
          slide.addShape(SHAPE_ELLIPSE, {
            x: px(x + 18),
            y: px(338),
            w: px(14),
            h: px(14),
            fill: { color: resolveHex(theme.palette.accent, '#f59e0b') },
            line: { color: resolveHex(theme.palette.accent, '#f59e0b'), transparency: 100 },
          });
          if (index < total - 1) {
            slide.addShape(SHAPE_LINE, {
              x: px(x + 32),
              y: px(345),
              w: px(width + 22),
              h: 0,
              line: { color: primaryColor, transparency: 55, width: 1.2 },
            });
          }
          addTextBox(slide, getTextValue(structuredSlide, item.labelKey), {
            x: x + 24,
            y: 386,
            w: width - 48,
            h: 34,
            fontSize: 17,
            color: primaryColor,
            fontFace: bodyFont,
            bold: true,
          });
          addTextBox(slide, getTextValue(structuredSlide, item.bodyKey), {
            x: x + 24,
            y: 430,
            w: width - 48,
            h: 128,
            fontSize: 15,
            color: titleColor,
            fontFace: bodyFont,
          });
        });
        break;
      default:
        addTextBox(slide, structuredSlide.title, {
          x: 88,
          y: 200,
          w: 1200,
          h: 120,
          fontSize: 28,
          color: titleColor,
          fontFace: titleFont,
          bold: true,
        });
    }

    if (footerText) {
      addFooterPill(slide, theme, footerText);
    }
  }

  const outputFileName = fileName || 'paper2ppt_structured_editable.pptx';
  const result = await pres.write({ outputType, compression: true });
  if (outputType === 'nodebuffer') {
    return {
      buffer: result as Uint8Array,
      fileName: outputFileName,
    };
  }
  return {
    blob: result as Blob,
    fileName: outputFileName,
  };
};
