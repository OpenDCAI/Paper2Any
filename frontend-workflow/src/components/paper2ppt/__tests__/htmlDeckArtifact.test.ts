import { describe, expect, it } from 'vitest';
import { buildHtmlDeckArtifact } from '../htmlDeckArtifact';

describe('buildHtmlDeckArtifact', () => {
  it('writes a full HTML document with structured slides rendered', () => {
    const html = buildHtmlDeckArtifact([
      {
        slideId: 'slide-1',
        pageNum: 1,
        title: 'Slide 1',
        layoutType: 'cover',
        layoutData: {
          type: 'cover',
          titleKey: 'title',
          subtitleKey: 'subtitle',
        },
        editableFields: [
          { key: 'title', label: 'Title', type: 'text', value: 'Slide 1', items: [] },
          { key: 'subtitle', label: 'Subtitle', type: 'text', value: 'Subtitle 1', items: [] },
        ],
        visualAssets: [],
        status: 'done',
      },
      {
        slideId: 'slide-2',
        pageNum: 2,
        title: 'Slide 2',
        layoutType: 'section',
        layoutData: {
          type: 'section',
          titleKey: 'title',
          summaryKey: 'summary',
        },
        editableFields: [
          { key: 'title', label: 'Title', type: 'text', value: 'Slide 2', items: [] },
          { key: 'summary', label: 'Summary', type: 'textarea', value: 'Section summary', items: [] },
        ],
        visualAssets: [],
        status: 'done',
      },
    ] as any);

    expect(html).toContain('<!doctype html>');
    expect(html).toContain('paper2ppt-html-deck');
    expect(html).toContain('slide-root');
    expect(html).toContain('Slide 1');
    expect(html).toContain('Slide 2');
    expect(html).toContain('Subtitle 1');
    expect(html).toContain('Section summary');
  });

  it('uses the deck theme for the html shell background', () => {
    const html = buildHtmlDeckArtifact([
      {
        slideId: 'slide-1',
        pageNum: 1,
        title: 'Slide 1',
        layoutType: 'cover',
        layoutData: {
          type: 'cover',
          titleKey: 'title',
        },
        editableFields: [
          { key: 'title', label: 'Title', type: 'text', value: 'Slide 1', items: [] },
        ],
        visualAssets: [],
        status: 'done',
      },
    ] as any, {
      styleFamily: 'creative',
      palette: {
        bg: '#111827',
        panel: '#1f2937',
        primary: '#22d3ee',
        secondary: '#a78bfa',
        accent: '#f59e0b',
        text: '#f8fafc',
        muted: '#94a3b8',
      },
      typography: {
        titleFontStack: 'Inter',
        bodyFontStack: 'Inter',
        eyebrowSize: 18,
        titleSize: 56,
        summarySize: 24,
        bodySize: 20,
      },
      themeName: 'test',
      visualMood: 'test',
      footerText: '',
      sectionLabelTemplate: '',
      themeLock: {
        mustKeep: [],
        preferredLayoutPatterns: [],
        componentSignature: '',
        avoid: [],
      },
    } as any);

    expect(html).toContain('#111827');
    expect(html).toContain('radial-gradient');
  });

  it('can build a pptx-compatible artifact with solid slide backgrounds', () => {
    const html = (buildHtmlDeckArtifact as any)([
      {
        slideId: 'slide-1',
        pageNum: 1,
        title: 'Slide 1',
        layoutType: 'cover',
        layoutData: {
          type: 'cover',
          titleKey: 'title',
        },
        editableFields: [
          { key: 'title', label: 'Title', type: 'text', value: 'Slide 1', items: [] },
        ],
        visualAssets: [],
        status: 'done',
      },
    ], {
      styleFamily: 'creative',
      palette: {
        bg: '#F6F1E6',
        panel: '#FFFFFF',
        primary: '#2B2B2B',
        secondary: '#7A8C63',
        accent: '#C65A3A',
        text: '#1F2328',
        muted: '#6B7280',
      },
      typography: {
        titleFontStack: 'Inter',
        bodyFontStack: 'Inter',
        eyebrowSize: 18,
        titleSize: 56,
        summarySize: 24,
        bodySize: 20,
      },
      themeName: 'test',
      visualMood: 'test',
      footerText: '',
      sectionLabelTemplate: '',
      themeLock: {
        mustKeep: [],
        preferredLayoutPatterns: [],
        componentSignature: '',
        avoid: [],
      },
    }, { pptxCompatible: true });

    expect(html).toContain('background:#F6F1E6');
    expect(html).not.toContain('repeating-linear-gradient');
    expect(html).not.toContain('radial-gradient');
  });
});
