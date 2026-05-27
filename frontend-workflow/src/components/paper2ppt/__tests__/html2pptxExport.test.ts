import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const { buildHtmlDeckArtifactMock } = vi.hoisted(() => ({
  buildHtmlDeckArtifactMock: vi.fn(),
}));

vi.mock('../htmlDeckArtifact', () => ({
  buildHtmlDeckArtifact: buildHtmlDeckArtifactMock,
}));

import { exportHtmlDeckToPptx } from '../html2pptxExport';

describe('exportHtmlDeckToPptx', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  beforeEach(() => {
    buildHtmlDeckArtifactMock.mockReset();
    buildHtmlDeckArtifactMock.mockReturnValue(`
      <!doctype html>
      <html>
        <head>
          <style>.paper2ppt-html-slide { color: rgb(1, 2, 3); }</style>
        </head>
        <body>
          <main class="paper2ppt-html-deck">
            <section class="paper2ppt-html-slide">
              <div class="slide-root">slide</div>
            </section>
          </main>
        </body>
      </html>
    `);

    vi.stubGlobal('window', {});
    vi.stubGlobal('document', {
      body: {
        appendChild: vi.fn(),
      },
      createElement: vi.fn(() => {
        const sandbox: any = {
          dataset: {},
          style: {},
          firstChild: null,
          innerHTML: '',
          insertedNodes: [] as unknown[],
          insertBefore(node: unknown) {
            sandbox.insertedNodes.push(node);
          },
          querySelectorAll: () => [{ nodeName: 'DIV' }],
          remove: vi.fn(),
        };
        return sandbox;
      }),
    });
    vi.stubGlobal('DOMParser', class {
      parseFromString(html: string) {
        return {
          head: {
            querySelectorAll: () => [{
              cloneNode: () => ({ cloned: true }),
            }],
          },
          body: {
            innerHTML: html,
          },
        };
      }
    });
    (window as any).domToPptx = {
      exportToPptx: vi.fn().mockResolvedValue(new Blob(['pptx'], { type: 'application/vnd.openxmlformats-officedocument.presentationml.presentation' })),
    };
  });

  it('passes the deck theme through to the HTML deck builder', async () => {
    const deckTheme = {
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
    } as const;

    const blob = await exportHtmlDeckToPptx({
      slides: [
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
        } as any,
      ],
      deckTheme: deckTheme as any,
      fileName: 'deck.pptx',
    });

    expect(blob).toBeInstanceOf(Blob);
    expect(buildHtmlDeckArtifactMock).toHaveBeenCalledWith(
      expect.any(Array),
      deckTheme,
      expect.objectContaining({ pptxCompatible: true }),
    );
    expect((window as any).domToPptx.exportToPptx).toHaveBeenCalledWith(
      expect.any(Array),
      expect.objectContaining({
        fileName: 'deck.pptx',
        skipDownload: true,
        svgAsVector: true,
      }),
    );
  });
});
