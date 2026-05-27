import { buildHtmlDeckArtifact } from './htmlDeckArtifact';
import { FrontendDeckTheme, FrontendSlide } from './types';

const DOM_TO_PPTX_BUNDLE_URL = '/vendor/dom-to-pptx.bundle.js';

declare global {
  interface Window {
    domToPptx?: {
      exportToPptx: (target: Element[] | Element | string, options?: Record<string, unknown>) => Promise<Blob>;
    };
  }
}

let domToPptxLoadPromise: Promise<void> | null = null;

const loadDomToPptxBundle = async () => {
  if (window.domToPptx?.exportToPptx) {
    return;
  }

  if (!domToPptxLoadPromise) {
    domToPptxLoadPromise = new Promise<void>((resolve, reject) => {
      const existing = document.querySelector<HTMLScriptElement>('script[data-dom-to-pptx="true"]');
      if (existing && window.domToPptx?.exportToPptx) {
        resolve();
        return;
      }

      const script = document.createElement('script');
      script.src = DOM_TO_PPTX_BUNDLE_URL;
      script.async = true;
      script.dataset.domToPptx = 'true';
      script.onload = () => resolve();
      script.onerror = () => reject(new Error('dom-to-pptx 资源加载失败'));
      document.head.appendChild(script);
    }).catch((error) => {
      domToPptxLoadPromise = null;
      throw error;
    });
  }

  await domToPptxLoadPromise;

  if (!window.domToPptx?.exportToPptx) {
    throw new Error('dom-to-pptx 未正确加载');
  }
};

const createExportSandbox = (html: string) => {
  const parser = new DOMParser();
  const parsed = parser.parseFromString(html, 'text/html');
  const sandbox = document.createElement('div');
  sandbox.dataset.paper2pptHtml2Pptx = 'true';
  sandbox.style.position = 'fixed';
  sandbox.style.left = '-100000px';
  sandbox.style.top = '0';
  sandbox.style.width = '1600px';
  sandbox.style.minHeight = '900px';
  sandbox.style.pointerEvents = 'none';
  sandbox.style.overflow = 'hidden';
  sandbox.style.zIndex = '-1';
  sandbox.style.background = 'transparent';
  sandbox.innerHTML = parsed.body.innerHTML;

  Array.from(parsed.head.querySelectorAll('style')).forEach((style) => {
    sandbox.insertBefore(style.cloneNode(true), sandbox.firstChild);
  });

  document.body.appendChild(sandbox);
  const slideRoots = Array.from(sandbox.querySelectorAll<HTMLElement>('.paper2ppt-html-slide .slide-root'));

  if (slideRoots.length === 0) {
    sandbox.remove();
    throw new Error('HTML 转 PPTX 失败：未找到任何可导出的幻灯片');
  }

  return { sandbox, slideRoots };
};

export const exportHtmlDeckToPptx = async ({
  slides,
  deckTheme,
  fileName,
}: {
  slides: FrontendSlide[];
  deckTheme?: FrontendDeckTheme | null;
  fileName?: string;
}): Promise<Blob> => {
  if (slides.length === 0) {
    throw new Error('没有可导出的 HTML 页面');
  }

  await loadDomToPptxBundle();

  const html = buildHtmlDeckArtifact(slides, deckTheme, { pptxCompatible: true });
  const { sandbox, slideRoots } = createExportSandbox(html);

  try {
    const blob = await window.domToPptx!.exportToPptx(slideRoots, {
      fileName: fileName || 'paper2ppt_html2pptx.pptx',
      skipDownload: true,
      svgAsVector: true,
    });
    return blob;
  } finally {
    sandbox.remove();
  }
};
