import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';

import StructuredSlideCanvas from './StructuredSlideCanvas';
import { DESIGN_HEIGHT, DESIGN_WIDTH, ensureDeckTheme, getDeckStyleFamily } from './structuredSlideModel';
import { FrontendDeckTheme, FrontendSlide } from './types';

export interface HtmlDeckArtifactOptions {
  pptxCompatible?: boolean;
}

const buildDeckShellCss = (
  deckTheme: FrontendDeckTheme,
  options: HtmlDeckArtifactOptions = {},
) => {
  const theme = ensureDeckTheme(deckTheme);
  const palette = theme.palette;
  const family = getDeckStyleFamily(theme);
  const background = options.pptxCompatible
    ? palette.bg
    : family === 'academic'
      ? `linear-gradient(180deg, ${palette.bg}, ${palette.bg}), repeating-linear-gradient(180deg, transparent 0, transparent 35px, ${palette.primary}08 36px)`
      : family === 'business'
        ? `linear-gradient(135deg, ${palette.bg} 0%, ${palette.bg} 74%, ${palette.accent}12 100%)`
        : family === 'creative'
          ? `radial-gradient(circle at 12% 18%, ${palette.secondary}28 0%, transparent 24%), radial-gradient(circle at 84% 14%, ${palette.accent}24 0%, transparent 22%), linear-gradient(160deg, ${palette.bg} 0%, ${palette.bg} 62%, ${palette.primary}10 100%)`
          : `
              radial-gradient(circle at top right, ${palette.secondary}33 0%, transparent 28%),
              radial-gradient(circle at bottom left, ${palette.accent}22 0%, transparent 32%),
              ${palette.bg}
            `;

  return `
  html, body {
    margin: 0;
    padding: 0;
    background: ${background};
    color: ${palette.text};
  }
  body {
    overflow: auto;
  }
  .paper2ppt-html-deck {
    display: flex;
    flex-direction: column;
    gap: 24px;
    align-items: center;
    padding: 24px 0 40px;
    box-sizing: border-box;
  }
  .paper2ppt-html-slide {
    width: ${DESIGN_WIDTH}px;
    height: ${DESIGN_HEIGHT}px;
    flex: 0 0 auto;
    ${options.pptxCompatible ? `background:${palette.bg};` : ''}
  }
  .slide-root {
    width: ${DESIGN_WIDTH}px;
    height: ${DESIGN_HEIGHT}px;
    overflow: hidden;
    ${options.pptxCompatible ? `background:${palette.bg};` : ''}
  }
`.trim();
};

const renderSlideMarkup = (
  slide: FrontendSlide,
  index: number,
  deckTheme: FrontendDeckTheme | null | undefined,
  options: HtmlDeckArtifactOptions = {},
) => {
  const section = React.createElement(
    'section',
    { className: 'paper2ppt-html-slide', 'data-paper2ppt-slide-index': index },
    React.createElement(
      'div',
      { className: 'slide-root' },
      React.createElement(StructuredSlideCanvas, {
        slide,
        deckTheme: ensureDeckTheme(deckTheme),
        pptxCompatible: Boolean(options.pptxCompatible),
      }),
    ),
  );
  return renderToStaticMarkup(section);
};

export const buildHtmlDeckArtifact = (
  slides: FrontendSlide[],
  deckTheme?: FrontendDeckTheme | null,
  options: HtmlDeckArtifactOptions = {},
): string => {
  const theme = ensureDeckTheme(deckTheme);
  const bodyHtml = slides
    .map((slide, index) => renderSlideMarkup(slide, index, theme, options))
    .join('\n');

  return `<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Paper2Any HTML Deck</title>
    <style>${buildDeckShellCss(theme, options)}</style>
  </head>
  <body>
    <main class="paper2ppt-html-deck">
      ${bodyHtml}
    </main>
  </body>
</html>`;
};
