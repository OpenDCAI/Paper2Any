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
  const family = getDeckStyleFamily(deckTheme);
  const background = options.pptxCompatible
    ? deckTheme.palette.bg
    : family === 'academic'
      ? `linear-gradient(180deg, ${deckTheme.palette.bg}, ${deckTheme.palette.bg}), repeating-linear-gradient(180deg, transparent 0, transparent 35px, ${deckTheme.palette.primary}08 36px)`
      : family === 'business'
        ? `linear-gradient(135deg, ${deckTheme.palette.bg} 0%, ${deckTheme.palette.bg} 74%, ${deckTheme.palette.accent}12 100%)`
        : family === 'creative'
          ? `radial-gradient(circle at 12% 18%, ${deckTheme.palette.secondary}28 0%, transparent 24%), radial-gradient(circle at 84% 14%, ${deckTheme.palette.accent}24 0%, transparent 22%), linear-gradient(160deg, ${deckTheme.palette.bg} 0%, ${deckTheme.palette.bg} 62%, ${deckTheme.palette.primary}10 100%)`
          : `
              radial-gradient(circle at top right, ${deckTheme.palette.secondary}33 0%, transparent 28%),
              radial-gradient(circle at bottom left, ${deckTheme.palette.accent}22 0%, transparent 32%),
              ${deckTheme.palette.bg}
            `;

  return `
  html, body {
    margin: 0;
    padding: 0;
    background: ${background};
    color: ${deckTheme.palette.text};
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
    ${options.pptxCompatible ? `background:${deckTheme.palette.bg};` : ''}
  }
  .slide-root {
    width: ${DESIGN_WIDTH}px;
    height: ${DESIGN_HEIGHT}px;
    overflow: hidden;
    ${options.pptxCompatible ? `background:${deckTheme.palette.bg};` : ''}
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
