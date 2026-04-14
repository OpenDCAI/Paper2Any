import React from 'react';
import {
  FrontendDeckTheme,
  FrontendEditableField,
  FrontendSlide,
  FrontendVisualAsset,
} from './types';
import {
  DESIGN_HEIGHT,
  DESIGN_WIDTH,
  ensureDeckTheme,
  getDeckStyleFamily,
  getField,
  getListValue,
  getTextValue,
  getVisualAsset,
} from './structuredSlideModel';

interface StructuredSlideCanvasProps {
  slide: FrontendSlide;
  deckTheme?: FrontendDeckTheme | null;
  useOriginalAssets?: boolean;
}

const toCssFont = (value: string) => value || 'system-ui, sans-serif';

const basePanelStyle = (theme: FrontendDeckTheme): React.CSSProperties => {
  const family = getDeckStyleFamily(theme);
  if (family === 'academic') {
    return {
      background: theme.palette.panel,
      border: `1.5px solid ${theme.palette.primary}22`,
      borderRadius: 18,
      boxShadow: '0 16px 34px rgba(15, 23, 42, 0.08)',
    };
  }
  if (family === 'business') {
    return {
      background: theme.palette.panel,
      border: `1px solid ${theme.palette.primary}28`,
      borderRadius: 18,
      boxShadow: '0 22px 44px rgba(15, 23, 42, 0.16)',
    };
  }
  if (family === 'creative') {
    return {
      background: theme.palette.panel,
      border: `1px solid ${theme.palette.primary}26`,
      borderRadius: 32,
      boxShadow: '0 28px 60px rgba(99, 102, 241, 0.12)',
    };
  }
  return {
    background: theme.palette.panel,
    border: `1px solid ${theme.palette.primary}33`,
    borderRadius: 28,
    boxShadow: '0 30px 60px rgba(15, 23, 42, 0.35)',
  };
};

const editableText = (
  field: FrontendEditableField | undefined,
  className: string,
  style: React.CSSProperties,
  tag: 'div' | 'p' | 'h1' | 'h2' | 'span' = 'div',
) => {
  if (!field) return null;
  const Tag = tag;
  const value = field.type === 'list' ? field.items.filter(Boolean).join(' • ') : field.value;
  return (
    <Tag
      className={className}
      style={style}
      data-edit-key={field.key}
      data-edit-type={field.type}
    >
      {value}
    </Tag>
  );
};

const editableList = (
  field: FrontendEditableField | undefined,
  theme: FrontendDeckTheme,
  compact = false,
) => {
  if (!field) return null;
  const items = field.type === 'list'
    ? field.items.filter((item) => item.trim())
    : (field.value ? [field.value] : []);
  return (
    <ul
      style={{
        margin: 0,
        paddingLeft: compact ? 22 : 26,
        display: 'grid',
        gap: compact ? 10 : 14,
        fontSize: compact ? Math.max(18, theme.typography.bodySize - 2) : theme.typography.bodySize,
        lineHeight: 1.35,
        color: theme.palette.text,
        fontFamily: toCssFont(theme.typography.bodyFontStack),
      }}
    >
      {items.map((item, index) => (
        <li key={`${field.key}-${index}`}>
          <span data-edit-key={field.key} data-edit-type="list" data-edit-index={index}>
            {item}
          </span>
        </li>
      ))}
    </ul>
  );
};

const imageToken = (asset: FrontendVisualAsset | undefined) => {
  if (!asset || !asset.src) {
    return (
      <div
        data-image-key={asset?.key || 'main_visual'}
        style={{
          width: '100%',
          height: '100%',
          minHeight: 220,
          borderRadius: 24,
          border: '1px dashed rgba(148,163,184,0.35)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: 'linear-gradient(135deg, rgba(15,23,42,0.18), rgba(15,23,42,0.32))',
          color: 'rgba(226,232,240,0.7)',
          fontSize: 16,
          fontWeight: 600,
          letterSpacing: '0.04em',
        }}
      >
        点击替换图片
      </div>
    );
  }

  return (
    <div
      data-image-key={asset.key}
      style={{
        width: '100%',
        height: '100%',
        minHeight: 220,
        overflow: 'hidden',
        borderRadius: 24,
        border: '1px solid rgba(148,163,184,0.18)',
        background: 'rgba(15,23,42,0.12)',
      }}
    >
      <img
        src={asset.src}
        alt={asset.alt || asset.label || asset.key}
        style={{
          width: '100%',
          height: '100%',
          display: 'block',
          objectFit: 'cover',
        }}
      />
    </div>
  );
};

export const StructuredSlideCanvas: React.FC<StructuredSlideCanvasProps> = ({
  slide,
  deckTheme,
  useOriginalAssets = false,
}) => {
  const theme = ensureDeckTheme(deckTheme);
  const styleFamily = getDeckStyleFamily(theme);
  const field = (key?: string) => getField(slide, key);
  const text = (key?: string) => getTextValue(slide, key);
  const list = (key?: string) => getListValue(slide, key);
  const visual = (key?: string) => getVisualAsset(slide, key, useOriginalAssets);
  const panel = basePanelStyle(theme);

  const shellStyle: React.CSSProperties = {
    position: 'relative',
    width: DESIGN_WIDTH,
    height: DESIGN_HEIGHT,
    overflow: 'hidden',
    borderRadius: styleFamily === 'academic' ? 18 : styleFamily === 'business' ? 20 : 28,
    background:
      styleFamily === 'academic'
        ? `linear-gradient(180deg, ${theme.palette.bg}, ${theme.palette.bg}), repeating-linear-gradient(180deg, transparent 0, transparent 35px, ${theme.palette.primary}08 36px)`
        : styleFamily === 'business'
          ? `linear-gradient(135deg, ${theme.palette.bg} 0%, ${theme.palette.bg} 74%, ${theme.palette.accent}12 100%)`
          : styleFamily === 'creative'
            ? `radial-gradient(circle at 12% 18%, ${theme.palette.secondary}28 0%, transparent 24%), radial-gradient(circle at 84% 14%, ${theme.palette.accent}24 0%, transparent 22%), linear-gradient(160deg, ${theme.palette.bg} 0%, ${theme.palette.bg} 62%, ${theme.palette.primary}10 100%)`
            : `
                radial-gradient(circle at top right, ${theme.palette.secondary}33 0%, transparent 28%),
                radial-gradient(circle at bottom left, ${theme.palette.accent}22 0%, transparent 32%),
                ${theme.palette.bg}
              `,
    color: theme.palette.text,
    fontFamily: toCssFont(theme.typography.bodyFontStack),
  };

  const slideShellStyle: React.CSSProperties = {
    position: 'relative',
    width: '100%',
    height: '100%',
    padding: '68px 72px',
    boxSizing: 'border-box',
  };

  const gridLayer = (
    <div
      style={{
        position: 'absolute',
        inset: 0,
        backgroundImage:
          styleFamily === 'academic'
            ? `linear-gradient(${theme.palette.primary}10 1px, transparent 1px)`
            : styleFamily === 'business'
              ? `linear-gradient(90deg, ${theme.palette.primary}10 1px, transparent 1px)`
              : 'linear-gradient(rgba(148,163,184,0.08) 1px, transparent 1px), linear-gradient(90deg, rgba(148,163,184,0.08) 1px, transparent 1px)',
        backgroundSize:
          styleFamily === 'academic'
            ? '100% 44px'
            : styleFamily === 'business'
              ? '84px 100%'
              : '48px 48px',
        opacity: styleFamily === 'creative' ? 0 : styleFamily === 'academic' ? 0.12 : 0.22,
      }}
    />
  );

  const chromeLayer = styleFamily === 'business'
    ? (
      <>
        <div style={{ position: 'absolute', inset: '0 0 auto 0', height: 28, background: theme.palette.accent, opacity: 0.92 }} />
        <div style={{ position: 'absolute', inset: '110px auto 96px 52px', width: 4, borderRadius: 999, background: `${theme.palette.primary}55` }} />
      </>
    )
    : styleFamily === 'creative'
      ? (
        <>
          <div style={{ position: 'absolute', top: -80, right: -40, width: 320, height: 320, borderRadius: '50%', background: `${theme.palette.secondary}22`, filter: 'blur(12px)' }} />
          <div style={{ position: 'absolute', bottom: -60, left: -30, width: 260, height: 260, borderRadius: '50%', background: `${theme.palette.accent}20`, filter: 'blur(10px)' }} />
        </>
      )
      : styleFamily === 'academic'
        ? <div style={{ position: 'absolute', inset: '114px 72px auto 72px', height: 2, background: `${theme.palette.primary}18` }} />
        : null;

  const eyebrowStyle: React.CSSProperties = {
    display: 'inline-flex',
    alignSelf: 'flex-start',
    padding: styleFamily === 'business' ? '9px 16px' : '8px 14px',
    borderRadius: styleFamily === 'academic' ? 12 : 999,
    background: styleFamily === 'academic' ? `${theme.palette.primary}10` : `${theme.palette.secondary}22`,
    border: `1px solid ${theme.palette.primary}${styleFamily === 'academic' ? '38' : '55'}`,
    color: theme.palette.primary,
    fontSize: theme.typography.eyebrowSize,
    fontWeight: 700,
    letterSpacing: styleFamily === 'creative' ? '0.04em' : '0.08em',
    textTransform: styleFamily === 'creative' ? 'none' : 'uppercase',
  };

  const titleStyle: React.CSSProperties = {
    margin: 0,
    fontSize: theme.typography.titleSize,
    lineHeight: 1.04,
    letterSpacing: '-0.04em',
    fontFamily: toCssFont(theme.typography.titleFontStack),
    color: theme.palette.text,
    whiteSpace: 'pre-wrap',
    textWrap: 'balance',
  };

  const summaryStyle: React.CSSProperties = {
    margin: 0,
    fontSize: theme.typography.summarySize,
    lineHeight: 1.4,
    color: theme.palette.muted,
    whiteSpace: 'pre-wrap',
    fontFamily: toCssFont(theme.typography.bodyFontStack),
  };

  const footerPill = (footerKey?: string) => (
    <div
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        minWidth: 220,
        padding: '14px 18px',
        borderRadius: styleFamily === 'academic' ? 12 : 999,
        border: `1px solid ${theme.palette.accent}${styleFamily === 'academic' ? '40' : '55'}`,
        color: theme.palette.accent,
        fontSize: theme.typography.eyebrowSize,
        fontWeight: 700,
        background:
          styleFamily === 'academic'
            ? `${theme.palette.panel}`
            : styleFamily === 'business'
              ? `${theme.palette.bg}E6`
              : 'rgba(15, 23, 42, 0.45)',
      }}
    >
      <span data-edit-key={footerKey || ''} data-edit-type={field(footerKey)?.type || 'text'}>
        {text(footerKey)}
      </span>
    </div>
  );

  let body: React.ReactNode = null;

  switch (slide.layoutType) {
    case 'cover': {
      const layout = slide.layoutData.type === 'cover' ? slide.layoutData : null;
      body = (
        <div style={{ position: 'relative', zIndex: 1, display: 'flex', flexDirection: 'column', height: '100%', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            {editableText(field(layout?.eyebrowKey), 'eyebrow', eyebrowStyle, 'div')}
            {layout?.footerKey ? footerPill(layout.footerKey) : null}
          </div>
          <div style={{ maxWidth: 980, margin: '0 auto', textAlign: 'center', display: 'grid', gap: 22 }}>
            {editableText(field(layout?.titleKey), 'title', { ...titleStyle, fontSize: 72 }, 'h1')}
            {editableText(field(layout?.subtitleKey), 'subtitle', { ...summaryStyle, fontSize: 30, color: theme.palette.text }, 'p')}
            {layout?.presenterKey
              ? editableText(field(layout.presenterKey), 'presenter', { ...summaryStyle, fontSize: 22 }, 'p')
              : null}
          </div>
          <div />
        </div>
      );
      break;
    }
    case 'section': {
      const layout = slide.layoutData.type === 'section' ? slide.layoutData : null;
      body = (
        <div style={{ position: 'relative', zIndex: 1, display: 'grid', gridTemplateRows: 'auto 1fr auto', gap: 24, height: '100%' }}>
          {editableText(field(layout?.eyebrowKey), 'eyebrow', eyebrowStyle, 'div')}
          <div style={{ display: 'grid', gridTemplateColumns: '1.25fr 0.75fr', gap: 28, alignItems: 'center' }}>
            <div style={{ display: 'grid', gap: 20 }}>
              {editableText(field(layout?.titleKey), 'title', { ...titleStyle, fontSize: 68 }, 'h1')}
              {editableText(field(layout?.summaryKey), 'summary', summaryStyle, 'p')}
            </div>
            <div style={{ ...panel, padding: 28, minHeight: 220, display: 'flex', alignItems: 'center' }}>
              {editableText(field(layout?.quoteKey), 'quote', { ...summaryStyle, fontSize: 28, color: theme.palette.text }, 'p')}
            </div>
          </div>
          <div style={{ display: 'flex', justifyContent: 'flex-end' }}>{layout?.footerKey ? footerPill(layout.footerKey) : null}</div>
        </div>
      );
      break;
    }
    case 'bullets': {
      const layout = slide.layoutData.type === 'bullets' ? slide.layoutData : null;
      body = (
        <div style={{ position: 'relative', zIndex: 1, display: 'grid', gridTemplateRows: 'auto 1fr auto', gap: 24, height: '100%' }}>
          {editableText(field(layout?.eyebrowKey), 'eyebrow', eyebrowStyle, 'div')}
          <div style={{ display: 'grid', gridTemplateColumns: '1.18fr 0.82fr', gap: 28, alignItems: 'start' }}>
            <div style={{ display: 'grid', gap: 20 }}>
              {editableText(field(layout?.titleKey), 'title', titleStyle, 'h1')}
              {editableText(field(layout?.summaryKey), 'summary', summaryStyle, 'p')}
              {editableList(field(layout?.bulletsKey), theme)}
            </div>
            <div style={{ ...panel, padding: 28, display: 'grid', gap: 16, minHeight: 240 }}>
              <div style={{ ...eyebrowStyle, width: 'fit-content' }}>Takeaway</div>
              {editableText(field(layout?.takeawayKey), 'takeaway', { ...summaryStyle, color: theme.palette.text }, 'p')}
            </div>
          </div>
          <div style={{ display: 'flex', justifyContent: 'flex-end' }}>{layout?.footerKey ? footerPill(layout.footerKey) : null}</div>
        </div>
      );
      break;
    }
    case 'two_column': {
      const layout = slide.layoutData.type === 'two_column' ? slide.layoutData : null;
      body = (
        <div style={{ position: 'relative', zIndex: 1, display: 'grid', gridTemplateRows: 'auto auto 1fr auto', gap: 24, height: '100%' }}>
          {editableText(field(layout?.eyebrowKey), 'eyebrow', eyebrowStyle, 'div')}
          <div style={{ display: 'grid', gap: 16, maxWidth: 1050 }}>
            {editableText(field(layout?.titleKey), 'title', titleStyle, 'h1')}
            {editableText(field(layout?.summaryKey), 'summary', summaryStyle, 'p')}
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
            <div style={{ ...panel, padding: 26, display: 'grid', gap: 16 }}>
              {editableText(field(layout?.leftHeadingKey), 'left-heading', { ...summaryStyle, color: theme.palette.text, fontSize: 30, fontWeight: 700 }, 'h2')}
              {editableText(field(layout?.leftBodyKey), 'left-body', { ...summaryStyle, fontSize: 22 }, 'p')}
              {editableList(field(layout?.leftPointsKey), theme, true)}
            </div>
            <div style={{ ...panel, padding: 26, display: 'grid', gap: 16 }}>
              {editableText(field(layout?.rightHeadingKey), 'right-heading', { ...summaryStyle, color: theme.palette.text, fontSize: 30, fontWeight: 700 }, 'h2')}
              {editableText(field(layout?.rightBodyKey), 'right-body', { ...summaryStyle, fontSize: 22 }, 'p')}
              {editableList(field(layout?.rightPointsKey), theme, true)}
            </div>
          </div>
          <div style={{ display: 'flex', justifyContent: 'flex-end' }}>{layout?.footerKey ? footerPill(layout.footerKey) : null}</div>
        </div>
      );
      break;
    }
    case 'cards_2x2': {
      const layout = slide.layoutData.type === 'cards_2x2' ? slide.layoutData : null;
      body = (
        <div style={{ position: 'relative', zIndex: 1, display: 'grid', gridTemplateRows: 'auto auto 1fr auto', gap: 22, height: '100%' }}>
          {editableText(field(layout?.eyebrowKey), 'eyebrow', eyebrowStyle, 'div')}
          <div style={{ display: 'grid', gap: 14, maxWidth: 1040 }}>
            {editableText(field(layout?.titleKey), 'title', titleStyle, 'h1')}
            {editableText(field(layout?.summaryKey), 'summary', summaryStyle, 'p')}
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 22, alignContent: 'stretch' }}>
            {(layout?.cards || []).map((card, index) => (
              <div key={`${card.titleKey}-${index}`} style={{ ...panel, padding: 24, display: 'grid', gap: 12, minHeight: 180 }}>
                {editableText(field(card.titleKey), 'card-title', { ...summaryStyle, color: theme.palette.text, fontSize: 26, fontWeight: 700 }, 'h2')}
                {editableText(field(card.bodyKey), 'card-body', { ...summaryStyle, fontSize: 20 }, 'p')}
              </div>
            ))}
          </div>
          <div style={{ display: 'flex', justifyContent: 'flex-end' }}>{layout?.footerKey ? footerPill(layout.footerKey) : null}</div>
        </div>
      );
      break;
    }
    case 'image_focus': {
      const layout = slide.layoutData.type === 'image_focus' ? slide.layoutData : null;
      const imageAsset = visual(layout?.visualKey);
      body = (
        <div style={{ position: 'relative', zIndex: 1, display: 'grid', gridTemplateRows: 'auto 1fr auto', gap: 24, height: '100%' }}>
          {editableText(field(layout?.eyebrowKey), 'eyebrow', eyebrowStyle, 'div')}
          <div style={{ display: 'grid', gridTemplateColumns: '1.02fr 0.98fr', gap: 28, alignItems: 'stretch' }}>
            <div style={{ display: 'grid', gap: 18, alignContent: 'center' }}>
              {editableText(field(layout?.titleKey), 'title', titleStyle, 'h1')}
              {editableText(field(layout?.summaryKey), 'summary', summaryStyle, 'p')}
              {layout?.bulletsKey ? editableList(field(layout.bulletsKey), theme, true) : null}
            </div>
            <div style={{ ...panel, padding: 18, minHeight: 460, display: 'grid', gridTemplateRows: '1fr auto', gap: 12 }}>
              {imageToken(imageAsset)}
              {layout?.visualCaptionKey
                ? editableText(field(layout.visualCaptionKey), 'visual-caption', { ...summaryStyle, fontSize: 18 }, 'p')
                : null}
            </div>
          </div>
          <div style={{ display: 'flex', justifyContent: 'flex-end' }}>{layout?.footerKey ? footerPill(layout.footerKey) : null}</div>
        </div>
      );
      break;
    }
    case 'comparison': {
      const layout = slide.layoutData.type === 'comparison' ? slide.layoutData : null;
      body = (
        <div style={{ position: 'relative', zIndex: 1, display: 'grid', gridTemplateRows: 'auto auto 1fr auto', gap: 24, height: '100%' }}>
          {editableText(field(layout?.eyebrowKey), 'eyebrow', eyebrowStyle, 'div')}
          <div style={{ display: 'grid', gap: 14 }}>
            {editableText(field(layout?.titleKey), 'title', titleStyle, 'h1')}
            {editableText(field(layout?.summaryKey), 'summary', summaryStyle, 'p')}
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 22 }}>
            <div style={{ ...panel, padding: 26, display: 'grid', gap: 14 }}>
              {editableText(field(layout?.leftTitleKey), 'left-title', { ...summaryStyle, fontSize: 28, color: theme.palette.text, fontWeight: 700 }, 'h2')}
              {editableList(field(layout?.leftPointsKey), theme)}
            </div>
            <div style={{ ...panel, padding: 26, display: 'grid', gap: 14 }}>
              {editableText(field(layout?.rightTitleKey), 'right-title', { ...summaryStyle, fontSize: 28, color: theme.palette.text, fontWeight: 700 }, 'h2')}
              {editableList(field(layout?.rightPointsKey), theme)}
            </div>
          </div>
          <div style={{ display: 'flex', justifyContent: 'flex-end' }}>{layout?.footerKey ? footerPill(layout.footerKey) : null}</div>
        </div>
      );
      break;
    }
    case 'timeline': {
      const layout = slide.layoutData.type === 'timeline' ? slide.layoutData : null;
      body = (
        <div style={{ position: 'relative', zIndex: 1, display: 'grid', gridTemplateRows: 'auto auto 1fr auto', gap: 24, height: '100%' }}>
          {editableText(field(layout?.eyebrowKey), 'eyebrow', eyebrowStyle, 'div')}
          <div style={{ display: 'grid', gap: 14, maxWidth: 1040 }}>
            {editableText(field(layout?.titleKey), 'title', titleStyle, 'h1')}
            {editableText(field(layout?.summaryKey), 'summary', summaryStyle, 'p')}
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: `repeat(${Math.max(1, layout?.timeline.length || 1)}, 1fr)`, gap: 18, alignItems: 'stretch' }}>
            {(layout?.timeline || []).map((item, index) => (
              <div key={`${item.labelKey}-${index}`} style={{ position: 'relative', paddingTop: 18 }}>
                <div style={{ position: 'absolute', left: 22, top: 0, width: 12, height: 12, borderRadius: '50%', background: theme.palette.accent, boxShadow: `0 0 0 6px ${theme.palette.accent}22` }} />
                <div style={{ position: 'absolute', left: 27, top: 12, bottom: -18, width: 2, background: `${theme.palette.primary}44` }} />
                <div style={{ ...panel, padding: 22, marginLeft: 18, minHeight: 220, display: 'grid', gap: 12 }}>
                  {editableText(field(item.labelKey), 'timeline-label', { ...eyebrowStyle, padding: '6px 12px', width: 'fit-content' }, 'div')}
                  {editableText(field(item.bodyKey), 'timeline-body', { ...summaryStyle, fontSize: 20, color: theme.palette.text }, 'p')}
                </div>
              </div>
            ))}
          </div>
          <div style={{ display: 'flex', justifyContent: 'flex-end' }}>{layout?.footerKey ? footerPill(layout.footerKey) : null}</div>
        </div>
      );
      break;
    }
    default:
      body = (
        <div style={{ position: 'relative', zIndex: 1, display: 'grid', placeItems: 'center', height: '100%' }}>
          <div style={{ ...panel, padding: 32, width: 760, textAlign: 'center' }}>
            {editableText(field(slide.layoutData.titleKey), 'title', titleStyle, 'h1')}
          </div>
        </div>
      );
  }

  return (
    <div style={shellStyle}>
      <div style={slideShellStyle}>
        {gridLayer}
        {chromeLayer}
        {body}
      </div>
    </div>
  );
};

export default StructuredSlideCanvas;
