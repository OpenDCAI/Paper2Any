import { FrontendEditableField, FrontendSlide, FrontendVisualAsset } from './types';

const FIELD_PLACEHOLDER_RE = /\{\{(?:field|list):([a-zA-Z0-9_]+)\}\}/g;
const IMAGE_PLACEHOLDER_RE = /\{\{image:([a-zA-Z0-9_]+)\}\}/g;
const ATTRIBUTE_RE = /([^\s"'<>/=]+)\s*=\s*(["'])([\s\S]*?)\2/g;
const FORBIDDEN_HTML_RE = /<\s*(script|iframe|img|video|audio|canvas|svg)\b|on[a-z]+\s*=/i;
const FORBIDDEN_CSS_RE = /@import|url\s*\(|(?:^|[,{])\s*(?:body|html|:root|#root)\b|position\s*:\s*fixed/i;

const escapeHtml = (value: string) =>
  value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');

const formatTextValue = (value: string) => escapeHtml(value).replace(/\n/g, '<br />');

const formatAttributeValue = (value: string) =>
  escapeHtml(value.replace(/\s+/g, ' ').trim());

const sanitizeTemplate = (value: string) =>
  value
    .replace(/<\s*\/?\s*(html|head|body)\b[^>]*>/gi, '')
    .replace(/<script[\s\S]*?<\/script>/gi, '')
    .replace(/\son[a-z]+\s*=\s*(['"]).*?\1/gi, '');

const sanitizeCss = (value: string) =>
  value
    .replace(/@import[^;]+;/gi, '')
    .replace(/url\s*\(([^)]*)\)/gi, 'none');

const ensureSlideRoot = (value: string) => {
  const trimmed = value.trim();
  if (!trimmed) return '<div class="slide-root"></div>';
  if (trimmed.includes('class="slide-root"') || trimmed.includes("class='slide-root'")) {
    return trimmed;
  }
  return `<div class="slide-root">${trimmed}</div>`;
};

const wrapEditableText = (
  field: FrontendEditableField,
  renderedValue: string,
  itemIndex?: number,
) => {
  const itemAttr = typeof itemIndex === 'number' ? ` data-edit-index="${itemIndex}"` : '';
  const itemClass = typeof itemIndex === 'number' ? ' ppt-inline-editable-list' : '';
  return `<span class="ppt-inline-editable${itemClass}" data-edit-key="${field.key}" data-edit-type="${field.type}"${itemAttr}>${renderedValue}</span>`;
};

const renderFieldValue = (field: FrontendEditableField) => {
  if (field.type === 'list') {
    return field.items
      .filter((item) => item.trim())
      .map((item, index) => `<li>${wrapEditableText(field, formatTextValue(item), index)}</li>`)
      .join('');
  }
  return wrapEditableText(field, formatTextValue(field.value || ''));
};

const getAttributeFieldValue = (field: FrontendEditableField | undefined) => {
  if (!field) return '';
  if (field.type === 'list') {
    return field.items.filter((item) => item.trim()).join(' • ');
  }
  return field.value || '';
};

const replaceAttributePlaceholders = (
  html: string,
  editableFields: FrontendEditableField[],
) => {
  const fieldMap = new Map(editableFields.map((field) => [field.key, field]));
  return html.replace(ATTRIBUTE_RE, (match, attrName: string, quote: string, attrValue: string) => {
    const nextValue = attrValue
      .replace(/\{\{field:([a-zA-Z0-9_]+)\}\}/g, (_token, key: string) =>
        formatAttributeValue(getAttributeFieldValue(fieldMap.get(key))),
      )
      .replace(/\{\{list:([a-zA-Z0-9_]+)\}\}/g, (_token, key: string) =>
        formatAttributeValue(getAttributeFieldValue(fieldMap.get(key))),
      )
      .replace(/\{\{image:([a-zA-Z0-9_]+)\}\}/g, '');
    if (nextValue === attrValue) {
      return match;
    }
    return `${attrName}=${quote}${nextValue}${quote}`;
  });
};

const collectAttributePlaceholderKeys = (html: string) => {
  const fieldKeys = new Set<string>();
  const imageKeys = new Set<string>();
  for (const match of html.matchAll(ATTRIBUTE_RE)) {
    const attrValue = match[3] || '';
    for (const fieldMatch of attrValue.matchAll(FIELD_PLACEHOLDER_RE)) {
      fieldKeys.add(fieldMatch[1]);
    }
    for (const imageMatch of attrValue.matchAll(IMAGE_PLACEHOLDER_RE)) {
      imageKeys.add(imageMatch[1]);
    }
  }
  return {
    fieldKeys: Array.from(fieldKeys),
    imageKeys: Array.from(imageKeys),
  };
};

const getVisualSourceLabel = (asset: FrontendVisualAsset) => {
  switch (asset.sourceType) {
    case 'paper_asset':
      return '论文图表';
    case 'upload':
      return '用户上传';
    default:
      return 'AI 配图';
  }
};

const renderVisualAsset = (asset: FrontendVisualAsset) => {
  const assetKey = escapeHtml(asset.key || 'main_visual');
  const assetLabel = escapeHtml(asset.label || asset.key || 'Image');
  const assetAlt = escapeHtml(asset.alt || asset.label || asset.key || 'Slide image');
  const sourceLabel = escapeHtml(getVisualSourceLabel(asset));
  const previewSrc = (asset.previewSrc || asset.src || '').trim();
  const originalSrc = (asset.originalSrc || previewSrc || '').trim();

  if (!previewSrc) {
    return `
<div class="ppt-managed-image" data-image-key="${assetKey}" data-image-label="${assetLabel}">
  <div class="ppt-managed-image-frame ppt-managed-image-frame-empty">
    <div class="ppt-managed-image-empty-text">点击上传图片</div>
  </div>
  <div class="ppt-managed-image-badge">${sourceLabel}</div>
</div>
`.trim();
  }

  return `
<div class="ppt-managed-image" data-image-key="${assetKey}" data-image-label="${assetLabel}">
  <div class="ppt-managed-image-frame">
    <img src="${escapeHtml(previewSrc)}" data-preview-src="${escapeHtml(previewSrc)}" data-original-src="${escapeHtml(originalSrc)}" alt="${assetAlt}" class="ppt-managed-image-el" />
  </div>
  <div class="ppt-managed-image-badge">${sourceLabel}</div>
</div>
`.trim();
};

export const buildFrontendSlideMarkup = (slide: FrontendSlide) => {
  let html = ensureSlideRoot(sanitizeTemplate(slide.htmlTemplate || ''));
  html = replaceAttributePlaceholders(html, slide.editableFields);
  slide.editableFields.forEach((field) => {
    const listToken = `{{list:${field.key}}}`;
    const fieldToken = `{{field:${field.key}}}`;
    const renderedValue = renderFieldValue(field);
    if (field.type === 'list') {
      html = html.split(listToken).join(renderedValue);
      html = html.split(fieldToken).join(
        wrapEditableText(
          field,
          formatTextValue(field.items.filter((item) => item.trim()).join(' • ')),
        ),
      );
    } else {
      html = html.split(fieldToken).join(renderedValue);
      html = html.split(listToken).join(`<li>${renderedValue}</li>`);
    }
  });
  slide.visualAssets.forEach((asset) => {
    const imageToken = `{{image:${asset.key}}}`;
    html = html.split(imageToken).join(renderVisualAsset(asset));
  });
  html = html.replace(/\{\{(?:field|list|image):[^}]+\}\}/g, '');
  const css = sanitizeCss(slide.cssCode || '');
  const editableHintCss = `
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
  min-width: 140px;
  min-height: 140px;
  cursor: pointer;
}
.slide-root .ppt-managed-image-frame {
  position: relative;
  width: 100%;
  height: 100%;
  overflow: hidden;
  border-radius: inherit;
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
  return `<style>${css}\n${editableHintCss}</style>${html}`;
};

export interface FrontendCodeValidationResult {
  ok: boolean;
  sanitizedHtml: string;
  sanitizedCss: string;
  issues: string[];
  warnings: string[];
}

export const validateFrontendSlideCode = (
  slide: FrontendSlide,
  htmlTemplate: string,
  cssCode: string,
): FrontendCodeValidationResult => {
  const issues: string[] = [];
  const warnings: string[] = [];

  if (!htmlTemplate.trim()) {
    issues.push('HTML 模板不能为空。');
  }
  if (!cssCode.trim()) {
    warnings.push('CSS 为空，将只使用默认样式约束。');
  }
  if (htmlTemplate.length > 20000) {
    issues.push('HTML 代码过长，请控制在 20000 字符以内。');
  }
  if (cssCode.length > 24000) {
    issues.push('CSS 代码过长，请控制在 24000 字符以内。');
  }
  if (FORBIDDEN_HTML_RE.test(htmlTemplate)) {
    issues.push('HTML 中包含不允许的标签或内联事件，例如 script/img/svg/iframe。');
  }
  if (FORBIDDEN_CSS_RE.test(cssCode)) {
    issues.push('CSS 中包含不允许的全局选择器、远程资源或 fixed 定位。');
  }

  const sanitizedHtml = ensureSlideRoot(sanitizeTemplate(htmlTemplate));
  const sanitizedCss = sanitizeCss(cssCode);
  const availableKeys = new Set(slide.editableFields.map((field) => field.key));
  const availableImageKeys = new Set(slide.visualAssets.map((asset) => asset.key));
  const fieldPlaceholderKeys = Array.from(sanitizedHtml.matchAll(FIELD_PLACEHOLDER_RE)).map((match) => match[1]);
  const imagePlaceholderKeys = Array.from(sanitizedHtml.matchAll(IMAGE_PLACEHOLDER_RE)).map((match) => match[1]);
  const attributePlaceholders = collectAttributePlaceholderKeys(sanitizedHtml);

  if (fieldPlaceholderKeys.length === 0 && imagePlaceholderKeys.length === 0) {
    issues.push('HTML 中至少需要保留一个 `{{field:...}}`、`{{list:...}}` 或 `{{image:...}}` 占位符。');
  }

  const unknownFieldKeys = Array.from(new Set(fieldPlaceholderKeys.filter((key) => !availableKeys.has(key))));
  if (unknownFieldKeys.length > 0) {
    issues.push(`发现未知文本占位符字段：${unknownFieldKeys.join('、')}。`);
  }

  const unknownImageKeys = Array.from(new Set(imagePlaceholderKeys.filter((key) => !availableImageKeys.has(key))));
  if (unknownImageKeys.length > 0) {
    issues.push(`发现未知图片占位符字段：${unknownImageKeys.join('、')}。`);
  }

  if (attributePlaceholders.fieldKeys.length > 0) {
    issues.push(
      `文本占位符不能放在 HTML 属性里，例如 aria-label/title/alt。请改成正文节点占位符：${attributePlaceholders.fieldKeys.join('、')}。`,
    );
  }

  if (attributePlaceholders.imageKeys.length > 0) {
    issues.push(
      `图片占位符不能放在 HTML 属性里，只能放在元素内容区域：${attributePlaceholders.imageKeys.join('、')}。`,
    );
  }

  const unusedKeys = slide.editableFields
    .map((field) => field.key)
    .filter((key) => !fieldPlaceholderKeys.includes(key) && !attributePlaceholders.fieldKeys.includes(key));
  if (unusedKeys.length > 0) {
    warnings.push(`以下字段当前未被模板使用：${unusedKeys.slice(0, 6).join('、')}。`);
  }

  const unusedImageKeys = slide.visualAssets
    .map((asset) => asset.key)
    .filter((key) => !imagePlaceholderKeys.includes(key) && !attributePlaceholders.imageKeys.includes(key));
  if (unusedImageKeys.length > 0) {
    warnings.push(`以下图片槽位当前未被模板使用：${unusedImageKeys.slice(0, 4).join('、')}。`);
  }

  if (!sanitizedHtml.includes('class="slide-root"') && !sanitizedHtml.includes("class='slide-root'")) {
    issues.push('HTML 根节点必须包含 `.slide-root`。');
  }

  return {
    ok: issues.length === 0,
    sanitizedHtml,
    sanitizedCss,
    issues,
    warnings,
  };
};

export const buildFrontendCodeRepairPrompt = (
  slide: FrontendSlide,
  validation: FrontendCodeValidationResult,
) => {
  const issueText = [...validation.issues, ...validation.warnings].join('；') || '请整体检查当前 HTML/CSS 的结构、占位符和版式。';
  return [
    `Keep the same slide topic "${slide.title}" and keep the same deck theme.`,
    'Repair the current HTML/CSS slide implementation while preserving editable text placeholders.',
    'Fix any invalid placeholder mapping, unsafe code, overflow risk, and readability problems.',
    `Specific issues: ${issueText}`,
  ].join(' ');
};

const describeElement = (node: HTMLElement) => {
  const className = (node.className || '').toString().trim().split(/\s+/).filter(Boolean)[0];
  if (className) {
    return `${node.tagName.toLowerCase()}.${className}`;
  }
  return node.tagName.toLowerCase();
};

export const inspectSlideLayout = (
  node: HTMLElement,
  width: number = 1600,
  height: number = 900,
) => {
  const root = (node.querySelector('.slide-root') as HTMLElement | null) || node;
  const issues: string[] = [];

  if (root.scrollWidth > width + 4) {
    issues.push(`画布横向内容溢出，实际宽度约 ${root.scrollWidth}px。`);
  }
  if (root.scrollHeight > height + 4) {
    issues.push(`画布纵向内容溢出，实际高度约 ${root.scrollHeight}px。`);
  }

  const rootRect = root.getBoundingClientRect();
  const overflowElements: string[] = [];
  const textCrowdedElements: string[] = [];

  root.querySelectorAll<HTMLElement>('*').forEach((element) => {
    const rect = element.getBoundingClientRect();
    if (rect.width <= 1 || rect.height <= 1) {
      return;
    }

    const style = window.getComputedStyle(element);
    const fontSize = Number.parseFloat(style.fontSize || '0');
    if (fontSize > 72) {
      textCrowdedElements.push(describeElement(element));
    }

    const overflowRight = rect.right - rootRect.right > 2;
    const overflowBottom = rect.bottom - rootRect.bottom > 2;
    const overflowLeft = rootRect.left - rect.left > 2;
    const overflowTop = rootRect.top - rect.top > 2;

    if (overflowRight || overflowBottom || overflowLeft || overflowTop) {
      overflowElements.push(describeElement(element));
      return;
    }

    if (element.scrollHeight - element.clientHeight > 4 || element.scrollWidth - element.clientWidth > 4) {
      overflowElements.push(describeElement(element));
    }
  });

  if (overflowElements.length > 0) {
    issues.push(
      `检测到 ${overflowElements.length} 个元素超出或挤出容器，例如：${overflowElements.slice(0, 3).join('、')}。`,
    );
  }
  if (textCrowdedElements.length > 0) {
    issues.push(
      `检测到字体过大的元素，例如：${textCrowdedElements.slice(0, 3).join('、')}。`,
    );
  }

  return {
    passed: issues.length === 0,
    issues,
  };
};

export const captureSlideToPngBlob = async (
  node: HTMLElement,
  width: number = 1600,
  height: number = 900,
  options?: {
    mimeType?: string;
    quality?: number;
    useOriginalAssets?: boolean;
  },
) => {
  const mimeType = options?.mimeType || 'image/png';
  const quality = options?.quality;
  const clone = node.cloneNode(true) as HTMLElement;
  clone.setAttribute('xmlns', 'http://www.w3.org/1999/xhtml');
  clone.style.width = `${width}px`;
  clone.style.height = `${height}px`;
  clone.style.margin = '0';

  const images = Array.from(clone.querySelectorAll<HTMLImageElement>('img'));
  await Promise.all(
    images.map(async (image) => {
      const originalSrc = image.getAttribute('data-original-src') || '';
      const previewSrc = image.getAttribute('data-preview-src') || '';
      const preferredSrc = options?.useOriginalAssets ? originalSrc || previewSrc : previewSrc || originalSrc;
      if (preferredSrc) {
        image.setAttribute('src', preferredSrc);
      }
      const src = image.getAttribute('src') || '';
      if (!src || src.startsWith('data:')) {
        return;
      }
      try {
        const res = await fetch(src);
        if (!res.ok) {
          return;
        }
        const blob = await res.blob();
        const dataUrl = await new Promise<string>((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(String(reader.result || ''));
          reader.onerror = () => reject(reader.error);
          reader.readAsDataURL(blob);
        });
        image.setAttribute('src', dataUrl);
      } catch {
        // Best effort: leave the original src if inlining fails.
      }
    }),
  );

  const serialized = new XMLSerializer().serializeToString(clone);
  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">
      <foreignObject width="100%" height="100%">${serialized}</foreignObject>
    </svg>
  `.trim();
  const url = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;

  const img = await new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('无法将前端页面转换为截图'));
    image.src = url;
  });

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('无法创建截图画布');
  }
  ctx.fillStyle = '#0b1020';
  ctx.fillRect(0, 0, width, height);
  ctx.drawImage(img, 0, 0, width, height);

  const blob = await new Promise<Blob | null>((resolve) => {
    canvas.toBlob((value) => resolve(value), mimeType, quality);
  });
  if (!blob) {
    throw new Error('截图导出失败');
  }
  return blob;
};
