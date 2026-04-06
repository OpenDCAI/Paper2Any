import type { Edge, Node } from 'reactflow';

export interface MindMapTreeNode {
  id: string;
  label: string;
  summary?: string;
  children: MindMapTreeNode[];
}

export interface MindMapNodeData {
  label: string;
  summary?: string;
  depth: number;
  branch: 'center' | 'left' | 'right';
  isRoot: boolean;
}

export interface MindMapSvgExportOptions {
  title?: string;
  subtitle?: string;
  highlights?: string[];
  width?: number;
  height?: number;
}

const NODE_WIDTH = 260;
const NODE_HEIGHT = 96;
const HORIZONTAL_GAP = 220;
const VERTICAL_GAP = 28;
const SVG_ROOT_WIDTH = 240;
const SVG_ROOT_HEIGHT = 112;
const SVG_CHILD_WIDTH = 230;
const SVG_CHILD_HEIGHT = 92;
const SVG_PADDING = 92;

const nextId = (prefix: string) => `${prefix}_${Date.now()}_${Math.random().toString(16).slice(2, 8)}`;

const coerceText = (value: unknown, fallback: string) => {
  const text = typeof value === 'string' ? value.trim() : '';
  return text || fallback;
};

const normalizeNode = (raw: unknown, prefix: string, depth = 0, maxDepth = 6): MindMapTreeNode => {
  if (!raw || typeof raw !== 'object') {
    return { id: prefix, label: 'Topic', summary: '', children: [] };
  }

  const rawNode = raw as Record<string, unknown>;
  const label = coerceText(rawNode.label ?? rawNode.title ?? rawNode.name, 'Topic');
  const summary = coerceText(rawNode.summary ?? rawNode.note ?? rawNode.description, '');
  const childrenInput = Array.isArray(rawNode.children) ? rawNode.children : [];
  const children =
    depth >= maxDepth
      ? []
      : childrenInput
          .slice(0, 8)
          .map((child, index) => normalizeNode(child, `${prefix}_${index + 1}`, depth + 1, maxDepth));

  return {
    id: coerceText(rawNode.id, prefix),
    label,
    summary,
    children,
  };
};

export const normalizeMindMapTree = (raw: unknown, maxDepth = 6): MindMapTreeNode => {
  const root = normalizeNode(raw, 'root', 0, maxDepth);
  if (!root.id) {
    root.id = 'root';
  }
  return root;
};

const cloneNode = (node: MindMapTreeNode): MindMapTreeNode => ({
  ...node,
  children: node.children.map(cloneNode),
});

export const findNodeById = (node: MindMapTreeNode | null, nodeId: string): MindMapTreeNode | null => {
  if (!node) return null;
  if (node.id === nodeId) return node;
  for (const child of node.children) {
    const match = findNodeById(child, nodeId);
    if (match) return match;
  }
  return null;
};

const mapTree = (
  node: MindMapTreeNode,
  nodeId: string,
  updater: (target: MindMapTreeNode) => MindMapTreeNode,
): MindMapTreeNode => {
  if (node.id === nodeId) {
    return updater(cloneNode(node));
  }
  return {
    ...node,
    children: node.children.map((child) => mapTree(child, nodeId, updater)),
  };
};

export const updateNodeById = (
  root: MindMapTreeNode,
  nodeId: string,
  patch: Partial<Pick<MindMapTreeNode, 'label' | 'summary'>>,
): MindMapTreeNode =>
  mapTree(root, nodeId, (node) => ({
    ...node,
    label: patch.label ?? node.label,
    summary: patch.summary ?? node.summary,
  }));

export const addChildToNode = (root: MindMapTreeNode, parentId: string): MindMapTreeNode =>
  mapTree(root, parentId, (node) => ({
    ...node,
    children: [
      ...node.children,
      {
        id: nextId(node.id),
        label: '新节点',
        summary: '',
        children: [],
      },
    ],
  }));

const addSiblingRecursive = (
  node: MindMapTreeNode,
  nodeId: string,
): { changed: boolean; node: MindMapTreeNode } => {
  const childIndex = node.children.findIndex((child) => child.id === nodeId);
  if (childIndex >= 0) {
    const nextChildren = [...node.children];
    nextChildren.splice(childIndex + 1, 0, {
      id: nextId(node.id),
      label: '新分支',
      summary: '',
      children: [],
    });
    return { changed: true, node: { ...node, children: nextChildren } };
  }

  let changed = false;
  const nextChildren = node.children.map((child) => {
    const result = addSiblingRecursive(child, nodeId);
    if (result.changed) changed = true;
    return result.node;
  });

  return { changed, node: changed ? { ...node, children: nextChildren } : node };
};

export const addSiblingToNode = (root: MindMapTreeNode, nodeId: string): MindMapTreeNode => {
  if (root.id === nodeId) return root;
  return addSiblingRecursive(root, nodeId).node;
};

const removeNodeRecursive = (
  node: MindMapTreeNode,
  nodeId: string,
): { changed: boolean; node: MindMapTreeNode } => {
  const filteredChildren = node.children.filter((child) => child.id !== nodeId);
  if (filteredChildren.length !== node.children.length) {
    return { changed: true, node: { ...node, children: filteredChildren } };
  }

  let changed = false;
  const nextChildren = node.children.map((child) => {
    const result = removeNodeRecursive(child, nodeId);
    if (result.changed) changed = true;
    return result.node;
  });

  return { changed, node: changed ? { ...node, children: nextChildren } : node };
};

export const removeNodeById = (root: MindMapTreeNode, nodeId: string): MindMapTreeNode => {
  if (root.id === nodeId) return root;
  return removeNodeRecursive(root, nodeId).node;
};

const measureHeight = (node: MindMapTreeNode): number => {
  if (node.children.length === 0) return NODE_HEIGHT;
  const childrenHeight =
    node.children.reduce((total, child) => total + measureHeight(child), 0) +
    VERTICAL_GAP * Math.max(node.children.length - 1, 0);
  return Math.max(NODE_HEIGHT, childrenHeight);
};

const splitTopLevelChildren = (children: MindMapTreeNode[]) => {
  const left: MindMapTreeNode[] = [];
  const right: MindMapTreeNode[] = [];
  let leftHeight = 0;
  let rightHeight = 0;

  [...children]
    .sort((a, b) => measureHeight(b) - measureHeight(a))
    .forEach((child, index) => {
      const childHeight = measureHeight(child);
      if ((index % 2 === 0 && rightHeight <= leftHeight) || leftHeight > rightHeight) {
        right.push(child);
        rightHeight += childHeight;
      } else {
        left.push(child);
        leftHeight += childHeight;
      }
    });

  return { left, right };
};

const layoutGroup = (
  children: MindMapTreeNode[],
  side: 'left' | 'right',
  parentId: string,
  startDepth: number,
  nodes: Node<MindMapNodeData>[],
  edges: Edge[],
) => {
  if (children.length === 0) return;
  const totalHeight =
    children.reduce((sum, child) => sum + measureHeight(child), 0) +
    VERTICAL_GAP * Math.max(children.length - 1, 0);
  let cursorY = -totalHeight / 2;

  const walk = (node: MindMapTreeNode, depth: number, centerY: number, parentNodeId: string) => {
    const x = side === 'right' ? depth * (NODE_WIDTH + HORIZONTAL_GAP) : -depth * (NODE_WIDTH + HORIZONTAL_GAP);
    nodes.push({
      id: node.id,
      type: 'mindMapNode',
      position: { x, y: centerY - NODE_HEIGHT / 2 },
      draggable: false,
      data: {
        label: node.label,
        summary: node.summary || '',
        depth,
        branch: side,
        isRoot: false,
      },
    });
    edges.push({
      id: `${parentNodeId}-${node.id}`,
      source: parentNodeId,
      target: node.id,
      sourceHandle: side === 'right' ? 'right-source' : 'left-source',
      targetHandle: side === 'right' ? 'left-target' : 'right-target',
      type: 'smoothstep',
      animated: false,
      style: {
        stroke: side === 'right' ? '#38bdf8' : '#a78bfa',
        strokeWidth: depth <= 1 ? 2.4 : 1.8,
      },
    });

    if (node.children.length === 0) return;

    const totalChildrenHeight =
      node.children.reduce((sum, child) => sum + measureHeight(child), 0) +
      VERTICAL_GAP * Math.max(node.children.length - 1, 0);
    let childCursor = centerY - totalChildrenHeight / 2;

    node.children.forEach((child) => {
      const childHeight = measureHeight(child);
      const childCenter = childCursor + childHeight / 2;
      walk(child, depth + 1, childCenter, node.id);
      childCursor += childHeight + VERTICAL_GAP;
    });
  };

  children.forEach((child) => {
    const childHeight = measureHeight(child);
    const centerY = cursorY + childHeight / 2;
    walk(child, startDepth, centerY, parentId);
    cursorY += childHeight + VERTICAL_GAP;
  });
};

export const buildMindMapFlow = (root: MindMapTreeNode): { nodes: Node<MindMapNodeData>[]; edges: Edge[] } => {
  const nodes: Node<MindMapNodeData>[] = [
    {
      id: root.id,
      type: 'mindMapNode',
      position: { x: 0, y: -NODE_HEIGHT / 2 },
      draggable: false,
      data: {
        label: root.label,
        summary: root.summary || '',
        depth: 0,
        branch: 'center',
        isRoot: true,
      },
    },
  ];
  const edges: Edge[] = [];

  const { left, right } = splitTopLevelChildren(root.children);
  layoutGroup(right, 'right', root.id, 1, nodes, edges);
  layoutGroup(left, 'left', root.id, 1, nodes, edges);

  return { nodes, edges };
};

const markdownLines = (node: MindMapTreeNode, depth: number, lines: string[]) => {
  const indent = '  '.repeat(depth);
  lines.push(`${indent}- ${node.label}`);
  if (node.summary) {
    lines.push(`${indent}  - ${node.summary}`);
  }
  node.children.forEach((child) => markdownLines(child, depth + 1, lines));
};

export const buildMindMapMarkdown = (root: MindMapTreeNode): string => {
  const lines: string[] = [`# ${root.label}`];
  if (root.summary) {
    lines.push('', root.summary);
  }
  if (root.children.length > 0) {
    lines.push('', '## Branches');
    root.children.forEach((child) => markdownLines(child, 0, lines));
  }
  return lines.join('\n');
};

export const countTreeNodes = (node: MindMapTreeNode): number =>
  1 + node.children.reduce((sum, child) => sum + countTreeNodes(child), 0);

export const getTreeDepth = (node: MindMapTreeNode): number =>
  node.children.length === 0 ? 1 : 1 + Math.max(...node.children.map(getTreeDepth));

const SVG_NODE_WIDTH = 260;
const SVG_NODE_HEIGHT = 96;
const SVG_TITLE_MAX_CHARS = 22;
const SVG_SUMMARY_MAX_CHARS = 34;
const SVG_HEADER_HEIGHT = 100;
const SVG_FOOTER_HEIGHT = 72;
const SVG_PADDING_X = 96;
const SVG_PADDING_Y = 88;

const escapeXml = (value: string) =>
  value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');

const wrapText = (value: string, maxChars: number, maxLines: number) => {
  const text = value.trim();
  if (!text) return [];
  const segments: string[] = [];
  let cursor = 0;
  while (cursor < text.length && segments.length < maxLines) {
    const next = text.slice(cursor, cursor + maxChars);
    segments.push(next);
    cursor += maxChars;
  }
  if (cursor < text.length && segments.length > 0) {
    segments[segments.length - 1] = `${segments[segments.length - 1]}…`;
  }
  return segments;
};

const nodeBoxWidth = (isRoot: boolean) => (isRoot ? 280 : SVG_NODE_WIDTH);
const nodeBoxHeight = (isRoot: boolean) => (isRoot ? 104 : SVG_NODE_HEIGHT);

const getNodeBounds = (node: Node<MindMapNodeData>) => {
  const width = nodeBoxWidth(node.data.isRoot);
  const height = nodeBoxHeight(node.data.isRoot);
  return {
    left: node.position.x,
    top: node.position.y,
    right: node.position.x + width,
    bottom: node.position.y + height,
    width,
    height,
  };
};

export const buildMindMapSvg = (
  root: MindMapTreeNode,
  options: MindMapSvgExportOptions = {},
): string => {
  const { nodes, edges } = buildMindMapFlow(root);
  const title = options.title?.trim() || root.label || 'Mind Map';
  const subtitle = options.subtitle?.trim() || root.summary?.trim() || '';
  const highlights = (options.highlights || []).map((item) => item.trim()).filter(Boolean).slice(0, 4);

  const bounds = nodes.reduce(
    (acc, node) => {
      const box = getNodeBounds(node);
      return {
        minX: Math.min(acc.minX, box.left),
        minY: Math.min(acc.minY, box.top),
        maxX: Math.max(acc.maxX, box.right),
        maxY: Math.max(acc.maxY, box.bottom),
      };
    },
    { minX: 0, minY: 0, maxX: 0, maxY: 0 },
  );

  const contentWidth = bounds.maxX - bounds.minX;
  const contentHeight = bounds.maxY - bounds.minY;
  const width = options.width || Math.max(1400, Math.ceil(contentWidth + SVG_PADDING_X * 2));
  const height =
    options.height ||
    Math.max(980, Math.ceil(contentHeight + SVG_PADDING_Y * 2 + SVG_HEADER_HEIGHT + SVG_FOOTER_HEIGHT));
  const shiftX = Math.round((width - contentWidth) / 2 - bounds.minX);
  const shiftY = Math.round(SVG_HEADER_HEIGHT + Math.max(0, (height - SVG_HEADER_HEIGHT - SVG_FOOTER_HEIGHT - contentHeight) / 2) - bounds.minY);

  const nodeMap = new Map(nodes.map((node) => [node.id, node]));
  const branchColor = (branch: MindMapNodeData['branch']) =>
    branch === 'left'
      ? { stroke: '#a78bfa', fill: 'rgba(91, 33, 182, 0.24)' }
      : branch === 'right'
        ? { stroke: '#22d3ee', fill: 'rgba(14, 165, 233, 0.24)' }
        : { stroke: '#67e8f9', fill: 'rgba(6, 182, 212, 0.28)' };

  const edgeMarkup = edges
    .map((edge) => {
      const source = nodeMap.get(edge.source);
      const target = nodeMap.get(edge.target);
      if (!source || !target) return '';
      const sourceBox = getNodeBounds(source);
      const targetBox = getNodeBounds(target);
      const sourceBranch = target.data.branch === 'left' ? 'left' : 'right';
      const sx = source.data.isRoot
        ? sourceBox.left + sourceBox.width / 2
        : sourceBranch === 'left'
          ? sourceBox.left
          : sourceBox.right;
      const tx = target.data.branch === 'left' ? targetBox.right : targetBox.left;
      const sy = sourceBox.top + sourceBox.height / 2;
      const ty = targetBox.top + targetBox.height / 2;
      const midX = (sx + tx) / 2;
      const color = sourceBranch === 'left' ? '#a78bfa' : '#22d3ee';
      return `<path d="M ${sx} ${sy} C ${midX} ${sy}, ${midX} ${ty}, ${tx} ${ty}" fill="none" stroke="${color}" stroke-width="${target.data.depth <= 1 ? 3 : 2}" stroke-linecap="round" />`;
    })
    .filter(Boolean)
    .join('\n');

  const nodeMarkup = nodes
    .map((node) => {
      const box = getNodeBounds(node);
      const x = node.position.x;
      const y = node.position.y;
      const palette = branchColor(node.data.branch);
      const titleLines = wrapText(node.data.label, node.data.isRoot ? 18 : SVG_TITLE_MAX_CHARS, 2);
      const summaryLines = wrapText(node.data.summary || '', SVG_SUMMARY_MAX_CHARS, 2);
      const textX = x + box.width / 2;
      const titleY = y + (node.data.isRoot ? 34 : 30);
      const summaryY = titleY + (titleLines.length > 0 ? titleLines.length * 16 : 0) + 10;
      return `
        <g>
          <rect x="${x}" y="${y}" width="${box.width}" height="${box.height}" rx="24" ry="24" fill="${palette.fill}" stroke="${palette.stroke}" stroke-width="${node.data.isRoot ? 2.4 : 1.8}" />
          <text x="${x + 18}" y="${y + 24}" font-size="10" font-weight="700" fill="rgba(255,255,255,0.42)" letter-spacing="3">${node.data.isRoot ? 'ROOT' : `DEPTH ${node.data.depth}`}</text>
          ${titleLines
            .map(
              (line, index) =>
                `<text x="${textX}" y="${titleY + index * 16}" text-anchor="middle" font-size="${node.data.isRoot ? 18 : 14}" font-weight="700" fill="#f8fafc">${escapeXml(line)}</text>`,
            )
            .join('\n')}
          ${summaryLines
            .map(
              (line, index) =>
                `<text x="${textX}" y="${summaryY + index * 14}" text-anchor="middle" font-size="11" fill="rgba(226,232,240,0.74)">${escapeXml(line)}</text>`,
            )
            .join('\n')}
        </g>
      `;
    })
    .join('\n');

  const highlightMarkup = highlights.length
    ? `
      <g transform="translate(${shiftX + 32}, ${height - 130})">
        <rect x="0" y="0" width="${Math.min(460, width - 64)}" height="74" rx="20" ry="20" fill="rgba(15, 23, 42, 0.72)" stroke="rgba(125, 211, 252, 0.2)" />
        <text x="18" y="26" font-size="12" font-weight="700" fill="#7dd3fc">Key Highlights</text>
        ${highlights
          .map((line, index) => `<text x="18" y="${48 + index * 16}" font-size="11" fill="rgba(226,232,240,0.8)">• ${escapeXml(line)}</text>`)
          .join('\n')}
      </g>
    `
    : '';

  return `
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  <defs>
    <linearGradient id="mindmapBg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#050816" />
      <stop offset="48%" stop-color="#0a1022" />
      <stop offset="100%" stop-color="#111827" />
    </linearGradient>
    <radialGradient id="mindmapGlow" cx="50%" cy="22%" r="82%">
      <stop offset="0%" stop-color="rgba(34,211,238,0.18)" />
      <stop offset="100%" stop-color="rgba(34,211,238,0)" />
    </radialGradient>
  </defs>
  <rect width="100%" height="100%" fill="url(#mindmapBg)" />
  <rect width="100%" height="100%" fill="url(#mindmapGlow)" />
  <text x="${width / 2}" y="42" text-anchor="middle" font-size="24" font-weight="800" fill="#f8fafc">${escapeXml(title)}</text>
  ${subtitle ? `<text x="${width / 2}" y="68" text-anchor="middle" font-size="13" fill="rgba(226,232,240,0.72)">${escapeXml(subtitle)}</text>` : ''}
  <g transform="translate(${shiftX}, ${shiftY})">
    ${edgeMarkup}
    ${nodeMarkup}
  </g>
  ${highlightMarkup}
</svg>
  `.trim();
};

export const estimateMindMapPoints = (sourceCount: number, maxDepth: number): number => {
  const safeSourceCount = Math.max(1, Math.trunc(sourceCount));
  const safeDepth = Math.max(2, Math.trunc(maxDepth));
  const estimatedNodeCount = Math.max(4, safeSourceCount * 6 + Math.max(0, safeDepth - 2) * 3);

  let tierPoints = 2;
  if (estimatedNodeCount <= 8) {
    tierPoints = 2;
  } else if (estimatedNodeCount <= 16) {
    tierPoints = 3;
  } else if (estimatedNodeCount <= 32) {
    tierPoints = 4;
  } else if (estimatedNodeCount <= 64) {
    tierPoints = 6;
  } else {
    tierPoints = 8;
  }

  const depthBonus = Math.min(2, Math.max(0, safeDepth - 4));
  return Math.min(10, tierPoints + depthBonus);
};

const escapeSvg = (value: string) =>
  value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');

const isCjk = (value: string) => /[\u4e00-\u9fff]/.test(value);

const wrapSvgLines = (value: string, maxChars = 14, maxLines = 3): string[] => {
  const text = value.trim();
  if (!text) return [];
  const chunks = isCjk(text)
    ? text.split('')
    : text.split(/\s+/).filter(Boolean);
  const lines: string[] = [];
  let current = '';

  chunks.forEach((chunk) => {
    const next = current ? `${current}${isCjk(text) ? '' : ' '}${chunk}` : chunk;
    if (next.length > maxChars && current) {
      lines.push(current);
      current = chunk;
      return;
    }
    current = next;
  });

  if (current) {
    lines.push(current);
  }

  if (lines.length <= maxLines) {
    return lines;
  }

  const clipped = lines.slice(0, maxLines);
  clipped[maxLines - 1] = `${clipped[maxLines - 1].slice(0, Math.max(0, maxChars - 1))}…`;
  return clipped;
};

const getNodeBox = (node: Node<MindMapNodeData>) => {
  const width = node.data.isRoot ? SVG_ROOT_WIDTH : SVG_CHILD_WIDTH;
  const height = node.data.isRoot ? SVG_ROOT_HEIGHT : SVG_CHILD_HEIGHT;
  return {
    x: node.position.x - width / 2,
    y: node.position.y - height / 2,
    width,
    height,
  };
};

const edgePath = (source: { x: number; y: number }, target: { x: number; y: number }) => {
  const dx = target.x - source.x;
  const control = Math.max(70, Math.min(220, Math.abs(dx) * 0.45));
  const dir = dx >= 0 ? 1 : -1;
  return `M ${source.x} ${source.y} C ${source.x + dir * control} ${source.y}, ${target.x - dir * control} ${target.y}, ${target.x} ${target.y}`;
};

export const buildMindMapSvgMarkup = (
  root: MindMapTreeNode,
  options: { title?: string; subtitle?: string; background?: string } = {},
): string =>
  buildMindMapSvg(root, {
    title: options.title,
    subtitle: options.subtitle,
  });
