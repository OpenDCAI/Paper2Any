import { useEffect, useMemo, useRef, useState } from 'react';
import ReactFlow, { Background, Controls, MiniMap, type ReactFlowInstance } from 'reactflow';
import 'reactflow/dist/style.css';
import {
  BrainCircuit,
  Copy,
  Download,
  FileText,
  Loader2,
  Plus,
  Save,
  Split,
  Trash2,
  UploadCloud,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';

import ManagedApiNotice from './ManagedApiNotice';
import MindMapNodeCard from './mindmap/MindMapNodeCard';
import { DEFAULT_LLM_API_URL } from '../config/api';
import { useRuntimeBilling } from '../hooks/useRuntimeBilling';
import { appendManagedApiConfig, appendManagedModel } from '../utils/runtimeBillingForm';
import { useAuthStore } from '../stores/authStore';
import { getApiSettings, saveApiSettings } from '../services/apiSettingsService';
import { backendFetch, normalizeBackendAssetUrl } from '../services/backendClient';
import { verifyLlmConnection } from '../services/llmService';
import { checkQuota } from '../services/quotaService';
import {
  addChildToNode,
  addSiblingToNode,
  buildMindMapFlow,
  buildMindMapMarkdown,
  buildMindMapSvg,
  countTreeNodes,
  estimateMindMapPoints,
  findNodeById,
  getTreeDepth,
  normalizeMindMapTree,
  removeNodeById,
  updateNodeById,
  type MindMapTreeNode,
} from '../utils/mindmapTree';

const MINDMAP_MODELS = ['gpt-5.4', 'gpt-5.2'] as const;

type InputMode = 'files' | 'text';
type MindMapStyle = 'default' | 'flowchart' | 'tree';
type OutputLanguage = 'zh' | 'en';

const nodeTypes = { mindMapNode: MindMapNodeCard };

export default function MindMapPage() {
  const { t, i18n } = useTranslation('mindmap');
  const { user } = useAuthStore();
  const { runtimeConfig, userApiConfigRequired } = useRuntimeBilling();

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const reactFlowRef = useRef<ReactFlowInstance | null>(null);
  const [inputMode, setInputMode] = useState<InputMode>('files');
  const [files, setFiles] = useState<File[]>([]);
  const [textContent, setTextContent] = useState('');
  const [apiUrl, setApiUrl] = useState(DEFAULT_LLM_API_URL);
  const [apiKey, setApiKey] = useState('');
  const [model, setModel] = useState<string>(MINDMAP_MODELS[0]);
  const [mindmapStyle, setMindmapStyle] = useState<MindMapStyle>('default');
  const [maxDepth, setMaxDepth] = useState(3);
  const [language, setLanguage] = useState<OutputLanguage>(i18n.language.startsWith('zh') ? 'zh' : 'en');
  const [tree, setTree] = useState<MindMapTreeNode | null>(null);
  const [highlights, setHighlights] = useState<string[]>([]);
  const [mindmapFileUrl, setMindmapFileUrl] = useState('');
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [chargeInfo, setChargeInfo] = useState<{
    nodeCount: number;
    depth: number;
    points: number;
    basePoints: number;
    depthBonus: number;
    tierLabel: string;
    rule: { threshold: number; perLevel: number; maxBonus: number };
  } | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isExporting, setIsExporting] = useState<'svg' | 'png' | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  useEffect(() => {
    const settings = getApiSettings(user?.id || null);
    if (!settings) return;
    setApiUrl(settings.apiUrl || DEFAULT_LLM_API_URL);
    setApiKey(settings.apiKey === '__managed_by_backend__' ? '' : settings.apiKey || '');
  }, [user?.id]);

  const hasInput = useMemo(() => files.length > 0 || textContent.trim().length > 0, [files.length, textContent]);
  const flow = useMemo(() => (tree ? buildMindMapFlow(tree) : { nodes: [], edges: [] }), [tree]);
  const selectedNode = useMemo(() => findNodeById(tree, selectedNodeId || ''), [selectedNodeId, tree]);
  const treeMarkdown = useMemo(() => (tree ? buildMindMapMarkdown(tree) : ''), [tree]);
  const sourceCount = useMemo(() => Math.max(files.length + (textContent.trim() ? 1 : 0), 0), [files.length, textContent]);
  const estimatedPoints = useMemo(() => estimateMindMapPoints(Math.max(1, sourceCount), maxDepth), [sourceCount, maxDepth]);
  const metrics = useMemo(
    () =>
      tree
        ? {
            nodes: countTreeNodes(tree),
            depth: getTreeDepth(tree),
            branches: tree.children.length,
          }
        : { nodes: 0, depth: 0, branches: 0 },
    [tree],
  );

  useEffect(() => {
    if (!flow.nodes.length || !reactFlowRef.current) return;
    window.setTimeout(() => reactFlowRef.current?.fitView({ padding: 0.18, duration: 300 }), 50);
  }, [flow.nodes.length, tree]);

  const appendFiles = (picked: FileList | null) => {
    if (!picked || picked.length === 0) return;
    setFiles((prev) => {
      const existing = new Set(prev.map((file) => `${file.name}-${file.size}-${file.lastModified}`));
      const merged = [...prev];
      Array.from(picked).forEach((file) => {
        const key = `${file.name}-${file.size}-${file.lastModified}`;
        if (!existing.has(key)) {
          existing.add(key);
          merged.push(file);
        }
      });
      return merged;
    });
  };

  const handleGenerate = async () => {
    if (isGenerating) return;
    setError(null);
    setStatus(null);
    setChargeInfo(null);

    if (!hasInput) {
      setError(t('errors.missingInput'));
      return;
    }

    const quota = await checkQuota(user?.id || null, user?.is_anonymous || false);
    if (quota.remaining < estimatedPoints) {
      setError(t('errors.quota', { points: estimatedPoints }));
      return;
    }

    if (userApiConfigRequired) {
      if (!apiUrl.trim() || !apiKey.trim()) {
        setError(t('errors.missingApi'));
        return;
      }
      try {
        await verifyLlmConnection(apiUrl.trim(), apiKey.trim(), model);
      } catch (err) {
        setError(err instanceof Error ? err.message : t('errors.server'));
        return;
      }
    }

    setIsGenerating(true);
    setStatus(t('status.generating', { points: estimatedPoints }));
    try {
      if (userApiConfigRequired) {
        saveApiSettings(user?.id || null, { apiUrl: apiUrl.trim(), apiKey: apiKey.trim() });
      }

      const formData = new FormData();
      appendManagedModel(formData, userApiConfigRequired, 'model', model);
      formData.append('mindmap_style', mindmapStyle);
      formData.append('max_depth', String(maxDepth));
      formData.append('language', language);
      appendManagedApiConfig(formData, userApiConfigRequired, apiUrl, apiKey);
      if (textContent.trim()) {
        formData.append('text', textContent.trim());
      }
      files.forEach((file) => formData.append('files', file));

      const response = await backendFetch('/api/v1/mindmap/generate', {
        method: 'POST',
        headers: {
          'X-Workflow-Amount': String(estimatedPoints),
        },
        body: formData,
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok || !data?.success) {
        throw new Error(data?.detail || data?.error || t('errors.server'));
      }

      const normalizedTree = normalizeMindMapTree(data.tree);
      setTree(normalizedTree);
      setSelectedNodeId(normalizedTree.id);
      setHighlights(Array.isArray(data.highlights) ? data.highlights : []);
      setMindmapFileUrl(normalizeBackendAssetUrl(data.mindmap_path || ''));
      const billing = data.billing && typeof data.billing === 'object' ? data.billing : null;
      if (billing) {
        setChargeInfo({
          nodeCount: Number((billing as Record<string, unknown>).node_count || countTreeNodes(normalizedTree)),
          depth: Number((billing as Record<string, unknown>).depth || getTreeDepth(normalizedTree)),
          points: Number((billing as Record<string, unknown>).points || estimatedPoints),
          basePoints: Number((billing as Record<string, unknown>).base_points || estimatedPoints),
          depthBonus: Number((billing as Record<string, unknown>).depth_bonus || 0),
          tierLabel: String((billing as Record<string, unknown>).tier_label || ''),
          rule: {
            threshold: Number(((billing as Record<string, unknown>).rule as Record<string, unknown> | undefined)?.threshold || 4),
            perLevel: Number(((billing as Record<string, unknown>).rule as Record<string, unknown> | undefined)?.per_level || 1),
            maxBonus: Number(((billing as Record<string, unknown>).rule as Record<string, unknown> | undefined)?.max_bonus || 2),
          },
        });
      }
      setStatus(t('status.generated', { points: Number(data.estimated_points || estimatedPoints) }));
    } catch (err) {
      setError(err instanceof Error ? err.message : t('errors.server'));
    } finally {
      setIsGenerating(false);
    }
  };

  const handleSave = async () => {
    if (isSaving || !mindmapFileUrl || !tree) return;
    setError(null);
    setStatus(null);
    setIsSaving(true);
    try {
      const response = await backendFetch('/api/v1/mindmap/save', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          file_url: mindmapFileUrl,
          tree,
          highlights,
        }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok || !data?.success) {
        throw new Error(data?.detail || data?.error || t('errors.server'));
      }
      if (data.mindmap_path) {
        setMindmapFileUrl(normalizeBackendAssetUrl(data.mindmap_path));
      }
      if (data.tree) {
        setTree(normalizeMindMapTree(data.tree));
      }
      setHighlights(Array.isArray(data.highlights) ? data.highlights : highlights);
      setStatus(t('status.saved'));
    } catch (err) {
      setError(err instanceof Error ? err.message : t('errors.server'));
    } finally {
      setIsSaving(false);
    }
  };

  const handleCopyOutline = async () => {
    if (!treeMarkdown) return;
    try {
      await navigator.clipboard.writeText(treeMarkdown);
      setStatus(t('status.copied'));
    } catch {
      setError(t('errors.server'));
    }
  };

  const handleDownloadJson = () => {
    if (!tree) return;
    const blob = new Blob([JSON.stringify({ root: tree, highlights }, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `mindmap_${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const triggerDownload = (blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  const svgToPngBlob = async (svgMarkup: string): Promise<Blob> => {
    const svgBlob = new Blob([svgMarkup], { type: 'image/svg+xml;charset=utf-8' });
    const svgUrl = URL.createObjectURL(svgBlob);
    try {
      const image = new Image();
      const widthMatch = svgMarkup.match(/width="(\d+(?:\.\d+)?)"/i);
      const heightMatch = svgMarkup.match(/height="(\d+(?:\.\d+)?)"/i);
      const width = Math.max(1200, Math.round(Number(widthMatch?.[1] || 1400)));
      const height = Math.max(900, Math.round(Number(heightMatch?.[1] || 980)));
      await new Promise<void>((resolve, reject) => {
        image.onload = () => resolve();
        image.onerror = () => reject(new Error('mindmap svg rasterize failed'));
        image.src = svgUrl;
      });
      const ratio = Math.min(2, window.devicePixelRatio || 1);
      const canvas = document.createElement('canvas');
      canvas.width = Math.max(1, Math.round(width * ratio));
      canvas.height = Math.max(1, Math.round(height * ratio));
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        throw new Error('canvas context unavailable');
      }
      ctx.scale(ratio, ratio);
      ctx.fillStyle = '#050816';
      ctx.fillRect(0, 0, width, height);
      ctx.drawImage(image, 0, 0, width, height);

      const pngBlob = await new Promise<Blob>((resolve, reject) => {
        canvas.toBlob((output) => {
          if (!output) {
            reject(new Error('png blob unavailable'));
            return;
          }
          resolve(output);
        }, 'image/png');
      });
      return pngBlob;
    } finally {
      URL.revokeObjectURL(svgUrl);
    }
  };

  const handleDownloadSvg = () => {
    if (!tree || isExporting) return;
    try {
      setError(null);
      setIsExporting('svg');
      const svgMarkup = buildMindMapSvg(tree, {
        title: tree.label,
        subtitle: tree.summary || t('hero.description'),
        highlights,
      });
      triggerDownload(new Blob([svgMarkup], { type: 'image/svg+xml;charset=utf-8' }), `mindmap_${Date.now()}.svg`);
      setStatus(t('status.exportedSvg'));
    } catch (err) {
      setError(err instanceof Error ? err.message : t('errors.exportFailed'));
    } finally {
      setIsExporting(null);
    }
  };

  const handleDownloadPng = async () => {
    if (!tree || isExporting) return;
    setError(null);
    setIsExporting('png');
    try {
      const svgMarkup = buildMindMapSvg(tree, {
        title: tree.label,
        subtitle: tree.summary || t('hero.description'),
        highlights,
      });
      const pngBlob = await svgToPngBlob(svgMarkup);
      triggerDownload(pngBlob, `mindmap_${Date.now()}.png`);
      setStatus(t('status.exportedPng'));
    } catch (err) {
      setError(err instanceof Error ? err.message : t('errors.exportFailed'));
    } finally {
      setIsExporting(null);
    }
  };

  const updateSelectedNode = (patch: Partial<Pick<MindMapTreeNode, 'label' | 'summary'>>) => {
    if (!tree || !selectedNodeId) return;
    setTree(updateNodeById(tree, selectedNodeId, patch));
  };

  const handleAddChild = () => {
    if (!tree || !selectedNodeId) return;
    setTree(addChildToNode(tree, selectedNodeId));
  };

  const handleAddSibling = () => {
    if (!tree || !selectedNodeId) return;
    setTree(addSiblingToNode(tree, selectedNodeId));
  };

  const handleDeleteNode = () => {
    if (!tree || !selectedNodeId || selectedNodeId === tree.id) return;
    setTree(removeNodeById(tree, selectedNodeId));
    setSelectedNodeId(tree.id);
  };

  return (
    <div className="h-full overflow-y-auto overflow-x-hidden">
      <div className="mx-auto flex min-h-full w-full max-w-[1600px] flex-col gap-6 px-5 pb-12 pt-6 md:px-8">
        <section className="relative overflow-hidden rounded-[32px] border border-white/10 bg-[linear-gradient(135deg,rgba(9,12,24,0.92),rgba(18,27,41,0.78))] p-6 shadow-[0_40px_120px_rgba(0,0,0,0.45)] backdrop-blur-2xl md:p-8">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_18%_22%,rgba(34,211,238,0.18),transparent_24%),radial-gradient(circle_at_82%_30%,rgba(167,139,250,0.18),transparent_28%),radial-gradient(circle_at_50%_80%,rgba(59,130,246,0.12),transparent_32%)]" />
          <div className="relative grid gap-6 lg:grid-cols-[1.02fr_0.98fr]">
            <div className="space-y-5">
              <div className="inline-flex items-center gap-2 rounded-full border border-cyan-300/25 bg-cyan-300/10 px-4 py-2 text-xs font-semibold uppercase tracking-[0.24em] text-cyan-100/85">
                <BrainCircuit size={14} />
                <span>{t('hero.badge')}</span>
              </div>
              <div className="space-y-3">
                <h1 className="text-4xl font-semibold tracking-tight text-white md:text-5xl">{t('hero.title')}</h1>
                <p className="max-w-3xl text-base leading-7 text-white/65 md:text-lg">{t('hero.description')}</p>
              </div>
              <div className="rounded-2xl border border-emerald-300/15 bg-emerald-300/10 px-4 py-3 text-sm text-emerald-100/90">
                <div className="text-sm font-medium text-emerald-50">{t('pricing.title')}</div>
                <div className="mt-1 text-xs leading-6 text-emerald-50/85">{t('pricing.description')}</div>
                <div className="mt-1 text-xs leading-6 text-emerald-50/80">
                  {t('pricing.preview', { points: estimatedPoints })}
                </div>
                <div className="mt-2 grid gap-1 text-xs leading-5 text-emerald-50/75">
                  <span>{t('pricing.tierSummary')}</span>
                  <span>{t('pricing.depthSummary')}</span>
                  <span>{t('pricing.previewNote', { points: estimatedPoints })}</span>
                </div>
                {chargeInfo ? (
                  <div className="mt-3 rounded-2xl border border-emerald-200/15 bg-black/15 px-3 py-2 text-xs leading-6 text-emerald-50/90">
                    {t('pricing.actual', {
                      nodes: chargeInfo.nodeCount,
                      depth: chargeInfo.depth,
                      points: chargeInfo.points,
                    })}
                  </div>
                ) : null}
                {runtimeConfig.points_purchase_url ? (
                  <>
                    {' '}
                    <a href={runtimeConfig.points_purchase_url} target="_blank" rel="noreferrer" className="font-semibold text-white underline decoration-emerald-200/60 underline-offset-4">
                      {runtimeConfig.points_purchase_url}
                    </a>
                  </>
                ) : null}
              </div>
              {status ? (
                <div className="rounded-2xl border border-cyan-300/15 bg-cyan-300/10 px-4 py-3 text-sm text-cyan-100/90">{status}</div>
              ) : null}
              {error ? (
                <div className="rounded-2xl border border-rose-300/15 bg-rose-300/10 px-4 py-3 text-sm text-rose-100/90">{error}</div>
              ) : null}
            </div>
            <div className="grid gap-4 sm:grid-cols-3">
              <div className="rounded-[26px] border border-white/10 bg-white/5 p-4 backdrop-blur-xl">
                <div className="text-xs font-semibold uppercase tracking-[0.22em] text-white/45">{t('stats.nodes')}</div>
                <div className="mt-3 text-3xl font-semibold text-white">{metrics.nodes}</div>
              </div>
              <div className="rounded-[26px] border border-white/10 bg-white/5 p-4 backdrop-blur-xl">
                <div className="text-xs font-semibold uppercase tracking-[0.22em] text-white/45">{t('stats.depth')}</div>
                <div className="mt-3 text-3xl font-semibold text-white">{metrics.depth}</div>
              </div>
              <div className="rounded-[26px] border border-white/10 bg-white/5 p-4 backdrop-blur-xl">
                <div className="text-xs font-semibold uppercase tracking-[0.22em] text-white/45">{t('stats.branches')}</div>
                <div className="mt-3 text-3xl font-semibold text-white">{metrics.branches}</div>
              </div>
              <div className="sm:col-span-3 rounded-[26px] border border-white/10 bg-[linear-gradient(135deg,rgba(14,165,233,0.14),rgba(59,130,246,0.06))] p-5 backdrop-blur-xl">
                <div className="flex flex-wrap items-center gap-3 text-sm text-white/80">
                  <span className="rounded-full border border-white/12 bg-black/20 px-3 py-1">React Flow</span>
                  <span className="rounded-full border border-white/12 bg-black/20 px-3 py-1">{t('editor.techRoute')}</span>
                  <span className="rounded-full border border-white/12 bg-black/20 px-3 py-1">{t('editor.treeJson')}</span>
                  <span className="rounded-full border border-white/12 bg-black/20 px-3 py-1">{t('editor.costBadge', { points: estimatedPoints })}</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="grid gap-6 2xl:grid-cols-[420px_minmax(0,1fr)_360px]">
          <div className="space-y-5 rounded-[30px] border border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.08),rgba(255,255,255,0.03))] p-5 shadow-[0_24px_80px_rgba(0,0,0,0.35)] backdrop-blur-2xl">
            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={() => setInputMode('files')}
                className={`rounded-full px-4 py-2 text-sm font-semibold transition-all ${inputMode === 'files' ? 'border border-cyan-300/30 bg-cyan-300/20 text-cyan-100' : 'border border-white/10 bg-white/5 text-white/60 hover:bg-white/10'}`}
              >
                {t('input.filesTab')}
              </button>
              <button
                type="button"
                onClick={() => setInputMode('text')}
                className={`rounded-full px-4 py-2 text-sm font-semibold transition-all ${inputMode === 'text' ? 'border border-cyan-300/30 bg-cyan-300/20 text-cyan-100' : 'border border-white/10 bg-white/5 text-white/60 hover:bg-white/10'}`}
              >
                {t('input.textTab')}
              </button>
            </div>

            {inputMode === 'files' ? (
              <div className="space-y-3">
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="flex min-h-[180px] w-full flex-col items-center justify-center gap-4 rounded-[26px] border border-dashed border-white/15 bg-black/20 px-6 py-8 text-center transition-all hover:border-cyan-300/30 hover:bg-cyan-300/5"
                >
                  <UploadCloud className="text-cyan-300" size={30} />
                  <div className="space-y-2">
                    <div className="text-base font-medium text-white">{t('input.dropzone')}</div>
                    <div className="text-sm text-white/50">{t('input.subtitle')}</div>
                  </div>
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  className="hidden"
                  accept=".pdf,.doc,.docx,.ppt,.pptx,.txt,.md"
                  onChange={(event) => appendFiles(event.target.files)}
                />
                {files.length > 0 ? (
                  <div className="rounded-2xl border border-white/10 bg-black/20 px-4 py-3 text-sm text-white/75">
                    <div>{t('input.picked', { count: files.length })}</div>
                    <div className="mt-2 flex flex-wrap gap-2">
                      {files.map((file) => (
                        <span key={`${file.name}-${file.size}-${file.lastModified}`} className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-white/70">
                          {file.name}
                        </span>
                      ))}
                    </div>
                    <button type="button" className="mt-3 text-xs font-medium text-cyan-200 hover:text-cyan-100" onClick={() => setFiles([])}>
                      {t('input.clear')}
                    </button>
                  </div>
                ) : null}
              </div>
            ) : (
              <textarea
                value={textContent}
                onChange={(event) => setTextContent(event.target.value)}
                placeholder={t('input.textPlaceholder')}
                className="min-h-[240px] w-full rounded-[26px] border border-white/10 bg-black/20 px-4 py-4 text-sm leading-7 text-white outline-none transition-all placeholder:text-white/30 focus:border-cyan-300/25 focus:bg-black/28"
              />
            )}

            <div className="space-y-4 rounded-[26px] border border-white/10 bg-black/20 p-4">
              <div>
                <div className="mb-2 text-xs font-semibold uppercase tracking-[0.22em] text-white/45">{t('settings.model')}</div>
                <select
                  value={model}
                  onChange={(event) => setModel(event.target.value)}
                  disabled={!userApiConfigRequired}
                  className="w-full rounded-2xl border border-white/10 bg-white/5 px-3 py-3 text-sm text-white outline-none focus:border-cyan-300/35"
                >
                  {MINDMAP_MODELS.map((item) => (
                    <option key={item} value={item} className="bg-slate-950">
                      {item}
                    </option>
                  ))}
                </select>
                {!userApiConfigRequired ? (
                  <p className="mt-2 text-[11px] leading-5 text-emerald-100/70">Free 模式下由后端统一选择思维导图模型。</p>
                ) : null}
              </div>

              <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-1 2xl:grid-cols-2">
                <div>
                  <div className="mb-2 text-xs font-semibold uppercase tracking-[0.22em] text-white/45">{t('settings.style')}</div>
                  <select
                    value={mindmapStyle}
                    onChange={(event) => setMindmapStyle(event.target.value as MindMapStyle)}
                    className="w-full rounded-2xl border border-white/10 bg-white/5 px-3 py-3 text-sm text-white outline-none focus:border-cyan-300/35"
                  >
                    <option value="default" className="bg-slate-950">{t('settings.styleDefault')}</option>
                    <option value="flowchart" className="bg-slate-950">{t('settings.styleFlowchart')}</option>
                    <option value="tree" className="bg-slate-950">{t('settings.styleTree')}</option>
                  </select>
                </div>
                <div>
                  <div className="mb-2 text-xs font-semibold uppercase tracking-[0.22em] text-white/45">{t('settings.language')}</div>
                  <select
                    value={language}
                    onChange={(event) => setLanguage(event.target.value as OutputLanguage)}
                    className="w-full rounded-2xl border border-white/10 bg-white/5 px-3 py-3 text-sm text-white outline-none focus:border-cyan-300/35"
                  >
                    <option value="zh" className="bg-slate-950">中文</option>
                    <option value="en" className="bg-slate-950">English</option>
                  </select>
                </div>
              </div>

              <div>
                <div className="mb-2 flex items-center justify-between text-xs font-semibold uppercase tracking-[0.22em] text-white/45">
                  <span>{t('settings.depth')}</span>
                  <span className="text-cyan-200">{maxDepth}</span>
                </div>
                <input
                  type="range"
                  min={2}
                  max={6}
                  value={maxDepth}
                  onChange={(event) => setMaxDepth(Number(event.target.value))}
                  className="w-full accent-cyan-300"
                />
              </div>

              {userApiConfigRequired ? (
                <div className="space-y-3">
                  <div>
                    <div className="mb-2 text-xs font-semibold uppercase tracking-[0.22em] text-white/45">{t('settings.apiUrl')}</div>
                    <input
                      value={apiUrl}
                      onChange={(event) => setApiUrl(event.target.value)}
                      className="w-full rounded-2xl border border-white/10 bg-white/5 px-3 py-3 text-sm text-white outline-none focus:border-cyan-300/35"
                    />
                  </div>
                  <div>
                    <div className="mb-2 text-xs font-semibold uppercase tracking-[0.22em] text-white/45">{t('settings.apiKey')}</div>
                    <input
                      type="password"
                      value={apiKey}
                      onChange={(event) => setApiKey(event.target.value)}
                      className="w-full rounded-2xl border border-white/10 bg-white/5 px-3 py-3 text-sm text-white outline-none focus:border-cyan-300/35"
                    />
                  </div>
                </div>
              ) : (
                <ManagedApiNotice />
              )}

              <button
                type="button"
                onClick={handleGenerate}
                disabled={isGenerating}
                className="flex w-full items-center justify-center gap-2 rounded-[22px] bg-[linear-gradient(135deg,#0ea5e9,#2563eb)] px-4 py-3 text-sm font-semibold text-white shadow-[0_18px_44px_rgba(14,165,233,0.35)] transition-all hover:scale-[1.01] disabled:cursor-not-allowed disabled:opacity-60"
              >
                {isGenerating ? <Loader2 size={18} className="animate-spin" /> : <BrainCircuit size={18} />}
                <span>{isGenerating ? t('actions.generating') : t('actions.generate')}</span>
              </button>
              <div className="text-xs leading-6 text-white/45">
                {t('pricing.ruleSummary')}
              </div>
            </div>
          </div>

          <div className="rounded-[30px] border border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.08),rgba(255,255,255,0.03))] p-5 shadow-[0_24px_80px_rgba(0,0,0,0.35)] backdrop-blur-2xl">
            <div className="mb-4 flex items-center justify-between gap-4">
              <div>
                <div className="text-lg font-semibold text-white">{t('editor.title')}</div>
                <div className="text-sm text-white/55">{t('editor.subtitle')}</div>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <button type="button" onClick={() => reactFlowRef.current?.fitView({ padding: 0.18, duration: 300 })} className="rounded-full border border-white/10 bg-white/5 px-3 py-2 text-xs font-medium text-white/75 hover:bg-white/10">
                  {t('actions.fitView')}
                </button>
                <button type="button" onClick={handleCopyOutline} disabled={!tree} className="rounded-full border border-white/10 bg-white/5 px-3 py-2 text-xs font-medium text-white/75 hover:bg-white/10 disabled:opacity-40">
                  <Copy size={14} className="mr-1 inline" />
                  {t('actions.copyOutline')}
                </button>
                <button type="button" onClick={handleDownloadJson} disabled={!tree} className="rounded-full border border-white/10 bg-white/5 px-3 py-2 text-xs font-medium text-white/75 hover:bg-white/10 disabled:opacity-40">
                  <Download size={14} className="mr-1 inline" />
                  {t('actions.downloadJson')}
                </button>
                <button
                  type="button"
                  onClick={handleDownloadSvg}
                  disabled={!tree || isExporting !== null}
                  className="rounded-full border border-white/10 bg-white/5 px-3 py-2 text-xs font-medium text-white/75 hover:bg-white/10 disabled:opacity-40"
                >
                  <Download size={14} className="mr-1 inline" />
                  {isExporting === 'svg' ? t('actions.exportingSvg') : t('actions.downloadSvg')}
                </button>
                <button
                  type="button"
                  onClick={handleDownloadPng}
                  disabled={!tree || isExporting !== null}
                  className="rounded-full border border-white/10 bg-white/5 px-3 py-2 text-xs font-medium text-white/75 hover:bg-white/10 disabled:opacity-40"
                >
                  <Download size={14} className="mr-1 inline" />
                  {isExporting === 'png' ? t('actions.exportingPng') : t('actions.downloadPng')}
                </button>
              </div>
            </div>

            {tree ? (
              <div className="h-[760px] overflow-hidden rounded-[26px] border border-white/10 bg-[radial-gradient(circle_at_top,rgba(34,211,238,0.12),transparent_35%),linear-gradient(180deg,rgba(3,7,18,0.96),rgba(15,23,42,0.92))]">
                <ReactFlow
                  nodes={flow.nodes}
                  edges={flow.edges}
                  nodeTypes={nodeTypes}
                  fitView
                  onInit={(instance) => {
                    reactFlowRef.current = instance;
                  }}
                  onNodeClick={(_, node) => setSelectedNodeId(node.id)}
                  panOnScroll
                  zoomOnScroll
                  nodesDraggable={false}
                  elementsSelectable
                  proOptions={{ hideAttribution: true }}
                >
                  <MiniMap nodeColor="#38bdf8" maskColor="rgba(2,6,23,0.45)" />
                  <Controls />
                  <Background gap={22} size={1} color="rgba(148,163,184,0.18)" />
                </ReactFlow>
              </div>
            ) : (
              <div className="flex h-[760px] flex-col items-center justify-center rounded-[26px] border border-dashed border-white/12 bg-black/20 px-6 text-center">
                <FileText size={34} className="text-cyan-300/75" />
                <div className="mt-5 text-lg font-semibold text-white">{t('editor.emptyTitle')}</div>
                <div className="mt-2 max-w-lg text-sm leading-7 text-white/55">{t('editor.emptyDesc')}</div>
              </div>
            )}
          </div>

          <div className="space-y-5 rounded-[30px] border border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.08),rgba(255,255,255,0.03))] p-5 shadow-[0_24px_80px_rgba(0,0,0,0.35)] backdrop-blur-2xl">
            <div>
              <div className="text-lg font-semibold text-white">{t('details.title')}</div>
              <div className="text-sm text-white/55">{t('details.subtitle')}</div>
            </div>

            <div className="space-y-3 rounded-[24px] border border-white/10 bg-black/20 p-4">
              <div className="flex items-center justify-between">
                <div className="text-sm font-medium text-white">{t('details.selectedNode')}</div>
                {selectedNode && selectedNode.id !== tree?.id ? (
                  <button type="button" onClick={handleDeleteNode} className="rounded-full border border-rose-300/15 bg-rose-300/10 p-2 text-rose-100 hover:bg-rose-300/15">
                    <Trash2 size={14} />
                  </button>
                ) : null}
              </div>
              {selectedNode ? (
                <div className="space-y-3">
                  <div>
                    <div className="mb-2 text-xs font-semibold uppercase tracking-[0.22em] text-white/45">{t('details.label')}</div>
                    <input
                      value={selectedNode.label}
                      onChange={(event) => updateSelectedNode({ label: event.target.value })}
                      className="w-full rounded-2xl border border-white/10 bg-white/5 px-3 py-3 text-sm text-white outline-none focus:border-cyan-300/35"
                    />
                  </div>
                  <div>
                    <div className="mb-2 text-xs font-semibold uppercase tracking-[0.22em] text-white/45">{t('details.summary')}</div>
                    <textarea
                      value={selectedNode.summary || ''}
                      onChange={(event) => updateSelectedNode({ summary: event.target.value })}
                      className="min-h-[110px] w-full rounded-2xl border border-white/10 bg-white/5 px-3 py-3 text-sm leading-6 text-white outline-none focus:border-cyan-300/35"
                    />
                  </div>
                  <div className="grid gap-2 sm:grid-cols-2">
                    <button type="button" onClick={handleAddChild} className="rounded-2xl border border-white/10 bg-white/5 px-3 py-3 text-sm font-medium text-white/80 hover:bg-white/10">
                      <Plus size={15} className="mr-1 inline" />
                      {t('actions.addChild')}
                    </button>
                    <button type="button" onClick={handleAddSibling} disabled={!selectedNodeId || selectedNodeId === tree?.id} className="rounded-2xl border border-white/10 bg-white/5 px-3 py-3 text-sm font-medium text-white/80 hover:bg-white/10 disabled:opacity-40">
                      <Split size={15} className="mr-1 inline" />
                      {t('actions.addSibling')}
                    </button>
                  </div>
                </div>
              ) : (
                <div className="text-sm leading-6 text-white/55">{t('details.empty')}</div>
              )}
            </div>

            <div className="space-y-3 rounded-[24px] border border-white/10 bg-black/20 p-4">
              <div className="text-sm font-medium text-white">{t('details.highlights')}</div>
              {highlights.length > 0 ? (
                <div className="space-y-2">
                  {highlights.map((item, index) => (
                    <div key={`${item}-${index}`} className="rounded-2xl border border-white/10 bg-white/5 px-3 py-3 text-sm leading-6 text-white/72">
                      {item}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-sm leading-6 text-white/55">{t('details.highlightsEmpty')}</div>
              )}
            </div>

            <div className="space-y-3 rounded-[24px] border border-white/10 bg-black/20 p-4">
              <div className="text-sm font-medium text-white">{t('details.source')}</div>
              <div className="rounded-2xl border border-white/10 bg-white/5 px-3 py-3 text-xs leading-6 text-white/60">
                {mindmapFileUrl || t('details.sourceEmpty')}
              </div>
              <div className="grid gap-2 sm:grid-cols-2">
                <button
                  type="button"
                  onClick={handleSave}
                  disabled={!tree || !mindmapFileUrl || isSaving}
                  className="rounded-2xl bg-[linear-gradient(135deg,#14b8a6,#0ea5e9)] px-3 py-3 text-sm font-semibold text-white shadow-[0_18px_40px_rgba(20,184,166,0.25)] disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {isSaving ? <Loader2 size={15} className="mr-1 inline animate-spin" /> : <Save size={15} className="mr-1 inline" />}
                  {isSaving ? t('actions.saving') : t('actions.save')}
                </button>
                {mindmapFileUrl ? (
                  <a href={mindmapFileUrl} target="_blank" rel="noreferrer" className="rounded-2xl border border-white/10 bg-white/5 px-3 py-3 text-center text-sm font-medium text-white/80 hover:bg-white/10">
                    {t('actions.openFile')}
                  </a>
                ) : (
                  <div className="rounded-2xl border border-white/10 bg-white/5 px-3 py-3 text-center text-sm font-medium text-white/35">
                    {t('actions.openFile')}
                  </div>
                )}
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
