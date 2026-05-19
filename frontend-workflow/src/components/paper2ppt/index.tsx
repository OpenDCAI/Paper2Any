import React, { useState, useEffect, ChangeEvent, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { uploadAndSaveFile } from '../../services/fileService';
import { DEFAULT_LLM_API_URL } from '../../config/api';
import { DEFAULT_PAPER2PPT_GEN_FIG_MODEL, DEFAULT_PAPER2PPT_MODEL } from '../../config/models';
import { checkQuota, recordUsage } from '../../services/quotaService';
import { verifyLlmConnection } from '../../services/llmService';
import { useAuthStore } from '../../stores/authStore';
import { getApiSettings, saveApiSettings } from '../../services/apiSettingsService';
import { backendFetch } from '../../services/backendClient';
import { useRuntimeBilling } from '../../hooks/useRuntimeBilling';
import {
  buildInsufficientPointsMessage,
  buildQuotaExhaustedMessage,
  resolvePointsPurchaseUrl,
} from '../../utils/pointsMessaging';

import {
  FrontendBlockChild,
  FrontendDeckTheme,
  FrontendEditableField,
  FrontendCanvasNode,
  FrontendCanvasVisualSpec,
  FrontendCanvasVisualStyle,
  FrontendSlide,
  FrontendSlideBlock,
  FrontendTableData,
  parseFrontendInsertZoneTarget,
  PptGenerationMode,
  Step,
  SlideOutline,
  GenerateResult,
  UploadMode,
  StyleMode,
  StylePreset,
  Paper2PPTTaskResponse,
} from './types';
import { MAX_FILE_SIZE, STORAGE_KEY } from './constants';

import Banner from './Banner';
import StepIndicator from './StepIndicator';
import UploadStep from './UploadStep';
import OutlineStep from './OutlineStep';
import GenerateStep from './GenerateStep';
import CompleteStep from './CompleteStep';
import FrontendGenerateStep from './FrontendGenerateStep';
import FrontendCompleteStep from './FrontendCompleteStep';
import FrontendSlidePreview from './FrontendSlidePreview';
import {
  buildFrontendCodeRepairPrompt,
  captureSlideToPngBlob,
  inspectSlideLayout,
  validateFrontendSlideCode,
} from './frontendSlideUtils';
import {
  buildCanvasSlidesPptxBlob,
  canExportCanvasSlidesToPptx,
} from './canvasPptxExporter';

const MANAGED_CREDENTIAL_SCOPE = 'paper2ppt';

export interface Paper2PptPageProps {
  initialMode?: PptGenerationMode;
}

const Paper2PptPage: React.FC<Paper2PptPageProps> = ({ initialMode }) => {
  const { user, refreshQuota } = useAuthStore();
  const { userApiConfigRequired, runtimeConfig } = useRuntimeBilling();
  const modeLocked = Boolean(initialMode);
  const purchaseUrl = runtimeConfig.billing_mode === 'free'
    ? resolvePointsPurchaseUrl(runtimeConfig)
    : '';
  
  // Step 状态
  const [currentStep, setCurrentStep] = useState<Step>('upload');
  const [pptMode, setPptMode] = useState<PptGenerationMode>(initialMode || 'image');
  
  // Step 1: 上传相关状态
  const [uploadMode, setUploadMode] = useState<UploadMode>('file');
  const [textContent, setTextContent] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [styleMode, setStyleMode] = useState<StyleMode>('prompt');
  const [stylePreset, setStylePreset] = useState<StylePreset>('modern');
  const [globalPrompt, setGlobalPrompt] = useState('');
  const [referenceImage, setReferenceImage] = useState<File | null>(null);
  const [referenceImagePreview, setReferenceImagePreview] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [pageCount, setPageCount] = useState(6);
  const [useLongPaper, setUseLongPaper] = useState(false);
  const [frontendIncludeImages, setFrontendIncludeImages] = useState(false);
  const [frontendAutoReviewEnabled, setFrontendAutoReviewEnabled] = useState(false);
  const [frontendImageStyle, setFrontendImageStyle] = useState('academic_illustration');
  const [progress, setProgress] = useState(0);
  const [progressStatus, setProgressStatus] = useState('');
  
  // Step 2: Outline 相关状态
  const [outlineData, setOutlineData] = useState<SlideOutline[]>([]);
  const [confirmedOutlineSnapshot, setConfirmedOutlineSnapshot] = useState<SlideOutline[]>([]);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editContent, setEditContent] = useState<{
    title: string;
    layout_description: string;
    key_points: string[];
  }>({ title: '', layout_description: '', key_points: [] });
  const [outlineFeedback, setOutlineFeedback] = useState('');
  const [isRefiningOutline, setIsRefiningOutline] = useState(false);
  
  // Step 3: 生成相关状态
  const [currentSlideIndex, setCurrentSlideIndex] = useState(0);
  const [generateResults, setGenerateResults] = useState<GenerateResult[]>([]);
  const [frontendSlides, setFrontendSlides] = useState<FrontendSlide[]>([]);
  const [frontendDeckTheme, setFrontendDeckTheme] = useState<FrontendDeckTheme | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isReviewingFrontendSlide, setIsReviewingFrontendSlide] = useState(false);
  const [slidePrompt, setSlidePrompt] = useState('');
  const [generateTaskMessage, setGenerateTaskMessage] = useState('');
  
  // Step 4: 完成状态
  const [isGeneratingFinal, setIsGeneratingFinal] = useState(false);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [pdfPreviewUrl, setPdfPreviewUrl] = useState<string | null>(null);
  const [finalTaskMessage, setFinalTaskMessage] = useState('');

  // 通用状态
  const [error, setError] = useState<string | null>(null);
  const [showBanner, setShowBanner] = useState(true);

  // API 配置状态 - 从环境变量读取默认值
  const [llmApiUrl, setLlmApiUrl] = useState(DEFAULT_LLM_API_URL);
  const [apiKey, setApiKey] = useState('');
  const [model, setModel] = useState(DEFAULT_PAPER2PPT_MODEL);
  const [genFigModel, setGenFigModel] = useState(DEFAULT_PAPER2PPT_GEN_FIG_MODEL);
  const [language, setLanguage] = useState<'zh' | 'en'>('en');
  const [resultPath, setResultPath] = useState<string | null>(null);
  const frontendCaptureRefs = useRef<Array<HTMLDivElement | null>>([]);
  const uploadSubmitGuardRef = useRef(false);
  const uploadSubmitGuardTimerRef = useRef<number | null>(null);
  const [isUploadSubmitLocked, setIsUploadSubmitLocked] = useState(false);
  const outlineSubmitGuardRef = useRef(false);
  const outlineSubmitGuardTimerRef = useRef<number | null>(null);
  const [isOutlineSubmitLocked, setIsOutlineSubmitLocked] = useState(false);

  // GitHub Stars
  const [stars, setStars] = useState<{dataflow: number | null, agent: number | null, dataflex: number | null}>({
    dataflow: null,
    agent: null,
    dataflex: null,
  });
  const [copySuccess, setCopySuccess] = useState('');

  const shareText = `发现一个超好用的AI工具 DataFlow-Agent！🚀
支持论文转PPT、PDF转PPT、PPT美化等功能，科研打工人的福音！

🔗 在线体验：https://dcai-paper2any.nas.cpolar.cn/
⭐ GitHub Agent：https://github.com/OpenDCAI/Paper2Any
🌟 GitHub Core：https://github.com/OpenDCAI/DataFlow

转发本文案+截图，联系微信群管理员即可获取免费Key！🎁
#AI工具 #PPT制作 #科研效率 #开源项目`;

  const getQuotaContext = () => ({
    userId: user?.id || null,
    isAnonymous: user?.is_anonymous || false,
  });

  const ensureQuotaForAction = async (required: number, action: string) => {
    const { userId, isAnonymous } = getQuotaContext();
    const quota = await checkQuota(userId, isAnonymous);
    if (quota.remaining < required) {
      setError(buildInsufficientPointsMessage(required, quota.remaining, action, purchaseUrl));
      return false;
    }
    return true;
  };

  const consumeQuotaForAction = async (workflowType: string, amount: number, warningMessage: string) => {
    const { userId, isAnonymous } = getQuotaContext();
    const ok = await recordUsage(userId, workflowType, { amount, isAnonymous });
    refreshQuota();
    if (!ok) {
      setError((prev) => prev || warningMessage);
    }
    return ok;
  };

  const normalizeBackendErrorDetail = (detail: unknown): string | null => {
    if (typeof detail === 'string' && detail.trim()) {
      return detail.trim();
    }
    if (Array.isArray(detail)) {
      const messages = detail
        .map((item) => {
          if (!item || typeof item !== 'object') {
            return '';
          }
          const entry = item as { loc?: unknown; msg?: unknown; type?: unknown };
          const loc = Array.isArray(entry.loc) ? entry.loc.slice(1).join('.') : '';
          const msg = typeof entry.msg === 'string' ? entry.msg.trim() : '';
          const type = typeof entry.type === 'string' ? entry.type.trim() : '';
          return [loc, msg || type].filter(Boolean).join(': ');
        })
        .filter(Boolean);
      return messages.length ? messages.join('；') : null;
    }
    if (detail && typeof detail === 'object') {
      const entry = detail as { message?: unknown; detail?: unknown; error?: unknown };
      if (typeof entry.message === 'string' && entry.message.trim()) {
        return entry.message.trim();
      }
      if (typeof entry.detail === 'string' && entry.detail.trim()) {
        return entry.detail.trim();
      }
      if (typeof entry.error === 'string' && entry.error.trim()) {
        return entry.error.trim();
      }
    }
    return null;
  };

  const extractErrorMessage = async (res: Response, fallback: string) => {
    if (res.status === 403) {
      return '邀请码不正确或已失效';
    }
    if (res.status === 429) {
      return '请求过于频繁，请稍后再试';
    }
    try {
      const errBody = await res.json();
      const detailMessage = normalizeBackendErrorDetail(errBody?.detail);
      if (detailMessage) {
        return detailMessage;
      }
      if (typeof errBody?.error === 'string' && errBody.error.trim()) {
        return errBody.error;
      }
      if (typeof errBody?.message === 'string' && errBody.message.trim()) {
        return errBody.message;
      }
    } catch {
      // ignore parse error
    }
    return fallback;
  };

  useEffect(() => {
    return () => {
      if (uploadSubmitGuardTimerRef.current !== null) {
        window.clearTimeout(uploadSubmitGuardTimerRef.current);
      }
      if (outlineSubmitGuardTimerRef.current !== null) {
        window.clearTimeout(outlineSubmitGuardTimerRef.current);
      }
    };
  }, []);

  const releaseUploadSubmitGuard = (cooldownMs: number = 1200) => {
    if (uploadSubmitGuardTimerRef.current !== null) {
      window.clearTimeout(uploadSubmitGuardTimerRef.current);
    }
    uploadSubmitGuardTimerRef.current = window.setTimeout(() => {
      uploadSubmitGuardRef.current = false;
      setIsUploadSubmitLocked(false);
      uploadSubmitGuardTimerRef.current = null;
    }, cooldownMs);
  };

  const releaseOutlineSubmitGuard = (cooldownMs: number = 1500) => {
    if (outlineSubmitGuardTimerRef.current !== null) {
      window.clearTimeout(outlineSubmitGuardTimerRef.current);
    }
    outlineSubmitGuardTimerRef.current = window.setTimeout(() => {
      outlineSubmitGuardRef.current = false;
      setIsOutlineSubmitLocked(false);
      outlineSubmitGuardTimerRef.current = null;
    }, cooldownMs);
  };

  const handleCopyShareText = async () => {
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(shareText);
      } else {
        const textArea = document.createElement("textarea");
        textArea.value = shareText;
        textArea.style.position = "fixed";
        textArea.style.left = "-9999px";
        textArea.style.top = "0";
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        try {
          document.execCommand('copy');
        } catch (err) {
          console.error('Fallback: Oops, unable to copy', err);
          throw err;
        } finally {
          document.body.removeChild(textArea);
        }
      }
      setCopySuccess('文案已复制！快去分享吧');
      setTimeout(() => setCopySuccess(''), 2000);
    } catch (err) {
      console.error('复制失败', err);
      setCopySuccess('复制失败，请手动复制');
    }
  };

  useEffect(() => {
    const fetchStars = async () => {
      try {
        const [res1, res2, res3] = await Promise.all([
          fetch('https://api.github.com/repos/OpenDCAI/DataFlow'),
          fetch('https://api.github.com/repos/OpenDCAI/Paper2Any'),
          fetch('https://api.github.com/repos/OpenDCAI/DataFlex')
        ]);
        const data1 = await res1.json();
        const data2 = await res2.json();
        const data3 = await res3.json();
        setStars({
          dataflow: data1.stargazers_count,
          agent: data2.stargazers_count,
          dataflex: data3.stargazers_count,
        });
      } catch (e) {
        console.error('Failed to fetch stars', e);
      }
    };
    fetchStars();
  }, []);

  // 从 localStorage 恢复配置
  useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const saved = JSON.parse(raw);
        
        if (saved.pptMode && !initialMode) setPptMode(saved.pptMode);
        if (saved.uploadMode) setUploadMode(saved.uploadMode);
        if (saved.textContent) setTextContent(saved.textContent);
        if (saved.styleMode) setStyleMode(saved.styleMode);
        if (saved.stylePreset) setStylePreset(saved.stylePreset);
        if (saved.globalPrompt) setGlobalPrompt(saved.globalPrompt);
        if (saved.pageCount) setPageCount(saved.pageCount);
        if (saved.useLongPaper !== undefined) setUseLongPaper(saved.useLongPaper);
        if (saved.frontendIncludeImages !== undefined) setFrontendIncludeImages(Boolean(saved.frontendIncludeImages));
        if (saved.frontendAutoReviewEnabled !== undefined) {
          setFrontendAutoReviewEnabled(Boolean(saved.frontendAutoReviewEnabled));
        }
        if (saved.frontendImageStyle) setFrontendImageStyle(saved.frontendImageStyle);
        if (saved.model) setModel(saved.model);
        if (saved.genFigModel) setGenFigModel(saved.genFigModel);
        if (saved.language) setLanguage(saved.language);

        // API settings: prioritize user-specific settings from apiSettingsService
        const userApiSettings = getApiSettings(user?.id || null);
        if (userApiSettings) {
          if (userApiSettings.apiUrl) setLlmApiUrl(userApiSettings.apiUrl);
          if (userApiSettings.apiKey) setApiKey(userApiSettings.apiKey);
        } else {
          if (saved.llmApiUrl) setLlmApiUrl(saved.llmApiUrl);
          if (saved.apiKey) setApiKey(saved.apiKey);
        }
      }
    } catch (e) {
      console.error('Failed to restore paper2ppt config', e);
    }
  }, [user?.id, userApiConfigRequired]);

  useEffect(() => {
    if (initialMode) {
      setPptMode(initialMode);
    }
  }, [initialMode]);

  // 将配置写入 localStorage
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const data = {
      pptMode,
      uploadMode,
      textContent,
      styleMode,
      stylePreset,
      globalPrompt,
      pageCount,
      useLongPaper,
      frontendIncludeImages,
      frontendAutoReviewEnabled,
      frontendImageStyle,
      llmApiUrl,
      apiKey,
      model,
      genFigModel,
      language
    };
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
      if (user?.id && llmApiUrl && apiKey) {
        saveApiSettings(user.id, { apiUrl: llmApiUrl, apiKey });
      }
    } catch (e) {
      console.error('Failed to persist paper2ppt config', e);
    }
  }, [
    pptMode, uploadMode, textContent, styleMode, stylePreset, globalPrompt,
    pageCount, useLongPaper, frontendIncludeImages, frontendAutoReviewEnabled, frontendImageStyle, llmApiUrl, apiKey,
    model, genFigModel, language, user?.id
  ]);

  // 自动加载版本历史
  useEffect(() => {
    if (currentStep === 'generate' && currentSlideIndex >= 0 && generateResults[currentSlideIndex]) {
      const currentResult = generateResults[currentSlideIndex];
      // 如果版本历史为空且页面已生成，则自动加载版本历史
      if (currentResult.versionHistory.length === 0 && currentResult.afterImage) {
        console.log(`[Paper2PptPage] 自动加载页面 ${currentSlideIndex} 的版本历史`);
        fetchVersionHistory(currentSlideIndex);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentStep, currentSlideIndex]); // 移除 generateResults 依赖，避免无限循环

  const sleep = (ms: number) => new Promise(resolve => window.setTimeout(resolve, ms));

  const parseErrorMessage = async (res: Response, fallback: string) => {
    return extractErrorMessage(res, fallback);
  };

  const submitPaper2PptTask = async (
    formData: FormData,
    workflowAmount?: number,
  ): Promise<Paper2PPTTaskResponse> => {
    const res = await backendFetch('/api/v1/paper2ppt/generate-task', {
      method: 'POST',
      headers: workflowAmount && workflowAmount > 0
        ? { 'X-Workflow-Amount': String(workflowAmount) }
        : undefined,
      body: formData,
    });

    if (!res.ok) {
      throw new Error(await parseErrorMessage(res, '服务器繁忙，请稍后再试'));
    }

    const data = await res.json() as Paper2PPTTaskResponse;
    if (!data.success || !data.task_id) {
      throw new Error(data.error || data.message || '任务提交失败');
    }
    return data;
  };

  const pollPaper2PptTask = async (
    taskId: string,
    onUpdate?: (task: Paper2PPTTaskResponse) => void,
  ) => {
    let transientFailures = 0;

    for (let attempt = 0; attempt < 720; attempt += 1) {
      try {
        const res = await backendFetch(`/api/v1/paper2ppt/tasks/${taskId}`);
        if (!res.ok) {
          throw new Error(await parseErrorMessage(res, '任务状态查询失败'));
        }

        const data = await res.json() as Paper2PPTTaskResponse;
        onUpdate?.(data);
        transientFailures = 0;

        if (data.status === 'done') {
          if (!data.result) {
            throw new Error('任务已完成，但缺少结果文件');
          }
          return data.result;
        }

        if (data.status === 'failed') {
          throw new Error(data.error || data.message || '任务执行失败');
        }
      } catch (err) {
        transientFailures += 1;
        if (transientFailures >= 5) {
          throw err instanceof Error ? err : new Error('任务轮询失败');
        }
      }

      await sleep(attempt < 20 ? 1500 : 2500);
    }

    throw new Error('任务执行超时，请稍后到历史输出目录检查结果');
  };

  const preloadGeneratedImages = (outputFiles?: string[]) => {
    if (!outputFiles || !Array.isArray(outputFiles)) return;
    console.log('预加载所有生成的图片...');
    outputFiles.forEach((url: string) => {
      if (url.endsWith('.png') || url.endsWith('.jpg') || url.endsWith('.jpeg')) {
        const img = new Image();
        img.src = url;
      }
    });
  };

  const getPreviewPath = (item: any, key: string) =>
    String(item?.[`${key}_preview_path`] || item?.[`${key}PreviewPath`] || '').trim();

  const SUPPORTED_SCHEMA_TEMPLATE_KEYS = new Set([
    'title_cover',
    'section_divider',
    'text_focus',
    'hero_visual',
    'split_media',
    'visual_compare',
    'insight_grid',
    'metrics_dashboard',
    'timeline_overview',
    'stacked_cards',
    'quote_focus',
    'dual_list',
  ]);

  const TEMPLATE_KEY_ALIASES: Record<string, string> = {
    cover: 'title_cover',
    cover_slide: 'title_cover',
    title_slide: 'title_cover',
    divider: 'section_divider',
    section: 'section_divider',
    section_break: 'section_divider',
    text_only: 'text_focus',
    text_heavy: 'text_focus',
    hero: 'hero_visual',
    single_visual: 'hero_visual',
    media_split: 'split_media',
    split_layout: 'split_media',
    compare: 'visual_compare',
    comparison: 'visual_compare',
    image_compare: 'visual_compare',
    grid: 'insight_grid',
    card_grid: 'insight_grid',
    dashboard: 'metrics_dashboard',
    metrics: 'metrics_dashboard',
    timeline: 'timeline_overview',
    process_timeline: 'timeline_overview',
    cards: 'stacked_cards',
    card_stack: 'stacked_cards',
    quote: 'quote_focus',
    quote_slide: 'quote_focus',
    two_lists: 'dual_list',
    dual_column_list: 'dual_list',
  };

  const PREFERRED_SCHEMA_FIELD_KEYS: Record<string, string> = {
    title: 'title',
    summary: 'summary',
    key_points: 'key_points',
    takeaway: 'takeaway',
    footer: 'footer',
    eyebrow: 'eyebrow',
  };

  const normalizeStringList = (value: unknown): string[] =>
    Array.isArray(value)
      ? value.map((item) => String(item || '').trim()).filter(Boolean)
      : [];

  const normalizeTableRows = (value: unknown): string[][] =>
    Array.isArray(value)
      ? value
          .map((row) =>
            Array.isArray(row)
              ? row.map((cell) => String(cell ?? '').trim())
              : [],
          )
          .filter((row) => row.length > 0)
      : [];

  const normalizeSchemaTableData = (value: unknown): FrontendTableData | undefined => {
    const source = (value && typeof value === 'object') ? value as Record<string, unknown> : {};
    const headers = normalizeStringList(
      source.headers
      || source.columns
      || source.cols,
    );
    const rows = normalizeTableRows(
      source.rows
      || source.data
      || source.values,
    );
    const maxColumns = Math.max(headers.length, ...rows.map((row) => row.length), 0);
    if (maxColumns === 0) {
      return undefined;
    }
    return {
      headers: Array.from({ length: maxColumns }, (_, index) => headers[index] || `列 ${index + 1}`),
      rows: rows.length > 0
        ? rows.map((row) => Array.from({ length: maxColumns }, (_, index) => row[index] || ''))
        : [Array.from({ length: maxColumns }, () => '')],
    };
  };

  const toFiniteNumber = (value: unknown, fallback: number) => {
    const num = Number(value);
    return Number.isFinite(num) ? num : fallback;
  };

  const slugifySchemaToken = (value: unknown) =>
    String(value || '')
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '_')
      .replace(/^_+|_+$/g, '');

  const normalizeSchemaTemplateKey = (value: unknown) => {
    const candidate = slugifySchemaToken(value);
    if (!candidate) {
      return '';
    }
    if (SUPPORTED_SCHEMA_TEMPLATE_KEYS.has(candidate)) {
      return candidate;
    }
    return TEMPLATE_KEY_ALIASES[candidate] || '';
  };

  const pickSchemaTemplateKeyFromBlocks = (
    blocks: FrontendSlideBlock[],
    visualAssetCount: number,
  ) => {
    const imageCount = blocks.filter((block) => block.type === 'image').length;
    const listCount = blocks.filter((block) => block.type === 'list').length;
    const statCount = blocks.filter((block) => block.type === 'stat').length;
    const quoteCount = blocks.filter((block) => block.type === 'quote').length;

    if (quoteCount > 0) return 'quote_focus';
    if (imageCount >= 2) return 'visual_compare';
    if (statCount >= 2) return 'metrics_dashboard';
    if (listCount >= 2) return 'dual_list';
    if (imageCount === 1 && listCount > 0) return 'split_media';
    if (imageCount === 1 || visualAssetCount > 0) return 'hero_visual';
    if (blocks.length <= 3) return 'section_divider';
    if (blocks.length >= 6) return 'insight_grid';
    return 'text_focus';
  };

  const normalizeSchemaLayoutMode = (value: unknown): FrontendSlide['layoutMode'] => {
    const candidate = String(value || '').trim().toLowerCase();
    if (candidate === 'fixed' || candidate === 'hybrid' || candidate === 'fluid') {
      return candidate;
    }
    return 'fluid';
  };

  const normalizeSchemaBlockType = (value: unknown): FrontendSlideBlock['type'] => {
    const candidate = String(value || '').trim().toLowerCase();
    const aliases: Record<string, FrontendSlideBlock['type']> = {
      paragraph: 'text',
      textarea: 'text',
      body: 'text',
      bullets: 'list',
      bullet_list: 'list',
      points: 'list',
      figure: 'image',
      visual: 'image',
      chart: 'image',
      metric: 'stat',
      number: 'stat',
      note: 'callout',
    };
    const normalized = aliases[candidate] || candidate || 'text';
    if (
      normalized === 'text'
      || normalized === 'list'
      || normalized === 'image'
      || normalized === 'quote'
      || normalized === 'stat'
      || normalized === 'callout'
      || normalized === 'table'
    ) {
      return normalized;
    }
    return 'text';
  };

  const normalizeSchemaBlockLayout = (value: unknown, fallbackOrder: number): FrontendSlideBlock['layout'] => {
    const layout = (value && typeof value === 'object') ? value as Record<string, unknown> : {};
    const zoneCandidate = String(
      layout.zone
      || layout.slot
      || layout.region
      || layout.area
      || 'main',
    ).trim().toLowerCase();
    const zone = (
      zoneCandidate === 'header'
      || zoneCandidate === 'main'
      || zoneCandidate === 'aside'
      || zoneCandidate === 'footer'
      || zoneCandidate === 'full'
      || zoneCandidate === 'left'
      || zoneCandidate === 'right'
    ) ? zoneCandidate : 'main';

    const span = Math.max(1, Math.min(12, Math.round(toFiniteNumber(layout.span || layout.columns, zone === 'header' || zone === 'footer' || zone === 'full' ? 12 : 6))));
    const order = Math.max(1, Math.round(toFiniteNumber(layout.order, fallbackOrder)));
    const widthCandidate = String(
      layout.preferred_width
      || layout.preferredWidth
      || layout.width
      || '',
    ).trim().toLowerCase();
    const preferredWidth = (
      widthCandidate === 'full'
      || widthCandidate === 'wide'
      || widthCandidate === 'half'
      || widthCandidate === 'third'
      || widthCandidate === 'narrow'
      || widthCandidate === 'auto'
    ) ? widthCandidate : span >= 12 ? 'full' : span >= 8 ? 'wide' : span >= 6 ? 'half' : span >= 4 ? 'third' : 'auto';

    const sideCandidate = String(
      layout.preferred_side
      || layout.preferredSide
      || layout.side
      || '',
    ).trim().toLowerCase();
    const preferredSide = (
      sideCandidate === 'left'
      || sideCandidate === 'right'
      || sideCandidate === 'center'
      || sideCandidate === 'auto'
    ) ? sideCandidate : zone === 'left' ? 'left' : zone === 'right' || zone === 'aside' ? 'right' : 'auto';

    const emphasisCandidate = String(layout.emphasis || '').trim().toLowerCase();
    const emphasis = (
      emphasisCandidate === 'high'
      || emphasisCandidate === 'medium'
      || emphasisCandidate === 'low'
    ) ? emphasisCandidate : 'medium';

    return {
      zone,
      span,
      order,
      preferredWidth,
      preferredSide,
      emphasis,
    };
  };

  const normalizeSchemaBlockChildren = (rawChildren: unknown): FrontendBlockChild[] => {
    if (!Array.isArray(rawChildren)) {
      return [];
    }
    const seenIds = new Set<string>();
    const children: FrontendBlockChild[] = [];
    rawChildren.forEach((rawChild, index) => {
      if (!rawChild || typeof rawChild !== 'object') {
        return;
      }
      const child = rawChild as Record<string, unknown>;
      let id = slugifySchemaToken(
        child.id
        || child.key
        || child.field_key
        || child.fieldKey
        || child.role
        || `child_${index + 1}`,
      ) || `child_${index + 1}`;
      if (seenIds.has(id)) {
        id = `${id}_${index + 1}`;
      }
      seenIds.add(id);

      const type = normalizeSchemaBlockType(
        child.type
        || child.block_type
        || child.blockType
        || child.kind,
      );
      const role = slugifySchemaToken(
        child.role
        || child.semantic_role
        || child.semanticRole
        || id,
      ) || id;
      const items = normalizeStringList(child.items || child.bullets || child.points);
      const content = String(
        child.content
        || child.text
        || child.value
        || child.body
        || '',
      ).trim();
      const assetKey = type === 'image'
        ? slugifySchemaToken(
            child.asset_key
            || child.assetKey
            || child.image_key
            || child.imageKey
            || child.visual_key
            || child.visualKey
            || id,
          ) || id
        : undefined;
      const tableData = type === 'table'
        ? normalizeSchemaTableData(
            child.table_data
            || child.tableData
            || child.table
          )
        : undefined;

      if (type === 'table' && !tableData) {
        return;
      }
      if (type !== 'image' && type !== 'list' && type !== 'table' && !content) {
        return;
      }
      if (type === 'list' && items.length === 0 && !content) {
        return;
      }

      children.push({
        id,
        type,
        role,
        content,
        items: type === 'list' && items.length === 0 && content
          ? content
              .split(/\n+/)
              .map((item) => item.replace(/^[\s\-•]+/, '').trim())
              .filter(Boolean)
          : items,
        assetKey,
        tableData,
      });
    });
    return children;
  };

  const normalizeSchemaBlocks = (
    rawBlocks: unknown,
    visualAssets: Array<{ key: string }>,
  ): FrontendSlideBlock[] => {
    if (!Array.isArray(rawBlocks)) {
      return [];
    }

    const availableAssetKeys = visualAssets
      .map((asset) => slugifySchemaToken(asset.key) || String(asset.key || '').trim())
      .filter(Boolean);
    const usedBlockIds = new Set<string>();
    const usedAssetKeys: string[] = [];

    const pickAssetKey = (preferred: string, fallbackId: string) => {
      const normalizedPreferred = slugifySchemaToken(preferred);
      if (normalizedPreferred) {
        return normalizedPreferred;
      }
      const nextUnused = availableAssetKeys.find((key) => !usedAssetKeys.includes(key));
      return nextUnused || fallbackId;
    };

    const normalizedBlocks: FrontendSlideBlock[] = [];

    rawBlocks.forEach((rawBlock, index) => {
      if (!rawBlock || typeof rawBlock !== 'object') {
        return;
      }
      const block = rawBlock as Record<string, unknown>;
      const fallbackId = `block_${index + 1}`;
      let id = slugifySchemaToken(
        block.id
        || block.key
        || block.field_key
        || block.fieldKey
        || block.role
        || fallbackId,
      ) || fallbackId;
      if (usedBlockIds.has(id)) {
        id = `${id}_${index + 1}`;
      }
      usedBlockIds.add(id);

      const type = normalizeSchemaBlockType(
        block.type
        || block.block_type
        || block.blockType
        || block.kind,
      );
      const role = slugifySchemaToken(
        block.role
        || block.semantic_role
        || block.semanticRole
        || id,
      ) || id;

      const items = normalizeStringList(
        block.items
        || block.bullets
        || block.points,
      );
      const content = String(
        block.content
        || block.text
        || block.value
        || block.body
        || '',
      ).trim();

      const normalizedItems = type === 'list' && items.length === 0 && content
        ? content
            .split(/\n+/)
            .map((item) => item.replace(/^[\s\-•]+/, '').trim())
            .filter(Boolean)
        : items;

      if (type === 'list' && normalizedItems.length === 0) {
        return;
      }

      const tableData = type === 'table'
        ? normalizeSchemaTableData(
            block.table_data
            || block.tableData
            || block.table
          )
        : undefined;

      if (type === 'table' && !tableData) {
        return;
      }
      if (type !== 'image' && type !== 'table' && !content && normalizedItems.length === 0) {
        return;
      }

      const assetKey = type === 'image'
        ? pickAssetKey(
            String(
              block.asset_key
              || block.assetKey
              || block.image_key
              || block.imageKey
              || block.visual_key
              || block.visualKey
              || '',
            ),
            id,
          )
        : undefined;

      if (assetKey && !usedAssetKeys.includes(assetKey)) {
        usedAssetKeys.push(assetKey);
      }

      normalizedBlocks.push({
        id,
        type,
        role,
        content,
        items: normalizedItems,
        assetKey,
        tableData,
        children: normalizeSchemaBlockChildren(
          block.children
          || block.content_items
          || block.contentItems
          || block.elements,
        ),
        layout: normalizeSchemaBlockLayout(
          block.layout
          || block.layout_hint
          || block.layoutHint,
          index + 1,
        ),
      });
    });

    return normalizedBlocks
      .sort((a, b) => a.layout.order - b.layout.order || a.id.localeCompare(b.id))
      .map((block, index) => ({
        ...block,
        layout: {
          ...block.layout,
          order: index + 1,
        },
      }));
  };

  const getEditableFieldKeyForBlock = (slide: FrontendSlide, block: FrontendSlideBlock) => {
    const fieldKeys = new Set(slide.editableFields.map((field) => field.key));
    const candidates = [
      PREFERRED_SCHEMA_FIELD_KEYS[block.role] || '',
      block.role || '',
      block.id || '',
      slugifySchemaToken(block.role || ''),
      slugifySchemaToken(block.id || ''),
    ].filter(Boolean);
    return candidates.find((candidate) => fieldKeys.has(candidate));
  };

  const buildIdleFrontendReview = (): NonNullable<FrontendSlide['review']> => ({
    status: 'idle',
    summary: '',
    issues: [],
  });

  const buildUniqueBlockId = (slide: FrontendSlide, prefix: string) => {
    const base = slugifySchemaToken(prefix) || 'block';
    const existingIds = new Set([
      ...slide.blocks.map((block) => block.id),
      ...slide.blocks.flatMap((block) => (block.children || []).map((child) => child.id)),
      ...slide.editableFields.map((field) => field.key),
      ...slide.visualAssets.map((asset) => asset.key),
    ]);
    let candidate = `${base}_${Date.now().toString(36)}`;
    let counter = 1;
    while (existingIds.has(candidate)) {
      counter += 1;
      candidate = `${base}_${Date.now().toString(36)}_${counter}`;
    }
    return candidate;
  };

  const buildUniqueChildId = (slide: FrontendSlide, prefix: string) => {
    const base = slugifySchemaToken(prefix) || 'item';
    const existingIds = new Set([
      ...slide.blocks.map((block) => block.id),
      ...slide.blocks.flatMap((block) => (block.children || []).map((child) => child.id)),
      ...slide.editableFields.map((field) => field.key),
      ...slide.visualAssets.map((asset) => asset.key),
    ]);
    let candidate = `${base}_${Date.now().toString(36)}`;
    let counter = 1;
    while (existingIds.has(candidate)) {
      counter += 1;
      candidate = `${base}_${Date.now().toString(36)}_${counter}`;
    }
    return candidate;
  };

  const collectCanvasNodeIds = (node?: FrontendCanvasNode): string[] => {
    if (!node) return [];
    return [
      node.id,
      ...(node.children || []).flatMap((child) => collectCanvasNodeIds(child)),
    ].filter(Boolean);
  };

  const buildUniqueCanvasNodeId = (slide: FrontendSlide, prefix: string) => {
    const base = slugifySchemaToken(prefix) || 'node';
    const existingIds = new Set([
      ...collectCanvasNodeIds(slide.root),
      ...slide.blocks.map((block) => block.id),
      ...slide.editableFields.map((field) => field.key),
      ...slide.visualAssets.map((asset) => asset.key),
    ]);
    let candidate = `${base}_${Date.now().toString(36)}`;
    let counter = 1;
    while (existingIds.has(candidate)) {
      counter += 1;
      candidate = `${base}_${Date.now().toString(36)}_${counter}`;
    }
    return candidate;
  };

  const cloneTableData = (tableData: FrontendTableData): FrontendTableData => ({
    headers: [...tableData.headers],
    rows: tableData.rows.map((row) => [...row]),
  });

  const normalizeContentTableData = (value: unknown): FrontendTableData | null => {
    if (!value || typeof value !== 'object') return null;
    const source = value as Record<string, unknown>;
    const headers = Array.isArray(source.headers)
      ? source.headers.map((item) => String(item ?? ''))
      : Array.isArray(source.columns)
        ? source.columns.map((item) => String(item ?? ''))
        : [];
    const rows = Array.isArray(source.rows)
      ? source.rows
          .filter((row): row is unknown[] => Array.isArray(row))
          .map((row) => row.map((cell) => String(cell ?? '')))
      : [];
    if (headers.length === 0 && rows.length === 0) return null;
    const maxCols = Math.max(headers.length, ...rows.map((row) => row.length), 1);
    return {
      headers: Array.from({ length: maxCols }, (_, index) => headers[index] || `列 ${index + 1}`),
      rows: rows.length > 0
        ? rows.map((row) => Array.from({ length: maxCols }, (_, index) => row[index] || ''))
        : [Array.from({ length: maxCols }, () => '')],
    };
  };

  const mergeTableCellFieldIntoGroups = (
    groups: Record<string, FrontendTableData>,
    fieldKey: string,
    value: string,
  ) => {
    const match = /^(.+)_cell_(h|\d+)_(\d+)$/.exec(fieldKey);
    if (!match) return false;
    const ownerId = match[1];
    const rowIndex = match[2] === 'h' ? 'h' : Number.parseInt(match[2], 10);
    const colIndex = Number.parseInt(match[3], 10);
    if (!Number.isFinite(colIndex) || (rowIndex !== 'h' && !Number.isFinite(rowIndex))) {
      return true;
    }
    const tableData = groups[ownerId] || { headers: [], rows: [] };
    if (rowIndex === 'h') {
      while (tableData.headers.length <= colIndex) {
        tableData.headers.push(`列 ${tableData.headers.length + 1}`);
      }
      tableData.headers[colIndex] = value;
    } else {
      while (tableData.rows.length <= rowIndex) {
        tableData.rows.push([]);
      }
      while (tableData.rows[rowIndex].length <= colIndex) {
        tableData.rows[rowIndex].push('');
      }
      tableData.rows[rowIndex][colIndex] = value;
    }
    groups[ownerId] = tableData;
    return true;
  };

  const buildCanvasContentFromSlide = (slide: FrontendSlide): Record<string, unknown> => {
    const content: Record<string, unknown> = {
      ...(slide.content || {}),
    };
    Object.keys(content).forEach((key) => {
      if (/^(.+)_cell_(h|\d+)_(\d+)$/.test(key)) {
        delete content[key];
      }
    });
    const tableFieldGroups: Record<string, FrontendTableData> = {};
    slide.editableFields.forEach((field) => {
      if (mergeTableCellFieldIntoGroups(tableFieldGroups, field.key, field.value)) {
        return;
      }
      content[field.key] = field.type === 'list' ? [...field.items] : field.value;
    });
    slide.blocks.forEach((block) => {
      if (block.type === 'table' && block.tableData) {
        content[block.role || block.id] = cloneTableData(block.tableData);
      } else if (block.type === 'list' && block.items.length > 0 && !(block.role in content)) {
        content[block.role || block.id] = [...block.items];
      } else if (block.content && !(block.role in content)) {
        content[block.role || block.id] = block.content;
      }
    });
    Object.entries(tableFieldGroups).forEach(([ownerId, tableData]) => {
      content[ownerId] = {
        ...(content[ownerId] && typeof content[ownerId] === 'object' ? content[ownerId] as Record<string, unknown> : {}),
        ...cloneTableData(tableData),
      };
    });
    content.assets = {
      ...((content.assets && typeof content.assets === 'object') ? content.assets as Record<string, unknown> : {}),
      ...Object.fromEntries(
        slide.visualAssets.map((asset) => [
          asset.key,
          {
            type: 'image',
            asset_key: asset.key,
            src: asset.src,
            preview_src: asset.previewSrc || asset.src,
            original_src: asset.originalSrc || asset.storagePath || asset.src,
            alt: asset.alt,
          },
        ]),
      ),
    };
    return content;
  };

  const buildCanvasRootFromSlide = (slide: FrontendSlide): FrontendCanvasNode => {
    if (slide.root) return slide.root;
    const headerNodes: FrontendCanvasNode[] = [];
    const mainNodes: FrontendCanvasNode[] = [];
    const asideNodes: FrontendCanvasNode[] = [];
    const footerNodes: FrontendCanvasNode[] = [];
    slide.blocks.forEach((block) => {
      const component = block.role === 'title'
        ? 'heading'
        : block.type === 'list'
          ? 'bullets'
          : block.type === 'image'
            ? 'figure'
            : block.type === 'table'
              ? 'table'
              : block.type === 'stat'
                ? 'stat'
                : block.type === 'callout'
                  ? 'callout'
                  : 'text';
      const props = component === 'bullets'
        ? { items_ref: block.role || block.id }
        : component === 'figure'
          ? { asset_ref: block.assetKey || block.id, asset_key: block.assetKey || block.id }
          : component === 'table'
            ? { table_ref: block.role || block.id }
            : { text_ref: block.role || block.id };
      const node: FrontendCanvasNode = {
        type: 'component',
        id: block.id,
        component,
        props,
      };
      const zone = block.layout?.zone || 'main';
      if (zone === 'header') headerNodes.push(node);
      else if (zone === 'footer') footerNodes.push(node);
      else if (zone === 'aside' || zone === 'right') asideNodes.push(node);
      else mainNodes.push(node);
    });
    const mainChildren: FrontendCanvasNode[] = [
      {
        type: 'container',
        id: 'main_left',
        style: { direction: 'column', gap: 18, weight: 1, align: 'stretch' },
        children: mainNodes,
      },
    ];
    if (asideNodes.length > 0) {
      mainChildren.push({
        type: 'container',
        id: 'main_right',
        style: { direction: 'column', gap: 18, weight: 1, align: 'stretch' },
        children: asideNodes,
      });
    }
    return {
      type: 'container',
      id: 'root',
      style: { direction: 'column', gap: 24, align: 'stretch' },
      children: [
        ...(headerNodes.length > 0 ? [{
          type: 'container' as const,
          id: 'header',
          style: { direction: 'column' as const, gap: 12, align: 'stretch' as const },
          children: headerNodes,
        }] : []),
        {
          type: 'container',
          id: 'main',
          style: { direction: mainChildren.length > 1 ? 'row' : 'column', gap: 24, weight: 1, align: 'stretch' },
          children: mainChildren,
        },
        ...(footerNodes.length > 0 ? [{
          type: 'container' as const,
          id: 'footer',
          style: { direction: 'row' as const, gap: 16, align: 'end' as const, justify: 'between' as const },
          children: footerNodes,
        }] : []),
      ],
    };
  };

  const appendCanvasNodeToContainer = (
    root: FrontendCanvasNode,
    node: FrontendCanvasNode,
    targetId?: string,
  ): FrontendCanvasNode => {
    let inserted = false;
    const appendToFirstMain = (current: FrontendCanvasNode): FrontendCanvasNode => {
      if ((current.id === 'main_left' || current.id === 'main' || current.id === 'root') && current.type === 'container') {
        inserted = true;
        return { ...current, children: [...(current.children || []), node] };
      }
      return {
        ...current,
        children: (current.children || []).map((child) => inserted ? child : appendToFirstMain(child)),
      };
    };
    const visit = (current: FrontendCanvasNode): FrontendCanvasNode => {
      const children = current.children || [];
      if (targetId && current.id === targetId && current.type === 'container') {
        inserted = true;
        return { ...current, children: [...children, node] };
      }
      const targetIndex = targetId ? children.findIndex((child) => child.id === targetId) : -1;
      if (targetIndex >= 0) {
        const target = children[targetIndex];
        inserted = true;
        if (target.type === 'container') {
          const nextTarget = { ...target, children: [...(target.children || []), node] };
          return {
            ...current,
            children: children.map((child, index) => index === targetIndex ? nextTarget : child),
          };
        }
        return {
          ...current,
          children: [
            ...children.slice(0, targetIndex + 1),
            node,
            ...children.slice(targetIndex + 1),
          ],
        };
      }
      return {
        ...current,
        children: children.map((child) => inserted ? child : visit(child)),
      };
    };
    const nextRoot = targetId ? visit(root) : root;
    return inserted ? nextRoot : appendToFirstMain(nextRoot);
  };

  const insertCanvasNode = (
    slide: FrontendSlide,
    node: FrontendCanvasNode,
    options: {
      targetId?: string;
      contentPatch?: Record<string, unknown>;
      editableFields?: FrontendEditableField | FrontendEditableField[];
      visualAssets?: FrontendSlide['visualAssets'];
    } = {},
  ): FrontendSlide => {
    const editableFields = Array.isArray(options.editableFields)
      ? options.editableFields
      : options.editableFields
        ? [options.editableFields]
        : [];
    const root = buildCanvasRootFromSlide(slide);
    const targetId = parseFrontendInsertZoneTarget(options.targetId) ? undefined : options.targetId;
    return {
      ...slide,
      renderEngine: 'canvas',
      schemaVersion: slide.schemaVersion || 'ppt_canvas_schema_v1',
      root: appendCanvasNodeToContainer(root, node, targetId),
      content: {
        ...buildCanvasContentFromSlide(slide),
        ...(options.contentPatch || {}),
      },
      blocks: [],
      editableFields: editableFields.length > 0
        ? [...slide.editableFields, ...editableFields]
        : slide.editableFields,
      visualAssets: options.visualAssets || slide.visualAssets,
      layoutIr: undefined,
      generationNote: '当前页 Canvas 内容已手动编辑。',
      review: buildIdleFrontendReview(),
    };
  };

  const getDefaultInsertionBlockId = (slide: FrontendSlide) =>
    slide.blocks.find((block) => ['main', 'aside', 'left', 'right', 'full'].includes(block.layout.zone))?.id
    || slide.blocks[0]?.id
    || '';

  const blockToChild = (block: FrontendSlideBlock): FrontendBlockChild | null => {
    if (block.type === 'image') {
      return {
        id: `${block.id}_content`,
        type: 'image',
        role: block.role,
        content: '',
        items: [],
        assetKey: block.assetKey || block.id,
      };
    }
    if (block.type === 'list') {
      if (block.items.length === 0) {
        return null;
      }
      return {
        id: `${block.id}_content`,
        type: 'list',
        role: block.role,
        content: '',
        items: [...block.items],
      };
    }
    if (block.type === 'table') {
      if (!block.tableData) {
        return null;
      }
      return {
        id: `${block.id}_content`,
        type: 'table',
        role: block.role,
        content: '',
        items: [],
        tableData: {
          headers: [...block.tableData.headers],
          rows: block.tableData.rows.map((row) => [...row]),
        },
      };
    }
    if (!block.content) {
      return null;
    }
    return {
      id: `${block.id}_content`,
      type: block.type,
      role: block.role,
      content: block.content,
      items: [],
    };
  };

  const ensureBlockChildren = (block: FrontendSlideBlock): FrontendSlideBlock => {
    if (block.children && block.children.length > 0) {
      return block;
    }
    const legacyChild = blockToChild(block);
    return {
      ...block,
      children: legacyChild ? [legacyChild] : [],
    };
  };

  const insertChildIntoBlock = (
    slide: FrontendSlide,
    targetBlockId: string | undefined,
    child: FrontendBlockChild,
    editableField?: FrontendEditableField | FrontendEditableField[],
  ): FrontendSlide => {
    const fallbackBlockId = getDefaultInsertionBlockId(slide);
    const resolvedBlockId = targetBlockId && slide.blocks.some((block) => block.id === targetBlockId)
      ? targetBlockId
      : fallbackBlockId;
    if (!resolvedBlockId) {
      return slide;
    }

    const editableFields = Array.isArray(editableField)
      ? editableField
      : editableField
        ? [editableField]
        : [];

    return {
      ...slide,
      schemaVersion: slide.schemaVersion || 'frontend_slide_schema_v2',
      layoutMode: slide.layoutMode || 'fluid',
      blocks: slide.blocks.map((block) => {
        if (block.id !== resolvedBlockId) {
          return block;
        }
        const blockWithChildren = ensureBlockChildren(block);
        return {
          ...blockWithChildren,
          children: [...(blockWithChildren.children || []), child],
        };
      }),
      editableFields: editableFields.length > 0
        ? [...slide.editableFields, ...editableFields]
        : slide.editableFields,
      generationNote: '当前页内容已手动编辑。',
      review: buildIdleFrontendReview(),
    };
  };

  const insertTopLevelBlockIntoZone = (
    slide: FrontendSlide,
    block: FrontendSlideBlock,
    editableField?: FrontendEditableField | FrontendEditableField[],
    visualAssets?: FrontendSlide['visualAssets'],
  ): FrontendSlide => {
    const editableFields = Array.isArray(editableField)
      ? editableField
      : editableField
        ? [editableField]
        : [];
    return {
      ...slide,
      schemaVersion: slide.schemaVersion || 'frontend_slide_schema_v2',
      layoutMode: slide.layoutMode || 'fluid',
      blocks: [...slide.blocks, block],
      editableFields: editableFields.length > 0
        ? [...slide.editableFields, ...editableFields]
        : slide.editableFields,
      visualAssets: visualAssets || slide.visualAssets,
      generationNote: '当前页内容已手动编辑。',
      review: buildIdleFrontendReview(),
    };
  };

  const buildInsertedBlockLayout = (
    slide: FrontendSlide,
    overrides: Partial<FrontendSlideBlock['layout']> = {},
  ): FrontendSlideBlock['layout'] => ({
    zone: 'main',
    span: 6,
    order: slide.blocks.length + 1,
    preferredWidth: 'half',
    preferredSide: 'auto',
    emphasis: 'medium',
    ...overrides,
  });

  const resolveTemplateAfterManualInsert = (
    slide: FrontendSlide,
    blocks: FrontendSlideBlock[],
    visualAssetCount: number,
    insertedType: FrontendSlideBlock['type'],
  ) => {
    const imageCount = blocks.filter((block) => block.type === 'image').length;
    const nonImageCount = blocks.filter((block) => block.type !== 'image').length;
    if (insertedType === 'image') {
      const listCount = blocks.filter((block) => block.type === 'list').length;
      if (imageCount >= 2) return 'visual_compare';
      return listCount > 0 ? 'split_media' : 'hero_visual';
    }
    if (imageCount > 0) {
      return imageCount >= 2 ? 'visual_compare' : 'split_media';
    }
    if (slide.templateKey === 'title_cover' || slide.templateKey === 'section_divider') {
      return 'stacked_cards';
    }
    if (nonImageCount >= 5) {
      return 'insight_grid';
    }
    return slide.templateKey || pickSchemaTemplateKeyFromBlocks(blocks, visualAssetCount);
  };

  const parseTableCellFieldKey = (fieldKey: string) => {
    const match = /^(.+)_cell_(h|\d+)_(\d+)$/.exec(fieldKey);
    if (!match) {
      return null;
    }
    return {
      ownerId: match[1],
      row: match[2] === 'h' ? 'h' as const : Number.parseInt(match[2], 10),
      col: Number.parseInt(match[3], 10),
    };
  };

  const applyTableCellValue = <T extends FrontendSlideBlock | FrontendBlockChild>(
    item: T,
    fieldKey: string,
    value: string,
  ): T => {
    const parsed = parseTableCellFieldKey(fieldKey);
    if (!parsed || parsed.ownerId !== item.id || item.type !== 'table' || !item.tableData) {
      return item;
    }
    const nextTableData: FrontendTableData = {
      headers: [...item.tableData.headers],
      rows: item.tableData.rows.map((row) => [...row]),
    };
    if (parsed.row === 'h') {
      if (parsed.col >= 0 && parsed.col < nextTableData.headers.length) {
        nextTableData.headers[parsed.col] = value;
      }
    } else if (
      parsed.row >= 0
      && parsed.row < nextTableData.rows.length
      && parsed.col >= 0
      && parsed.col < nextTableData.rows[parsed.row].length
    ) {
      nextTableData.rows[parsed.row][parsed.col] = value;
    }
    return {
      ...item,
      tableData: nextTableData,
    };
  };

  const applyTableCellValueToCanvasContent = (
    content: Record<string, unknown>,
    fieldKey: string,
    value: string,
  ) => {
    const parsed = parseTableCellFieldKey(fieldKey);
    if (!parsed) {
      return content;
    }
    const tableData = normalizeContentTableData(content[parsed.ownerId]) || { headers: [], rows: [] };
    const nextTableData = cloneTableData(tableData);
    const nextContent = { ...content };
    delete nextContent[fieldKey];
    if (parsed.row === 'h') {
      while (nextTableData.headers.length <= parsed.col) {
        nextTableData.headers.push(`列 ${nextTableData.headers.length + 1}`);
      }
      nextTableData.headers[parsed.col] = value;
    } else {
      while (nextTableData.rows.length <= parsed.row) {
        nextTableData.rows.push([]);
      }
      while (nextTableData.rows[parsed.row].length <= parsed.col) {
        nextTableData.rows[parsed.row].push('');
      }
      nextTableData.rows[parsed.row][parsed.col] = value;
      while (nextTableData.headers.length <= parsed.col) {
        nextTableData.headers.push(`列 ${nextTableData.headers.length + 1}`);
      }
    }
    return {
      ...nextContent,
      [parsed.ownerId]: {
        ...(content[parsed.ownerId] && typeof content[parsed.ownerId] === 'object'
          ? content[parsed.ownerId] as Record<string, unknown>
          : {}),
        ...nextTableData,
      },
    };
  };

  const syncCanvasContentWithEditableField = (
    slide: FrontendSlide,
    field: FrontendEditableField,
  ) => {
    const content = { ...(slide.content || {}) };
    const tableSyncedContent = applyTableCellValueToCanvasContent(content, field.key, field.value);
    if (tableSyncedContent !== content) {
      return tableSyncedContent;
    }
    return {
      ...content,
      [field.key]: field.type === 'list' ? [...field.items] : field.value,
    };
  };

  const syncBlockWithEditableField = (
    block: FrontendSlideBlock,
    field: FrontendEditableField,
  ): FrontendSlideBlock => {
    const tableSyncedBlock = applyTableCellValue(block, field.key, field.value);
    if (tableSyncedBlock !== block) {
      return tableSyncedBlock;
    }

    if (field.type === 'list') {
      if (block.type === 'list') {
        return {
          ...block,
          content: '',
          items: [...field.items],
        };
      }
      return {
        ...block,
        content: field.items.join(' • '),
        items: [...field.items],
      };
    }

    if (block.type === 'list') {
      return {
        ...block,
        content: field.value,
        items: field.value ? [field.value] : [],
      };
    }

    return {
      ...block,
      content: field.value,
    };
  };

  const applyFrontendEditableFieldMutation = (
    slide: FrontendSlide,
    fieldKey: string,
    updater: (field: FrontendEditableField) => FrontendEditableField,
  ): FrontendSlide => {
    const currentField = slide.editableFields.find((field) => field.key === fieldKey);
    if (!currentField) {
      return slide;
    }
    const nextMatchedField = updater(currentField);
    const nextEditableFields = slide.editableFields.map((field) => {
      if (field.key !== fieldKey) {
        return field;
      }
      return nextMatchedField;
    });
    const isCanvasSlide = slide.renderEngine === 'canvas';

    return {
      ...slide,
      title: fieldKey === 'title' ? nextMatchedField.value || slide.title : slide.title,
      content: isCanvasSlide
        ? syncCanvasContentWithEditableField(slide, nextMatchedField)
        : slide.content,
      blocks: slide.blocks.map((block) =>
        getEditableFieldKeyForBlock(slide, block) === fieldKey
          ? syncBlockWithEditableField(block, nextMatchedField)
          : {
              ...block,
              children: (block.children || []).map((child) => {
                const tableSyncedChild = applyTableCellValue(child, fieldKey, nextMatchedField.value);
                if (tableSyncedChild !== child) {
                  return tableSyncedChild;
                }
                if (child.id !== fieldKey && child.role !== fieldKey) {
                  return child;
                }
                if (nextMatchedField.type === 'list') {
                  return {
                    ...child,
                    content: '',
                    items: [...nextMatchedField.items],
                  };
                }
                return {
                  ...child,
                  content: nextMatchedField.value,
                };
              }),
            },
      ),
      editableFields: nextEditableFields,
      layoutIr: isCanvasSlide ? undefined : slide.layoutIr,
      generationNote: '当前页内容已手动编辑。',
      review: buildIdleFrontendReview(),
    };
  };

  const normalizeFrontendSlides = (slides: any[]): FrontendSlide[] =>
    slides.map((slide: any, index: number) => {
      const editableFields = Array.isArray(slide.editable_fields || slide.editableFields)
        ? (slide.editable_fields || slide.editableFields).map((field: any) => ({
            key: String(field.key || ''),
            label: String(field.label || field.key || ''),
            type: field.type === 'list' || field.type === 'textarea' ? field.type : 'text',
            value: String(field.value || ''),
            items: normalizeStringList(field.items),
          }))
        : [];
      const visualAssets = Array.isArray(slide.visual_assets || slide.visualAssets)
        ? (slide.visual_assets || slide.visualAssets).map((asset: any, assetIndex: number) => ({
            key: String(asset.key || `main_visual_${assetIndex + 1}`),
            label: String(asset.label || asset.key || `Image ${assetIndex + 1}`),
            src: String(asset.src || ''),
            previewSrc: String(asset.preview_src || asset.previewSrc || asset.src || ''),
            originalSrc: String(asset.original_src || asset.originalSrc || asset.storage_path || asset.storagePath || asset.src || ''),
            alt: String(asset.alt || asset.label || asset.key || ''),
            sourceType: asset.source_type === 'paper_asset' || asset.sourceType === 'paper_asset'
              ? 'paper_asset'
              : asset.source_type === 'upload' || asset.sourceType === 'upload'
                ? 'upload'
                : 'generated',
            storagePath: asset.storage_path || asset.storagePath || undefined,
            previewStoragePath: asset.preview_storage_path || asset.previewStoragePath || undefined,
            prompt: asset.prompt || undefined,
            style: asset.style || undefined,
          }))
        : [];
      const blocks = normalizeSchemaBlocks(
        slide.blocks
        || slide.elements
        || slide.content_blocks
        || slide.contentBlocks,
        visualAssets,
      );
      if (!blocks.some((block) => block.type === 'list')) {
        const keyPointField = editableFields.find((field: FrontendEditableField) => field.key === 'key_points' && field.items.length > 0);
        const contentKeyPoints = Array.isArray(slide.content?.key_points)
          ? slide.content.key_points.map((item: unknown) => String(item || '').trim()).filter(Boolean)
          : [];
        const keyPointItems = keyPointField?.items?.length ? keyPointField.items : contentKeyPoints;
        if (keyPointItems.length > 0) {
          blocks.push({
            id: 'key_points',
            type: 'list',
            role: 'key_points',
            content: '',
            items: keyPointItems,
            layout: {
              zone: visualAssets.length > 0 ? 'main' : 'full',
              span: visualAssets.length > 0 ? 6 : 12,
              order: blocks.length + 1,
              preferredWidth: visualAssets.length > 0 ? 'wide' : 'full',
              preferredSide: 'auto',
              emphasis: 'medium',
            },
          });
        }
      }
      const templateKey = normalizeSchemaTemplateKey(
        slide.template_key
        || slide.templateKey
        || slide.layout_template
        || slide.layoutTemplate,
      ) || (blocks.length > 0 ? pickSchemaTemplateKeyFromBlocks(blocks, visualAssets.length) : '');
      const rawRenderEngine = String(slide.render_engine || slide.renderEngine || '').trim().toLowerCase();
      const renderEngine = rawRenderEngine === 'blocks' ? 'blocks' : 'canvas';
      const visualSpec = normalizeCanvasVisualSpec(slide.visual_spec || slide.visualSpec);

      return {
        slideId: String(slide.slide_id || slide.slideId || index + 1),
        pageNum: Number(slide.page_num || slide.pageNum || index + 1),
        title: String(slide.title || `第 ${index + 1} 页`),
        schemaVersion: String(slide.schema_version || slide.schemaVersion || '').trim() || undefined,
        renderEngine,
        templateKey: templateKey || undefined,
        layoutMode: blocks.length > 0
          ? normalizeSchemaLayoutMode(slide.layout_mode || slide.layoutMode)
          : undefined,
        blocks: renderEngine === 'canvas' ? [] : blocks,
        layoutFamily: String(slide.layout_family || slide.layoutFamily || '').trim() || undefined,
        root: (slide.root && typeof slide.root === 'object') ? slide.root : undefined,
        content: (slide.content && typeof slide.content === 'object') ? slide.content : undefined,
        visualSpec,
        constraints: (slide.constraints && typeof slide.constraints === 'object') ? slide.constraints : undefined,
        editableMap: (slide.editable_map || slide.editableMap) && typeof (slide.editable_map || slide.editableMap) === 'object'
          ? (slide.editable_map || slide.editableMap)
          : undefined,
        canvasValidation: normalizeCanvasValidation(slide.canvas_validation || slide.canvasValidation),
        layoutIr: (slide.layout_ir || slide.layoutIr) && typeof (slide.layout_ir || slide.layoutIr) === 'object'
          ? (slide.layout_ir || slide.layoutIr)
          : undefined,
        htmlTemplate: slide.html_template || slide.htmlTemplate || '',
        cssCode: slide.css_code || slide.cssCode || '',
        editableFields,
        visualAssets,
        generationNote: slide.generation_note || slide.generationNote || '',
        status: slide.status === 'processing' || slide.status === 'pending' ? slide.status : 'done',
        review: buildIdleFrontendReview(),
      };
    });

  const normalizeFrontendDeckTheme = (theme: any): FrontendDeckTheme | null => {
    if (!theme || typeof theme !== 'object') {
      return null;
    }
    const themeLockSource = theme.theme_lock || theme.themeLock;
    const themeLock = typeof themeLockSource === 'object' && themeLockSource ? themeLockSource : {};
    const palette = theme.palette && typeof theme.palette === 'object'
      ? {
          bg: String(theme.palette.bg || ''),
          panel: String(theme.palette.panel || ''),
          primary: String(theme.palette.primary || ''),
          secondary: String(theme.palette.secondary || ''),
          accent: String(theme.palette.accent || ''),
          text: String(theme.palette.text || ''),
          muted: String(theme.palette.muted || ''),
        }
      : undefined;
    const typography = theme.typography && typeof theme.typography === 'object'
      ? {
          titleFontStack: String(theme.typography.title_font_stack || theme.typography.titleFontStack || ''),
          bodyFontStack: String(theme.typography.body_font_stack || theme.typography.bodyFontStack || ''),
          eyebrowSize: toFiniteNumber(theme.typography.eyebrow_size || theme.typography.eyebrowSize, 18),
          titleSize: toFiniteNumber(theme.typography.title_size || theme.typography.titleSize, 56),
          summarySize: toFiniteNumber(theme.typography.summary_size || theme.typography.summarySize, 26),
          bodySize: toFiniteNumber(theme.typography.body_size || theme.typography.bodySize, 24),
        }
      : undefined;
    return {
      themeName: String(theme.theme_name || theme.themeName || 'locked_deck_theme'),
      stylePrompt: String(theme.style_prompt || theme.stylePrompt || ''),
      visualMood: String(theme.visual_mood || theme.visualMood || ''),
      footerText: String(theme.footer_text || theme.footerText || ''),
      sectionLabelTemplate: String(theme.section_label_template || theme.sectionLabelTemplate || ''),
      palette,
      typography,
      layoutRules: normalizeStringList(theme.layout_rules || theme.layoutRules),
      componentRules: normalizeStringList(theme.component_rules || theme.componentRules),
      themeLock: {
        mustKeep: normalizeStringList(themeLock.must_keep || themeLock.mustKeep),
        preferredLayoutPatterns: normalizeStringList(
          themeLock.preferred_layout_patterns || themeLock.preferredLayoutPatterns,
        ),
        componentSignature: String(themeLock.component_signature || themeLock.componentSignature || ''),
        avoid: normalizeStringList(themeLock.avoid),
      },
    };
  };

  const normalizeCanvasValidation = (value: any) => {
    if (!value || typeof value !== 'object') {
      return undefined;
    }
    return {
      ok: Boolean(value.ok),
      usedRefs: normalizeStringList(value.used_refs || value.usedRefs),
      definedContentKeys: normalizeStringList(value.defined_content_keys || value.definedContentKeys),
      missingRefs: normalizeStringList(value.missing_refs || value.missingRefs),
      orphanContentKeys: normalizeStringList(value.orphan_content_keys || value.orphanContentKeys),
      emptyComponents: normalizeStringList(value.empty_components || value.emptyComponents),
      issues: Array.isArray(value.issues)
        ? value.issues
            .filter((issue: any) => issue && typeof issue === 'object')
            .map((issue: any) => ({
              severity: issue.severity === 'error' || issue.severity === 'warning' || issue.severity === 'info'
                ? issue.severity
                : 'repairable',
              code: String(issue.code || ''),
              nodeId: issue.node_id || issue.nodeId ? String(issue.node_id || issue.nodeId) : undefined,
              ref: issue.ref ? String(issue.ref) : undefined,
              suggestedRef: issue.suggested_ref || issue.suggestedRef ? String(issue.suggested_ref || issue.suggestedRef) : undefined,
              message: String(issue.message || issue.code || ''),
            }))
        : [],
    };
  };

  const normalizeCanvasVisualStyle = (value: any): FrontendCanvasVisualStyle => {
    if (!value || typeof value !== 'object') {
      return {};
    }
    const source = value as Record<string, unknown>;
    const textAlign = String(source.textAlign || source.text_align || '').trim().toLowerCase();
    const fontStyle = String(source.fontStyle || source.font_style || '').trim().toLowerCase();
    const imageFit = String(source.imageFit || source.image_fit || '').trim().toLowerCase();
    const style: FrontendCanvasVisualStyle = {};
    const fill = String(source.fill || source.background || source.backgroundColor || '').trim();
    const color = String(source.color || source.textColor || source.text_color || '').trim();
    const borderColor = String(source.borderColor || source.border_color || '').trim();
    const fontFamily = String(source.fontFamily || source.font_family || '').trim();
    if (fill) style.fill = fill;
    if (color) style.color = color;
    if (borderColor) style.borderColor = borderColor;
    if (source.borderWidth !== undefined || source.border_width !== undefined) {
      style.borderWidth = toFiniteNumber(source.borderWidth ?? source.border_width, 0);
    }
    if (source.radius !== undefined || source.borderRadius !== undefined || source.border_radius !== undefined) {
      style.radius = toFiniteNumber(source.radius ?? source.borderRadius ?? source.border_radius, 0);
    }
    if (source.padding !== undefined) {
      style.padding = toFiniteNumber(source.padding, 0);
    }
    if (fontFamily) style.fontFamily = fontFamily;
    if (source.fontSize !== undefined || source.font_size !== undefined) {
      style.fontSize = toFiniteNumber(source.fontSize ?? source.font_size, 0);
    }
    if (source.fontWeight !== undefined || source.font_weight !== undefined) {
      const fontWeight = source.fontWeight ?? source.font_weight;
      style.fontWeight = typeof fontWeight === 'number' || typeof fontWeight === 'string'
        ? fontWeight
        : String(fontWeight);
    }
    if (fontStyle === 'italic' || fontStyle === 'normal') {
      style.fontStyle = fontStyle;
    }
    if (source.lineHeight !== undefined || source.line_height !== undefined) {
      style.lineHeight = toFiniteNumber(source.lineHeight ?? source.line_height, 0);
    }
    if (textAlign === 'left' || textAlign === 'center' || textAlign === 'right' || textAlign === 'justify') {
      style.textAlign = textAlign;
    }
    if (source.opacity !== undefined) {
      style.opacity = toFiniteNumber(source.opacity, 1);
    }
    if (imageFit === 'contain' || imageFit === 'cover' || imageFit === 'fill') {
      style.imageFit = imageFit;
    }
    if (source.emphasis === 'high' || source.emphasis === 'medium' || source.emphasis === 'low') {
      style.emphasis = source.emphasis;
    }
    return style;
  };

  const normalizeCanvasVisualSpec = (value: any): FrontendCanvasVisualSpec | undefined => {
    if (!value || typeof value !== 'object') {
      return undefined;
    }
    const source = value as Record<string, unknown>;
    const spec: FrontendCanvasVisualSpec = {};

    const paletteSource = source.palette && typeof source.palette === 'object'
      ? source.palette as Record<string, unknown>
      : undefined;
    if (paletteSource) {
      spec.palette = {
        bg: String(paletteSource.bg || '').trim() || undefined,
        panel: String(paletteSource.panel || '').trim() || undefined,
        primary: String(paletteSource.primary || '').trim() || undefined,
        secondary: String(paletteSource.secondary || '').trim() || undefined,
        accent: String(paletteSource.accent || '').trim() || undefined,
        text: String(paletteSource.text || '').trim() || undefined,
        muted: String(paletteSource.muted || '').trim() || undefined,
      };
    }

    const typographySource = source.typography && typeof source.typography === 'object'
      ? source.typography as Record<string, unknown>
      : undefined;
    if (typographySource) {
      spec.typography = {
        titleFontStack: String(typographySource.titleFontStack || typographySource.title_font_stack || '').trim() || undefined,
        bodyFontStack: String(typographySource.bodyFontStack || typographySource.body_font_stack || '').trim() || undefined,
        eyebrowSize: typographySource.eyebrowSize !== undefined || typographySource.eyebrow_size !== undefined
          ? toFiniteNumber(typographySource.eyebrowSize ?? typographySource.eyebrow_size, 18)
          : undefined,
        titleSize: typographySource.titleSize !== undefined || typographySource.title_size !== undefined
          ? toFiniteNumber(typographySource.titleSize ?? typographySource.title_size, 56)
          : undefined,
        summarySize: typographySource.summarySize !== undefined || typographySource.summary_size !== undefined
          ? toFiniteNumber(typographySource.summarySize ?? typographySource.summary_size, 26)
          : undefined,
        bodySize: typographySource.bodySize !== undefined || typographySource.body_size !== undefined
          ? toFiniteNumber(typographySource.bodySize ?? typographySource.body_size, 24)
          : undefined,
      };
    }

    const surfaceSource = source.surface && typeof source.surface === 'object'
      ? source.surface as Record<string, unknown>
      : undefined;
    if (surfaceSource) {
      spec.surface = {
        background: String(surfaceSource.background || '').trim() || undefined,
        panel: String(surfaceSource.panel || '').trim() || undefined,
        primary: String(surfaceSource.primary || '').trim() || undefined,
        secondary: String(surfaceSource.secondary || '').trim() || undefined,
        accent: String(surfaceSource.accent || '').trim() || undefined,
        text: String(surfaceSource.text || '').trim() || undefined,
        muted: String(surfaceSource.muted || '').trim() || undefined,
        cardRadius: surfaceSource.cardRadius !== undefined || surfaceSource.card_radius !== undefined
          ? toFiniteNumber(surfaceSource.cardRadius ?? surfaceSource.card_radius, 0)
          : undefined,
        cardPadding: surfaceSource.cardPadding !== undefined || surfaceSource.card_padding !== undefined
          ? toFiniteNumber(surfaceSource.cardPadding ?? surfaceSource.card_padding, 0)
          : undefined,
        sectionGap: surfaceSource.sectionGap !== undefined || surfaceSource.section_gap !== undefined
          ? toFiniteNumber(surfaceSource.sectionGap ?? surfaceSource.section_gap, 0)
          : undefined,
      };
    }

    const layoutSource = source.layout && typeof source.layout === 'object'
      ? source.layout as Record<string, unknown>
      : undefined;
    if (layoutSource) {
      spec.layout = {
        safeMargin: layoutSource.safeMargin !== undefined || layoutSource.safe_margin !== undefined
          ? toFiniteNumber(layoutSource.safeMargin ?? layoutSource.safe_margin, 72)
          : undefined,
        sectionGap: layoutSource.sectionGap !== undefined || layoutSource.section_gap !== undefined
          ? toFiniteNumber(layoutSource.sectionGap ?? layoutSource.section_gap, 24)
          : undefined,
        contentGap: layoutSource.contentGap !== undefined || layoutSource.content_gap !== undefined
          ? toFiniteNumber(layoutSource.contentGap ?? layoutSource.content_gap, 18)
          : undefined,
        maxColumns: layoutSource.maxColumns !== undefined || layoutSource.max_columns !== undefined
          ? toFiniteNumber(layoutSource.maxColumns ?? layoutSource.max_columns, 0)
          : undefined,
      };
    }

    const nodeStylesSource = source.node_styles || source.nodeStyles;
    if (nodeStylesSource && typeof nodeStylesSource === 'object') {
      const nodeStyles = Object.fromEntries(
        Object.entries(nodeStylesSource as Record<string, unknown>)
          .map(([key, raw]) => [String(key), normalizeCanvasVisualStyle(raw)])
          .filter(([, style]) => Object.keys(style as Record<string, unknown>).length > 0),
      );
      if (Object.keys(nodeStyles).length > 0) {
        spec.nodeStyles = nodeStyles as FrontendCanvasVisualSpec['nodeStyles'];
      }
    }

    const componentStylesSource = source.component_styles || source.componentStyles;
    if (componentStylesSource && typeof componentStylesSource === 'object') {
      const componentStyles = Object.fromEntries(
        Object.entries(componentStylesSource as Record<string, unknown>)
          .map(([key, raw]) => [String(key), normalizeCanvasVisualStyle(raw)])
          .filter(([, style]) => Object.keys(style as Record<string, unknown>).length > 0),
      );
      if (Object.keys(componentStyles).length > 0) {
        spec.componentStyles = componentStyles as FrontendCanvasVisualSpec['componentStyles'];
      }
    }

    return Object.keys(spec).length > 0 ? spec : undefined;
  };

  const serializeFrontendSlide = (slide: FrontendSlide) => ({
    slide_id: slide.slideId,
    page_num: slide.pageNum,
    title: slide.title,
    schema_version: slide.schemaVersion || '',
    render_engine: slide.renderEngine || 'canvas',
    template_key: slide.templateKey || '',
    layout_mode: slide.layoutMode || '',
    layout_family: slide.layoutFamily || '',
    root: slide.root || undefined,
    content: slide.content || undefined,
    visual_spec: slide.visualSpec || undefined,
    constraints: slide.constraints || undefined,
    editable_map: slide.editableMap || undefined,
    canvas_validation: slide.canvasValidation || undefined,
    layout_ir: slide.layoutIr || undefined,
    blocks: slide.blocks.map((block) => ({
      id: block.id,
      type: block.type,
      role: block.role,
      content: block.content,
      items: block.items,
      asset_key: block.assetKey || '',
      table_data: block.tableData
        ? {
            headers: block.tableData.headers,
            rows: block.tableData.rows,
          }
        : undefined,
      children: (block.children || []).map((child) => ({
        id: child.id,
        type: child.type,
        role: child.role,
        content: child.content,
        items: child.items,
        asset_key: child.assetKey || '',
        table_data: child.tableData
          ? {
              headers: child.tableData.headers,
              rows: child.tableData.rows,
            }
          : undefined,
      })),
      layout: {
        zone: block.layout.zone,
        span: block.layout.span,
        order: block.layout.order,
        preferred_width: block.layout.preferredWidth,
        preferred_side: block.layout.preferredSide,
        emphasis: block.layout.emphasis,
      },
    })),
    html_template: slide.htmlTemplate,
    css_code: slide.cssCode,
    editable_fields: slide.editableFields.map((field) => ({
      key: field.key,
      label: field.label,
      type: field.type,
      value: field.value,
      items: field.items,
    })),
    visual_assets: slide.visualAssets.map((asset) => ({
      key: asset.key,
      label: asset.label,
      src: asset.src,
      preview_src: asset.previewSrc || asset.src,
      original_src: asset.originalSrc || asset.storagePath || asset.src,
      alt: asset.alt,
      source_type: asset.sourceType,
      storage_path: asset.storagePath || '',
      preview_storage_path: asset.previewStoragePath || '',
      prompt: asset.prompt || '',
      style: asset.style || '',
    })),
    generation_note: slide.generationNote || '',
    status: slide.status,
  });

  const buildFrontendPagecontentPayload = () =>
    JSON.stringify(
      outlineData.map((slide) => ({
        title: slide.title,
        layout_description: slide.layout_description,
        key_points: slide.key_points,
        asset_ref: slide.asset_ref,
      })),
    );

  const cloneOutlineSnapshot = (slides: SlideOutline[]) =>
    slides.map((slide) => ({
      ...slide,
      key_points: [...slide.key_points],
    }));

  const getUnchangedPageIndices = (
    current: SlideOutline[],
    snapshot: SlideOutline[],
  ): number[] => {
    if (snapshot.length === 0) return [];
    const unchanged: number[] = [];
    const minLength = Math.min(current.length, snapshot.length);
    for (let index = 0; index < minLength; index += 1) {
      const currentSlide = current[index];
      const snapshotSlide = snapshot[index];
      if (
        currentSlide.id === snapshotSlide.id &&
        currentSlide.title === snapshotSlide.title &&
        currentSlide.layout_description === snapshotSlide.layout_description &&
        currentSlide.asset_ref === snapshotSlide.asset_ref &&
        JSON.stringify(currentSlide.key_points) === JSON.stringify(snapshotSlide.key_points)
      ) {
        unchanged.push(index);
      }
    }
    return unchanged;
  };

  const buildPagecontentForGeneration = () =>
    outlineData.map((slide, index) => {
      const result = generateResults[index];
      const generatedPath = result?.afterImage || '';
      return {
        title: slide.title,
        layout_description: slide.layout_description,
        key_points: slide.key_points,
        asset_ref: slide.asset_ref,
        generated_img_path: generatedPath || undefined,
      };
    });

  const getEffectiveStylePrompt = (mode: PptGenerationMode = pptMode) =>
    globalPrompt || (mode === 'frontend' ? '' : getStyleDescription(stylePreset));

  const getFrontendGenerationCostPerPage = () => (frontendIncludeImages ? 2 : 1);

  const waitForFrontendCaptureNodes = async (count: number, timeoutMs: number = 6000) => {
    const startedAt = Date.now();
    while (Date.now() - startedAt < timeoutMs) {
      const ready = Array.from({ length: count }).every((_, index) => Boolean(frontendCaptureRefs.current[index]));
      if (ready) {
        return true;
      }
      await sleep(80);
    }
    return false;
  };

  const requestFrontendSlideGeneration = async ({
    slideIndex,
    prompt,
    resultPathValue,
    slideSnapshot,
  }: {
    slideIndex: number;
    prompt: string;
    resultPathValue: string;
    slideSnapshot: FrontendSlide;
  }) => {
    const formData = new FormData();
    formData.append('credential_scope', MANAGED_CREDENTIAL_SCOPE);
    formData.append('chat_api_url', llmApiUrl.trim());
    formData.append('api_key', apiKey.trim());
    formData.append('model', model);
    formData.append('language', language);
    formData.append('style', getEffectiveStylePrompt('frontend'));
    formData.append('email', user?.id || user?.email || '');
    formData.append('result_path', resultPathValue);
    formData.append('include_images', String(frontendIncludeImages));
    formData.append('image_style', frontendImageStyle);
    formData.append('image_model', genFigModel);
    formData.append('page_id', String(slideIndex));
    formData.append('edit_prompt', prompt.trim());
    formData.append('current_slide', JSON.stringify(serializeFrontendSlide(slideSnapshot)));
    formData.append('pagecontent', buildFrontendPagecontentPayload());

    const res = await backendFetch('/api/v1/paper2ppt/frontend/generate', {
      method: 'POST',
      headers: { 'X-Workflow-Amount': '1' },
      body: formData,
    });
    if (!res.ok) {
      throw new Error(await extractErrorMessage(res, '前端页面重生成失败'));
    }

    const data = await res.json();
    if (!data.success || !Array.isArray(data.slides) || data.slides.length === 0) {
      throw new Error(data.error || '前端页面重生成失败');
    }

    return {
      updatedSlide: normalizeFrontendSlides(data.slides)[0],
      nextTheme: normalizeFrontendDeckTheme(data.theme),
    };
  };

  const requestFrontendSlideReview = async ({
    slide,
    resultPathValue,
    layoutIssues,
    screenshot,
  }: {
    slide: FrontendSlide;
    resultPathValue: string;
    layoutIssues: string[];
    screenshot: Blob;
  }) => {
    const formData = new FormData();
    formData.append('credential_scope', MANAGED_CREDENTIAL_SCOPE);
    formData.append('chat_api_url', llmApiUrl.trim());
    formData.append('api_key', apiKey.trim());
    formData.append('language', language);
    formData.append('result_path', resultPathValue);
    formData.append('slide', JSON.stringify(serializeFrontendSlide(slide)));
    if (layoutIssues.length > 0) {
      formData.append('layout_issues', JSON.stringify(layoutIssues));
    }
    const reviewMimeType = screenshot.type || 'image/jpeg';
    const reviewExt = reviewMimeType === 'image/png' ? 'png' : 'jpg';
    formData.append(
      'screenshot',
      new File([screenshot], `review_page_${String(slide.pageNum - 1).padStart(3, '0')}.${reviewExt}`, {
        type: reviewMimeType,
      }),
    );

    const res = await backendFetch('/api/v1/paper2ppt/frontend/review', {
      method: 'POST',
      headers: { 'X-Workflow-Amount': '1' },
      body: formData,
    });
    if (!res.ok) {
      throw new Error(await extractErrorMessage(res, '前端页面视觉检查失败'));
    }
    return res.json();
  };

  const runWithConcurrency = async <T,>(
    items: T[],
    limit: number,
    worker: (item: T, index: number) => Promise<void>,
  ) => {
    let cursor = 0;
    const runners = Array.from({ length: Math.min(limit, items.length) }, async () => {
      while (cursor < items.length) {
        const currentIndex = cursor;
        cursor += 1;
        await worker(items[currentIndex], currentIndex);
      }
    });
    await Promise.all(runners);
  };

  const autoReviewAndRepairFrontendSlide = async (
    slideIndex: number,
    slideSnapshot: FrontendSlide,
    resultPathValue: string,
  ) => {
    const node = frontendCaptureRefs.current[slideIndex];
    if (!node) {
      updateFrontendSlideReview(slideIndex, {
        status: 'needs_repair',
        summary: '首轮自动检查跳过：预览节点尚未就绪。',
        issues: [],
      });
      return false;
    }

    updateFrontendSlideReview(slideIndex, {
      status: 'repairing',
      summary: '正在做首轮视觉检查...',
      issues: [],
    });

    try {
      await sleep(40);
      const localLayoutCheck = inspectSlideLayout(node, 1600, 900);
      const blob = await captureSlideToPngBlob(node, 1280, 720, {
        mimeType: 'image/jpeg',
        quality: 0.82,
      });
      const data = await requestFrontendSlideReview({
        slide: slideSnapshot,
        resultPathValue,
        layoutIssues: localLayoutCheck.issues,
        screenshot: blob,
      });

      const reviewIssues = Array.isArray(data.issues)
        ? data.issues.map((item: unknown) => String(item || '').trim()).filter(Boolean)
        : [];
      const reviewSummary = typeof data.summary === 'string' && data.summary.trim()
        ? data.summary.trim()
        : (data.passed ? '首轮检查通过。' : '检测到需要修复的版式问题。');

      if (data.passed) {
        updateFrontendSlideReview(slideIndex, {
          status: 'passed',
          summary: reviewSummary,
          issues: reviewIssues,
        });
        return true;
      }

      const repairPrompt = typeof data.repair_prompt === 'string' ? data.repair_prompt.trim() : '';
      if (!repairPrompt) {
        updateFrontendSlideReview(slideIndex, {
          status: 'needs_repair',
          summary: reviewSummary || '首轮检查发现问题，但没有收到修复指令。',
          issues: reviewIssues,
        });
        return false;
      }

      updateFrontendSlideReview(slideIndex, {
        status: 'repairing',
        summary: '首轮检查发现版式问题，正在自动修正...',
        issues: reviewIssues,
      });

      const { updatedSlide, nextTheme } = await requestFrontendSlideGeneration({
        slideIndex,
        prompt: repairPrompt,
        resultPathValue,
        slideSnapshot,
      });

      setFrontendSlides((prev) =>
        prev.map((slide, index) =>
          index === slideIndex
            ? {
                ...updatedSlide,
                review: {
                  status: 'passed',
                  summary: '首轮视觉检查已自动修正当前页。',
                  issues: [],
                },
              }
            : slide,
        ),
      );

      if (nextTheme) {
        setFrontendDeckTheme(nextTheme);
      }
      return true;
    } catch (err) {
      const message = err instanceof Error ? err.message : '首轮自动视觉检查失败';
      updateFrontendSlideReview(slideIndex, {
        status: 'needs_repair',
        summary: `首轮自动检查失败：${message}`,
        issues: [],
      });
      return false;
    }
  };

  const runInitialFrontendReviewPass = async (
    slides: FrontendSlide[],
    resultPathValue: string,
  ) => {
    if (slides.length === 0) {
      return;
    }

    setGenerateTaskMessage('首轮生成完成，正在并行做视觉检查与自动调整...');
    await sleep(180);
    const ready = await waitForFrontendCaptureNodes(slides.length);
    if (!ready) {
      setError('前端页面已生成，但自动视觉检查未能拿到全部预览节点，请手动逐页检查。');
      return;
    }

    const reviewResults: boolean[] = new Array(slides.length).fill(false);
    let completed = 0;
    await runWithConcurrency(slides, 2, async (slide, index) => {
      reviewResults[index] = await autoReviewAndRepairFrontendSlide(index, slide, resultPathValue);
      completed += 1;
      setGenerateTaskMessage(`首轮视觉检查进行中（${completed}/${slides.length}）...`);
    });

    const failedCount = reviewResults.filter((item) => !item).length;
    if (failedCount > 0) {
      setError(`首轮自动视觉检查已完成，但仍有 ${failedCount} 页需要你手动复查。`);
    } else {
      setError(null);
    }
  };

  const uploadGeneratedResultFile = async (filePath: string | null | undefined, defaultName: string) => {
    if (!filePath) return;
    try {
      let fetchUrl = filePath;
      if (window.location.protocol === 'https:' && filePath.startsWith('http:')) {
        fetchUrl = filePath.replace('http:', 'https:');
      }

      const fileRes = await fetch(fetchUrl);
      if (!fileRes.ok) {
        console.error('[Paper2PptPage] Failed to fetch file for upload:', fileRes.status, fileRes.statusText);
        return;
      }

      const fileBlob = await fileRes.blob();
      const fileName = filePath.split('/').pop() || defaultName;
      await uploadAndSaveFile(fileBlob, fileName, 'paper2ppt');
    } catch (e) {
      console.error('[Paper2PptPage] Failed to upload file:', e);
    }
  };

  // ============== Step 1: 上传处理 ==============
  const validateDocFile = (file: File): boolean => {
    const ext = file.name.split('.').pop()?.toLowerCase();
    if (ext !== 'pdf') {
      setError('仅支持 PDF 格式');
      return false;
    }
    return true;
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !validateDocFile(file)) return;
    if (file.size > MAX_FILE_SIZE) {
      setError('文件大小超过 50MB 限制');
      return;
    }
    setSelectedFile(file);
    setError(null);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (!file || !validateDocFile(file)) return;
    if (file.size > MAX_FILE_SIZE) {
      setError('文件大小超过 50MB 限制');
      return;
    }
    setSelectedFile(file);
    setError(null);
  };

  const handleReferenceImageChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const ext = file.name.split('.').pop()?.toLowerCase();
    if (!['jpg', 'jpeg', 'png', 'webp', 'gif'].includes(ext || '')) {
      setError('参考图片仅支持 JPG/PNG/WEBP/GIF 格式');
      return;
    }
    setReferenceImage(file);
    setReferenceImagePreview(URL.createObjectURL(file));
    setError(null);
  };

  const handleRemoveReferenceImage = () => {
    if (referenceImagePreview) {
      URL.revokeObjectURL(referenceImagePreview);
    }
    setReferenceImage(null);
    setReferenceImagePreview(null);
  };

  const getStyleDescription = (preset: string): string => {
    const imageStyles: Record<string, string> = {
      modern: '现代简约风格，使用干净的线条和充足的留白',
      business: '商务专业风格，稳重大气，适合企业演示',
      academic: '学术报告风格，清晰的层次结构，适合论文汇报',
      creative: '创意设计风格，活泼生动，色彩丰富',
    };
    return imageStyles[preset] || imageStyles.modern;
  };

  const handleUploadAndParse = async () => {
    if (uploadMode === 'file' && !selectedFile) {
      setError('请先选择 PDF 文件');
      return;
    }
    if ((uploadMode === 'text' || uploadMode === 'topic') && !textContent.trim()) {
      setError(uploadMode === 'text' ? '请输入长文本内容' : '请输入 Topic 主题');
      return;
    }
    
    if (userApiConfigRequired && !apiKey.trim()) {
      setError('请输入 API Key');
      return;
    }

    if (isUploading || isValidating || isUploadSubmitLocked || uploadSubmitGuardRef.current) {
      return;
    }

    uploadSubmitGuardRef.current = true;
    setIsUploadSubmitLocked(true);

    let progressInterval: number | null = null;

    try {
      const quota = await checkQuota(user?.id || null, user?.is_anonymous || false);
      if (quota.remaining <= 0) {
        setError(buildQuotaExhaustedMessage(purchaseUrl));
        return;
      }

      try {
        setIsValidating(true);
        setError(null);
        await verifyLlmConnection(llmApiUrl, apiKey, import.meta.env.VITE_DEFAULT_LLM_MODEL || 'deepseek-v3.2');
      } catch (err) {
        const message = err instanceof Error ? err.message : 'API 验证失败';
        setError(message);
        return;
      }

      setIsUploading(true);
      setError(null);
      setGenerateResults([]);
      setFrontendSlides([]);
      setFrontendDeckTheme(null);
      frontendCaptureRefs.current = [];
      setDownloadUrl((previousUrl) => {
        if (previousUrl?.startsWith('blob:')) {
          URL.revokeObjectURL(previousUrl);
        }
        return null;
      });
      setPdfPreviewUrl(null);
      setResultPath(null);
      setProgress(0);
      setProgressStatus('正在初始化...');

      const requestStartedAt = Date.now();
      progressInterval = window.setInterval(() => {
        setProgress(prev => {
          const elapsedSec = Math.floor((Date.now() - requestStartedAt) / 1000);
          if (prev >= 90) {
            if (elapsedSec >= 90) {
              setProgressStatus(`AI 正在生成大纲，已等待 ${Math.floor(elapsedSec / 60)} 分 ${elapsedSec % 60} 秒，请稍候`);
            } else {
              setProgressStatus('AI 正在生成大纲，请稍候');
            }
            return 90;
          }
          const messages = [
            '正在准备输入内容...',
            '正在解析论文内容...',
            '正在提取关键信息...',
            '正在请求大模型生成大纲...',
          ];
          const msgIndex = Math.min(messages.length - 1, Math.floor(prev / 25));
          if (elapsedSec >= 90) {
            setProgressStatus(`AI 正在生成大纲，已等待 ${Math.floor(elapsedSec / 60)} 分 ${elapsedSec % 60} 秒，请稍候`);
          } else if (elapsedSec >= 45) {
            setProgressStatus('AI 正在生成大纲，模型响应较慢，请稍候');
          } else {
            setProgressStatus(messages[msgIndex]);
          }
          return prev + (Math.random() * 0.6 + 0.2);
        });
      }, 1000);

      const formData = new FormData();
      if (uploadMode === 'file' && selectedFile) {
        formData.append('file', selectedFile);
        formData.append('input_type', 'pdf');
      } else {
        formData.append('text', textContent.trim());
        formData.append('input_type', uploadMode); // 'text' or 'topic'
      }
      
      formData.append('email', user?.id || user?.email || '');
      formData.append('credential_scope', MANAGED_CREDENTIAL_SCOPE);
      if (userApiConfigRequired) {
        formData.append('chat_api_url', llmApiUrl.trim());
        formData.append('api_key', apiKey.trim());
      }
      formData.append('model', model);
      formData.append('language', language);
      formData.append('style', getEffectiveStylePrompt());
      formData.append('gen_fig_model', genFigModel);
      formData.append('page_count', String(pageCount));
      formData.append('use_long_paper', String(useLongPaper));

      if (styleMode === 'reference' && referenceImage) {
        formData.append('reference_img', referenceImage);
        // 参考图模式下：保留用户显式输入的风格提示词（globalPrompt），但去掉默认 preset 描述
        formData.set('style', globalPrompt || '');
      }

      console.log(`Sending request to /api/v1/paper2ppt/page-content with input_type=${uploadMode}`);
      
      const res = await backendFetch('/api/v1/paper2ppt/page-content', {
        method: 'POST',
        body: formData,
      });
      
      if (!res.ok) {
        throw new Error(await extractErrorMessage(res, '服务器繁忙，请稍后再试'));
      }

      const data = await res.json();
      console.log('API Response:', JSON.stringify(data, null, 2));

      if (!data.success) {
        throw new Error(data.error || '服务器繁忙，请稍后再试');
      }
      
      const currentResultPath = data.result_path || '';
      if (currentResultPath) {
        setResultPath(currentResultPath);
      } else {
        throw new Error('后端未返回 result_path');
      }
      
      if (!data.pagecontent || data.pagecontent.length === 0) {
        throw new Error('解析结果为空，请检查输入内容是否正确');
      }
      
      const convertedSlides: SlideOutline[] = data.pagecontent.map((item: any, index: number) => ({
        id: String(index + 1),
        pageNum: index + 1,
        title: item.title || `第 ${index + 1} 页`,
        layout_description: item.layout_description || '',
        key_points: item.key_points || [],
        asset_ref: item.asset_ref || null,
      }));
      
      window.clearInterval(progressInterval);
      progressInterval = null;
      setProgress(100);
      setProgressStatus('解析完成！');
      
      // 稍微延迟一下跳转，让用户看到 100%
      setTimeout(() => {
        setOutlineData(convertedSlides);
        setConfirmedOutlineSnapshot([]);
        setGenerateResults([]);
        setFrontendSlides([]);
        setFrontendDeckTheme(null);
        setCurrentStep('outline');
      }, 500);
      
    } catch (err) {
      if (progressInterval !== null) {
        window.clearInterval(progressInterval);
        progressInterval = null;
      }
      setProgress(0);
      const message = err instanceof Error ? err.message : '服务器繁忙，请稍后再试';
      setError(message);
      console.error(err);
    } finally {
      if (progressInterval !== null) {
        window.clearInterval(progressInterval);
      }
      setIsValidating(false);
      setIsUploading(false);
      releaseUploadSubmitGuard();
    }
  };

  // ============== Step 2: Outline 编辑处理 ==============
  const handleEditStart = (slide: SlideOutline) => {
    setEditingId(slide.id);
    setEditContent({ 
      title: slide.title, 
      layout_description: slide.layout_description,
      key_points: [...slide.key_points]
    });
  };

  const handleEditSave = () => {
    if (!editingId) return;
    setOutlineData(prev => prev.map(s => 
      s.id === editingId 
        ? { ...s, title: editContent.title, layout_description: editContent.layout_description, key_points: editContent.key_points }
        : s
    ));
    setEditingId(null);
  };

  const handleKeyPointChange = (index: number, value: string) => {
    setEditContent(prev => {
      const newKeyPoints = [...prev.key_points];
      newKeyPoints[index] = value;
      return { ...prev, key_points: newKeyPoints };
    });
  };

  const handleAddKeyPoint = () => {
    setEditContent(prev => ({ ...prev, key_points: [...prev.key_points, ''] }));
  };

  const handleRemoveKeyPoint = (index: number) => {
    setEditContent(prev => ({ ...prev, key_points: prev.key_points.filter((_, i) => i !== index) }));
  };

  const handleEditCancel = () => setEditingId(null);
  
  const handleDeleteSlide = (id: string) => {
    setOutlineData(prev => prev.filter(s => s.id !== id).map((s, i) => ({ ...s, pageNum: i + 1 })));
  };

  const handleAddSlide = (index: number) => {
    setOutlineData(prev => {
      const newSlide: SlideOutline = {
        id: String(Date.now()),
        pageNum: 0, 
        title: '新页面',
        layout_description: '左右图文，左边是：，右边是：',
        key_points: [''],
        asset_ref: null,
      };
      const newData = [...prev];
      newData.splice(index + 1, 0, newSlide);
      return newData.map((s, i) => ({ ...s, pageNum: i + 1, title: s.title === '新页面' ? `第 ${i + 1} 页` : s.title }));
    });
  };
  
  const handleMoveSlide = (index: number, direction: 'up' | 'down') => {
    const newData = [...outlineData];
    const targetIndex = direction === 'up' ? index - 1 : index + 1;
    if (targetIndex < 0 || targetIndex >= newData.length) return;
    [newData[index], newData[targetIndex]] = [newData[targetIndex], newData[index]];
    setOutlineData(newData.map((s, i) => ({ ...s, pageNum: i + 1 })));
  };

  const handleRefineOutline = async () => {
    if (isRefiningOutline) return;
    if (!outlineFeedback.trim()) {
      setError('请输入修改需求');
      return;
    }
    if (!resultPath) {
      setError('缺少 result_path，请重新上传文件');
      return;
    }

    setError(null);
    setIsRefiningOutline(true);

    const currentOutline = editingId
      ? outlineData.map(s =>
          s.id === editingId
            ? {
                ...s,
                title: editContent.title,
                layout_description: editContent.layout_description,
                key_points: editContent.key_points,
              }
            : s
        )
      : outlineData;

    if (editingId) {
      setOutlineData(currentOutline);
      setEditingId(null);
    }

    const pagecontent = currentOutline.map((slide) => ({
      title: slide.title,
      layout_description: slide.layout_description,
      key_points: slide.key_points,
      asset_ref: slide.asset_ref,
    }));

    try {
      const formData = new FormData();
      formData.append('outline_feedback', outlineFeedback.trim());
      formData.append('pagecontent', JSON.stringify(pagecontent));
      formData.append('credential_scope', MANAGED_CREDENTIAL_SCOPE);
      formData.append('chat_api_url', llmApiUrl.trim());
      formData.append('api_key', apiKey.trim());
      formData.append('model', model);
      formData.append('language', language);
      formData.append('email', user?.email || '');
      formData.append('result_path', resultPath);

      const res = await backendFetch('/api/v1/paper2ppt/outline-refine', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        let msg = '服务器繁忙，请稍后再试';
        if (res.status === 429) {
          msg = '请求过于频繁，请稍后再试';
        } else {
          try {
            const errBody = await res.json();
            if (errBody?.error) msg = errBody.error;
          } catch { /* ignore parse error */ }
        }
        throw new Error(msg);
      }

      const data = await res.json();
      if (!data.success) {
        throw new Error(data.error || '服务器繁忙，请稍后再试');
      }

      if (!data.pagecontent || data.pagecontent.length === 0) {
        throw new Error('AI 调整失败，请重试');
      }

      const refinedSlides: SlideOutline[] = data.pagecontent.map((item: any, index: number) => ({
        id: String(index + 1),
        pageNum: index + 1,
        title: item.title || `第 ${index + 1} 页`,
        layout_description: item.layout_description || '',
        key_points: item.key_points || [],
        asset_ref: item.asset_ref || null,
      }));

      setOutlineData(refinedSlides);
      setOutlineFeedback('');
    } catch (err) {
      const message = err instanceof Error ? err.message : '服务器繁忙，请稍后再试';
      setError(message);
    } finally {
      setIsRefiningOutline(false);
    }
  };

  const updateFrontendFieldValue = (slideIndex: number, fieldKey: string, value: string) => {
    setFrontendSlides((prev) =>
      prev.map((slide, idx) =>
        idx === slideIndex
          ? applyFrontendEditableFieldMutation(slide, fieldKey, (field) => ({ ...field, value }))
          : slide,
      ),
    );
  };

  const updateFrontendListItem = (slideIndex: number, fieldKey: string, itemIndex: number, value: string) => {
    setFrontendSlides((prev) =>
      prev.map((slide, idx) => {
        if (idx !== slideIndex) return slide;
        return applyFrontendEditableFieldMutation(slide, fieldKey, (field) => {
          const nextItems = [...field.items];
          nextItems[itemIndex] = value;
          return { ...field, items: nextItems };
        });
      }),
    );
  };

  const addFrontendListItem = (slideIndex: number, fieldKey: string) => {
    setFrontendSlides((prev) =>
      prev.map((slide, idx) => {
        if (idx !== slideIndex) return slide;
        return applyFrontendEditableFieldMutation(slide, fieldKey, (field) => ({
          ...field,
          items: [...field.items, ''],
        }));
      }),
    );
  };

  const replaceFrontendListItems = (slideIndex: number, fieldKey: string, items: string[]) => {
    setFrontendSlides((prev) =>
      prev.map((slide, idx) => {
        if (idx !== slideIndex) return slide;
        return applyFrontendEditableFieldMutation(slide, fieldKey, (field) => ({
          ...field,
          items,
        }));
      }),
    );
  };

  const removeFrontendListItem = (slideIndex: number, fieldKey: string, itemIndex: number) => {
    setFrontendSlides((prev) =>
      prev.map((slide, idx) => {
        if (idx !== slideIndex) return slide;
        return applyFrontendEditableFieldMutation(slide, fieldKey, (field) => ({
          ...field,
          items: field.items.filter((_, idx2) => idx2 !== itemIndex),
        }));
      }),
    );
  };

  const updateFrontendLayoutIr = (slideIndex: number, layoutIr: FrontendSlide['layoutIr']) => {
    if (!layoutIr) return;
    setFrontendSlides((prev) =>
      prev.map((slide, idx) => {
        if (idx !== slideIndex) return slide;
        const previous = slide.layoutIr ? JSON.stringify(slide.layoutIr) : '';
        const next = JSON.stringify(layoutIr);
        if (previous === next) return slide;
        return {
          ...slide,
          layoutIr,
        };
      }),
    );
  };

  const buildDefaultTableData = (): FrontendTableData => ({
    headers: ['指标', '当前值', '说明'],
    rows: [
      ['样本量', 'N/A', '补充说明'],
      ['效果', 'N/A', '补充说明'],
    ],
  });

  const buildTableEditableFields = (tableId: string, tableData: FrontendTableData): FrontendEditableField[] => [
    ...tableData.headers.map((header, colIndex) => ({
      key: `${tableId}_cell_h_${colIndex}`,
      label: `表头 ${colIndex + 1}`,
      type: 'text' as const,
      value: header,
      items: [],
    })),
    ...tableData.rows.flatMap((row, rowIndex) =>
      row.map((cell, colIndex) => ({
        key: `${tableId}_cell_${rowIndex}_${colIndex}`,
        label: `表格 R${rowIndex + 1}C${colIndex + 1}`,
        type: 'text' as const,
        value: cell,
        items: [],
      })),
    ),
  ];

  const insertFrontendTableBlock = (slideIndex: number, targetBlockId?: string) => {
    setFrontendSlides((prev) =>
      prev.map((slide, idx) => {
        if (idx !== slideIndex) return slide;
        const tableData = buildDefaultTableData();
        if (slide.renderEngine === 'canvas') {
          const nodeId = buildUniqueCanvasNodeId(slide, 'table');
          return insertCanvasNode(
            slide,
            {
              type: 'component',
              id: nodeId,
              component: 'table',
              props: { table_ref: nodeId },
            },
            {
              targetId: targetBlockId,
              contentPatch: {
                [nodeId]: {
                  headers: tableData.headers,
                  rows: tableData.rows,
                },
              },
              editableFields: buildTableEditableFields(nodeId, tableData),
            },
          );
        }
        const targetZone = parseFrontendInsertZoneTarget(targetBlockId);
        if (targetZone) {
          const blockId = buildUniqueBlockId(slide, 'table_block');
          const block: FrontendSlideBlock = {
            id: blockId,
            type: 'table',
            role: blockId,
            content: '',
            items: [],
            tableData,
            layout: buildInsertedBlockLayout(slide, {
              zone: targetZone,
              order: slide.blocks.length + 1,
              preferredWidth: targetZone === 'full' ? 'full' : 'half',
              preferredSide: targetZone === 'left' || targetZone === 'right' ? targetZone : 'auto',
            }),
          };
          return insertTopLevelBlockIntoZone(
            slide,
            block,
            buildTableEditableFields(blockId, tableData),
          );
        }
        const childId = buildUniqueChildId(slide, 'table_item');
        const child: FrontendBlockChild = {
          id: childId,
          type: 'table',
          role: childId,
          content: '',
          items: [],
          tableData,
        };
        return insertChildIntoBlock(
          slide,
          targetBlockId,
          child,
          buildTableEditableFields(childId, tableData),
        );
      }),
    );
  };

  const insertFrontendTextBlock = (slideIndex: number, targetBlockId?: string) => {
    setFrontendSlides((prev) =>
      prev.map((slide, idx) => {
        if (idx !== slideIndex) return slide;
        if (slide.renderEngine === 'canvas') {
          const nodeId = buildUniqueCanvasNodeId(slide, 'text');
          const value = '新的文本块';
          return insertCanvasNode(
            slide,
            {
              type: 'component',
              id: nodeId,
              component: 'text',
              props: { text_ref: nodeId },
            },
            {
              targetId: targetBlockId,
              contentPatch: { [nodeId]: value },
              editableFields: {
                key: nodeId,
                label: '文本块',
                type: 'textarea',
                value,
                items: [],
              },
            },
          );
        }
        const targetZone = parseFrontendInsertZoneTarget(targetBlockId);
        if (targetZone) {
          const blockId = buildUniqueBlockId(slide, 'text_block');
          const block: FrontendSlideBlock = {
            id: blockId,
            type: 'text',
            role: blockId,
            content: '新的文本块',
            items: [],
            layout: buildInsertedBlockLayout(slide, {
              zone: targetZone,
              order: slide.blocks.length + 1,
              preferredSide: targetZone === 'left' || targetZone === 'right' ? targetZone : 'auto',
            }),
          };
          return insertTopLevelBlockIntoZone(slide, block, {
            key: blockId,
            label: '文本块',
            type: 'textarea',
            value: block.content,
            items: [],
          });
        }
        const childId = buildUniqueChildId(slide, 'text_item');
        const child: FrontendBlockChild = {
          id: childId,
          type: 'text',
          role: childId,
          content: '新的文本块',
          items: [],
        };
        return insertChildIntoBlock(slide, targetBlockId, child, {
          key: childId,
          label: '文本块',
          type: 'textarea',
          value: child.content,
          items: [],
        });
      }),
    );
  };

  const insertFrontendCalloutBlock = (slideIndex: number, targetBlockId?: string) => {
    setFrontendSlides((prev) =>
      prev.map((slide, idx) => {
        if (idx !== slideIndex) return slide;
        if (slide.renderEngine === 'canvas') {
          const nodeId = buildUniqueCanvasNodeId(slide, 'callout');
          const value = '新的重点内容';
          return insertCanvasNode(
            slide,
            {
              type: 'component',
              id: nodeId,
              component: 'callout',
              props: { text_ref: nodeId },
            },
            {
              targetId: targetBlockId,
              contentPatch: { [nodeId]: value },
              editableFields: {
                key: nodeId,
                label: '重点内容',
                type: 'textarea',
                value,
                items: [],
              },
            },
          );
        }
        const targetZone = parseFrontendInsertZoneTarget(targetBlockId);
        if (targetZone) {
          const blockId = buildUniqueBlockId(slide, 'callout_block');
          const block: FrontendSlideBlock = {
            id: blockId,
            type: 'callout',
            role: blockId,
            content: '新的重点内容',
            items: [],
            layout: buildInsertedBlockLayout(slide, {
              zone: targetZone,
              order: slide.blocks.length + 1,
              preferredSide: targetZone === 'left' || targetZone === 'right' ? targetZone : 'auto',
              emphasis: 'high',
            }),
          };
          return insertTopLevelBlockIntoZone(slide, block, {
            key: blockId,
            label: '重点内容',
            type: 'textarea',
            value: block.content,
            items: [],
          });
        }
        const childId = buildUniqueChildId(slide, 'callout_item');
        const child: FrontendBlockChild = {
          id: childId,
          type: 'callout',
          role: childId,
          content: '新的重点内容',
          items: [],
        };
        return insertChildIntoBlock(slide, targetBlockId, child, {
          key: childId,
          label: '重点内容',
          type: 'textarea',
          value: child.content,
          items: [],
        });
      }),
    );
  };

  const insertFrontendImageBlock = async (slideIndex: number, file: File, targetBlockId?: string) => {
    if (!resultPath) {
      setError('缺少 result_path，请重新上传文件');
      return;
    }
    if (!file.type.startsWith('image/')) {
      setError('仅支持上传图片文件');
      return;
    }

    const currentSlide = frontendSlides[slideIndex];
    if (!currentSlide) {
      setError('当前前端页面不存在');
      return;
    }

    const assetKey = currentSlide.renderEngine === 'canvas'
      ? buildUniqueCanvasNodeId(currentSlide, 'user_image')
      : buildUniqueBlockId(currentSlide, 'user_image');
    setError(null);

    try {
      const formData = new FormData();
      formData.append('result_path', resultPath);
      formData.append('asset_key', assetKey);
      formData.append('file', file);

      const res = await backendFetch('/api/v1/paper2ppt/frontend/upload-asset', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error(await extractErrorMessage(res, '图片上传失败'));
      }

      const data = await res.json();
      if (!data.success || !data.asset) {
        throw new Error(data.error || '图片上传失败');
      }

      setFrontendSlides((prev) =>
        prev.map((slide, idx) => {
          if (idx !== slideIndex) return slide;
          const nextAsset = {
            key: assetKey,
            label: String(data.asset.label || data.asset.key || file.name || assetKey),
            src: String(data.asset.src || ''),
            previewSrc: String(data.asset.preview_src || data.asset.previewSrc || data.asset.src || ''),
            originalSrc: String(data.asset.original_src || data.asset.originalSrc || data.asset.storage_path || data.asset.storagePath || data.asset.src || ''),
            alt: String(data.asset.alt || file.name || assetKey),
            sourceType: 'upload' as const,
            storagePath: String(data.asset.storage_path || data.asset.storagePath || ''),
            previewStoragePath: String(data.asset.preview_storage_path || data.asset.previewStoragePath || ''),
            prompt: typeof data.asset.prompt === 'string' ? data.asset.prompt : undefined,
            style: typeof data.asset.style === 'string' ? data.asset.style : undefined,
          };
          const child: FrontendBlockChild = {
            id: assetKey,
            type: 'image',
            role: 'supporting_visual',
            content: '',
            items: [],
            assetKey,
          };
          const visualAssets = [...slide.visualAssets, nextAsset];
          if (slide.renderEngine === 'canvas') {
            return insertCanvasNode(
              slide,
              {
                type: 'component',
                id: assetKey,
                component: 'figure',
                props: {
                  asset_ref: assetKey,
                  asset_key: assetKey,
                  fit: 'contain',
                },
              },
              {
                targetId: targetBlockId,
                visualAssets,
                contentPatch: {
                  assets: {
                    ...(((slide.content?.assets && typeof slide.content.assets === 'object')
                      ? slide.content.assets as Record<string, unknown>
                      : {})),
                    [assetKey]: {
                      type: 'image',
                      asset_key: assetKey,
                      src: nextAsset.src,
                      preview_src: nextAsset.previewSrc || nextAsset.src,
                      original_src: nextAsset.originalSrc || nextAsset.storagePath || nextAsset.src,
                      alt: nextAsset.alt,
                    },
                  },
                },
              },
            );
          }
          const targetZone = parseFrontendInsertZoneTarget(targetBlockId);
          if (targetZone) {
            const block: FrontendSlideBlock = {
              id: assetKey,
              type: 'image',
              role: 'supporting_visual',
              content: '',
              items: [],
              assetKey,
              layout: buildInsertedBlockLayout(slide, {
                zone: targetZone,
                order: slide.blocks.length + 1,
                preferredSide: targetZone === 'left' || targetZone === 'right' ? targetZone : 'auto',
              }),
            };
            return {
              ...insertTopLevelBlockIntoZone(slide, block, undefined, visualAssets),
              generationNote: '当前页已插入图片块。',
            };
          }
          return {
            ...insertChildIntoBlock(slide, targetBlockId, child),
            visualAssets,
            generationNote: '当前页已插入图片块。',
          };
        }),
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : '图片上传失败';
      setError(message);
    }
  };

  const replaceFrontendVisualAsset = async (slideIndex: number, imageKey: string, file: File) => {
    if (!resultPath) {
      setError('缺少 result_path，请重新上传文件');
      return;
    }
    if (!file.type.startsWith('image/')) {
      setError('仅支持上传图片文件');
      return;
    }

    const currentSlide = frontendSlides[slideIndex];
    if (!currentSlide) {
      setError('当前前端页面不存在');
      return;
    }

    setError(null);

    try {
      const formData = new FormData();
      formData.append('result_path', resultPath);
      formData.append('asset_key', imageKey);
      formData.append('file', file);

      const res = await backendFetch('/api/v1/paper2ppt/frontend/upload-asset', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error(await extractErrorMessage(res, '图片上传失败'));
      }

      const data = await res.json();
      if (!data.success || !data.asset) {
        throw new Error(data.error || '图片上传失败');
      }

      setFrontendSlides((prev) =>
        prev.map((slide, idx) => {
          if (idx !== slideIndex) return slide;
          const normalizedAsset = {
            key: imageKey,
            label: String(data.asset.label || data.asset.key || imageKey),
            src: String(data.asset.src || ''),
            previewSrc: String(data.asset.preview_src || data.asset.previewSrc || data.asset.src || ''),
            originalSrc: String(data.asset.original_src || data.asset.originalSrc || data.asset.storage_path || data.asset.storagePath || data.asset.src || ''),
            alt: String(data.asset.alt || file.name || imageKey),
            sourceType: 'upload' as const,
            storagePath: String(data.asset.storage_path || data.asset.storagePath || ''),
            previewStoragePath: String(data.asset.preview_storage_path || data.asset.previewStoragePath || ''),
            prompt: typeof data.asset.prompt === 'string' ? data.asset.prompt : undefined,
            style: typeof data.asset.style === 'string' ? data.asset.style : undefined,
          };
          const hasExistingAsset = slide.visualAssets.some((asset) => asset.key === imageKey);
          return {
            ...slide,
            generationNote: '当前页图片已替换为用户上传版本。',
            review: buildIdleFrontendReview(),
            visualAssets: hasExistingAsset
              ? slide.visualAssets.map((asset) =>
                  asset.key === imageKey
                    ? {
                        ...asset,
                        ...normalizedAsset,
                      }
                    : asset,
                )
              : [...slide.visualAssets, normalizedAsset],
          };
        }),
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : '图片上传失败';
      setError(message);
    }
  };

  const applyFrontendCodeEdit = (slideIndex: number, htmlTemplate: string, cssCode: string) => {
    const currentSlide = frontendSlides[slideIndex];
    if (!currentSlide) {
      setError('当前前端页面不存在');
      return false;
    }

    const validation = validateFrontendSlideCode(currentSlide, htmlTemplate, cssCode);
    if (!validation.ok) {
      setError(validation.issues.join(' '));
      return false;
    }

    setError(
      validation.warnings.length > 0
        ? `代码已应用，但请留意：${validation.warnings.join(' ')}`
        : null,
    );
    setFrontendSlides((prev) =>
      prev.map((slide, index) =>
        index === slideIndex
          ? {
              ...slide,
              htmlTemplate: validation.sanitizedHtml,
              cssCode: validation.sanitizedCss,
              generationNote: validation.warnings.length > 0
                ? `本地代码已应用。${validation.warnings.join(' ')}`
                : '本地代码已应用。',
              review: {
                status: 'idle',
                summary: '',
                issues: [],
              },
            }
          : slide,
      ),
    );
    return true;
  };

  const handleConfirmFrontendOutline = async () => {
    const unchangedIndices = getUnchangedPageIndices(outlineData, confirmedOutlineSnapshot);
    const hasExistingSlides = frontendSlides.some((slide) => slide.status === 'done');
    const skipSlides = hasExistingSlides ? unchangedIndices : [];
    const pagesToGenerate = outlineData.length - skipSlides.length;
    const requiredPoints = pagesToGenerate * getFrontendGenerationCostPerPage();

    if (
      requiredPoints > 0 &&
      !(await ensureQuotaForAction(requiredPoints, `批量生成前端 PPT（${pagesToGenerate} 页，预计 ${requiredPoints} 点）`))
    ) {
      return;
    }

    setCurrentStep('generate');
    setCurrentSlideIndex(0);
    setIsGenerating(true);
    if (skipSlides.length > 0) {
      setGenerateTaskMessage(`复用 ${skipSlides.length} 页未修改内容，重新生成 ${pagesToGenerate} 页可编辑版页面...`);
    } else {
      setGenerateTaskMessage(frontendIncludeImages ? '正在生成可编辑版页面代码与配图...' : '正在生成可编辑版页面代码...');
    }
    setError(null);

    const skipSet = new Set(skipSlides);
    const pendingSlides: FrontendSlide[] = outlineData.map((slide, index) => {
      if (skipSet.has(index) && index < frontendSlides.length && frontendSlides[index].status === 'done') {
        return { ...frontendSlides[index] };
      }
      return {
        slideId: slide.id,
        pageNum: index + 1,
        title: slide.title,
        blocks: [],
        htmlTemplate: '',
        cssCode: '',
        editableFields: [],
        visualAssets: [],
        status: 'processing',
        generationNote: '',
        review: {
          status: 'idle',
          summary: '',
          issues: [],
        },
      };
    });
    frontendCaptureRefs.current = [];
    setFrontendSlides(pendingSlides);

    try {
      const formData = new FormData();
      formData.append('credential_scope', MANAGED_CREDENTIAL_SCOPE);
      formData.append('chat_api_url', llmApiUrl.trim());
      formData.append('api_key', apiKey.trim());
      formData.append('model', model);
      formData.append('language', language);
      formData.append('style', getEffectiveStylePrompt('frontend'));
      formData.append('email', user?.id || user?.email || '');
      formData.append('result_path', resultPath || '');
      formData.append('include_images', String(frontendIncludeImages));
      formData.append('image_style', frontendImageStyle);
      formData.append('image_model', genFigModel);
      formData.append('pagecontent', buildFrontendPagecontentPayload());
      if (skipSlides.length > 0) {
        formData.append('skip_slides', JSON.stringify(skipSlides));
      }

      const res = await backendFetch('/api/v1/paper2ppt/frontend/generate', {
        method: 'POST',
        headers: requiredPoints > 0 ? { 'X-Workflow-Amount': String(requiredPoints) } : undefined,
        body: formData,
      });

      if (!res.ok) {
        throw new Error(await extractErrorMessage(res, '可编辑版 PPT 生成失败'));
      }

      const data = await res.json();
      if (!data.success || !Array.isArray(data.slides) || data.slides.length === 0) {
        throw new Error(data.error || '可编辑版 PPT 生成失败');
      }

      if (data.result_path) {
        setResultPath(data.result_path);
      }
      const normalizedTheme = normalizeFrontendDeckTheme(data.theme);
      const normalizedSlides = normalizeFrontendSlides(data.slides);
      const mergedSlides = pendingSlides.map((pendingSlide, index) => {
        if (skipSet.has(index) && pendingSlide.status === 'done') {
          return pendingSlide;
        }
        return normalizedSlides.find((slide) => slide.pageNum === index + 1) || pendingSlide;
      });
      setFrontendDeckTheme(normalizedTheme);
      setFrontendSlides(mergedSlides);
      setConfirmedOutlineSnapshot(cloneOutlineSnapshot(outlineData));
      if (frontendAutoReviewEnabled) {
        await runInitialFrontendReviewPass(mergedSlides, data.result_path || resultPath || '');
      }
      if (requiredPoints > 0) {
        await consumeQuotaForAction(
          'paper2ppt',
          requiredPoints,
          `可编辑版 PPT 页面已生成，但 ${requiredPoints} 点扣费记录失败，请刷新余额确认。`,
        );
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : '可编辑版 PPT 生成失败';
      setError(message);
      setFrontendSlides(
        pendingSlides.map((slide) =>
          slide.status === 'done' ? slide : { ...slide, status: 'pending' as const },
        ),
      );
    } finally {
      setGenerateTaskMessage('');
      setIsGenerating(false);
    }
  };

  const handleConfirmOutline = async () => {
    try {
      if (isRefiningOutline || isGenerating || isOutlineSubmitLocked || outlineSubmitGuardRef.current) {
        return;
      }
      outlineSubmitGuardRef.current = true;
      setIsOutlineSubmitLocked(true);

      if (pptMode === 'frontend') {
        await handleConfirmFrontendOutline();
        return;
      }

      const unchangedIndices = getUnchangedPageIndices(outlineData, confirmedOutlineSnapshot);
      const hasExistingResults = generateResults.some((result) => result.status === 'done' && result.afterImage);
      const skipPages = hasExistingResults ? unchangedIndices : [];
      const pagesToGenerate = outlineData.length - skipPages.length;
      const requiredPoints = pagesToGenerate;

      if (
        requiredPoints > 0 &&
        !(await ensureQuotaForAction(requiredPoints, `批量生成 ${pagesToGenerate} 页 PPT`))
      ) {
        return;
      }
      setCurrentStep('generate');
      setCurrentSlideIndex(0);
      setIsGenerating(true);
      if (skipPages.length > 0) {
        setGenerateTaskMessage(`复用 ${skipPages.length} 页未修改内容，重新生成 ${pagesToGenerate} 页...`);
      } else {
        setGenerateTaskMessage('');
      }
      setError(null);

      const skipSet = new Set(skipPages);
      const results: GenerateResult[] = outlineData.map((slide, index) => {
        if (skipSet.has(index) && index < generateResults.length && generateResults[index].status === 'done') {
          return { ...generateResults[index] };
        }
        return {
          slideId: slide.id,
          beforeImage: slide.asset_ref || '',
          beforeImagePreview: slide.asset_ref_preview_path || slide.asset_ref || '',
          afterImage: '',
          afterImagePreview: '',
          status: 'processing' as const,
          versionHistory: [],
          currentVersionIndex: -1,
        };
      });
      setGenerateResults(results);
      
      try {
        const formData = new FormData();
        formData.append('img_gen_model_name', genFigModel);
        formData.append('credential_scope', MANAGED_CREDENTIAL_SCOPE);
        formData.append('chat_api_url', llmApiUrl.trim());
        formData.append('api_key', apiKey.trim());
        formData.append('model', model);
        formData.append('language', language);
        formData.append('style', getEffectiveStylePrompt());
        formData.append('aspect_ratio', '16:9');
        formData.append('email', user?.id || user?.email || '');
        formData.append('result_path', resultPath || '');
        formData.append('get_down', 'false');
        if (skipPages.length > 0) {
          formData.append('skip_pages', JSON.stringify(skipPages));
        }

        // 如果用户选的是参考图模式，附加参考图，保留用户显式输入的风格提示词
        if (styleMode === 'reference' && referenceImage) {
          formData.append('reference_img', referenceImage);
          formData.set('style', globalPrompt || '');
        }

        const pagecontent = outlineData.map((slide) => ({
          title: slide.title,
          layout_description: slide.layout_description,
          key_points: slide.key_points,
          asset_ref: slide.asset_ref,
        }));
        formData.append('pagecontent', JSON.stringify(pagecontent));

        const task = await submitPaper2PptTask(formData, requiredPoints > 0 ? requiredPoints : undefined);
        if (skipPages.length > 0) {
          setGenerateTaskMessage(`复用 ${skipPages.length} 页，正在生成 ${pagesToGenerate} 页...`);
        } else {
          setGenerateTaskMessage(task.message || '批量生成任务已提交');
        }

        const data = await pollPaper2PptTask(task.task_id, (status) => {
          setGenerateTaskMessage(status.message || '正在生成页面');
        });

        if (data.result_path) {
          setResultPath(data.result_path);
        }

        const updatedResults = results.map((result, index) => {
          if (skipSet.has(index) && result.status === 'done' && result.afterImage) {
            return result;
          }
          const pageNumStr = String(index).padStart(3, '0');
          let afterImage = '';
          let afterImagePreview = '';
          const pageMeta = Array.isArray(data.pagecontent) ? data.pagecontent[index] : null;
          
          if (data.all_output_files && Array.isArray(data.all_output_files)) {
            const pageImg = data.all_output_files.find((url: string) => 
              url.includes(`ppt_pages/page_${pageNumStr}.png`)
            );
            if (pageImg) {
              afterImage = pageImg;
            }
          }
          afterImagePreview =
            getPreviewPath(pageMeta, 'generated_img_path')
            || getPreviewPath(pageMeta, 'asset_ref')
            || afterImage;
          
          return {
            ...result,
            afterImage,
            afterImagePreview,
            status: 'done' as const,
          };
        });
        
        preloadGeneratedImages(data.all_output_files);
        
        setGenerateResults(updatedResults);
        setConfirmedOutlineSnapshot(cloneOutlineSnapshot(outlineData));
        if (requiredPoints > 0) {
          await consumeQuotaForAction(
            'paper2ppt',
            requiredPoints,
            `PPT 页面已生成，但 ${requiredPoints} 点扣费记录失败，请刷新余额确认。`,
          );
        }
        
      } catch (err) {
        const message = err instanceof Error ? err.message : '服务器繁忙，请稍后再试';
        setError(message);
        setGenerateResults(results.map((result) => (
          result.status === 'done' ? result : { ...result, status: 'pending' as const }
        )));
      } finally {
        setGenerateTaskMessage('');
        setIsGenerating(false);
      }
    } finally {
      releaseOutlineSubmitGuard();
    }
  };

  // ============== 版本历史相关函数 ==============
  const convertToHttpUrl = (path: string): string => {
    // 如果已经是HTTP URL，直接返回
    if (path.startsWith('http://') || path.startsWith('https://')) {
      return path;
    }

    // 如果是文件系统路径，转换为HTTP URL
    // 例如：/data/users/.../outputs/xxx/yyy.png -> http://localhost:9090/outputs/xxx/yyy.png
    const outputsIndex = path.indexOf('/outputs/');
    if (outputsIndex !== -1) {
      const relativePath = path.substring(outputsIndex);
      // 使用当前页面的协议和主机
      const baseUrl = window.location.origin.replace(':3005', ':9090');
      return `${baseUrl}${relativePath}`;
    }

    // 如果无法转换，返回原路径
    console.warn('[convertToHttpUrl] 无法转换路径:', path);
    return path;
  };

  const fetchVersionHistory = async (pageIndex: number) => {
    if (!resultPath) return;

    try {
      const encodedPath = btoa(resultPath);
      const res = await backendFetch(`/api/v1/paper2ppt/version-history/${encodedPath}/${pageIndex}`);

      if (!res.ok) return;

      const data = await res.json();
      if (data.success && data.versions) {
        setGenerateResults(prev => prev.map((result, idx) =>
          idx === pageIndex
            ? {
                ...result,
                versionHistory: data.versions.map((v: any) => ({
                  versionNumber: v.version,
                  imageUrl: convertToHttpUrl(v.imageUrl), // 转换文件系统路径为HTTP URL
                  prompt: v.prompt,
                  timestamp: v.timestamp,
                  isCurrentVersion: v.version === data.versions.length
                })),
                currentVersionIndex: data.versions.length - 1
              }
            : result
        ));
      }
    } catch (err) {
      console.error('Failed to fetch version history:', err);
    }
  };

  const handleRevertToVersion = async (versionNumber: number) => {
    if (!resultPath) {
      setError('缺少 result_path');
      return;
    }

    setIsGenerating(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('result_path', resultPath);
      formData.append('page_id', String(currentSlideIndex));
      formData.append('target_version', String(versionNumber));

      const res = await backendFetch('/api/v1/paper2ppt/revert-version', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) throw new Error('恢复版本失败');

      const data = await res.json();

      if (data.success) {
        const updatedResults = [...generateResults];
        updatedResults[currentSlideIndex] = {
          ...updatedResults[currentSlideIndex],
          afterImage: data.currentImageUrl + '?t=' + Date.now(),
          afterImagePreview: data.currentImageUrl + '?t=' + Date.now(),
          currentVersionIndex: versionNumber - 1,
        };
        setGenerateResults(updatedResults);

        // 不需要重新获取版本历史，因为版本历史不会改变
        // 只是切换了当前显示的版本
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : '恢复版本失败';
      setError(message);
    } finally {
      setIsGenerating(false);
    }
  };

  const updateFrontendSlideReview = (
    slideIndex: number,
    review: FrontendSlide['review'],
  ) => {
    setFrontendSlides((prev) =>
      prev.map((slide, index) => (index === slideIndex ? { ...slide, review } : slide)),
    );
  };

  const saveCurrentSlideEdits = (layoutDescription: string, keyPoints: string[]) => {
    setOutlineData((prev) =>
      prev.map((slide, slideIndex) =>
        slideIndex !== currentSlideIndex
          ? slide
          : {
              ...slide,
              layout_description: layoutDescription,
              key_points: keyPoints.length > 0 ? keyPoints : [''],
            },
      ),
    );
  };

  const regenerateFrontendSlideWithPrompt = async ({
    slideIndex,
    prompt,
    quotaAction,
    quotaWarningMessage,
    progressMessage,
    clearManualPrompt,
    slideOverride,
  }: {
    slideIndex: number;
    prompt: string;
    quotaAction: string;
    quotaWarningMessage: string;
    progressMessage: string;
    clearManualPrompt?: boolean;
    slideOverride?: FrontendSlide;
  }) => {
    if (!resultPath) {
      setError('缺少 result_path，请重新上传文件');
      return false;
    }
    if (!prompt.trim()) {
      setError('请输入重新生成的提示词');
      return false;
    }
    if (!(await ensureQuotaForAction(1, quotaAction))) {
      return false;
    }

    const slideSnapshot = slideOverride || frontendSlides[slideIndex];
    if (!slideSnapshot) {
      setError('当前前端页面不存在');
      return false;
    }

    setIsGenerating(true);
    setGenerateTaskMessage(progressMessage);
    setError(null);

    setFrontendSlides((prev) =>
      prev.map((slide, index) =>
        index === slideIndex
          ? {
              ...slide,
              status: 'processing',
              review: slide.review
                ? {
                    ...slide.review,
                    status: slide.review.status === 'idle' ? 'idle' : 'repairing',
                  }
                : slide.review,
            }
          : slide,
      ),
    );

    try {
      const { updatedSlide, nextTheme } = await requestFrontendSlideGeneration({
        slideIndex,
        prompt,
        resultPathValue: resultPath,
        slideSnapshot,
      });
      setFrontendSlides((prev) =>
        prev.map((slide, index) =>
          index === slideIndex
            ? {
                ...updatedSlide,
                review: {
                  status: 'idle',
                  summary: '',
                  issues: [],
                },
              }
            : slide,
        ),
      );
      if (nextTheme) {
        setFrontendDeckTheme(nextTheme);
      }
      if (clearManualPrompt && slideIndex === currentSlideIndex) {
        setSlidePrompt('');
      }
      await consumeQuotaForAction(
        'paper2ppt',
        1,
        quotaWarningMessage,
      );
      return true;
    } catch (err) {
      const message = err instanceof Error ? err.message : '前端页面重生成失败';
      setError(message);
      setFrontendSlides((prev) =>
        prev.map((slide, index) =>
          index === slideIndex
            ? {
                ...slide,
                status: 'done',
                review: slide.review && slide.review.status === 'repairing'
                  ? { ...slide.review, status: 'needs_repair' }
                  : slide.review,
              }
            : slide,
        ),
      );
      return false;
    } finally {
      setGenerateTaskMessage('');
      setIsGenerating(false);
    }
  };

  // ============== Step 3: 重新生成单页 ==============
  const handleRegenerateFrontendSlide = async () => {
    if (!slidePrompt.trim()) {
      setError('请输入重新生成的提示词');
      return;
    }
    await regenerateFrontendSlideWithPrompt({
      slideIndex: currentSlideIndex,
      prompt: slidePrompt.trim(),
      quotaAction: '重新生成当前前端页面',
      quotaWarningMessage: '前端页面已重新生成，但 1 点扣费记录失败，请刷新余额确认。',
      progressMessage: '正在重新生成当前前端页面...',
      clearManualPrompt: true,
    });
  };

  const handleDebugFrontendCodeEdit = async (htmlTemplate: string, cssCode: string) => {
    const targetIndex = currentSlideIndex;
    const currentSlide = frontendSlides[targetIndex];
    if (!currentSlide) {
      setError('当前前端页面不存在');
      return;
    }

    setError(null);
    const validation = validateFrontendSlideCode(currentSlide, htmlTemplate, cssCode);
    const draftSlide: FrontendSlide = {
      ...currentSlide,
      htmlTemplate: validation.sanitizedHtml,
      cssCode: validation.sanitizedCss,
    };

    updateFrontendSlideReview(targetIndex, {
      status: 'repairing',
      summary: '正在检查并修正当前代码...',
      issues: [...validation.issues, ...validation.warnings],
    });

    if (validation.ok) {
      const applied = applyFrontendCodeEdit(targetIndex, htmlTemplate, cssCode);
      if (applied) {
        updateFrontendSlideReview(targetIndex, {
          status: 'passed',
          summary: '当前代码已通过本地校验并成功应用。',
          issues: validation.warnings,
        });
      }
      return;
    }

    const repaired = await regenerateFrontendSlideWithPrompt({
      slideIndex: targetIndex,
      prompt: buildFrontendCodeRepairPrompt(draftSlide, validation),
      quotaAction: 'AI 调试当前前端代码',
      quotaWarningMessage: 'AI 已调试当前代码，但 1 点扣费记录失败，请刷新余额确认。',
      progressMessage: '本地校验发现代码问题，正在调用 AI 修正...',
      slideOverride: draftSlide,
    });

    if (repaired) {
      updateFrontendSlideReview(targetIndex, {
        status: 'passed',
        summary: 'AI 已根据代码问题完成修正。',
        issues: [],
      });
    } else {
      updateFrontendSlideReview(targetIndex, {
        status: 'needs_repair',
        summary: 'AI 调试未成功，请继续修改代码或重新生成当前页。',
        issues: validation.issues,
      });
    }
  };

  const handleReviewFrontendSlide = async () => {
    if (!resultPath) {
      setError('缺少 result_path，请重新上传文件');
      return;
    }

    const targetIndex = currentSlideIndex;
    const currentSlide = frontendSlides[targetIndex];
    const node = frontendCaptureRefs.current[targetIndex];

    if (!currentSlide) {
      setError('当前前端页面不存在');
      return;
    }
    if (!node) {
      setError('当前页面尚未渲染完成，请稍后重试');
      return;
    }

    setGenerateTaskMessage('当前页正在进行视觉检查，请稍候，检查完成后“确认并继续”会自动恢复可点击状态。');
    setIsReviewingFrontendSlide(true);
    setError(null);
    updateFrontendSlideReview(targetIndex, {
      status: 'repairing',
      summary: '正在检查当前页面的视觉版式...',
      issues: [],
    });

    try {
      await sleep(40);
      const localLayoutCheck = inspectSlideLayout(node, 1600, 900);
      const blob = await captureSlideToPngBlob(node, 1600, 900);
      const data = await requestFrontendSlideReview({
        slide: currentSlide,
        resultPathValue: resultPath,
        layoutIssues: localLayoutCheck.issues,
        screenshot: blob,
      });
      const reviewIssues = Array.isArray(data.issues)
        ? data.issues.map((item: unknown) => String(item || '').trim()).filter(Boolean)
        : [];
      const reviewSummary = typeof data.summary === 'string' && data.summary.trim()
        ? data.summary.trim()
        : (data.passed ? '未发现明显视觉问题。' : '检测到需要修复的版式问题。');

      if (data.passed) {
        updateFrontendSlideReview(targetIndex, {
          status: 'passed',
          summary: reviewSummary,
          issues: reviewIssues,
        });
        return;
      }

      updateFrontendSlideReview(targetIndex, {
        status: 'needs_repair',
        summary: reviewSummary,
        issues: reviewIssues,
      });

      const repairPrompt = typeof data.repair_prompt === 'string' ? data.repair_prompt.trim() : '';
      if (!repairPrompt) {
        setError('视觉检查发现问题，但未返回可执行的修复指令');
        return;
      }

      const repaired = await regenerateFrontendSlideWithPrompt({
        slideIndex: targetIndex,
        prompt: repairPrompt,
        quotaAction: '视觉检查后修复当前前端页面',
        quotaWarningMessage: '视觉检查已触发自动修复，但 1 点扣费记录失败，请刷新余额确认。',
        progressMessage: '视觉检查发现问题，正在自动修复当前页面...',
      });

      if (repaired) {
        updateFrontendSlideReview(targetIndex, {
          status: 'passed',
          summary: '视觉检查已完成，并根据问题自动修复当前页面。',
          issues: [],
        });
      } else {
        updateFrontendSlideReview(targetIndex, {
          status: 'needs_repair',
          summary: '视觉检查发现问题，但自动修复失败，请根据提示词继续调整。',
          issues: reviewIssues,
        });
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : '前端页面视觉检查失败';
      setError(message);
      updateFrontendSlideReview(targetIndex, {
        status: 'needs_repair',
        summary: '视觉检查失败，请稍后重试。',
        issues: [],
      });
    } finally {
      setIsReviewingFrontendSlide(false);
      setGenerateTaskMessage('');
    }
  };

  const handleRegenerateSlideFromOutline = async () => {
    if (!resultPath) {
      setError('缺少 result_path，请重新上传文件');
      return;
    }
    if (!(await ensureQuotaForAction(1, '按当前页面内容重新生成'))) {
      return;
    }

    setIsGenerating(true);
    setGenerateTaskMessage('正在按当前页面内容重新生成...');
    setError(null);

    const updatedResults = [...generateResults];
    updatedResults[currentSlideIndex] = {
      ...updatedResults[currentSlideIndex],
      status: 'processing',
    };
    setGenerateResults(updatedResults);

    try {
      const formData = new FormData();
      formData.append('img_gen_model_name', genFigModel);
      formData.append('credential_scope', MANAGED_CREDENTIAL_SCOPE);
      formData.append('chat_api_url', llmApiUrl.trim());
      formData.append('api_key', apiKey.trim());
      formData.append('model', model);
      formData.append('language', language);
      formData.append('style', getEffectiveStylePrompt());
      formData.append('aspect_ratio', '16:9');
      formData.append('email', user?.id || user?.email || '');
      formData.append('result_path', resultPath);
      formData.append('get_down', 'true');
      formData.append('page_id', String(currentSlideIndex));
      formData.append('regenerate_from_outline', 'true');

      if (styleMode === 'reference' && referenceImage) {
        formData.append('reference_img', referenceImage);
        formData.set('style', globalPrompt || '');
      }

      formData.append('pagecontent', JSON.stringify(buildPagecontentForGeneration()));

      const res = await backendFetch('/api/v1/paper2ppt/generate', {
        method: 'POST',
        headers: { 'X-Workflow-Amount': '1' },
        body: formData,
      });

      if (!res.ok) {
        throw new Error(await extractErrorMessage(res, '服务器繁忙，请稍后再试'));
      }

      const data = await res.json();
      if (!data.success) {
        throw new Error(data.error || '服务器繁忙，请稍后再试');
      }

      const pageNumStr = String(currentSlideIndex).padStart(3, '0');
      let afterImage = updatedResults[currentSlideIndex].afterImage;
      let afterImagePreview = updatedResults[currentSlideIndex].afterImagePreview || afterImage;

      if (data.all_output_files && Array.isArray(data.all_output_files)) {
        const pageImg = data.all_output_files.find((url: string) =>
          url.includes(`ppt_pages/page_${pageNumStr}.png`)
        );
        if (pageImg) {
          afterImage = `${pageImg}?t=${Date.now()}`;
        }
      }
      const pageMeta = Array.isArray(data.pagecontent) ? data.pagecontent[currentSlideIndex] : null;
      afterImagePreview =
        getPreviewPath(pageMeta, 'generated_img_path')
        || getPreviewPath(pageMeta, 'asset_ref')
        || afterImage;

      updatedResults[currentSlideIndex] = {
        ...updatedResults[currentSlideIndex],
        afterImage,
        afterImagePreview,
        status: 'done',
      };
      setGenerateResults([...updatedResults]);
      setConfirmedOutlineSnapshot((prev) => {
        const next = prev.length > 0 ? cloneOutlineSnapshot(prev) : cloneOutlineSnapshot(outlineData);
        if (currentSlideIndex < outlineData.length) {
          next[currentSlideIndex] = {
            ...outlineData[currentSlideIndex],
            key_points: [...outlineData[currentSlideIndex].key_points],
          };
        }
        return next;
      });
      await fetchVersionHistory(currentSlideIndex);
      await consumeQuotaForAction(
        'paper2ppt',
        1,
        '页面已按当前内容重新生成，但 1 点扣费记录失败，请刷新余额确认。',
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : '服务器繁忙，请稍后再试';
      setError(message);
      updatedResults[currentSlideIndex] = {
        ...updatedResults[currentSlideIndex],
        status: 'done',
      };
      setGenerateResults(updatedResults);
    } finally {
      setGenerateTaskMessage('');
      setIsGenerating(false);
    }
  };

  const handleRegenerateSlide = async () => {
    if (pptMode === 'frontend') {
      await handleRegenerateFrontendSlide();
      return;
    }
    if (!resultPath) {
      setError('缺少 result_path，请重新上传文件');
      return;
    }
    
    if (!slidePrompt.trim()) {
      setError('请输入重新生成的提示词');
      return;
    }
    if (!(await ensureQuotaForAction(1, '重新生成当前页面'))) {
      return;
    }
    
    setIsGenerating(true);
    setError(null);
    
    const updatedResults = [...generateResults];
    updatedResults[currentSlideIndex] = { 
      ...updatedResults[currentSlideIndex], 
      status: 'processing',
      userPrompt: slidePrompt,
    };
    setGenerateResults(updatedResults);
    
    try {
      const formData = new FormData();
      formData.append('img_gen_model_name', genFigModel);
      formData.append('credential_scope', MANAGED_CREDENTIAL_SCOPE);
      formData.append('chat_api_url', llmApiUrl.trim());
      formData.append('api_key', apiKey.trim());
      formData.append('model', model);
      formData.append('language', language);
      formData.append('style', getEffectiveStylePrompt());
      formData.append('aspect_ratio', '16:9');
      formData.append('email', user?.id || user?.email || '');
      formData.append('result_path', resultPath);
      formData.append('get_down', 'true');
      formData.append('page_id', String(currentSlideIndex));
      formData.append('edit_prompt', slidePrompt);

      // 如果用户选的是参考图模式，附加参考图，保留用户显式输入的风格提示词
      if (styleMode === 'reference' && referenceImage) {
        formData.append('reference_img', referenceImage);
        formData.set('style', globalPrompt || '');
      }

      formData.append('pagecontent', JSON.stringify(buildPagecontentForGeneration()));

      const res = await backendFetch('/api/v1/paper2ppt/generate', {
        method: 'POST',
        body: formData,
      });
      
      if (!res.ok) {
        throw new Error(await extractErrorMessage(res, '服务器繁忙，请稍后再试'));
      }

      const data = await res.json();

      if (!data.success) {
        throw new Error(data.error || '服务器繁忙，请稍后再试');
      }

      const pageNumStr = String(currentSlideIndex).padStart(3, '0');
      let afterImage = updatedResults[currentSlideIndex].afterImage;
      let afterImagePreview = updatedResults[currentSlideIndex].afterImagePreview || afterImage;
      
      if (data.all_output_files && Array.isArray(data.all_output_files)) {
        const pageImg = data.all_output_files.find((url: string) => 
          url.includes(`ppt_pages/page_${pageNumStr}.png`)
        );
        if (pageImg) {
          afterImage = pageImg + '?t=' + Date.now();
        }
      }
      const pageMeta = Array.isArray(data.pagecontent) ? data.pagecontent[currentSlideIndex] : null;
      afterImagePreview =
        getPreviewPath(pageMeta, 'generated_img_path')
        || getPreviewPath(pageMeta, 'asset_ref')
        || afterImage;
      
      updatedResults[currentSlideIndex] = {
        ...updatedResults[currentSlideIndex],
        afterImage,
        afterImagePreview,
        status: 'done',
      };
      setGenerateResults([...updatedResults]);
      setSlidePrompt('');

      // 获取更新的版本历史
      await fetchVersionHistory(currentSlideIndex);
      await consumeQuotaForAction(
        'paper2ppt',
        1,
        '页面已重新生成，但 1 点扣费记录失败，请刷新余额确认。',
      );

    } catch (err) {
      const message = err instanceof Error ? err.message : '服务器繁忙，请稍后再试';
      setError(message);
      updatedResults[currentSlideIndex] = { 
        ...updatedResults[currentSlideIndex], 
        status: 'done',
      };
      setGenerateResults([...updatedResults]);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleConfirmSlide = () => {
    setError(null);
    if (currentSlideIndex < outlineData.length - 1) {
      const nextIndex = currentSlideIndex + 1;
      setCurrentSlideIndex(nextIndex);
      setSlidePrompt('');
    } else {
      setCurrentStep('complete');
    }
  };

  // ============== Step 4: 完成处理 ==============
  const handleGenerateFrontendFinal = async () => {
    if (!resultPath) {
      setError('缺少 result_path');
      return;
    }
    if (frontendSlides.length === 0) {
      setError('当前没有可导出的前端页面');
      return;
    }

    setIsGeneratingFinal(true);
    setFinalTaskMessage('正在准备可编辑 PPTX...');
    setError(null);

    try {
      setFinalTaskMessage('正在解析 Canvas 布局...');
      await sleep(180);
      const slidesForExport = normalizeFrontendSlides(
        frontendSlides.map((slide) => serializeFrontendSlide(slide)),
      );
      if (canExportCanvasSlidesToPptx(slidesForExport)) {
        setFinalTaskMessage('正在生成可编辑 PPTX...');
        const pptxBlob = await buildCanvasSlidesPptxBlob(slidesForExport, frontendDeckTheme);
        const objectUrl = URL.createObjectURL(pptxBlob);
        setDownloadUrl((previousUrl) => {
          if (previousUrl?.startsWith('blob:')) {
            URL.revokeObjectURL(previousUrl);
          }
          return objectUrl;
        });
        setPdfPreviewUrl(null);
        await uploadAndSaveFile(pptxBlob, 'paper2ppt_editable.pptx', 'paper2ppt');
        setFinalTaskMessage('');
        return;
      }

      throw new Error('当前页面不是完整 Canvas schema，已停止导出，避免生成图片型 PPTX。请重新生成或稍后重试。');
    } catch (err) {
      const message = err instanceof Error ? err.message : '可编辑版 PPT 导出失败';
      setError(message);
    } finally {
      setFinalTaskMessage('');
      setIsGeneratingFinal(false);
    }
  };

  const handleGenerateFinal = async () => {
    if (pptMode === 'frontend') {
      await handleGenerateFrontendFinal();
      return;
    }
    if (!resultPath) {
      setError('缺少 result_path');
      return;
    }
    
    setIsGeneratingFinal(true);
    setFinalTaskMessage('');
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('img_gen_model_name', genFigModel);
      formData.append('credential_scope', MANAGED_CREDENTIAL_SCOPE);
      formData.append('chat_api_url', llmApiUrl.trim());
      formData.append('api_key', apiKey.trim());
      formData.append('model', model);
      formData.append('language', language);
      formData.append('style', getEffectiveStylePrompt());
      formData.append('aspect_ratio', '16:9');
      formData.append('email', user?.id || user?.email || '');
      formData.append('result_path', resultPath);
      formData.append('get_down', 'false');
      formData.append('all_edited_down', 'true');

      // 如果用户选的是参考图模式，附加参考图，保留用户显式输入的风格提示词
      if (styleMode === 'reference' && referenceImage) {
        formData.append('reference_img', referenceImage);
        formData.set('style', globalPrompt || '');
      }

      const pagecontent = outlineData.map((slide) => ({
        title: slide.title,
        layout_description: slide.layout_description,
        key_points: slide.key_points,
        asset_ref: slide.asset_ref,
      }));
      formData.append('pagecontent', JSON.stringify(pagecontent));

      const task = await submitPaper2PptTask(formData);
      setFinalTaskMessage(task.message || '最终导出任务已提交');

      const data = await pollPaper2PptTask(task.task_id, (status) => {
        setFinalTaskMessage(status.message || '正在生成最终文件');
      });

      // 优先使用后端直接返回的路径
      if (data.ppt_pptx_path) {
        setDownloadUrl(data.ppt_pptx_path);
      }
      if (data.ppt_pdf_path) {
        setPdfPreviewUrl(data.ppt_pdf_path);
      }
      
      // 备选：从 all_output_files 中查找
      if (data.all_output_files && Array.isArray(data.all_output_files)) {
        if (!data.ppt_pptx_path) {
          const pptxFile = data.all_output_files.find((url: string) => 
            url.endsWith('.pptx') || url.includes('editable.pptx')
          );
          if (pptxFile) {
            setDownloadUrl(pptxFile);
          }
        }
        if (!data.ppt_pdf_path) {
          const pdfFile = data.all_output_files.find((url: string) =>
            url.endsWith('.pdf') && !url.includes('input')
          );
          if (pdfFile) {
            setPdfPreviewUrl(pdfFile);
          }
        }
      }

      // 校验是否有有效的输出文件
      const hasOutput = data.ppt_pptx_path || data.ppt_pdf_path ||
        (data.all_output_files && data.all_output_files.some((url: string) =>
          url.endsWith('.pptx') || (url.endsWith('.pdf') && !url.includes('input'))
        ));
      if (!hasOutput) {
        throw new Error('生成失败：未能获取到有效的文件，请检查 API Key 余额后重试');
      }

      // Upload generated file to Supabase Storage (either PPTX or PDF)
      let filePath = data.ppt_pptx_path || (data.all_output_files?.find((url: string) =>
        url.endsWith('.pptx') || url.includes('editable.pptx')
      ));
      let defaultName = 'paper2ppt_result.pptx';

      if (!filePath) {
        filePath = data.ppt_pdf_path || (data.all_output_files?.find((url: string) =>
          url.endsWith('.pdf') && !url.includes('input')
        ));
        defaultName = 'paper2ppt_result.pdf';
      }

      await uploadGeneratedResultFile(filePath, defaultName);

    } catch (err) {
      const message = err instanceof Error ? err.message : '服务器繁忙，请稍后再试';
      setError(message);
    } finally {
      setFinalTaskMessage('');
      setIsGeneratingFinal(false);
    }
  };

  const handleDownloadPdf = () => {
    if (!pdfPreviewUrl) return;
    window.open(pdfPreviewUrl, '_blank');
  };

  const handleDownloadPptx = async () => {
    if (!downloadUrl) {
      setError('下载链接不存在');
      return;
    }

    try {
      if (downloadUrl.startsWith('blob:')) {
        const a = document.createElement('a');
        a.href = downloadUrl;
        a.download = 'paper2ppt_editable.pptx';
        document.body.appendChild(a);
        a.click();
        a.remove();
        return;
      }
      const res = await fetch(downloadUrl);
      if (!res.ok) {
        throw new Error('下载失败');
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'paper2ppt_editable.pptx';
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      const message = err instanceof Error ? err.message : '服务器繁忙，请稍后再试';
      setError(message);
    }
  };

  const handleReset = () => {
    setCurrentStep('upload');
    setSelectedFile(null);
    setOutlineData([]);
    setConfirmedOutlineSnapshot([]);
    setGenerateResults([]);
    setFrontendSlides([]);
    setFrontendDeckTheme(null);
    setDownloadUrl((previousUrl) => {
      if (previousUrl?.startsWith('blob:')) {
        URL.revokeObjectURL(previousUrl);
      }
      return null;
    });
    setPdfPreviewUrl(null);
    setResultPath(null);
    setError(null);
    setProgress(0);
    setProgressStatus('');
    setGenerateTaskMessage('');
    setFinalTaskMessage('');
    setIsReviewingFrontendSlide(false);
    frontendCaptureRefs.current = [];
  };

  return (
    <div className="w-full h-screen flex flex-col bg-[#050512] overflow-hidden">
      <Banner show={showBanner} onClose={() => setShowBanner(false)} stars={stars} />

      <div className="flex-1 overflow-auto">
        <div className="max-w-7xl mx-auto px-6 py-8 pb-24">
          <StepIndicator currentStep={currentStep} />
          
          {currentStep === 'upload' && (
            <UploadStep
              pptMode={pptMode}
              setPptMode={setPptMode}
              modeLocked={modeLocked}
              uploadMode={uploadMode} setUploadMode={setUploadMode}
              textContent={textContent} setTextContent={setTextContent}
              selectedFile={selectedFile}
              isDragOver={isDragOver} setIsDragOver={setIsDragOver}
              styleMode={styleMode} setStyleMode={setStyleMode}
              stylePreset={stylePreset} setStylePreset={setStylePreset}
              globalPrompt={globalPrompt} setGlobalPrompt={setGlobalPrompt}
              referenceImage={referenceImage} referenceImagePreview={referenceImagePreview}
              isUploading={isUploading} isValidating={isValidating}
              isUploadSubmitLocked={isUploadSubmitLocked}
              pageCount={pageCount} setPageCount={setPageCount}
              useLongPaper={useLongPaper} setUseLongPaper={setUseLongPaper}
              frontendIncludeImages={frontendIncludeImages}
              setFrontendIncludeImages={setFrontendIncludeImages}
              frontendAutoReviewEnabled={frontendAutoReviewEnabled}
              setFrontendAutoReviewEnabled={setFrontendAutoReviewEnabled}
              frontendImageStyle={frontendImageStyle}
              setFrontendImageStyle={setFrontendImageStyle}
              progress={progress} progressStatus={progressStatus}
              error={error}
              purchaseUrl={purchaseUrl}
              showApiConfig={userApiConfigRequired}
              llmApiUrl={llmApiUrl} setLlmApiUrl={setLlmApiUrl}
              apiKey={apiKey} setApiKey={setApiKey}
              model={model} setModel={setModel}
              genFigModel={genFigModel} setGenFigModel={setGenFigModel}
              language={language} setLanguage={setLanguage}
              handleFileChange={handleFileChange}
              handleDrop={handleDrop}
              handleReferenceImageChange={handleReferenceImageChange}
              handleRemoveReferenceImage={handleRemoveReferenceImage}
              handleUploadAndParse={handleUploadAndParse}
            />
          )}
          
      {currentStep === 'outline' && (
        <OutlineStep
          outlineData={outlineData}
          editingId={editingId}
          editContent={editContent}
          setEditContent={setEditContent}
          handleEditStart={handleEditStart}
          handleEditSave={handleEditSave}
          handleEditCancel={handleEditCancel}
          handleKeyPointChange={handleKeyPointChange}
          handleAddKeyPoint={handleAddKeyPoint}
          handleRemoveKeyPoint={handleRemoveKeyPoint}
          handleDeleteSlide={handleDeleteSlide}
          handleAddSlide={handleAddSlide}
          handleMoveSlide={handleMoveSlide}
          handleConfirmOutline={handleConfirmOutline}
          handleRefineOutline={handleRefineOutline}
          setCurrentStep={setCurrentStep}
          error={error}
          outlineFeedback={outlineFeedback}
          setOutlineFeedback={setOutlineFeedback}
          isRefiningOutline={isRefiningOutline}
          isGenerating={isGenerating || isOutlineSubmitLocked}
        />
      )}
          
          {currentStep === 'generate' && (
            pptMode === 'frontend' ? (
              <FrontendGenerateStep
                outlineData={outlineData}
                frontendSlides={frontendSlides}
                deckTheme={frontendDeckTheme}
                currentSlideIndex={currentSlideIndex}
                setCurrentSlideIndex={setCurrentSlideIndex}
                isGenerating={isGenerating}
                taskMessage={generateTaskMessage}
                slidePrompt={slidePrompt}
                setSlidePrompt={setSlidePrompt}
                handleRegenerateSlide={handleRegenerateSlide}
                handleReviewSlide={handleReviewFrontendSlide}
                applyCodeEdit={(htmlTemplate, cssCode) =>
                  applyFrontendCodeEdit(currentSlideIndex, htmlTemplate, cssCode)
                }
                handleDebugCodeEdit={handleDebugFrontendCodeEdit}
                handleConfirmSlide={handleConfirmSlide}
                setCurrentStep={setCurrentStep}
                error={error}
                isReviewing={isReviewingFrontendSlide}
                updateFieldValue={updateFrontendFieldValue}
                updateListItem={updateFrontendListItem}
                replaceListItems={replaceFrontendListItems}
                addListItem={addFrontendListItem}
                removeListItem={removeFrontendListItem}
                replaceVisualAsset={replaceFrontendVisualAsset}
                insertTextBlock={insertFrontendTextBlock}
                insertCalloutBlock={insertFrontendCalloutBlock}
                insertTableBlock={insertFrontendTableBlock}
                insertImageBlock={insertFrontendImageBlock}
                updateLayoutIr={updateFrontendLayoutIr}
              />
            ) : (
              <GenerateStep
                outlineData={outlineData}
                currentSlideIndex={currentSlideIndex}
                setCurrentSlideIndex={setCurrentSlideIndex}
                generateResults={generateResults}
                isGenerating={isGenerating}
                taskMessage={generateTaskMessage}
                slidePrompt={slidePrompt}
                setSlidePrompt={setSlidePrompt}
                saveCurrentSlideEdits={saveCurrentSlideEdits}
                handleRegenerateSlideFromOutline={handleRegenerateSlideFromOutline}
                handleRegenerateSlide={handleRegenerateSlide}
                handleConfirmSlide={handleConfirmSlide}
                setCurrentStep={setCurrentStep}
                error={error}
                handleRevertToVersion={handleRevertToVersion}
              />
            )
          )}
          
          {currentStep === 'complete' && (
            pptMode === 'frontend' ? (
              <FrontendCompleteStep
                slides={frontendSlides}
                deckTheme={frontendDeckTheme}
                downloadUrl={downloadUrl}
                pdfPreviewUrl={pdfPreviewUrl}
                isGeneratingFinal={isGeneratingFinal}
                taskMessage={finalTaskMessage}
                handleGenerateFinal={handleGenerateFinal}
                handleDownloadPptx={handleDownloadPptx}
                handleDownloadPdf={handleDownloadPdf}
                handleReset={handleReset}
                error={error}
              />
            ) : (
              <CompleteStep
                outlineData={outlineData}
                generateResults={generateResults}
                downloadUrl={downloadUrl}
                pdfPreviewUrl={pdfPreviewUrl}
                isGeneratingFinal={isGeneratingFinal}
                taskMessage={finalTaskMessage}
                handleGenerateFinal={handleGenerateFinal}
                handleDownloadPptx={handleDownloadPptx}
                handleDownloadPdf={handleDownloadPdf}
                handleReset={handleReset}
                error={error}
                handleCopyShareText={handleCopyShareText}
                copySuccess={copySuccess}
                stars={stars}
                showFreeApiPromo={userApiConfigRequired}
              />
            )
          )}
        </div>
      </div>

      {pptMode === 'frontend' && frontendSlides.length > 0 && (
        <div
          aria-hidden="true"
          style={{
            position: 'fixed',
            left: '-20000px',
            top: 0,
            width: '1600px',
            pointerEvents: 'none',
          }}
        >
          {frontendSlides.map((slide, index) => (
            <FrontendSlidePreview
              key={`${slide.slideId}-capture`}
              slide={slide}
              deckTheme={frontendDeckTheme}
              mode="capture"
              className="mb-4"
              captureRef={(node) => {
                frontendCaptureRefs.current[index] = node;
              }}
              onLayoutIrChange={(layoutIr) => updateFrontendLayoutIr(index, layoutIr)}
            />
          ))}
        </div>
      )}

      <style>{`
        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
        .animate-shimmer {
          animation: shimmer 3s infinite;
        }
        .animate-shimmer-fast {
          animation: shimmer 1.5s infinite;
        }
        .glass { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(10px); }
        .demo-input-placeholder {
          min-height: 80px;
        }
        .demo-output-placeholder {
          min-height: 80px;
        }
      `}</style>
    </div>
  );
};

export default Paper2PptPage;
