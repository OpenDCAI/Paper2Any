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
import { appendManagedApiConfig, appendManagedModel } from '../../utils/runtimeBillingForm';
import {
  buildInsufficientPointsMessage,
  buildQuotaExhaustedMessage,
  resolvePointsPurchaseUrl,
} from '../../utils/pointsMessaging';

import {
  FrontendDeckTheme,
  FrontendSlide,
  MaskSelectionSpec,
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
import { validateStructuredSlide, buildStructuredSlideRepairPrompt } from './structuredSlideModel';
import { exportStructuredSlidesToPptx } from './exportStructuredSlides';

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
  const [slideMaskSelection, setSlideMaskSelection] = useState<MaskSelectionSpec | null>(null);
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

  const extractOutlineText = (value: unknown): string => {
    if (typeof value === 'string') return value.trim();
    if (typeof value === 'number' || typeof value === 'boolean') return String(value);
    if (Array.isArray(value)) {
      return value.map((item) => extractOutlineText(item)).filter(Boolean).join(' ');
    }
    if (value && typeof value === 'object') {
      const record = value as Record<string, unknown>;
      for (const key of ['text', 'value', 'content', 'summary', 'title', 'label', 'body', 'description', 'reason', 'point']) {
        const text = extractOutlineText(record[key]);
        if (text) return text;
      }
      for (const item of Object.values(record)) {
        const text = extractOutlineText(item);
        if (text) return text;
      }
    }
    return '';
  };

  const normalizeOutlinePoints = (value: unknown): string[] => {
    const items = Array.isArray(value) ? value : [value];
    return items
      .map((item) => extractOutlineText(item))
      .map((item) => item.replace(/\s+/g, ' ').trim())
      .filter(Boolean);
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

  useEffect(() => {
    if (pptMode === 'image' && currentStep === 'generate') {
      setSlideMaskSelection(null);
    }
  }, [currentSlideIndex, currentStep, pptMode]);

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
    endpoint: string,
    formData: FormData,
    workflowAmount?: number,
  ): Promise<Paper2PPTTaskResponse> => {
    const res = await backendFetch(endpoint, {
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

  const normalizeLayoutData = (layoutData: any) => {
    if (!layoutData || typeof layoutData !== 'object') {
      return { type: 'bullets', titleKey: 'title' };
    }
    return {
      ...layoutData,
      eyebrowKey: layoutData.eyebrow_key || layoutData.eyebrowKey,
      titleKey: layoutData.title_key || layoutData.titleKey,
      footerKey: layoutData.footer_key || layoutData.footerKey,
      summaryKey: layoutData.summary_key || layoutData.summaryKey,
      subtitleKey: layoutData.subtitle_key || layoutData.subtitleKey,
      presenterKey: layoutData.presenter_key || layoutData.presenterKey,
      quoteKey: layoutData.quote_key || layoutData.quoteKey,
      bulletsKey: layoutData.bullets_key || layoutData.bulletsKey,
      takeawayKey: layoutData.takeaway_key || layoutData.takeawayKey,
      leftHeadingKey: layoutData.left_heading_key || layoutData.leftHeadingKey,
      leftBodyKey: layoutData.left_body_key || layoutData.leftBodyKey,
      leftPointsKey: layoutData.left_points_key || layoutData.leftPointsKey,
      rightHeadingKey: layoutData.right_heading_key || layoutData.rightHeadingKey,
      rightBodyKey: layoutData.right_body_key || layoutData.rightBodyKey,
      rightPointsKey: layoutData.right_points_key || layoutData.rightPointsKey,
      visualKey: layoutData.visual_key || layoutData.visualKey,
      visualCaptionKey: layoutData.visual_caption_key || layoutData.visualCaptionKey,
      leftTitleKey: layoutData.left_title_key || layoutData.leftTitleKey,
      rightTitleKey: layoutData.right_title_key || layoutData.rightTitleKey,
      cards: Array.isArray(layoutData.cards)
        ? layoutData.cards.map((card: any) => ({
            titleKey: card.title_key || card.titleKey,
            bodyKey: card.body_key || card.bodyKey,
          }))
        : [],
      timeline: Array.isArray(layoutData.timeline)
        ? layoutData.timeline.map((item: any) => ({
            labelKey: item.label_key || item.labelKey,
            bodyKey: item.body_key || item.bodyKey,
          }))
        : [],
    };
  };

  const normalizeThemeLock = (themeLock: any) => ({
    mustKeep: Array.isArray(themeLock?.must_keep || themeLock?.mustKeep)
      ? (themeLock.must_keep || themeLock.mustKeep).map((item: unknown) => String(item || '')).filter(Boolean)
      : [],
    preferredLayoutPatterns: Array.isArray(themeLock?.preferred_layout_patterns || themeLock?.preferredLayoutPatterns)
      ? (themeLock.preferred_layout_patterns || themeLock.preferredLayoutPatterns)
          .map((item: unknown) => String(item || ''))
          .filter(Boolean)
      : [],
    componentSignature: String(themeLock?.component_signature || themeLock?.componentSignature || ''),
    avoid: Array.isArray(themeLock?.avoid)
      ? themeLock.avoid.map((item: unknown) => String(item || '')).filter(Boolean)
      : [],
  });

  const normalizeTypography = (typography: any) => ({
    titleFontStack: String(typography?.title_font_stack || typography?.titleFontStack || ''),
    bodyFontStack: String(typography?.body_font_stack || typography?.bodyFontStack || ''),
    eyebrowSize: Number(typography?.eyebrow_size || typography?.eyebrowSize || 18),
    titleSize: Number(typography?.title_size || typography?.titleSize || 56),
    summarySize: Number(typography?.summary_size || typography?.summarySize || 26),
    bodySize: Number(typography?.body_size || typography?.bodySize || 24),
  });

  const normalizeFrontendSlides = (slides: any[]): FrontendSlide[] =>
    slides.map((slide: any, index: number) => ({
      slideId: String(slide.slide_id || slide.slideId || index + 1),
      pageNum: Number(slide.page_num || slide.pageNum || index + 1),
      title: slide.title || `第 ${index + 1} 页`,
      layoutType: slide.layout_type || slide.layoutType || 'bullets',
      layoutData: normalizeLayoutData(slide.layout_data || slide.layoutData || {}),
      editableFields: Array.isArray(slide.editable_fields || slide.editableFields)
        ? (slide.editable_fields || slide.editableFields).map((field: any) => ({
            key: String(field.key || ''),
            label: String(field.label || field.key || ''),
            type: field.type === 'list' || field.type === 'textarea' ? field.type : 'text',
            value: String(field.value || ''),
            items: Array.isArray(field.items) ? field.items.map((item: any) => String(item || '')) : [],
          }))
        : [],
      visualAssets: Array.isArray(slide.visual_assets || slide.visualAssets)
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
        : [],
      generationNote: slide.generation_note || slide.generationNote || '',
      status: slide.status === 'processing' || slide.status === 'pending' ? slide.status : 'done',
      review: {
        status: 'idle',
        summary: '',
        issues: [],
      },
    }));

  const normalizeFrontendDeckTheme = (theme: any): FrontendDeckTheme | null => {
    if (!theme || typeof theme !== 'object') {
      return null;
    }
    const themeLock = theme.theme_lock || theme.themeLock || {};
    return {
      themeName: String(theme.theme_name || theme.themeName || 'locked_deck_theme'),
      visualMood: String(theme.visual_mood || theme.visualMood || ''),
      styleFamily: String(theme.style_family || theme.styleFamily || 'modern') as FrontendDeckTheme['styleFamily'],
      footerText: String(theme.footer_text || theme.footerText || ''),
      sectionLabelTemplate: String(theme.section_label_template || theme.sectionLabelTemplate || ''),
      palette: {
        bg: String(theme.palette?.bg || '#0b1020'),
        panel: String(theme.palette?.panel || 'rgba(15, 23, 42, 0.92)'),
        primary: String(theme.palette?.primary || '#7dd3fc'),
        secondary: String(theme.palette?.secondary || '#38bdf8'),
        accent: String(theme.palette?.accent || '#f59e0b'),
        text: String(theme.palette?.text || '#e2e8f0'),
        muted: String(theme.palette?.muted || '#94a3b8'),
      },
      typography: normalizeTypography(theme.typography || {}),
      themeLock: normalizeThemeLock(themeLock),
    };
  };

  const serializeFrontendSlide = (slide: FrontendSlide) => ({
    slide_id: slide.slideId,
    page_num: slide.pageNum,
    title: slide.title,
    layout_type: slide.layoutType,
    layout_data: slide.layoutData,
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
    globalPrompt || getStyleDescription(stylePreset, mode);

  const getFrontendGenerationCostPerPage = () => (frontendIncludeImages ? 2 : 1);

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
    appendManagedApiConfig(formData, userApiConfigRequired, llmApiUrl, apiKey);
    appendManagedModel(formData, userApiConfigRequired, 'model', model);
    formData.append('language', language);
    formData.append('style', getEffectiveStylePrompt('frontend'));
    formData.append('email', user?.id || user?.email || '');
    formData.append('result_path', resultPathValue);
    formData.append('include_images', String(frontendIncludeImages));
    formData.append('image_style', frontendImageStyle);
    appendManagedModel(formData, userApiConfigRequired, 'image_model', genFigModel);
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
    updateFrontendSlideReview(slideIndex, {
      status: 'repairing',
      summary: '正在做结构检查...',
      issues: [],
    });

    try {
      const validation = validateStructuredSlide(slideSnapshot);
      if (validation.ok) {
        updateFrontendSlideReview(slideIndex, {
          status: 'passed',
          summary: '结构检查通过。',
          issues: [],
        });
        return true;
      }

      updateFrontendSlideReview(slideIndex, {
        status: 'repairing',
        summary: '结构检查发现问题，正在自动修正...',
        issues: validation.issues,
      });

      const { updatedSlide, nextTheme } = await requestFrontendSlideGeneration({
        slideIndex,
        prompt: buildStructuredSlideRepairPrompt(slideSnapshot, validation),
        resultPathValue,
        slideSnapshot,
      });
      const repairedValidation = validateStructuredSlide(updatedSlide);

      setFrontendSlides((prev) =>
        prev.map((slide, index) =>
          index === slideIndex
            ? {
                ...updatedSlide,
                review: {
                  status: repairedValidation.ok ? 'passed' : 'needs_repair',
                  summary: repairedValidation.ok
                    ? '结构检查已自动修正当前页。'
                    : '自动修正后仍有结构问题，请继续精简内容。',
                  issues: repairedValidation.issues,
                },
              }
            : slide,
        ),
      );

      if (nextTheme) {
        setFrontendDeckTheme(nextTheme);
      }
      return repairedValidation.ok;
    } catch (err) {
      const message = err instanceof Error ? err.message : '首轮自动结构检查失败';
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

    setGenerateTaskMessage('首轮生成完成，正在并行做结构检查与自动调整...');

    const reviewResults: boolean[] = new Array(slides.length).fill(false);
    let completed = 0;
    await runWithConcurrency(slides, 2, async (slide, index) => {
      reviewResults[index] = await autoReviewAndRepairFrontendSlide(index, slide, resultPathValue);
      completed += 1;
      setGenerateTaskMessage(`首轮结构检查进行中（${completed}/${slides.length}）...`);
    });

    const failedCount = reviewResults.filter((item) => !item).length;
    if (failedCount > 0) {
      setError(`首轮自动结构检查已完成，但仍有 ${failedCount} 页需要你手动复查。`);
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

  const uploadGeneratedResultBlob = async (blob: Blob, fileName: string) => {
    try {
      await uploadAndSaveFile(blob, fileName, 'paper2ppt');
    } catch (e) {
      console.error('[Paper2PptPage] Failed to upload generated blob:', e);
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

  const getStyleDescription = (preset: string, mode: PptGenerationMode = pptMode): string => {
    if (mode === 'frontend') {
      const frontendStyles: Record<string, string> = {
        modern: '暖白或象牙白背景，深石墨文字，赤陶强调色，克制的 keynote 学术汇报风，禁止青色玻璃拟态。',
        business: '午夜蓝或深海军蓝底色，冰灰文字，电蓝小面积强调，专业研究组汇报风，避免默认青绿色主调。',
        academic: '纸感米白背景，墨黑正文，酒红重点标注，像学术讲义与答辩 deck 的结合，禁止赛博青蓝。',
        creative: '森林绿或深橄榄主色，沙金点缀，奶油白底，组件统一且有高级研究报告气质，避免默认 cyan accent。',
      };
      return frontendStyles[preset] || frontendStyles.modern;
    }

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
        if (userApiConfigRequired) {
          await verifyLlmConnection(llmApiUrl, apiKey, import.meta.env.VITE_DEFAULT_LLM_MODEL || 'deepseek-v3.2');
        }
        setIsValidating(false);
      } catch (err) {
        setIsValidating(false);
        const message = err instanceof Error ? err.message : 'API 验证失败';
        setError(message);
        return;
      }

      setIsUploading(true);
      setError(null);
      setGenerateResults([]);
      setFrontendSlides([]);
      setFrontendDeckTheme(null);
      setDownloadUrl(null);
      setPdfPreviewUrl(null);
      setResultPath(null);
      setProgress(0);
      setProgressStatus('正在初始化...');

      if (uploadMode === 'text' && pageCount > 20 && textContent.trim().length < 200) {
        setError(`当前为文本模式，输入内容仅 ${textContent.trim().length} 个字符，不足以稳定生成 ${pageCount} 页大纲。请补充更完整的正文，或改用 Topic 模式。`);
        return;
      }

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
      appendManagedApiConfig(formData, userApiConfigRequired, llmApiUrl, apiKey);
      appendManagedModel(formData, userApiConfigRequired, 'model', model);
      formData.append('language', language);
      formData.append('style', getEffectiveStylePrompt());
      appendManagedModel(formData, userApiConfigRequired, 'gen_fig_model', genFigModel);
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
        title: extractOutlineText(item.title) || `第 ${index + 1} 页`,
        layout_description: extractOutlineText(item.layout_description) || '',
        key_points: normalizeOutlinePoints(item.key_points),
        asset_ref: extractOutlineText(item.asset_ref) || null,
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
      appendManagedApiConfig(formData, userApiConfigRequired, llmApiUrl, apiKey);
      appendManagedModel(formData, userApiConfigRequired, 'model', model);
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
        title: extractOutlineText(item.title) || `第 ${index + 1} 页`,
        layout_description: extractOutlineText(item.layout_description) || '',
        key_points: normalizeOutlinePoints(item.key_points),
        asset_ref: extractOutlineText(item.asset_ref) || null,
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
          ? {
              ...slide,
              generationNote: '当前页内容已手动编辑。',
              title: fieldKey === 'title' ? value : slide.title,
              review: {
                status: 'idle',
                summary: '',
                issues: [],
              },
              editableFields: slide.editableFields.map((field) =>
                field.key === fieldKey ? { ...field, value } : field,
              ),
            }
          : slide,
      ),
    );
  };

  const updateFrontendListItem = (slideIndex: number, fieldKey: string, itemIndex: number, value: string) => {
    setFrontendSlides((prev) =>
      prev.map((slide, idx) => {
        if (idx !== slideIndex) return slide;
        return {
          ...slide,
          generationNote: '当前页内容已手动编辑。',
          review: {
            status: 'idle',
            summary: '',
            issues: [],
          },
          editableFields: slide.editableFields.map((field) => {
            if (field.key !== fieldKey) return field;
            const nextItems = [...field.items];
            nextItems[itemIndex] = value;
            return { ...field, items: nextItems };
          }),
        };
      }),
    );
  };

  const addFrontendListItem = (slideIndex: number, fieldKey: string) => {
    setFrontendSlides((prev) =>
      prev.map((slide, idx) => {
        if (idx !== slideIndex) return slide;
        return {
          ...slide,
          generationNote: '当前页内容已手动编辑。',
          review: {
            status: 'idle',
            summary: '',
            issues: [],
          },
          editableFields: slide.editableFields.map((field) =>
            field.key === fieldKey ? { ...field, items: [...field.items, ''] } : field,
          ),
        };
      }),
    );
  };

  const replaceFrontendListItems = (slideIndex: number, fieldKey: string, items: string[]) => {
    setFrontendSlides((prev) =>
      prev.map((slide, idx) => {
        if (idx !== slideIndex) return slide;
        return {
          ...slide,
          generationNote: '当前页内容已手动编辑。',
          review: {
            status: 'idle',
            summary: '',
            issues: [],
          },
          editableFields: slide.editableFields.map((field) =>
            field.key === fieldKey ? { ...field, items } : field,
          ),
        };
      }),
    );
  };

  const removeFrontendListItem = (slideIndex: number, fieldKey: string, itemIndex: number) => {
    setFrontendSlides((prev) =>
      prev.map((slide, idx) => {
        if (idx !== slideIndex) return slide;
        return {
          ...slide,
          generationNote: '当前页内容已手动编辑。',
          review: {
            status: 'idle',
            summary: '',
            issues: [],
          },
          editableFields: slide.editableFields.map((field) =>
            field.key === fieldKey
              ? { ...field, items: field.items.filter((_, idx2) => idx2 !== itemIndex) }
              : field,
          ),
        };
      }),
    );
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
          return {
            ...slide,
            generationNote: '当前页图片已替换为用户上传版本。',
            review: {
              status: 'idle',
              summary: '',
              issues: [],
            },
            visualAssets: slide.visualAssets.map((asset) =>
              asset.key === imageKey
                ? {
                    ...asset,
                    src: String(data.asset.src || asset.src || ''),
                    previewSrc: String(data.asset.preview_src || data.asset.previewSrc || data.asset.src || asset.previewSrc || asset.src || ''),
                    originalSrc: String(data.asset.original_src || data.asset.originalSrc || data.asset.storage_path || data.asset.storagePath || asset.originalSrc || asset.storagePath || asset.src || ''),
                    alt: String(data.asset.alt || file.name || asset.alt || ''),
                    sourceType: 'upload',
                    storagePath: String(data.asset.storage_path || data.asset.storagePath || asset.storagePath || ''),
                    previewStoragePath: String(data.asset.preview_storage_path || data.asset.previewStoragePath || asset.previewStoragePath || ''),
                  }
                : asset,
            ),
          };
        }),
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : '图片上传失败';
      setError(message);
    }
  };

  const buildPendingFrontendSlide = (slide: SlideOutline, index: number): FrontendSlide => ({
    slideId: slide.id,
    pageNum: index + 1,
    title: slide.title,
    layoutType: 'bullets',
    layoutData: {
      type: 'bullets',
      eyebrowKey: 'eyebrow',
      titleKey: 'title',
      summaryKey: 'summary',
      bulletsKey: 'bullets',
      takeawayKey: 'takeaway',
      footerKey: 'footer',
    },
    editableFields: [
      { key: 'eyebrow', label: 'Eyebrow', type: 'text', value: `Slide ${String(index + 1).padStart(2, '0')}`, items: [] },
      { key: 'title', label: 'Title', type: 'text', value: slide.title, items: [] },
      { key: 'summary', label: 'Summary', type: 'textarea', value: slide.layout_description || '', items: [] },
      { key: 'bullets', label: 'Bullets', type: 'list', value: '', items: slide.key_points.length > 0 ? [...slide.key_points] : [''] },
      { key: 'takeaway', label: 'Takeaway', type: 'textarea', value: slide.key_points[0] || '', items: [] },
      { key: 'footer', label: 'Footer', type: 'text', value: frontendDeckTheme?.footerText || 'Paper2Any Structured PPT', items: [] },
    ],
    visualAssets: [],
    status: 'processing',
    generationNote: '',
    review: {
      status: 'idle',
      summary: '',
      issues: [],
    },
  });

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
      setGenerateTaskMessage(frontendIncludeImages ? '正在生成结构化 slide 与配图...' : '正在生成结构化 slide...');
    }
    setError(null);

    const skipSet = new Set(skipSlides);
    const pendingSlides: FrontendSlide[] = outlineData.map((slide, index) => {
      if (skipSet.has(index) && index < frontendSlides.length && frontendSlides[index].status === 'done') {
        return { ...frontendSlides[index] };
      }
      return buildPendingFrontendSlide(slide, index);
    });
    setFrontendSlides(pendingSlides);

    try {
      const formData = new FormData();
      formData.append('credential_scope', MANAGED_CREDENTIAL_SCOPE);
      appendManagedApiConfig(formData, userApiConfigRequired, llmApiUrl, apiKey);
      appendManagedModel(formData, userApiConfigRequired, 'model', model);
      formData.append('language', language);
      formData.append('style', getEffectiveStylePrompt('frontend'));
      formData.append('email', user?.id || user?.email || '');
      formData.append('result_path', resultPath || '');
      formData.append('include_images', String(frontendIncludeImages));
      formData.append('image_style', frontendImageStyle);
      appendManagedModel(formData, userApiConfigRequired, 'image_model', genFigModel);
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
        appendManagedModel(formData, userApiConfigRequired, 'img_gen_model_name', genFigModel);
        formData.append('credential_scope', MANAGED_CREDENTIAL_SCOPE);
        appendManagedApiConfig(formData, userApiConfigRequired, llmApiUrl, apiKey);
        appendManagedModel(formData, userApiConfigRequired, 'model', model);
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

        const task = await submitPaper2PptTask(
          '/api/v1/paper2ppt/slides/generate-task',
          formData,
          requiredPoints > 0 ? requiredPoints : undefined,
        );
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

  const handleReviewFrontendSlide = async () => {
    if (!resultPath) {
      setError('缺少 result_path，请重新上传文件');
      return;
    }

    const targetIndex = currentSlideIndex;
    const currentSlide = frontendSlides[targetIndex];

    if (!currentSlide) {
      setError('当前前端页面不存在');
      return;
    }

    setGenerateTaskMessage('当前页正在进行结构检查，请稍候，检查完成后“确认并继续”会自动恢复可点击状态。');
    setIsReviewingFrontendSlide(true);
    setError(null);
    updateFrontendSlideReview(targetIndex, {
      status: 'repairing',
      summary: '正在检查当前页面的结构约束...',
      issues: [],
    });

    try {
      const validation = validateStructuredSlide(currentSlide);
      if (validation.ok) {
        updateFrontendSlideReview(targetIndex, {
          status: 'passed',
          summary: '当前页结构检查通过。',
          issues: [],
        });
        return;
      }

      updateFrontendSlideReview(targetIndex, {
        status: 'needs_repair',
        summary: '检测到需要修复的结构问题。',
        issues: validation.issues,
      });

      const repaired = await regenerateFrontendSlideWithPrompt({
        slideIndex: targetIndex,
        prompt: buildStructuredSlideRepairPrompt(currentSlide, validation),
        quotaAction: '结构检查后修复当前前端页面',
        quotaWarningMessage: '结构检查已触发自动修复，但 1 点扣费记录失败，请刷新余额确认。',
        progressMessage: '结构检查发现问题，正在自动修复当前页面...',
      });

      if (repaired) {
        updateFrontendSlideReview(targetIndex, {
          status: 'passed',
          summary: '结构检查已完成，并根据问题自动修复当前页面。',
          issues: [],
        });
      } else {
        updateFrontendSlideReview(targetIndex, {
          status: 'needs_repair',
          summary: '结构检查发现问题，但自动修复失败，请根据提示词继续调整。',
          issues: validation.issues,
        });
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : '前端页面结构检查失败';
      setError(message);
      updateFrontendSlideReview(targetIndex, {
        status: 'needs_repair',
        summary: '结构检查失败，请稍后重试。',
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
      appendManagedModel(formData, userApiConfigRequired, 'img_gen_model_name', genFigModel);
      formData.append('credential_scope', MANAGED_CREDENTIAL_SCOPE);
      appendManagedApiConfig(formData, userApiConfigRequired, llmApiUrl, apiKey);
      appendManagedModel(formData, userApiConfigRequired, 'model', model);
      formData.append('language', language);
      formData.append('style', getEffectiveStylePrompt());
      formData.append('aspect_ratio', '16:9');
      formData.append('email', user?.id || user?.email || '');
      formData.append('result_path', resultPath);
      formData.append('regenerate_from_outline', 'true');

      if (styleMode === 'reference' && referenceImage) {
        formData.append('reference_img', referenceImage);
        formData.set('style', globalPrompt || '');
      }

      formData.append('pagecontent', JSON.stringify(buildPagecontentForGeneration()));

      const res = await backendFetch(`/api/v1/paper2ppt/slides/${currentSlideIndex}/edit`, {
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
      appendManagedModel(formData, userApiConfigRequired, 'img_gen_model_name', genFigModel);
      formData.append('credential_scope', MANAGED_CREDENTIAL_SCOPE);
      appendManagedApiConfig(formData, userApiConfigRequired, llmApiUrl, apiKey);
      appendManagedModel(formData, userApiConfigRequired, 'model', model);
      formData.append('language', language);
      formData.append('style', getEffectiveStylePrompt());
      formData.append('aspect_ratio', '16:9');
      formData.append('email', user?.id || user?.email || '');
      formData.append('result_path', resultPath);
      formData.append('edit_prompt', slidePrompt);
      if (slideMaskSelection) {
        formData.append('mask_spec', JSON.stringify(slideMaskSelection));
      }

      // 如果用户选的是参考图模式，附加参考图，保留用户显式输入的风格提示词
      if (styleMode === 'reference' && referenceImage) {
        formData.append('reference_img', referenceImage);
        formData.set('style', globalPrompt || '');
      }

      formData.append('pagecontent', JSON.stringify(buildPagecontentForGeneration()));

      const res = await backendFetch(`/api/v1/paper2ppt/slides/${currentSlideIndex}/edit`, {
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
      setSlideMaskSelection(null);
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
    if (frontendSlides.length === 0) {
      setError('当前没有可导出的前端页面');
      return;
    }

    setIsGeneratingFinal(true);
    setFinalTaskMessage('正在生成真可编辑 PPTX...');
    setError(null);

    try {
      const invalidSlides = frontendSlides
        .map((slide, index) => ({ index, validation: validateStructuredSlide(slide) }))
        .filter((item) => !item.validation.ok);
      if (invalidSlides.length > 0) {
        const first = invalidSlides[0];
        throw new Error(`第 ${first.index + 1} 页仍不满足结构导出要求：${first.validation.issues.join('；')}`);
      }

      const fileName = resultPath
        ? `${resultPath.split('/').pop() || 'paper2ppt'}_structured_editable.pptx`
        : 'paper2ppt_structured_editable.pptx';
      const exported = await exportStructuredSlidesToPptx({
        slides: frontendSlides,
        deckTheme: frontendDeckTheme,
        fileName,
      });
      if (!('blob' in exported) || !exported.blob) {
        throw new Error('前端导出未返回浏览器 Blob');
      }
      const { blob } = exported;
      if (downloadUrl?.startsWith('blob:')) {
        URL.revokeObjectURL(downloadUrl);
      }
      const objectUrl = URL.createObjectURL(blob);
      setDownloadUrl(objectUrl);
      setPdfPreviewUrl(null);
      await uploadGeneratedResultBlob(blob, fileName);
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
      appendManagedModel(formData, userApiConfigRequired, 'img_gen_model_name', genFigModel);
      formData.append('credential_scope', MANAGED_CREDENTIAL_SCOPE);
      appendManagedApiConfig(formData, userApiConfigRequired, llmApiUrl, apiKey);
      appendManagedModel(formData, userApiConfigRequired, 'model', model);
      formData.append('language', language);
      formData.append('style', getEffectiveStylePrompt());
      formData.append('aspect_ratio', '16:9');
      formData.append('email', user?.id || user?.email || '');
      formData.append('result_path', resultPath);
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

      const task = await submitPaper2PptTask('/api/v1/paper2ppt/finalize-task', formData);
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
      const a = document.createElement('a');
      a.href = downloadUrl;
      a.download = 'paper2ppt_editable.pptx';
      document.body.appendChild(a);
      a.click();
      a.remove();
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
    setDownloadUrl(null);
    setPdfPreviewUrl(null);
    setResultPath(null);
    setError(null);
    setProgress(0);
    setProgressStatus('');
    setGenerateTaskMessage('');
    setFinalTaskMessage('');
    setIsReviewingFrontendSlide(false);
    if (downloadUrl?.startsWith('blob:')) {
      URL.revokeObjectURL(downloadUrl);
    }
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
                slideMaskSelection={slideMaskSelection}
                setSlideMaskSelection={setSlideMaskSelection}
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
