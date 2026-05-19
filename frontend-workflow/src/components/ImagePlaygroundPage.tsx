import { useEffect, useState } from 'react';
import { Flame, Sparkles, Copy, Download, Loader2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import ManagedApiNotice from './ManagedApiNotice';
import { DEFAULT_LLM_API_URL } from '../config/api';
import { DEFAULT_IMAGE_PLAYGROUND_MODEL, IMAGE_PLAYGROUND_MODELS } from '../config/models';
import { useAuthStore } from '../stores/authStore';
import { backendFetch, normalizeBackendAssetUrl } from '../services/backendClient';
import { getApiSettings, saveApiSettings } from '../services/apiSettingsService';
import { checkQuota } from '../services/quotaService';
import { useRuntimeBilling } from '../hooks/useRuntimeBilling';
import { buildInsufficientPointsMessage, buildQuotaExhaustedMessage } from '../utils/pointsMessaging';

type TemplateKey = 'research' | 'cs' | 'bio';
type ChipKey = 'method' | 'model' | 'pipeline' | 'result' | 'cover';
type ImageTextLanguage = 'en' | 'zh';
type ImageAspectRatio =
  | '1:1'
  | '2:3'
  | '3:2'
  | '3:4'
  | '4:3'
  | '4:5'
  | '5:4'
  | '9:16'
  | '16:9'
  | '21:9'
  | '1:4'
  | '4:1'
  | '1:8'
  | '8:1';
type ImageResolution = '1K' | '2K' | '4K';
type GptImageSize = '1024x1024' | '1536x1024' | '1024x1536' | '2048x2048' | '2048x1152' | '1152x2048';
type GptImageQuality = 'auto' | 'low' | 'medium' | 'high';
type BatchCount = 1 | 2 | 4 | 8 | 16;

type GeneratedImageResult = {
  index: number;
  imageUrl: string;
  previewUrl: string;
  fileName: string;
  previewFileName: string;
  variantLabel: string;
};

const STORAGE_KEY = 'image_playground_settings';
const GEMINI_FLASH_ASPECT_RATIOS: ImageAspectRatio[] = ['1:1', '2:3', '3:2', '3:4', '4:3', '4:5', '5:4', '9:16', '16:9', '21:9', '1:4', '4:1', '1:8', '8:1'];
const GEMINI_PRO_ASPECT_RATIOS: ImageAspectRatio[] = ['1:1', '2:3', '3:2', '3:4', '4:3', '4:5', '5:4', '9:16', '16:9', '21:9'];
const GEMINI_RESOLUTIONS: ImageResolution[] = ['1K', '2K', '4K'];
const GPT_IMAGE_SIZES: GptImageSize[] = ['1024x1024', '1536x1024', '1024x1536', '2048x2048', '2048x1152', '1152x2048'];
const GPT_IMAGE_QUALITIES: GptImageQuality[] = ['auto', 'low', 'medium', 'high'];
const BATCH_COUNT_OPTIONS: BatchCount[] = [1, 2, 4, 8, 16];

const MODEL_META: Record<string, { titleKey: string; descKey: string; accent: string }> = {
  'gemini-3.1-flash-image-preview': {
    titleKey: 'models.gemini31',
    descKey: 'models.gemini31Desc',
    accent: 'from-cyan-500 to-sky-500',
  },
  'gemini-3-pro-image-preview': {
    titleKey: 'models.geminiPro',
    descKey: 'models.geminiProDesc',
    accent: 'from-fuchsia-500 to-rose-500',
  },
  'gpt-image-2': {
    titleKey: 'models.gptImage2',
    descKey: 'models.gptImage2Desc',
    accent: 'from-amber-500 to-orange-500',
  },
  'gpt-image-2-all': {
    titleKey: 'models.gptImage2All',
    descKey: 'models.gptImage2AllDesc',
    accent: 'from-emerald-500 to-lime-500',
  },
};

const TEMPLATE_CARD_KEYS: TemplateKey[] = ['research', 'cs', 'bio'];
const CHIP_KEYS: ChipKey[] = ['method', 'model', 'pipeline', 'result', 'cover'];

function buildTemplatePrompt(templateKey: TemplateKey): string {
  if (templateKey === 'cs') {
    return 'Create a clean computer-science research figure. Emphasize architecture blocks, data flow, concise labels, balanced whitespace, and presentation-ready composition.';
  }
  if (templateKey === 'bio') {
    return 'Create a polished biology or medical research figure. Emphasize mechanism clarity, experimental stages, scientific annotation, legible callouts, and a calm professional palette.';
  }
  return 'Create a polished academic research illustration or infographic. Keep hierarchy clear, labels concise, composition strong, and visual style presentation-ready.';
}

function buildChipPrompt(chipKey: ChipKey): string {
  if (chipKey === 'model') return 'Prefer a model architecture style with clear module grouping and directional relations.';
  if (chipKey === 'pipeline') return 'Prefer a pipeline-style composition with explicit stages and transitions.';
  if (chipKey === 'result') return 'Prefer an infographic that highlights findings, comparisons, and takeaways.';
  if (chipKey === 'cover') return 'Prefer a bold hero-style paper cover visual with strong visual atmosphere and minimal text.';
  return 'Prefer a method overview composition that summarizes the core workflow at a glance.';
}

function buildLanguagePrompt(textLanguage: ImageTextLanguage): string {
  if (textLanguage === 'zh') {
    return 'All visible text inside the generated image must be in simplified Chinese. Do not mix English labels unless the user explicitly asks for bilingual output.';
  }
  return 'All visible text inside the generated image must be in English. Do not mix Chinese labels unless the user explicitly asks for bilingual output.';
}

function buildPrompt(
  templateKey: TemplateKey,
  selectedChips: ChipKey[],
  paperContent: string,
  extraInstructions: string,
  textLanguage: ImageTextLanguage,
): string {
  const sections = [
    buildTemplatePrompt(templateKey),
    paperContent.trim() ? `Paper content or topic:\n${paperContent.trim()}` : '',
    selectedChips.length > 0 ? `Preferred direction:\n${selectedChips.map((chip) => `- ${buildChipPrompt(chip)}`).join('\n')}` : '',
    `Text language requirement:\n${buildLanguagePrompt(textLanguage)}`,
    extraInstructions.trim() ? `Extra instructions:\n${extraInstructions.trim()}` : '',
    'Output a single high-quality research visual. Avoid watermarks, broken typography, and cluttered composition.',
  ];

  return sections.filter(Boolean).join('\n\n');
}

function normalizeBatchCount(value: unknown): BatchCount {
  const parsed = Number(value);
  return BATCH_COUNT_OPTIONS.includes(parsed as BatchCount) ? (parsed as BatchCount) : 1;
}

function normalizeResultImages(data: any): GeneratedImageResult[] {
  const rawImages = Array.isArray(data?.images) ? data.images : [];
  if (rawImages.length === 0 && data?.image_url) {
    return [{
      index: 1,
      imageUrl: normalizeBackendAssetUrl(data.image_url || ''),
      previewUrl: normalizeBackendAssetUrl(data.preview_url || data.image_url || ''),
      fileName: data.file_name || 'generated.png',
      previewFileName: data.preview_file_name || data.file_name || 'generated.png',
      variantLabel: data.variant_label || 'Variant 1',
    }];
  }
  return rawImages.map((item: any, index: number) => ({
    index: Number(item?.index) || index + 1,
    imageUrl: normalizeBackendAssetUrl(item?.image_url || ''),
    previewUrl: normalizeBackendAssetUrl(item?.preview_url || item?.image_url || ''),
    fileName: item?.file_name || `generated_${index + 1}.png`,
    previewFileName: item?.preview_file_name || item?.file_name || `generated_${index + 1}.jpg`,
    variantLabel: item?.variant_label || `Variant ${index + 1}`,
  }));
}

export default function ImagePlaygroundPage() {
  const { t } = useTranslation('imagePlayground');
  const { user, refreshQuota } = useAuthStore();
  const { runtimeConfig, userApiConfigRequired } = useRuntimeBilling();

  const [selectedModel, setSelectedModel] = useState(DEFAULT_IMAGE_PLAYGROUND_MODEL);
  const [templateKey, setTemplateKey] = useState<TemplateKey>('research');
  const [selectedChips, setSelectedChips] = useState<ChipKey[]>(['method']);
  const [paperContent, setPaperContent] = useState('');
  const [extraInstructions, setExtraInstructions] = useState('');
  const [textLanguage, setTextLanguage] = useState<ImageTextLanguage>('en');
  const [batchCount, setBatchCount] = useState<BatchCount>(1);
  const [aspectRatio, setAspectRatio] = useState<ImageAspectRatio>('16:9');
  const [resolution, setResolution] = useState<ImageResolution>('2K');
  const [gptSize, setGptSize] = useState<GptImageSize>('2048x1152');
  const [gptQuality, setGptQuality] = useState<GptImageQuality>('medium');
  const [apiUrl, setApiUrl] = useState(DEFAULT_LLM_API_URL);
  const [apiKey, setApiKey] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [billingWarning, setBillingWarning] = useState<string | null>(null);
  const [resultImages, setResultImages] = useState<GeneratedImageResult[]>([]);
  const [resultZipUrl, setResultZipUrl] = useState('');
  const [resultZipFileName, setResultZipFileName] = useState('image-playground-batch.zip');
  const [resultSuccessCount, setResultSuccessCount] = useState(0);
  const [resultBatchCount, setResultBatchCount] = useState(0);
  const [lastPrompt, setLastPrompt] = useState('');

  const imageCost = Math.max(1, Number(runtimeConfig.workflow_costs?.image_playground || 2));
  const totalCost = imageCost * batchCount;
  const promptPreview = buildPrompt(templateKey, selectedChips, paperContent, extraInstructions, textLanguage);
  const supportsGeminiControls = selectedModel === 'gemini-3.1-flash-image-preview' || selectedModel === 'gemini-3-pro-image-preview';
  const supportsGptImage2Controls = selectedModel === 'gpt-image-2';
  const aspectRatioOptions = selectedModel === 'gemini-3.1-flash-image-preview' ? GEMINI_FLASH_ASPECT_RATIOS : GEMINI_PRO_ASPECT_RATIOS;
  const hasResults = resultImages.length > 0;
  const resultFailedCount = Math.max(0, resultBatchCount - resultSuccessCount);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const saved = JSON.parse(raw);
        if (saved.selectedModel) setSelectedModel(saved.selectedModel);
        if (saved.templateKey) setTemplateKey(saved.templateKey);
        if (Array.isArray(saved.selectedChips)) setSelectedChips(saved.selectedChips);
        if (saved.paperContent) setPaperContent(saved.paperContent);
        if (saved.extraInstructions) setExtraInstructions(saved.extraInstructions);
        if (saved.textLanguage) setTextLanguage(saved.textLanguage);
        if (saved.batchCount) setBatchCount(normalizeBatchCount(saved.batchCount));
        if (saved.aspectRatio) setAspectRatio(saved.aspectRatio);
        if (saved.resolution) setResolution(saved.resolution);
        if (saved.gptSize) setGptSize(saved.gptSize);
        if (saved.gptQuality) setGptQuality(saved.gptQuality);
        if (saved.apiUrl) setApiUrl(saved.apiUrl);
        if (saved.apiKey) setApiKey(saved.apiKey);
      }
      const userApi = getApiSettings(user?.id || null);
      if (userApi) {
        if (userApi.apiUrl) setApiUrl(userApi.apiUrl);
        if (userApi.apiKey) setApiKey(userApi.apiKey);
      }
    } catch (err) {
      console.error('Failed to restore image playground settings', err);
    }
  }, [user?.id, userApiConfigRequired]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const data = {
      selectedModel,
      templateKey,
      selectedChips,
      paperContent,
      extraInstructions,
      textLanguage,
      batchCount,
      aspectRatio,
      resolution,
      gptSize,
      gptQuality,
      apiUrl,
      apiKey,
    };
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
      if (user?.id && apiUrl && apiKey) {
        saveApiSettings(user.id, { apiUrl, apiKey });
      }
    } catch (err) {
      console.error('Failed to persist image playground settings', err);
    }
  }, [selectedModel, templateKey, selectedChips, paperContent, extraInstructions, textLanguage, batchCount, aspectRatio, resolution, gptSize, gptQuality, apiUrl, apiKey, user?.id]);

  useEffect(() => {
    if (selectedModel === 'gemini-3-pro-image-preview' && !GEMINI_PRO_ASPECT_RATIOS.includes(aspectRatio)) {
      setAspectRatio('16:9');
    }
    if (!GEMINI_RESOLUTIONS.includes(resolution)) {
      setResolution('2K');
    }
    if (!GPT_IMAGE_SIZES.includes(gptSize)) {
      setGptSize('2048x1152');
    }
    if (!GPT_IMAGE_QUALITIES.includes(gptQuality)) {
      setGptQuality('medium');
    }
  }, [selectedModel, aspectRatio, resolution, gptSize, gptQuality]);

  const toggleChip = (chipKey: ChipKey) => {
    setSelectedChips((current) => (
      current.includes(chipKey)
        ? current.filter((item) => item !== chipKey)
        : [...current, chipKey]
    ));
  };

  const handleGenerate = async () => {
    setError(null);
    setBillingWarning(null);
    setResultImages([]);
    setResultZipUrl('');
    setResultZipFileName('image-playground-batch.zip');
    setResultSuccessCount(0);
    setResultBatchCount(0);

    if (!user) {
      setError(t('errors.loginRequired'));
      return;
    }
    if (!paperContent.trim()) {
      setError(t('errors.promptRequired'));
      return;
    }
    if (userApiConfigRequired && (!apiUrl.trim() || !apiKey.trim())) {
      setError(t('errors.apiRequired'));
      return;
    }

    const quota = await checkQuota(user.id || null);
    if (quota.remaining < totalCost) {
      setError(
        quota.isAuthenticated
          ? buildInsufficientPointsMessage(totalCost, quota.remaining, t('hero.title'))
          : buildQuotaExhaustedMessage(runtimeConfig.points_purchase_url),
      );
      return;
    }

    setIsGenerating(true);
    try {
      const response = await backendFetch('/api/v1/image-playground/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Workflow-Amount': String(totalCost),
        },
        body: JSON.stringify({
          prompt: promptPreview,
          model: selectedModel,
          template_key: templateKey,
          domain_key: templateKey,
          batch_count: batchCount,
          ...(supportsGeminiControls ? { aspect_ratio: aspectRatio, resolution } : {}),
          ...(supportsGptImage2Controls ? { size: gptSize, quality: gptQuality } : {}),
          ...(userApiConfigRequired ? { chat_api_url: apiUrl.trim(), api_key: apiKey.trim() } : {}),
        }),
      });

      const data = await response.json().catch(() => null);
      if (!response.ok) {
        setError(data?.detail || t('errors.requestFailed'));
        return;
      }

      const normalizedImages = normalizeResultImages(data);
      setResultImages(normalizedImages);
      setResultSuccessCount(Number(data?.success_count) || normalizedImages.length);
      setResultBatchCount(Number(data?.batch_count) || batchCount);
      setResultZipUrl(normalizeBackendAssetUrl(data?.zip_path || ''));
      setResultZipFileName(data?.zip_file_name || 'image-playground-batch.zip');
      setBillingWarning(data?.billing_warning || null);
      setLastPrompt(data?.prompt || promptPreview);
      void refreshQuota();
    } catch (err) {
      setError(err instanceof Error ? err.message : t('errors.requestFailed'));
    } finally {
      setIsGenerating(false);
    }
  };

  const handleCopyPrompt = async () => {
    try {
      await navigator.clipboard.writeText(lastPrompt || promptPreview);
    } catch {
      setError(t('errors.copyFailed'));
    }
  };

  return (
    <div className="h-full overflow-y-auto overflow-x-hidden">
      <div className="mx-auto flex max-w-7xl flex-col gap-6 px-5 pb-14 pt-6 md:px-8 lg:px-10">
        <section className="relative overflow-hidden rounded-[32px] border border-white/10 bg-[linear-gradient(135deg,rgba(15,22,35,0.92),rgba(40,16,18,0.68))] p-6 shadow-[0_30px_100px_rgba(0,0,0,0.35)]">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(251,191,36,0.14),transparent_24%),radial-gradient(circle_at_bottom_left,rgba(248,113,113,0.16),transparent_30%)]" />
          <div className="relative flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-3xl">
              <div className="inline-flex items-center gap-2 rounded-full border border-orange-400/30 bg-orange-500/10 px-4 py-2 text-xs font-semibold uppercase tracking-[0.24em] text-orange-100">
                <Flame size={14} />
                <span>{t('hero.badge')}</span>
                <span className="rounded-full bg-orange-500 px-2 py-0.5 text-[10px] text-white">HOT</span>
              </div>
              <h2 className="mt-4 text-3xl font-semibold tracking-tight text-white md:text-4xl">{t('hero.title')}</h2>
              <p className="mt-3 max-w-2xl text-sm leading-7 text-white/70 md:text-base">{t('hero.description')}</p>
            </div>
            <div className="rounded-2xl border border-orange-400/25 bg-black/20 px-4 py-3 text-sm text-orange-100 shadow-[0_0_40px_rgba(251,146,60,0.12)]">
              <div className="flex items-center gap-2 font-medium">
                <Sparkles size={16} />
                <span>{t('meta.cost', { count: totalCost })}</span>
              </div>
            </div>
          </div>
        </section>

        {!userApiConfigRequired && (
          <ManagedApiNotice description={t('managedNotice')} />
        )}

        <div className="grid gap-6 xl:grid-cols-[1.05fr_0.95fr]">
          <div className="space-y-6">
            <section className="rounded-[28px] border border-white/10 bg-white/5 p-5 shadow-[0_20px_70px_rgba(0,0,0,0.22)] backdrop-blur-xl">
              <h3 className="text-lg font-semibold text-white">{t('models.title')}</h3>
              <div className="mt-4 grid gap-3 md:grid-cols-2">
                {IMAGE_PLAYGROUND_MODELS.map((model) => {
                  const meta = MODEL_META[model];
                  const active = selectedModel === model;
                  return (
                    <button
                      key={model}
                      type="button"
                      onClick={() => setSelectedModel(model)}
                      className={`rounded-2xl border p-4 text-left transition-all ${
                        active
                          ? `border-white/25 bg-gradient-to-r ${meta.accent} text-white shadow-[0_20px_60px_rgba(0,0,0,0.18)]`
                          : 'border-white/10 bg-white/5 text-slate-200 hover:border-white/20 hover:bg-white/10'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-3">
                        <div className="text-sm font-semibold">{t(meta.titleKey)}</div>
                        {active && <span className="rounded-full bg-white/20 px-2 py-0.5 text-[10px] font-semibold uppercase">On</span>}
                      </div>
                      <div className={`mt-2 text-xs leading-6 ${active ? 'text-white/85' : 'text-slate-400'}`}>{t(meta.descKey)}</div>
                    </button>
                  );
                })}
              </div>

              <div className="mt-5 grid gap-3 md:grid-cols-2">
                <label className="block">
                  <div className="mb-2 text-sm font-medium text-white">{t('controls.textLanguage')}</div>
                  <select
                    value={textLanguage}
                    onChange={(event) => setTextLanguage(event.target.value as ImageTextLanguage)}
                    className="w-full rounded-xl border border-white/10 bg-black/20 px-3 py-2 text-sm text-white outline-none transition focus:border-cyan-400/50"
                  >
                    <option value="en" className="bg-slate-900 text-white">{t('controls.languageOptions.en')}</option>
                    <option value="zh" className="bg-slate-900 text-white">{t('controls.languageOptions.zh')}</option>
                  </select>
                </label>
                <label className="block">
                  <div className="mb-2 text-sm font-medium text-white">{t('controls.batchCount')}</div>
                  <select
                    value={batchCount}
                    onChange={(event) => setBatchCount(normalizeBatchCount(event.target.value))}
                    className="w-full rounded-xl border border-white/10 bg-black/20 px-3 py-2 text-sm text-white outline-none transition focus:border-cyan-400/50"
                  >
                    {BATCH_COUNT_OPTIONS.map((option) => (
                      <option key={option} value={option} className="bg-slate-900 text-white">
                        {t('controls.batchOption', { count: option })}
                      </option>
                    ))}
                  </select>
                </label>
              </div>

              {(supportsGeminiControls || supportsGptImage2Controls) && (
                <div className="mt-5 grid gap-3 md:grid-cols-2">
                  {supportsGeminiControls && (
                    <>
                      <label className="block">
                        <div className="mb-2 text-sm font-medium text-white">{t('controls.aspectRatio')}</div>
                        <select
                          value={aspectRatio}
                          onChange={(event) => setAspectRatio(event.target.value as ImageAspectRatio)}
                          className="w-full rounded-xl border border-white/10 bg-black/20 px-3 py-2 text-sm text-white outline-none transition focus:border-cyan-400/50"
                        >
                          {aspectRatioOptions.map((option) => (
                            <option key={option} value={option} className="bg-slate-900 text-white">
                              {option}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label className="block">
                        <div className="mb-2 text-sm font-medium text-white">{t('controls.resolution')}</div>
                        <select
                          value={resolution}
                          onChange={(event) => setResolution(event.target.value as ImageResolution)}
                          className="w-full rounded-xl border border-white/10 bg-black/20 px-3 py-2 text-sm text-white outline-none transition focus:border-cyan-400/50"
                        >
                          {GEMINI_RESOLUTIONS.map((option) => (
                            <option key={option} value={option} className="bg-slate-900 text-white">
                              {option}
                            </option>
                          ))}
                        </select>
                      </label>
                    </>
                  )}

                  {supportsGptImage2Controls && (
                    <>
                      <label className="block">
                        <div className="mb-2 text-sm font-medium text-white">{t('controls.size')}</div>
                        <select
                          value={gptSize}
                          onChange={(event) => setGptSize(event.target.value as GptImageSize)}
                          className="w-full rounded-xl border border-white/10 bg-black/20 px-3 py-2 text-sm text-white outline-none transition focus:border-cyan-400/50"
                        >
                          {GPT_IMAGE_SIZES.map((option) => (
                            <option key={option} value={option} className="bg-slate-900 text-white">
                              {option}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label className="block">
                        <div className="mb-2 text-sm font-medium text-white">{t('controls.quality')}</div>
                        <select
                          value={gptQuality}
                          onChange={(event) => setGptQuality(event.target.value as GptImageQuality)}
                          className="w-full rounded-xl border border-white/10 bg-black/20 px-3 py-2 text-sm text-white outline-none transition focus:border-cyan-400/50"
                        >
                          {GPT_IMAGE_QUALITIES.map((option) => (
                            <option key={option} value={option} className="bg-slate-900 text-white">
                              {t(`controls.qualityOptions.${option}`)}
                            </option>
                          ))}
                        </select>
                      </label>
                    </>
                  )}
                </div>
              )}

              {selectedModel === 'gpt-image-2-all' && (
                <div className="mt-5 rounded-2xl border border-amber-400/25 bg-amber-500/10 px-4 py-3 text-sm leading-6 text-amber-100">
                  {t('controls.gptImage2AllNotice')}
                </div>
              )}
            </section>

            <section className="rounded-[28px] border border-white/10 bg-white/5 p-5 shadow-[0_20px_70px_rgba(0,0,0,0.22)] backdrop-blur-xl">
              <h3 className="text-lg font-semibold text-white">{t('templates.title')}</h3>
              <div className="mt-4 grid gap-3 md:grid-cols-3">
                {TEMPLATE_CARD_KEYS.map((key) => {
                  const active = templateKey === key;
                  return (
                    <button
                      key={key}
                      type="button"
                      onClick={() => setTemplateKey(key)}
                      className={`rounded-2xl border p-4 text-left transition-all ${
                        active
                          ? 'border-orange-300/45 bg-orange-500/12 text-white'
                          : 'border-white/10 bg-black/10 text-slate-200 hover:border-white/20 hover:bg-white/8'
                      }`}
                    >
                      <div className="text-sm font-semibold">{t(`templates.${key}`)}</div>
                      <div className={`mt-2 text-xs leading-6 ${active ? 'text-orange-100/90' : 'text-slate-400'}`}>{t(`templates.${key}Desc`)}</div>
                    </button>
                  );
                })}
              </div>

              <div className="mt-5">
                <div className="mb-3 text-sm font-medium text-white">{t('chips.title')}</div>
                <div className="flex flex-wrap gap-2">
                  {CHIP_KEYS.map((chipKey) => {
                    const active = selectedChips.includes(chipKey);
                    return (
                      <button
                        key={chipKey}
                        type="button"
                        onClick={() => toggleChip(chipKey)}
                        className={`rounded-full border px-3 py-2 text-xs font-medium transition-all ${
                          active
                            ? 'border-cyan-300/40 bg-cyan-500/15 text-cyan-100'
                            : 'border-white/10 bg-white/5 text-slate-300 hover:border-white/20 hover:text-white'
                        }`}
                      >
                        {t(`chips.${chipKey}`)}
                      </button>
                    );
                  })}
                </div>
              </div>
            </section>

            <section className="rounded-[28px] border border-white/10 bg-white/5 p-5 shadow-[0_20px_70px_rgba(0,0,0,0.22)] backdrop-blur-xl">
              {userApiConfigRequired && (
                <div className="mb-5 grid gap-3 md:grid-cols-2">
                  <label className="block">
                    <div className="mb-2 text-sm font-medium text-white">{t('inputs.apiUrl')}</div>
                    <input
                      value={apiUrl}
                      onChange={(event) => setApiUrl(event.target.value)}
                      className="w-full rounded-xl border border-white/10 bg-black/20 px-3 py-2 text-sm text-white outline-none transition focus:border-cyan-400/50"
                    />
                  </label>
                  <label className="block">
                    <div className="mb-2 text-sm font-medium text-white">{t('inputs.apiKey')}</div>
                    <input
                      type="password"
                      value={apiKey}
                      onChange={(event) => setApiKey(event.target.value)}
                      className="w-full rounded-xl border border-white/10 bg-black/20 px-3 py-2 text-sm text-white outline-none transition focus:border-cyan-400/50"
                    />
                  </label>
                </div>
              )}

              <label className="block">
                <div className="mb-2 text-sm font-medium text-white">{t('inputs.paperContent')}</div>
                <textarea
                  value={paperContent}
                  onChange={(event) => setPaperContent(event.target.value)}
                  placeholder={t('inputs.paperPlaceholder')}
                  rows={8}
                  className="w-full rounded-2xl border border-white/10 bg-black/20 px-4 py-3 text-sm leading-7 text-white outline-none transition placeholder:text-slate-500 focus:border-cyan-400/50"
                />
              </label>

              <label className="mt-4 block">
                <div className="mb-2 text-sm font-medium text-white">{t('inputs.extra')}</div>
                <textarea
                  value={extraInstructions}
                  onChange={(event) => setExtraInstructions(event.target.value)}
                  placeholder={t('inputs.extraPlaceholder')}
                  rows={4}
                  className="w-full rounded-2xl border border-white/10 bg-black/20 px-4 py-3 text-sm leading-7 text-white outline-none transition placeholder:text-slate-500 focus:border-cyan-400/50"
                />
              </label>

              <div className="mt-5">
                <div className="mb-2 text-sm font-medium text-white">{t('inputs.preview')}</div>
                <pre className="max-h-[280px] overflow-auto rounded-2xl border border-white/10 bg-[#09101a] p-4 text-xs leading-6 text-slate-200 whitespace-pre-wrap">
                  {promptPreview}
                </pre>
              </div>

              <div className="mt-5 flex flex-wrap items-center gap-3">
                <button
                  type="button"
                  onClick={handleGenerate}
                  disabled={isGenerating}
                  className="inline-flex items-center gap-2 rounded-2xl bg-gradient-to-r from-orange-500 via-amber-500 to-rose-500 px-5 py-3 text-sm font-semibold text-white shadow-[0_20px_60px_rgba(251,146,60,0.35)] transition hover:scale-[1.01] disabled:cursor-not-allowed disabled:opacity-70"
                >
                  {isGenerating ? <Loader2 size={16} className="animate-spin" /> : <Flame size={16} />}
                  <span>
                    {isGenerating
                      ? t('actions.generating', { count: batchCount })
                      : batchCount > 1
                        ? t('actions.generateBatch', { count: totalCost, images: batchCount })
                        : t('actions.generate', { count: totalCost })}
                  </span>
                </button>
                {error && <div className="text-sm text-rose-300">{error}</div>}
              </div>
            </section>
          </div>

          <div className="space-y-6">
            <section className="rounded-[28px] border border-white/10 bg-white/5 p-5 shadow-[0_20px_70px_rgba(0,0,0,0.22)] backdrop-blur-xl">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <h3 className="text-lg font-semibold text-white">{t('result.title')}</h3>
                  {hasResults && (
                    <p className="mt-1 text-sm text-slate-400">
                      {t('result.summary', { success: resultSuccessCount, total: resultBatchCount })}
                    </p>
                  )}
                </div>
                {hasResults && (
                  <div className="rounded-full border border-emerald-400/30 bg-emerald-500/10 px-3 py-1 text-xs text-emerald-100">
                    {t('meta.saved')}
                  </div>
                )}
              </div>

              <div className="mt-4 overflow-hidden rounded-[24px] border border-white/10 bg-[#08101a]">
                {isGenerating ? (
                  <div className="flex min-h-[440px] flex-col items-center justify-center px-6 text-center">
                    <Loader2 size={42} className="animate-spin text-orange-300" />
                    <p className="mt-4 text-base font-medium text-white">
                      {t('result.loadingTitle', { count: batchCount })}
                    </p>
                    <p className="mt-2 max-w-md text-sm leading-6 text-slate-400">
                      {t('result.loadingDesc', { count: totalCost })}
                    </p>
                  </div>
                ) : hasResults ? (
                  <div className="p-4">
                    <div className="grid gap-3 sm:grid-cols-2">
                      {resultImages.map((item) => (
                        <article key={`${item.index}-${item.fileName}`} className="overflow-hidden rounded-2xl border border-white/10 bg-white/5">
                          <div className="aspect-[4/3] overflow-hidden bg-black/30">
                            <img
                              src={item.previewUrl || item.imageUrl}
                              alt={`generated-${item.index}`}
                              className="h-full w-full object-cover"
                            />
                          </div>
                          <div className="flex items-center justify-between gap-3 px-4 py-3">
                            <div>
                              <div className="text-sm font-medium text-white">{item.variantLabel}</div>
                              <div className="text-xs text-slate-400">{item.fileName}</div>
                            </div>
                            <a
                              href={item.imageUrl}
                              download={item.fileName}
                              className="inline-flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-white transition hover:border-white/20 hover:bg-white/10"
                            >
                              <Download size={14} />
                              <span>{t('actions.download')}</span>
                            </a>
                          </div>
                        </article>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="flex min-h-[440px] items-center justify-center px-6 text-center text-sm leading-7 text-slate-500">
                    {t('result.empty')}
                  </div>
                )}
              </div>

              {hasResults && resultFailedCount > 0 && (
                <div className="mt-4 rounded-2xl border border-amber-400/25 bg-amber-500/10 px-4 py-3 text-sm leading-6 text-amber-100">
                  {t('result.partial', { success: resultSuccessCount, failed: resultFailedCount })}
                </div>
              )}

              {billingWarning && (
                <div className="mt-4 rounded-2xl border border-amber-400/25 bg-amber-500/10 px-4 py-3 text-sm leading-6 text-amber-100">
                  {billingWarning}
                </div>
              )}

              <div className="mt-4 flex flex-wrap items-center gap-3">
                <button
                  type="button"
                  onClick={handleCopyPrompt}
                  className="inline-flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-4 py-2 text-sm text-white transition hover:border-white/20 hover:bg-white/10"
                >
                  <Copy size={15} />
                  <span>{t('actions.copyPrompt')}</span>
                </button>
                {resultZipUrl && (
                  <a
                    href={resultZipUrl}
                    download={resultZipFileName}
                    className="inline-flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-4 py-2 text-sm text-white transition hover:border-white/20 hover:bg-white/10"
                  >
                    <Download size={15} />
                    <span>{t('actions.downloadAll')}</span>
                  </a>
                )}
              </div>
            </section>
          </div>
        </div>
      </div>
    </div>
  );
}
