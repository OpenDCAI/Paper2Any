import React, { ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';
import { API_URL_OPTIONS, getPurchaseUrl } from '../../config/api';
import { PAPER2PPT_GEN_FIG_MODELS, PAPER2PPT_MODELS, withModelOptions } from '../../config/models';
import {
  UploadCloud, Settings2, Loader2, AlertCircle, Sparkles,
  ArrowRight, FileText, Key, Globe, Cpu, Type, Lightbulb,
  Info, X
} from 'lucide-react';
import QRCodeTooltip from '../QRCodeTooltip';
import DemoCard from './DemoCard';
import { UploadMode, StyleMode, StylePreset } from './types';

interface UploadStepProps {
  uploadMode: UploadMode;
  setUploadMode: (mode: UploadMode) => void;
  textContent: string;
  setTextContent: (text: string) => void;
  selectedFile: File | null;
  isDragOver: boolean;
  setIsDragOver: (isDragOver: boolean) => void;
  styleMode: StyleMode;
  setStyleMode: (mode: StyleMode) => void;
  stylePreset: StylePreset;
  setStylePreset: (preset: StylePreset) => void;
  globalPrompt: string;
  setGlobalPrompt: (prompt: string) => void;
  referenceImage: File | null;
  referenceImagePreview: string | null;
  
  isUploading: boolean;
  isValidating: boolean;
  pageCount: number;
  setPageCount: (count: number) => void;
  useLongPaper: boolean;
  setUseLongPaper: (use: boolean) => void;
  progress: number;
  progressStatus: string;
  error: string | null;
  
  llmApiUrl: string;
  setLlmApiUrl: (url: string) => void;
  apiKey: string;
  setApiKey: (key: string) => void;
  model: string;
  setModel: (model: string) => void;
  genFigModel: string;
  setGenFigModel: (model: string) => void;
  language: 'zh' | 'en';
  setLanguage: (lang: 'zh' | 'en') => void;

  handleFileChange: (e: ChangeEvent<HTMLInputElement>) => void;
  handleDrop: (e: React.DragEvent<HTMLDivElement>) => void;
  handleReferenceImageChange: (e: ChangeEvent<HTMLInputElement>) => void;
  handleRemoveReferenceImage: () => void;
  handleUploadAndParse: () => void;
}

const UploadStep: React.FC<UploadStepProps> = ({
  uploadMode, setUploadMode,
  textContent, setTextContent,
  selectedFile,
  isDragOver, setIsDragOver,
  styleMode, setStyleMode,
  stylePreset, setStylePreset,
  globalPrompt, setGlobalPrompt,
  referenceImage, referenceImagePreview,
  
  isUploading, isValidating,
  pageCount, setPageCount,
  useLongPaper, setUseLongPaper,
  progress, progressStatus,
  error,
  
  llmApiUrl, setLlmApiUrl,
  apiKey, setApiKey,
  model, setModel,
  genFigModel, setGenFigModel,
  language, setLanguage,

  handleFileChange,
  handleDrop,
  handleReferenceImageChange,
  handleRemoveReferenceImage,
  handleUploadAndParse
}) => {
  const { t, i18n } = useTranslation(['paper2ppt', 'common']);
  const modelOptions = withModelOptions(PAPER2PPT_MODELS, model);
  const genFigModelOptions = withModelOptions(PAPER2PPT_GEN_FIG_MODELS, genFigModel);
  const genFigModelLabels: Record<string, string> = {
    'gemini-3-pro-image-preview': 'Gemini 3 Pro (中文必选)',
    'gemini-2.5-flash-image': 'Gemini 2.5 (Flash Image)',
  };
  const uiLang = i18n.language?.startsWith('zh') ? 'zh' : 'en';
  const stylePromptCards = uiLang === 'zh'
    ? [
        {
          title: '手绘卡通信息图',
          text: '手绘卡通风格的信息图。线条：素描感、粗糙笔触、卡通简化\n禁止写实、禁止照片级明暗、禁止 3D 渲染\n效果参考：涂鸦 / 蜡笔 / 马克笔 / 粉彩',
        },
        {
          title: '极简专业商务',
          text: '极简商务风格。大留白、清晰对比、2~3 色主辅配色\n强调对齐与网格、轻阴影、扁平图标\n禁止复杂纹理、禁止炫光、禁止杂乱背景',
        },
        {
          title: '科技蓝紫渐变',
          text: '科技感视觉：深蓝到青色渐变背景，发光线条/节点\n图表与关键数字高亮，模块卡片玻璃拟态\n禁止复古元素、禁止卡通元素',
        },
        {
          title: '学术论文风',
          text: '学术报告风格：白底、严谨排版、稳重配色（蓝/灰/黑）\n图表优先、标题清晰、关键结论加粗\n禁止花哨装饰、禁止大面积高饱和色',
        },
        {
          title: '品牌宣传风',
          text: '品牌宣传风格：高质感图片占比高，统一品牌色系\n标题大、层级分明，口号式短句\n禁止密集文字、禁止表格式排版',
        },
        {
          title: '自然柔和插画',
          text: '自然柔和插画风：米白背景、低饱和配色、柔和阴影\n插画/贴纸元素点缀，整体温暖亲和\n禁止强对比、禁止金属质感、禁止赛博霓虹',
        },
      ]
    : [
        {
          title: 'Hand-drawn Infographic',
          text: 'Hand-drawn cartoon infographic. Lines: sketchy, rough strokes, simplified shapes.\nNo realism, no photographic lighting, no 3D rendering.\nLook & feel: doodle / crayon / marker / pastel.',
        },
        {
          title: 'Minimal Business',
          text: 'Minimal business style. Spacious layout, strong contrast, 2–3 color palette.\nStrict alignment/grid, subtle shadows, flat icons.\nNo heavy textures, no glow effects, no busy backgrounds.',
        },
        {
          title: 'Tech Gradient',
          text: 'Futuristic tech look: deep blue to cyan gradients, glowing lines/nodes.\nHighlight charts and key numbers, glassmorphism cards.\nNo retro elements, no cartoon elements.',
        },
        {
          title: 'Academic Report',
          text: 'Academic report style: white background, rigorous layout, sober colors (blue/gray/black).\nChart-first, clear titles, bold key findings.\nNo fancy decorations, no highly saturated blocks.',
        },
        {
          title: 'Brand Promo',
          text: 'Brand promo style: high-quality visuals, consistent brand colors.\nBig titles, clear hierarchy, slogan-like short phrases.\nNo dense text, no table-like layouts.',
        },
        {
          title: 'Soft Illustration',
          text: 'Soft illustration style: off-white background, low-saturation palette, gentle shadows.\nLight stickers/illustrations as accents, warm and friendly tone.\nNo harsh contrast, no metallic textures, no cyber neon.',
        },
      ];

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-10 text-center">
        <p className="paper2ppt-kicker mb-3 text-xs font-semibold uppercase">{t('upload.subtitle')}</p>
        <h1 className="paper2ppt-title mb-4 text-4xl font-bold md:text-5xl">
          {t('upload.title')} <span className="paper2ppt-accent">{t('upload.descHighlight')}</span>
        </h1>
        <p className="paper2ppt-subtitle mx-auto max-w-3xl text-base leading-relaxed">
          {t('upload.desc')}
        </p>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="paper2ppt-panel relative overflow-hidden rounded-[30px] p-6">
          <div className="absolute left-8 right-8 top-0 h-px bg-[linear-gradient(90deg,transparent,rgba(140,29,64,0.38),transparent)]" />

          <div className="paper2ppt-tabbar mb-6 grid grid-cols-3 gap-3">
            {[
              { id: 'file', label: t('upload.tabs.file'), icon: FileText, sub: t('upload.tabs.fileSub') },
              { id: 'text', label: t('upload.tabs.text'), icon: Type, sub: t('upload.tabs.textSub') },
              { id: 'topic', label: t('upload.tabs.topic'), icon: Lightbulb, sub: t('upload.tabs.topicSub') },
            ].map((item) => (
              <button
                key={item.id}
                onClick={() => setUploadMode(item.id as UploadMode)}
                className={`paper2ppt-tab relative flex flex-col items-center justify-center overflow-hidden rounded-2xl py-3 ${
                  uploadMode === item.id ? 'paper2ppt-tab-active scale-[1.01]' : ''
                }`}
              >
                {uploadMode === item.id && (
                  <div className="animate-shimmer-fast absolute inset-0 h-full w-full -translate-x-full bg-gradient-to-r from-transparent via-white/18 to-transparent" />
                )}
                <item.icon size={22} className={`mb-1.5 transition-colors ${uploadMode === item.id ? 'text-white' : 'text-[#8c1d40]'}`} />
                <span className={`text-sm font-bold tracking-wide ${uploadMode === item.id ? 'text-white' : 'text-[#1d1c1a]'}`}>{item.label}</span>
                <span className={`text-[10px] uppercase tracking-wider font-medium ${uploadMode === item.id ? 'text-[#f7dcb1]' : 'text-[#675f58]'}`}>{item.sub}</span>
              </button>
            ))}
          </div>

          <div className="mb-3 flex items-center gap-2 px-1">
            <span className="h-4 w-1 rounded-full bg-[#8c1d40]" />
            <h3 className="text-sm font-semibold text-[#1d1c1a]">
              {uploadMode === 'file' ? t('upload.instruction.file') : uploadMode === 'text' ? t('upload.instruction.text') : t('upload.instruction.topic')}
            </h3>
          </div>

          {uploadMode === 'file' ? (
            <div
              className={`paper2ppt-dropzone flex h-[300px] flex-col items-center justify-center gap-4 rounded-[24px] p-8 text-center ${
                isDragOver ? 'paper2ppt-dropzone-active' : ''
              }`}
              onDragOver={e => { e.preventDefault(); setIsDragOver(true); }}
              onDragLeave={e => { e.preventDefault(); setIsDragOver(false); }}
              onDrop={handleDrop}
            >
              <div className="flex h-16 w-16 items-center justify-center rounded-full bg-[rgba(140,29,64,0.08)]">
                <UploadCloud size={32} className="text-[#8c1d40]" />
              </div>
              <div>
                <p className="mb-1 text-base font-semibold text-[#1d1c1a]">{t('upload.dropzone.dragText')}</p>
                <p className="text-sm text-[#675f58]">{t('upload.dropzone.supportText')}</p>
              </div>
              <label className="paper2ppt-button-primary cursor-pointer rounded-full px-6 py-2.5 text-sm font-medium">
                {t('upload.dropzone.button')}
                <input type="file" accept=".pdf" className="hidden" onChange={handleFileChange} />
              </label>
              {selectedFile && (
                <div className="paper2ppt-status-success px-4 py-2">
                  <p className="text-sm font-medium">✓ {selectedFile.name}</p>
                  <p className="mt-1 text-xs">{t('upload.dropzone.analyzing')}</p>
                </div>
              )}
            </div>
          ) : (
            <div className="flex h-[300px] flex-col">
              <textarea
                value={textContent}
                onChange={e => setTextContent(e.target.value)}
                placeholder={uploadMode === 'text'
                  ? t('upload.textInput.placeholderText')
                  : t('upload.textInput.placeholderTopic')}
                className="paper2ppt-input flex-1 rounded-[24px] px-4 py-3 text-sm resize-none"
              />
              <p className="mt-2 text-right text-xs text-[#675f58]">
                {uploadMode === 'text' ? `${textContent.length} ${t('upload.textInput.charCount')}` : t('upload.textInput.deepResearch')}
              </p>
            </div>
          )}
        </div>

        <div className="paper2ppt-panel space-y-4 rounded-[30px] p-6">
          <h3 className="flex items-center gap-2 text-lg font-semibold text-[#1d1c1a]">
            <Settings2 size={18} className="text-[#8c1d40]" /> {t('upload.config.title')}
          </h3>

          <div className="grid grid-cols-2 gap-3">
            <div className="col-span-2">
              <label className="paper2ppt-label mb-1 flex items-center gap-1">
                <Key size={12} /> {t('upload.config.apiKey')}
              </label>
              <input
                type="password"
                value={apiKey}
                onChange={e => setApiKey(e.target.value)}
                placeholder={t('upload.config.apiKeyPlaceholder')}
                className="paper2ppt-input rounded-xl px-3 py-2 text-sm"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <div className="mb-1 flex items-center justify-between">
                <label className="paper2ppt-label flex items-center gap-1">
                  <Globe size={12} /> {t('upload.config.apiUrl')}
                </label>
                <QRCodeTooltip>
                  <a
                    href={getPurchaseUrl(llmApiUrl)}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="paper2ppt-link text-[10px]"
                  >
                    {t('upload.config.buyLink')}
                  </a>
                </QRCodeTooltip>
              </div>
              <select
                value={llmApiUrl}
                onChange={e => {
                  const val = e.target.value;
                  setLlmApiUrl(val);
                  if (val.includes('123.129.219.111')) {
                    setGenFigModel('gemini-3-pro-image-preview');
                  }
                }}
                className="paper2ppt-input rounded-xl px-3 py-2 text-sm"
              >
                {API_URL_OPTIONS.map((url: string) => (
                  <option key={url} value={url}>{url}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="paper2ppt-label mb-1 flex items-center gap-1">
                <Cpu size={12} /> {t('upload.config.model')}
              </label>
              <div className="grid grid-cols-2 gap-2">
                <select
                  value={model}
                  onChange={e => setModel(e.target.value)}
                  className="paper2ppt-input rounded-xl px-3 py-2 text-sm"
                >
                  {modelOptions.map((option) => (
                    <option key={option} value={option}>{option}</option>
                  ))}
                </select>
                <div className="group relative">
                  <input
                    type="text"
                    value={model}
                    onChange={e => setModel(e.target.value)}
                    placeholder="自定义模型"
                    className="paper2ppt-input rounded-xl px-3 py-2 text-sm"
                  />
                  <div className="pointer-events-none absolute left-full top-1/2 z-20 ml-2 w-56 -translate-y-1/2 rounded-xl border border-[rgba(110,76,55,0.14)] bg-[rgba(255,250,245,0.98)] px-2 py-1.5 text-[10px] text-[#675f58] opacity-0 shadow-lg transition group-hover:opacity-100">
                    {t('upload.config.customModelTip')}
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="paper2ppt-label mb-1">{t('upload.config.genModel')}</label>
              <select
                value={genFigModel}
                onChange={e => setGenFigModel(e.target.value)}
                disabled={llmApiUrl.includes('123.129.219.111')}
                className="paper2ppt-input rounded-xl px-3 py-2 text-sm disabled:cursor-not-allowed disabled:opacity-50"
              >
                {genFigModelOptions.map((option) => (
                  <option key={option} value={option}>{genFigModelLabels[option] || option}</option>
                ))}
              </select>
              {llmApiUrl.includes('123.129.219.111') && (
                <p className="mt-1 text-[10px] text-[#675f58]">此源仅支持 gemini-3-pro</p>
              )}
            </div>
            <div>
              <label className="paper2ppt-label mb-1">{t('upload.config.pageCount')}</label>
              <input
                type="number"
                value={pageCount}
                onChange={e => setPageCount(parseInt(e.target.value) || 6)}
                min={1}
                max={20}
                className="paper2ppt-input rounded-xl px-3 py-2 text-sm"
              />
            </div>
          </div>

          <div className="flex items-center gap-2 px-1 py-1">
            <button
              onClick={() => setUseLongPaper(!useLongPaper)}
              className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
                useLongPaper ? 'bg-[#8c1d40]' : 'bg-[rgba(103,95,88,0.52)]'
              }`}
            >
              <span
                className={`inline-block h-3 w-3 transform rounded-full bg-white transition-transform ${
                  useLongPaper ? 'translate-x-5' : 'translate-x-1'
                }`}
              />
            </button>
            <span className="cursor-pointer text-xs text-[#1d1c1a]" onClick={() => setUseLongPaper(!useLongPaper)}>
              {t('upload.config.longPaper')}
            </span>
          </div>

          <div className="mt-2 border-t border-[rgba(110,76,55,0.14)] pt-4">
            <h4 className="paper2ppt-label mb-2">{t('upload.config.styleTitle')}</h4>

            <div className="mb-3">
              <label className="paper2ppt-label mb-1 block">{t('upload.config.language')}</label>
              <select
                value={language}
                onChange={e => setLanguage(e.target.value as 'zh' | 'en')}
                className="paper2ppt-input rounded-xl px-3 py-2 text-sm"
              >
                <option value="zh">中文</option>
                <option value="en">English</option>
              </select>
            </div>

            <div className="mb-3 flex gap-2">
              <button
                type="button"
                onClick={() => setStyleMode('prompt')}
                className={`flex-1 rounded-xl px-3 py-2.5 text-xs font-medium ${styleMode === 'prompt' ? 'paper2ppt-button-primary' : 'paper2ppt-button-secondary'}`}
              >
                <span className="flex items-center justify-center gap-1"><Sparkles size={14} /> {t('upload.config.styleMode.prompt')}</span>
              </button>
              <button
                type="button"
                onClick={() => setStyleMode('reference')}
                className={`flex-1 rounded-xl px-3 py-2.5 text-xs font-medium ${styleMode === 'reference' ? 'paper2ppt-button-primary' : 'paper2ppt-button-secondary'}`}
              >
                <span className="flex items-center justify-center gap-1"><UploadCloud size={14} /> {t('upload.config.styleMode.reference')}</span>
              </button>
            </div>

            {styleMode === 'prompt' ? (
              <>
                <div className="mb-3">
                  <label className="paper2ppt-label mb-1 block">{t('upload.config.stylePreset')}</label>
                  <select
                    value={stylePreset}
                    onChange={e => setStylePreset(e.target.value as StylePreset)}
                    className="paper2ppt-input rounded-xl px-3 py-2 text-sm"
                  >
                    <option value="modern">{t('upload.config.presets.modern')}</option>
                    <option value="business">{t('upload.config.presets.business')}</option>
                    <option value="academic">{t('upload.config.presets.academic')}</option>
                    <option value="creative">{t('upload.config.presets.creative')}</option>
                  </select>
                </div>
                <div className="mb-3">
                  <label className="paper2ppt-label mb-1 block">{t('upload.config.promptLabel')}</label>
                  <textarea
                    value={globalPrompt}
                    onChange={e => setGlobalPrompt(e.target.value)}
                    placeholder={t('upload.config.promptPlaceholder')}
                    rows={2}
                    className="paper2ppt-input rounded-xl px-3 py-2 text-sm resize-none"
                  />
                </div>
                <div>
                  <div className="mb-2 flex items-center justify-between">
                    <label className="paper2ppt-label">{t('upload.config.promptCardsTitle')}</label>
                    <span className="text-[10px] text-[#675f58]">{t('upload.config.promptCardsTip')}</span>
                  </div>
                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                    {stylePromptCards.map((card) => (
                      <button
                        key={card.title}
                        type="button"
                        onClick={() => {
                          setStyleMode('prompt');
                          setGlobalPrompt(card.text);
                        }}
                        className="paper2ppt-choice-card group rounded-2xl px-4 py-3 text-left"
                      >
                        <div className="mb-1 text-sm font-semibold text-[#1d1c1a]">{card.title}</div>
                        <div className="line-clamp-4 whitespace-pre-line text-[11px] leading-relaxed text-[#675f58]">
                          {card.text}
                        </div>
                        <div className="mt-2 text-[10px] text-[#8c1d40] opacity-0 transition-opacity group-hover:opacity-100">
                          {t('upload.config.promptCardsUse')}
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              </>
            ) : (
              <div>
                <label className="paper2ppt-label mb-1 block">{t('upload.config.referenceLabel')}</label>
                {referenceImagePreview ? (
                  <div className="relative">
                    <img
                      src={referenceImagePreview}
                      alt="参考风格"
                      className="h-32 w-full rounded-xl border border-[rgba(110,76,55,0.14)] object-cover"
                    />
                    <button
                      type="button"
                      onClick={handleRemoveReferenceImage}
                      className="absolute right-2 top-2 rounded-full bg-[rgba(140,29,64,0.92)] p-1.5 text-white transition-colors hover:bg-[#b12d3e]"
                    >
                      <X size={14} />
                    </button>
                    <p className="mt-1 text-[11px] text-[#8c1d40]">✓ {t('upload.config.referenceUploaded')}</p>
                  </div>
                ) : (
                  <label className="paper2ppt-dropzone flex cursor-pointer flex-col items-center justify-center gap-2 rounded-2xl p-4 text-center">
                    <UploadCloud size={20} className="text-[#8c1d40]" />
                    <span className="text-xs text-[#675f58]">{t('upload.config.referenceUpload')}</span>
                    <input
                      type="file"
                      accept="image/*"
                      className="hidden"
                      onChange={handleReferenceImageChange}
                    />
                  </label>
                )}
              </div>
            )}
          </div>

          <button
            onClick={handleUploadAndParse}
            disabled={(uploadMode === 'file' && !selectedFile) || ((uploadMode === 'text' || uploadMode === 'topic') && !textContent.trim()) || isUploading}
            className="paper2ppt-button-primary flex w-full items-center justify-center gap-2 rounded-xl py-3 font-semibold disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isUploading ? (
              <><Loader2 size={18} className="animate-spin" /> {uploadMode === 'topic' ? t('upload.config.startButton.researching') : t('upload.config.startButton.parsing')}</>
            ) : (
              <><ArrowRight size={18} /> {uploadMode === 'topic' ? t('upload.config.startButton.research') : t('upload.config.startButton.parse')}</>
            )}
          </button>

          <div className="mt-3 flex items-start gap-2 px-1 text-xs text-[#675f58]">
            <Info size={14} className="mt-0.5 flex-shrink-0 text-[#8c1d40]" />
            <p>{t('upload.config.tip')}</p>
          </div>

          {isUploading && (
            <div className="mt-4 animate-in fade-in slide-in-from-top-2">
              <div className="mb-1 flex justify-between text-xs text-[#675f58]">
                <span>{progressStatus}</span>
                <span>{Math.round(progress)}%</span>
              </div>
              <div className="paper2ppt-progress-track h-1.5 overflow-hidden rounded-full">
                <div
                  className="paper2ppt-progress-value h-full transition-all duration-300 ease-out"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )}
        </div>
      </div>

      {isValidating && (
        <div className="paper2ppt-status-info mt-4 flex animate-pulse items-center gap-2 px-4 py-3 text-sm">
          <Loader2 size={16} className="animate-spin" />
          <p>正在验证 API Key 有效性...</p>
        </div>
      )}

      {error && (
        <div className="paper2ppt-status-error mt-4 flex items-center gap-2 px-4 py-3 text-sm">
          <AlertCircle size={16} /> {error}
        </div>
      )}

      <div className="mt-8 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-3">
            <h3 className="text-sm font-semibold text-[#1d1c1a]">{t('upload.demo.title')}</h3>
            <a
              href="https://wcny4qa9krto.feishu.cn/wiki/VXKiwYndwiWAVmkFU6kcqsTenWh"
              target="_blank"
              rel="noopener noreferrer"
              className="paper2ppt-button-secondary inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-medium"
            >
              <Sparkles size={12} className="text-[#8c1d40]" />
              <span className="text-[#8c1d40]">{t('upload.demo.more')}</span>
            </a>
          </div>
          <span className="text-[11px] text-[#675f58]">
            {t('upload.demo.desc')}
          </span>
        </div>

        <div className="grid grid-cols-1 gap-4 text-xs md:grid-cols-2">
          <DemoCard
            title={t('upload.demo.card1.title')}
            desc={t('upload.demo.card1.desc')}
            inputImg="/paper2ppt/input_1.png"
            outputImg="/paper2ppt/ouput_1.png"
          />
          <DemoCard
            title={t('upload.demo.card2.title')}
            desc={t('upload.demo.card2.desc')}
            inputImg="/paper2ppt/input_3.png"
            outputImg="/paper2ppt/ouput_3.png"
          />
          <DemoCard
            title={t('upload.demo.card3.title')}
            desc={t('upload.demo.card3.desc')}
            inputImg="/paper2ppt/input_2.png"
            outputImg="/paper2ppt/ouput_2.png"
          />
          <DemoCard
            title={t('upload.demo.card4.title')}
            desc={t('upload.demo.card4.desc')}
            inputImg="/paper2ppt/input_4.png"
            outputImg="/paper2ppt/ouput_4.png"
          />
        </div>
      </div>
    </div>
  );
};

export default UploadStep;
