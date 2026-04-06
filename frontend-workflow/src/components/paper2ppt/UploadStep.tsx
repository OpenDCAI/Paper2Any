import React, { ChangeEvent, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { API_URL_OPTIONS, getPurchaseUrl } from '../../config/api';
import { PAPER2PPT_GEN_FIG_MODELS, PAPER2PPT_MODELS, withModelOptions } from '../../config/models';
import {
  UploadCloud, Settings2, Loader2, AlertCircle, Sparkles,
  ArrowRight, FileText, Key, Globe, Cpu, Type, Lightbulb,
  MonitorSmartphone,
  Info, X
} from 'lucide-react';
import QRCodeTooltip from '../QRCodeTooltip';
import ManagedApiNotice from '../ManagedApiNotice';
import DemoCard from './DemoCard';
import { PptGenerationMode, UploadMode, StyleMode, StylePreset } from './types';
import { getManagedValidationText, isInsufficientPointsError } from '../../utils/pointsMessaging';

interface UploadStepProps {
  pptMode: PptGenerationMode;
  setPptMode: (mode: PptGenerationMode) => void;
  modeLocked?: boolean;
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
  isUploadSubmitLocked: boolean;
  pageCount: number;
  setPageCount: (count: number) => void;
  useLongPaper: boolean;
  setUseLongPaper: (use: boolean) => void;
  frontendIncludeImages: boolean;
  setFrontendIncludeImages: (enabled: boolean) => void;
  frontendAutoReviewEnabled: boolean;
  setFrontendAutoReviewEnabled: (enabled: boolean) => void;
  frontendImageStyle: string;
  setFrontendImageStyle: (style: string) => void;
  progress: number;
  progressStatus: string;
  error: string | null;
  purchaseUrl?: string | null;
  showApiConfig: boolean;
  
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
  pptMode, setPptMode,
  modeLocked = false,
  uploadMode, setUploadMode,
  textContent, setTextContent,
  selectedFile,
  isDragOver, setIsDragOver,
  styleMode, setStyleMode,
  stylePreset, setStylePreset,
  globalPrompt, setGlobalPrompt,
  referenceImage, referenceImagePreview,
  
  isUploading, isValidating,
  isUploadSubmitLocked,
  pageCount, setPageCount,
  useLongPaper, setUseLongPaper,
  frontendIncludeImages,
  setFrontendIncludeImages,
  frontendAutoReviewEnabled,
  setFrontendAutoReviewEnabled,
  frontendImageStyle,
  setFrontendImageStyle,
  progress, progressStatus,
  error,
  purchaseUrl,
  showApiConfig,
  
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
  const isSubmitBusy = isUploading || isValidating;
  const modelOptions = withModelOptions(PAPER2PPT_MODELS, model);
  const genFigModelOptions = withModelOptions(PAPER2PPT_GEN_FIG_MODELS, genFigModel);
  const genFigModelLabels: Record<string, string> = {
    'gemini-3-pro-image-preview': 'Gemini 3 Pro (中文必选)',
    'gemini-2.5-flash-image': 'Gemini 2.5 (Flash Image)',
  };
  const uiLang = i18n.language?.startsWith('zh') ? 'zh' : 'en';
  const imageStylePromptCards = uiLang === 'zh'
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
  const frontendStylePromptCards = uiLang === 'zh'
    ? [
        {
          title: '暖白 + 石墨 + 赤陶',
          text: '请使用暖白或象牙白背景，深石墨文字，赤陶或赭红作为唯一强调色。整体像高质量 keynote 学术汇报，不要青色系，不要霓虹感，不要玻璃发光。',
          swatch: 'from-[#f4efe6] via-[#d8c3a5] to-[#b85c38]',
        },
        {
          title: '午夜蓝 + 冰灰 + 电蓝',
          text: '请使用午夜蓝或深海军蓝背景，冰灰文字，电蓝作为少量强调色。整体要克制、冷静、专业，像研究组年度汇报，不要默认青色玻璃卡片。',
          swatch: 'from-[#0f172a] via-[#334155] to-[#60a5fa]',
        },
        {
          title: '纸感米白 + 墨黑 + 酒红',
          text: '请做成纸感米白底色，墨黑正文，酒红或暗红作为重点强调。整体像学术讲义与答辩结合的风格，不要赛博蓝绿，不要荧光描边。',
          swatch: 'from-[#f8f2e7] via-[#d6c6b8] to-[#7f1d1d]',
        },
        {
          title: '森林绿 + 沙金 + 奶油白',
          text: '请使用森林绿或深橄榄作为主色，沙金做点缀，奶油白做底。整体要像高端研究报告封面延展到全 deck，避免青色与玻璃拟态默认风格。',
          swatch: 'from-[#1f3d2b] via-[#7c8f4e] to-[#d4b483]',
        },
        {
          title: '黑白灰 + 一点亮橙',
          text: '请使用黑白灰为主，亮橙只用于极少数重点标签或数字。布局要极简、留白多、组件统一，像极简发布会风格，不要任何青蓝色主导。',
          swatch: 'from-[#111827] via-[#6b7280] to-[#f97316]',
        },
        {
          title: '深紫红 + 雾粉 + 银灰',
          text: '请使用深紫红基底，雾粉或银灰作为辅助色。风格要成熟、优雅、有研究报告质感，不要默认青色边框和蓝绿高光。',
          swatch: 'from-[#3b0d2e] via-[#9d6381] to-[#c9c9d6]',
        },
      ]
    : [
        {
          title: 'Warm Ivory + Terracotta',
          text: 'Use an ivory or warm white background, deep graphite text, and terracotta as the only accent. Make it feel like a refined keynote-style academic talk. No cyan, no neon, no glass glow.',
          swatch: 'from-[#f4efe6] via-[#d8c3a5] to-[#b85c38]',
        },
        {
          title: 'Midnight Blue + Ice Gray',
          text: 'Use midnight blue or deep navy as the base, ice-gray text, and electric blue as a sparse accent. Keep it restrained and professional. Avoid the default cyan glass-card look.',
          swatch: 'from-[#0f172a] via-[#334155] to-[#60a5fa]',
        },
        {
          title: 'Parchment + Burgundy',
          text: 'Use a parchment-like off-white canvas, ink-black body text, and burgundy as emphasis. The deck should feel like an academic handout merged with a defense presentation. No cyber cyan/teal.',
          swatch: 'from-[#f8f2e7] via-[#d6c6b8] to-[#7f1d1d]',
        },
        {
          title: 'Forest Green + Sand Gold',
          text: 'Use forest green or dark olive as the main tone, sand-gold accents, and cream as the base. Make it feel like a premium research report. Avoid cyan and generic glassmorphism.',
          swatch: 'from-[#1f3d2b] via-[#7c8f4e] to-[#d4b483]',
        },
        {
          title: 'Monochrome + Bright Orange',
          text: 'Stay mostly black, white, and gray, with bright orange used only for sparse labels or key metrics. Minimal, quiet, and presentation-grade. No cyan-led palette.',
          swatch: 'from-[#111827] via-[#6b7280] to-[#f97316]',
        },
        {
          title: 'Plum Red + Mist Pink',
          text: 'Use a dark plum-red base with mist pink and silver-gray as supporting colors. Aim for a mature, elegant research deck. Avoid default cyan borders and aqua highlights.',
          swatch: 'from-[#3b0d2e] via-[#9d6381] to-[#c9c9d6]',
        },
      ];
  const modeTexts = uiLang === 'zh'
      ? {
          title: 'PPT 生成模式',
          imageTitle: '图片版 PPT',
          imageDesc: '沿用现有图像工作流，逐页生成视觉稿并导出。',
          frontendTitle: '可编辑版 PPT',
          frontendDesc: '生成 16:9 HTML/CSS 模板，文字可编辑，最终截图导出。',
          frontendTip: '可编辑版默认文本优先；若开启图像增强，会优先复用论文图表，否则按页面内容自动补示意图。',
        }
      : {
          title: 'PPT Mode',
          imageTitle: 'Image PPT',
          imageDesc: 'Use the current image workflow and export generated visual slides.',
          frontendTitle: 'Editable PPT',
          frontendDesc: 'Generate editable 16:9 HTML/CSS slides and export by screenshots.',
          frontendTip: 'Editable mode stays text-editable and can optionally reuse paper figures/tables or generate supporting images.',
        };
  const pageCopy = uiLang === 'zh'
    ? {
        image: {
          kicker: 'Image Deck Workflow',
          title: '图片版 PPT 生成',
          desc: '生成偏视觉稿路线的学术汇报页面，适合组会、答辩和展示型场景。',
          highlight: '这一页只保留图片版工作流，不再混入可编辑版模式切换。',
        },
        frontend: {
          kicker: 'Editable Deck Workflow',
          title: '可编辑版 PPT 生成',
          desc: '生成 16:9 HTML/CSS 可编辑页面，支持画布内直接改字、可选首轮视觉检查和截图导出。',
          highlight: '这一页只做可编辑版 deck，不再混入图片版配置。',
        },
      }
    : {
        image: {
          kicker: 'Image Deck Workflow',
          title: 'Image-style PPT Generation',
          desc: 'Generate image-first academic slides for presentation-heavy seminar and defense scenarios.',
          highlight: 'This page is dedicated to the image workflow only.',
        },
        frontend: {
          kicker: 'Editable Deck Workflow',
          title: 'Editable PPT Generation',
          desc: 'Generate 16:9 HTML/CSS text slides with inline editing, optional first-pass visual QA, and screenshot export.',
          highlight: 'This page is dedicated to the editable deck workflow only.',
        },
      };
  const currentPageCopy = pptMode === 'frontend' ? pageCopy.frontend : pageCopy.image;
  const promptCards = pptMode === 'frontend' ? frontendStylePromptCards : imageStylePromptCards;
  const presetOptions = pptMode === 'frontend'
    ? (
        uiLang === 'zh'
          ? [
              { value: 'modern', label: '暖白赤陶' },
              { value: 'business', label: '午夜蓝冰灰' },
              { value: 'academic', label: '纸感酒红' },
              { value: 'creative', label: '森林绿沙金' },
            ]
          : [
              { value: 'modern', label: 'Ivory + Terracotta' },
              { value: 'business', label: 'Midnight Blue' },
              { value: 'academic', label: 'Parchment + Burgundy' },
              { value: 'creative', label: 'Forest Green' },
            ]
      )
    : [
        { value: 'modern', label: t('upload.config.presets.modern') },
        { value: 'business', label: t('upload.config.presets.business') },
        { value: 'academic', label: t('upload.config.presets.academic') },
        { value: 'creative', label: t('upload.config.presets.creative') },
      ];
  const stylePresetLabel = pptMode === 'frontend'
    ? (uiLang === 'zh' ? '主题色方向' : 'Palette Direction')
    : t('upload.config.stylePreset');
  const promptLabel = pptMode === 'frontend'
    ? (uiLang === 'zh' ? '前端主题提示词' : 'Frontend Theme Prompt')
    : t('upload.config.promptLabel');
  const promptPlaceholder = pptMode === 'frontend'
    ? (uiLang === 'zh'
        ? '例如：米白背景，酒红强调，像答辩 keynote；标题克制、卡片边框更细...'
        : 'Example: ivory canvas, burgundy accents, keynote-like academic tone; restrained titles and thinner card borders...')
    : t('upload.config.promptPlaceholder');
  const promptCardsTitle = pptMode === 'frontend'
    ? (uiLang === 'zh' ? '推荐主题 / 配色候选' : 'Recommended Palette / Theme Directions')
    : t('upload.config.promptCardsTitle');
  const promptCardsTip = pptMode === 'frontend'
    ? (uiLang === 'zh' ? '可编辑版建议直接写颜色、材质和组件气质' : 'For editable decks, specify palette, material, and component language directly')
    : t('upload.config.promptCardsTip');
  const frontendImageStyleOptions = uiLang === 'zh'
    ? [
        { value: 'academic_illustration', label: '学术示意图' },
        { value: 'realistic', label: '写实' },
        { value: 'sci_fi', label: '科幻' },
        { value: 'flat_infographic', label: '扁平信息图' },
      ]
    : [
        { value: 'academic_illustration', label: 'Academic Illustration' },
        { value: 'realistic', label: 'Realistic' },
        { value: 'sci_fi', label: 'Sci-Fi' },
        { value: 'flat_infographic', label: 'Flat Infographic' },
      ];

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-10 text-center">
        <p className={`text-xs uppercase tracking-[0.2em] mb-3 font-semibold ${pptMode === 'frontend' ? 'text-amber-300' : 'text-purple-300'}`}>
          {currentPageCopy.kicker}
        </p>
        <h1 className="text-4xl md:text-5xl font-bold mb-4">
          <span className={`bg-gradient-to-r bg-clip-text text-transparent ${
            pptMode === 'frontend'
              ? 'from-amber-300 via-orange-300 to-yellow-200'
              : 'from-purple-400 via-pink-400 to-rose-400'
          }`}>
            {currentPageCopy.title}
          </span>
        </h1>
        <p className="text-base text-gray-300 max-w-2xl mx-auto leading-relaxed">
          {currentPageCopy.desc}<br />
          <span className={pptMode === 'frontend' ? 'text-amber-300' : 'text-purple-400'}>
            {currentPageCopy.highlight}
          </span>
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 左侧：输入区域 */}
        <div className="glass rounded-xl border border-white/10 p-6 relative overflow-hidden">
          {/* 装饰背景光 */}
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-2/3 h-1 bg-gradient-to-r from-transparent via-purple-500 to-transparent opacity-50 blur-sm"></div>

          {!modeLocked && (
            <div className="mb-6">
              <div className="mb-3 flex items-center gap-2 px-1">
                <span className="w-1 h-4 rounded-full bg-cyan-500"></span>
                <h3 className="text-white font-medium text-sm">{modeTexts.title}</h3>
              </div>
              <div className="grid grid-cols-2 gap-3 p-1.5 bg-black/40 rounded-2xl border border-white/5">
                <button
                  type="button"
                  onClick={() => setPptMode('image')}
                  className={`text-left rounded-xl px-4 py-4 transition-all ${
                    pptMode === 'image'
                      ? 'bg-gradient-to-br from-purple-600 to-pink-600 text-white shadow-lg shadow-purple-500/30 ring-1 ring-white/20'
                      : 'bg-white/5 text-gray-300 hover:bg-white/10'
                  }`}
                >
                  <div className="flex items-center gap-2 mb-2">
                    <Sparkles size={18} className={pptMode === 'image' ? 'text-white' : 'text-purple-300'} />
                    <span className="font-semibold text-sm">{modeTexts.imageTitle}</span>
                  </div>
                  <p className={`text-xs leading-relaxed ${pptMode === 'image' ? 'text-purple-100' : 'text-gray-400'}`}>
                    {modeTexts.imageDesc}
                  </p>
                </button>
                <button
                  type="button"
                  onClick={() => setPptMode('frontend')}
                  className={`text-left rounded-xl px-4 py-4 transition-all ${
                    pptMode === 'frontend'
                      ? 'bg-gradient-to-br from-amber-500 to-orange-500 text-white shadow-lg shadow-amber-500/30 ring-1 ring-white/20'
                      : 'bg-white/5 text-gray-300 hover:bg-white/10'
                  }`}
                >
                  <div className="flex items-center gap-2 mb-2">
                    <MonitorSmartphone size={18} className={pptMode === 'frontend' ? 'text-white' : 'text-amber-300'} />
                    <span className="font-semibold text-sm">{modeTexts.frontendTitle}</span>
                  </div>
                  <p className={`text-xs leading-relaxed ${pptMode === 'frontend' ? 'text-amber-100' : 'text-gray-400'}`}>
                    {modeTexts.frontendDesc}
                  </p>
                </button>
              </div>
              {pptMode === 'frontend' && (
                <p className="mt-3 text-xs text-amber-200 bg-amber-500/10 border border-amber-500/20 rounded-xl px-3 py-2">
                  {modeTexts.frontendTip}
                </p>
              )}
            </div>
          )}

          {/* 炫酷模式切换 Tabs */}
          <div className="grid grid-cols-3 gap-3 mb-6 p-1.5 bg-black/40 rounded-2xl border border-white/5">
            {[
              { id: 'file', label: t('upload.tabs.file'), icon: FileText, sub: t('upload.tabs.fileSub') },
              { id: 'text', label: t('upload.tabs.text'), icon: Type, sub: t('upload.tabs.textSub') },
              { id: 'topic', label: t('upload.tabs.topic'), icon: Lightbulb, sub: t('upload.tabs.topicSub') },
            ].map((item) => (
              <button 
                key={item.id}
                onClick={() => setUploadMode(item.id as any)}
                className={`relative group flex flex-col items-center justify-center py-3 rounded-xl transition-all duration-300 overflow-hidden ${
                  uploadMode === item.id 
                    ? 'bg-gradient-to-br from-purple-600 to-pink-600 text-white shadow-lg shadow-purple-500/30 scale-[1.02] ring-1 ring-white/20' 
                    : 'bg-white/5 text-gray-400 hover:bg-white/10 hover:text-gray-200 hover:scale-[1.02]'
                }`}
              >
                {/* 选中态的光效扫光动画 */}
                {uploadMode === item.id && (
                  <div className="absolute inset-0 w-full h-full bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full animate-shimmer-fast"></div>
                )}
                
                <item.icon size={22} className={`mb-1.5 transition-colors ${uploadMode === item.id ? 'text-white' : 'text-gray-500 group-hover:text-purple-400'}`} />
                <span className={`text-sm font-bold tracking-wide ${uploadMode === item.id ? 'text-white' : 'text-gray-300'}`}>{item.label}</span>
                <span className={`text-[10px] uppercase tracking-wider font-medium ${uploadMode === item.id ? 'text-purple-100' : 'text-gray-600'}`}>{item.sub}</span>
              </button>
            ))}
          </div>

          <div className="mb-3 flex items-center gap-2 px-1">
            <span className="w-1 h-4 rounded-full bg-purple-500"></span>
            <h3 className="text-white font-medium text-sm">
              {uploadMode === 'file' ? t('upload.instruction.file') : uploadMode === 'text' ? t('upload.instruction.text') : t('upload.instruction.topic')}
            </h3>
          </div>

          {uploadMode === 'file' ? (
            <div 
              className={`border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center text-center gap-4 transition-all h-[300px] ${
                isDragOver ? 'border-purple-500 bg-purple-500/10' : 'border-white/20 hover:border-purple-400'
              }`} 
              onDragOver={e => { e.preventDefault(); setIsDragOver(true); }} 
              onDragLeave={e => { e.preventDefault(); setIsDragOver(false); }} 
              onDrop={handleDrop}
            >
              <div className="w-16 h-16 rounded-full bg-gradient-to-br from-purple-500/20 to-pink-500/20 flex items-center justify-center">
                <UploadCloud size={32} className="text-purple-400" />
              </div>
              <div>
                <p className="text-white font-medium mb-1">{t('upload.dropzone.dragText')}</p>
                <p className="text-sm text-gray-400">{t('upload.dropzone.supportText')}</p>
              </div>
              <label className="px-6 py-2.5 rounded-full bg-gradient-to-r from-purple-600 to-pink-600 text-white text-sm font-medium cursor-pointer hover:from-purple-700 hover:to-pink-700 transition-all">
                {t('upload.dropzone.button')}
                <input type="file" accept=".pdf" className="hidden" onChange={handleFileChange} />
              </label>
              {selectedFile && (
                <div className="px-4 py-2 bg-purple-500/20 border border-purple-500/40 rounded-lg">
                  <p className="text-sm text-purple-300">✓ {selectedFile.name}</p>
                  <p className="text-xs text-gray-400 mt-1">✨ {t('upload.dropzone.analyzing')}</p>
                </div>
              )}
            </div>
          ) : (
            <div className="flex flex-col h-[300px]">
              <textarea
                value={textContent}
                onChange={e => setTextContent(e.target.value)}
                placeholder={uploadMode === 'text' 
                  ? t('upload.textInput.placeholderText')
                  : t('upload.textInput.placeholderTopic')}
                className="flex-1 w-full rounded-xl border border-white/20 bg-black/40 px-4 py-3 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-purple-500 resize-none"
              />
              <p className="text-xs text-gray-500 mt-2 text-right">
                {uploadMode === 'text' ? `${textContent.length} ${t('upload.textInput.charCount')}` : t('upload.textInput.deepResearch')}
              </p>
            </div>
          )}
        </div>

        {/* 右侧：配置区域 */}
        <div className="glass rounded-xl border border-white/10 p-6 space-y-4">
          <h3 className="text-white font-semibold flex items-center gap-2">
            <Settings2 size={18} className="text-purple-400" /> {t('upload.config.title')}
          </h3>
          
          {showApiConfig ? (
            <>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs text-gray-400 mb-1 flex items-center gap-1">
                    <Key size={12} /> {t('upload.config.apiKey')}
                  </label>
                  <input 
                    type="password" 
                    value={apiKey} 
                    onChange={e => setApiKey(e.target.value)}
                    placeholder={t('upload.config.apiKeyPlaceholder')}
                    className="w-full rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-purple-500"
                  />
                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <label className="block text-xs text-gray-400 flex items-center gap-1">
                      <Globe size={12} /> {t('upload.config.apiUrl')}
                    </label>
                    <QRCodeTooltip>
                      <a
                        href={getPurchaseUrl(llmApiUrl)}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-[10px] text-purple-300 hover:text-purple-200 hover:underline"
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
                    className="w-full rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-purple-500"
                  >
                    {API_URL_OPTIONS.map((url: string) => (
                      <option key={url} value={url}>{url}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-xs text-gray-400 mb-1 flex items-center gap-1">
                    <Cpu size={12} /> {t('upload.config.model')}
                  </label>
                  <div className="grid grid-cols-2 gap-2">
                    <select 
                      value={model} 
                      onChange={e => setModel(e.target.value)}
                      className="w-full rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-purple-500"
                    >
                      {modelOptions.map((option) => (
                        <option key={option} value={option}>{option}</option>
                      ))}
                    </select>
                    <div className="relative group">
                      <input
                        type="text"
                        value={model} 
                        onChange={e => setModel(e.target.value)}
                        placeholder="自定义模型"
                        className="w-full rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-purple-500"
                      />
                      <div className="pointer-events-none absolute left-full top-1/2 z-20 ml-2 w-56 -translate-y-1/2 rounded-md border border-white/10 bg-black/80 px-2 py-1.5 text-[10px] text-gray-100 opacity-0 shadow-lg transition group-hover:opacity-100">
                        {t('upload.config.customModelTip')}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </>
          ) : (
            <>
              <ManagedApiNotice />
              <div>
                <label className="block text-xs text-gray-400 mb-1 flex items-center gap-1">
                  <Cpu size={12} /> {t('upload.config.model')}
                </label>
                <div className="grid grid-cols-2 gap-2">
                  <select 
                    value={model} 
                    onChange={e => setModel(e.target.value)}
                    className="w-full rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-purple-500"
                  >
                    {modelOptions.map((option) => (
                      <option key={option} value={option}>{option}</option>
                    ))}
                  </select>
                  <div className="relative group">
                    <input
                      type="text"
                      value={model} 
                      onChange={e => setModel(e.target.value)}
                      placeholder="自定义模型"
                      className="w-full rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-purple-500"
                    />
                    <div className="pointer-events-none absolute left-full top-1/2 z-20 ml-2 w-56 -translate-y-1/2 rounded-md border border-white/10 bg-black/80 px-2 py-1.5 text-[10px] text-gray-100 opacity-0 shadow-lg transition group-hover:opacity-100">
                      {t('upload.config.customModelTip')}
                    </div>
                  </div>
                </div>
              </div>
            </>
          )}
          
          <div className={`grid gap-3 ${pptMode === 'image' ? 'grid-cols-2' : 'grid-cols-1'}`}>
            {(pptMode === 'image' || frontendIncludeImages) && (
            <div>
              <label className="block text-xs text-gray-400 mb-1">
                {pptMode === 'frontend' ? '可编辑版生图模型' : t('upload.config.genModel')}
              </label>
              <select
                value={genFigModel}
                onChange={e => setGenFigModel(e.target.value)}
                disabled={llmApiUrl.includes('123.129.219.111')}
                className="w-full rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-purple-500 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {genFigModelOptions.map((option) => (
                  <option key={option} value={option}>{genFigModelLabels[option] || option}</option>
                ))}
              </select>
              {llmApiUrl.includes('123.129.219.111') && (
                 <p className="text-[10px] text-gray-500 mt-1">此源仅支持 gemini-3-pro</p>
              )}
            </div>
            )}
            <div>
              <label className="block text-xs text-gray-400 mb-1">{t('upload.config.pageCount')}</label>
              <input 
                type="number" 
                value={pageCount} 
                onChange={e => setPageCount(parseInt(e.target.value) || 6)}
                min={1}
                max={100}
                className="w-full rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-purple-500"
              />
            </div>
          </div>

          <div className="flex items-center gap-2 px-1 py-1">
            <button
              onClick={() => setUseLongPaper(!useLongPaper)}
              className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
                useLongPaper ? 'bg-purple-600' : 'bg-gray-600'
              }`}
            >
              <span
                className={`inline-block h-3 w-3 transform rounded-full bg-white transition-transform ${
                  useLongPaper ? 'translate-x-5' : 'translate-x-1'
                }`}
              />
            </button>
            <span className="text-xs text-gray-300 cursor-pointer" onClick={() => setUseLongPaper(!useLongPaper)}>
              {t('upload.config.longPaper')}
            </span>
          </div>

          {pptMode === 'frontend' && (
            <div className="space-y-3">
              <div className="rounded-2xl border border-amber-400/20 bg-amber-500/5 p-4">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <div className="text-sm font-semibold text-white">图像增强</div>
                    <div className="mt-1 text-xs leading-5 text-amber-100/80">
                      开启后优先使用论文解析出的图/表；当前页没有可复用素材时，再按大纲自动生成示意图。
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => setFrontendIncludeImages(!frontendIncludeImages)}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                      frontendIncludeImages ? 'bg-amber-500' : 'bg-slate-600'
                    }`}
                  >
                    <span
                      className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                        frontendIncludeImages ? 'translate-x-6' : 'translate-x-1'
                      }`}
                    />
                  </button>
                </div>
                {frontendIncludeImages && (
                  <div className="mt-4 grid grid-cols-1 gap-3">
                    <div>
                      <label className="block text-xs text-gray-300 mb-1">图像风格</label>
                      <select
                        value={frontendImageStyle}
                        onChange={(e) => setFrontendImageStyle(e.target.value)}
                        className="w-full rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-amber-500"
                      >
                        {frontendImageStyleOptions.map((option) => (
                          <option key={option.value} value={option.value}>{option.label}</option>
                        ))}
                      </select>
                    </div>
                    <div className="rounded-xl border border-amber-400/15 bg-black/20 px-3 py-2 text-[11px] leading-5 text-amber-100/85">
                      开启图像增强后，批量生成阶段按 2 点 / 页计费；文字可继续直接编辑，图片可在画布里点击替换。
                    </div>
                  </div>
                )}
              </div>
              <div className="rounded-2xl border border-cyan-400/20 bg-cyan-500/5 p-4">
                <div className="flex items-start justify-between gap-3">
                  <div className="pr-2">
                    <div className="flex items-center gap-2">
                      <div className="text-sm font-semibold text-white">首轮自动视觉检查</div>
                      <span className="rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-[10px] text-gray-300">
                        默认关闭
                      </span>
                    </div>
                    <div className="mt-1 text-xs leading-5 text-cyan-100/80">
                      开启后，批量生成完成会对每一页并发做一次视觉检查，并在发现问题时自动尝试修复；等待时间会明显变长。
                    </div>
                    <div className="mt-2 text-[11px] leading-5 text-cyan-100/70">
                      关闭时会直接进入编辑页。需要时再使用编辑页面下方的“视觉检查并修复”按钮逐页处理。
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => setFrontendAutoReviewEnabled(!frontendAutoReviewEnabled)}
                    className={`relative mt-0.5 inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                      frontendAutoReviewEnabled ? 'bg-cyan-500' : 'bg-slate-600'
                    }`}
                  >
                    <span
                      className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                        frontendAutoReviewEnabled ? 'translate-x-6' : 'translate-x-1'
                      }`}
                    />
                  </button>
                </div>
              </div>
            </div>
          )}

          <div className="border-t border-white/10 pt-4 mt-2">
            <h4 className="text-xs text-gray-400 mb-2">{t('upload.config.styleTitle')}</h4>

            <div className="mb-3">
              <label className="block text-xs text-gray-400 mb-1">{t('upload.config.language')}</label>
              <select 
                value={language} 
                onChange={e => setLanguage(e.target.value as 'zh' | 'en')} 
                className="w-full rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-purple-500"
              >
                <option value="zh">中文</option>
                <option value="en">English</option>
              </select>
            </div>

            {pptMode === 'image' && (
              <div className="flex gap-2 mb-3">
                <button
                  type="button"
                  onClick={() => setStyleMode('prompt')}
                  className={`flex-1 py-2.5 px-3 rounded-lg text-xs font-medium flex items-center justify-center gap-1 transition-all ${
                    styleMode === 'prompt'
                      ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white shadow-sm'
                      : 'bg-white/5 text-gray-400 border border-white/10 hover:bg-white/10'
                  }`}
                >
                  <Sparkles size={14} /> {t('upload.config.styleMode.prompt')}
                </button>
                <button
                  type="button"
                  onClick={() => setStyleMode('reference')}
                  className={`flex-1 py-2.5 px-3 rounded-lg text-xs font-medium flex items-center justify-center gap-1 transition-all ${
                    styleMode === 'reference'
                      ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white shadow-sm'
                      : 'bg-white/5 text-gray-400 border border-white/10 hover:bg-white/10'
                  }`}
                >
                  <UploadCloud size={14} /> {t('upload.config.styleMode.reference')}
                </button>
              </div>
            )}

            {pptMode === 'frontend' || styleMode === 'prompt' ? (
              <>
                <div className="mb-3">
                  <label className="block text-xs text-gray-400 mb-1">{stylePresetLabel}</label>
                  <select 
                    value={stylePreset} 
                    onChange={e => setStylePreset(e.target.value as typeof stylePreset)} 
                    className="w-full rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-purple-500"
                  >
                    {presetOptions.map((option) => (
                      <option key={option.value} value={option.value}>{option.label}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-xs text-gray-400 mb-1">{promptLabel}</label>
                  <textarea 
                    value={globalPrompt} 
                    onChange={e => setGlobalPrompt(e.target.value)} 
                    placeholder={promptPlaceholder}
                    rows={2} 
                    className="w-full rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-purple-500 resize-none" 
                  />
                </div>
                {pptMode === 'frontend' && (
                  <div className="text-[11px] text-amber-200 bg-amber-500/10 border border-amber-500/20 rounded-lg px-3 py-2">
                    可编辑版建议把颜色、材质、留白感和卡片语言写清楚，不要只写“科技风 / 学术风”这类过泛描述。
                  </div>
                )}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <label className="block text-xs text-gray-400">{promptCardsTitle}</label>
                    <span className="text-[10px] text-gray-500">{promptCardsTip}</span>
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {promptCards.map((card) => (
                      <button
                        key={card.title}
                        type="button"
                        onClick={() => {
                          setStyleMode('prompt');
                          setGlobalPrompt(card.text);
                        }}
                        className="group text-left rounded-2xl border border-white/15 bg-white/5 px-4 py-3 shadow-[0_10px_30px_rgba(0,0,0,0.25)] backdrop-blur transition-all hover:-translate-y-0.5 hover:border-purple-400/60 hover:bg-white/10"
                      >
                        {'swatch' in card && (
                          <div className={`mb-3 h-2.5 rounded-full bg-gradient-to-r ${card.swatch}`} />
                        )}
                        <div className="text-sm font-semibold text-white mb-1">{card.title}</div>
                        <div className="text-[11px] leading-relaxed text-gray-300 whitespace-pre-line line-clamp-4">
                          {card.text}
                        </div>
                        <div className="mt-2 text-[10px] text-purple-300 opacity-0 transition-opacity group-hover:opacity-100">
                          {t('upload.config.promptCardsUse')}
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              </>
            ) : (
              <div>
                <label className="block text-xs text-gray-400 mb-1">{t('upload.config.referenceLabel')}</label>
                {referenceImagePreview ? (
                  <div className="relative">
                    <img
                      src={referenceImagePreview}
                      alt="参考风格"
                      className="w-full h-32 object-cover rounded-lg border border-white/20"
                    />
                    <button
                      type="button"
                      onClick={handleRemoveReferenceImage}
                      className="absolute top-2 right-2 p-1.5 rounded-full bg-black/60 text-white hover:bg-red-500 transition-colors"
                    >
                      <X size={14} />
                    </button>
                    <p className="text-[11px] text-purple-300 mt-1">✓ {t('upload.config.referenceUploaded')}</p>
                  </div>
                ) : (
                  <label className="border-2 border-dashed border-white/20 rounded-lg p-4 flex flex-col items-center justify-center text-center gap-2 cursor-pointer hover:border-purple-400 transition-all">
                    <UploadCloud size={20} className="text-gray-400" />
                    <span className="text-xs text-gray-400">{t('upload.config.referenceUpload')}</span>
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
            disabled={(uploadMode === 'file' && !selectedFile) || ((uploadMode === 'text' || uploadMode === 'topic') && !textContent.trim()) || isSubmitBusy || isUploadSubmitLocked} 
            className="w-full py-3 rounded-lg bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:from-gray-600 disabled:to-gray-700 text-white font-semibold flex items-center justify-center gap-2 transition-all"
          >
            {isSubmitBusy ? (
              <><Loader2 size={18} className="animate-spin" /> {uploadMode === 'topic' ? t('upload.config.startButton.researching') : t('upload.config.startButton.parsing')}</>
            ) : (
              <><ArrowRight size={18} /> {uploadMode === 'topic' ? t('upload.config.startButton.research') : t('upload.config.startButton.parse')}</>
            )}
          </button>

          <div className="flex items-start gap-2 text-xs text-gray-500 mt-3 px-1">
            <Info size={14} className="mt-0.5 text-gray-400 flex-shrink-0" />
            <p>{pptMode === 'frontend' ? '可编辑版会在下一步生成可编辑字段和 HTML/CSS 代码；若开启图像增强，会同时预留受控图片槽位；若开启首轮自动视觉检查，会在批量生成后先自动逐页检查，再进入编辑页。' : t('upload.config.tip')}</p>
          </div>

          {isUploading && (
            <div className="mt-4 animate-in fade-in slide-in-from-top-2">
              <div className="flex justify-between text-xs text-gray-400 mb-1">
                <span>{progressStatus}</span>
                <span>{Math.round(progress)}%</span>
              </div>
              <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-purple-500 to-pink-500 transition-all duration-300 ease-out"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )}
        </div>
      </div>

      {isValidating && (
        <div className="mt-4 flex items-center gap-2 text-sm text-blue-300 bg-blue-500/10 border border-blue-500/40 rounded-lg px-4 py-3 animate-pulse">
            <Loader2 size={16} className="animate-spin" />
            <p>{getManagedValidationText(showApiConfig)}</p>
        </div>
      )}

      {error && (
        <div className="mt-4 flex items-start gap-2 text-sm text-red-300 bg-red-500/10 border border-red-500/40 rounded-lg px-4 py-3">
          <AlertCircle size={16} className="mt-0.5 shrink-0" />
          <div className="flex-1">
            <p>{error}</p>
            {purchaseUrl && isInsufficientPointsError(error) && (
              <a
                href={purchaseUrl}
                target="_blank"
                rel="noreferrer"
                className="mt-2 inline-flex items-center gap-1 rounded-md border border-red-300/30 px-2.5 py-1 text-xs font-medium text-red-100 transition-colors hover:border-red-200/60 hover:text-white"
              >
                前往购买页获取兑换码
              </a>
            )}
          </div>
        </div>
      )}

      {/* 示例区 */}
      <div className="space-y-4 mt-8">
        <div className="flex items-center justify-between flex-wrap gap-2">
            <div className="flex items-center gap-3">
            <h3 className="text-sm font-medium text-gray-200">{t('upload.demo.title')}</h3>
            <a
              href="https://wcny4qa9krto.feishu.cn/wiki/VXKiwYndwiWAVmkFU6kcqsTenWh"
              target="_blank"
              rel="noopener noreferrer"
              className="group relative inline-flex items-center gap-2 px-3 py-1 rounded-full bg-black/50 border border-white/10 text-xs font-medium text-white overflow-hidden transition-all hover:border-white/30 hover:shadow-[0_0_15px_rgba(168,85,247,0.5)]"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 via-purple-500/20 to-pink-500/20 opacity-0 group-hover:opacity-100 transition-opacity" />
              <Sparkles size={12} className="text-yellow-300 animate-pulse" />
              <span className="bg-gradient-to-r from-blue-300 via-purple-300 to-pink-300 bg-clip-text text-transparent group-hover:from-blue-200 group-hover:via-purple-200 group-hover:to-pink-200">
                {t('upload.demo.more')}
              </span>
            </a>
          </div>
          <span className="text-[11px] text-gray-500">
            {t('upload.demo.desc')}
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
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
