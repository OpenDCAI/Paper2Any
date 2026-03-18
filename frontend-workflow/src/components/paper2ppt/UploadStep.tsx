import React, { ChangeEvent, useState } from 'react';
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
        <p className="text-xs uppercase tracking-[0.2em] text-purple-300 mb-3 font-semibold">{t('upload.subtitle')}</p>
        <h1 className="text-4xl md:text-5xl font-bold mb-4">
          <span className="bg-gradient-to-r from-purple-400 via-pink-400 to-rose-400 bg-clip-text text-transparent">
            {t('upload.title')}
          </span>
        </h1>
        <p className="text-base text-gray-300 max-w-2xl mx-auto leading-relaxed">
          {t('upload.desc')}<br />
          <span className="text-purple-400">{t('upload.descHighlight')}</span>
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 左侧：输入区域 */}
        <div className="glass rounded-xl border border-white/10 p-6 relative overflow-hidden">
          {/* 装饰背景光 */}
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-2/3 h-1 bg-gradient-to-r from-transparent via-purple-500 to-transparent opacity-50 blur-sm"></div>

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
          
          {/* API 配置 */}
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
          
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-gray-400 mb-1">{t('upload.config.genModel')}</label>
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

            {styleMode === 'prompt' ? (
              <>
                <div className="mb-3">
                  <label className="block text-xs text-gray-400 mb-1">{t('upload.config.stylePreset')}</label>
                  <select 
                    value={stylePreset} 
                    onChange={e => setStylePreset(e.target.value as typeof stylePreset)} 
                    className="w-full rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-purple-500"
                  >
                    <option value="modern">{t('upload.config.presets.modern')}</option>
                    <option value="business">{t('upload.config.presets.business')}</option>
                    <option value="academic">{t('upload.config.presets.academic')}</option>
                    <option value="creative">{t('upload.config.presets.creative')}</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs text-gray-400 mb-1">{t('upload.config.promptLabel')}</label>
                  <textarea 
                    value={globalPrompt} 
                    onChange={e => setGlobalPrompt(e.target.value)} 
                    placeholder={t('upload.config.promptPlaceholder')}
                    rows={2} 
                    className="w-full rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-purple-500 resize-none" 
                  />
                </div>
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <label className="block text-xs text-gray-400">{t('upload.config.promptCardsTitle')}</label>
                    <span className="text-[10px] text-gray-500">{t('upload.config.promptCardsTip')}</span>
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {stylePromptCards.map((card) => (
                      <button
                        key={card.title}
                        type="button"
                        onClick={() => {
                          setStyleMode('prompt');
                          setGlobalPrompt(card.text);
                        }}
                        className="group text-left rounded-2xl border border-white/15 bg-white/5 px-4 py-3 shadow-[0_10px_30px_rgba(0,0,0,0.25)] backdrop-blur transition-all hover:-translate-y-0.5 hover:border-purple-400/60 hover:bg-white/10"
                      >
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
            disabled={(uploadMode === 'file' && !selectedFile) || ((uploadMode === 'text' || uploadMode === 'topic') && !textContent.trim()) || isUploading} 
            className="w-full py-3 rounded-lg bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:from-gray-600 disabled:to-gray-700 text-white font-semibold flex items-center justify-center gap-2 transition-all"
          >
            {isUploading ? (
              <><Loader2 size={18} className="animate-spin" /> {uploadMode === 'topic' ? t('upload.config.startButton.researching') : t('upload.config.startButton.parsing')}</>
            ) : (
              <><ArrowRight size={18} /> {uploadMode === 'topic' ? t('upload.config.startButton.research') : t('upload.config.startButton.parse')}</>
            )}
          </button>

          <div className="flex items-start gap-2 text-xs text-gray-500 mt-3 px-1">
            <Info size={14} className="mt-0.5 text-gray-400 flex-shrink-0" />
            <p>{t('upload.config.tip')}</p>
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
            <p>正在验证 API Key 有效性...</p>
        </div>
      )}

      {error && (
        <div className="mt-4 flex items-center gap-2 text-sm text-red-300 bg-red-500/10 border border-red-500/40 rounded-lg px-4 py-3">
          <AlertCircle size={16} /> {error}
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
