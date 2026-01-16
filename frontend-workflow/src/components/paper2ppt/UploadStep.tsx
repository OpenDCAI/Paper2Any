import React, { ChangeEvent, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  UploadCloud, Settings2, Loader2, AlertCircle, Sparkles,
  ArrowRight, FileText, Key, Globe, Cpu, Type, Lightbulb,
  Info, X
} from 'lucide-react';
import QRCodeTooltip from '../QRCodeTooltip';
import { PricingDisplay } from '../PricingDisplay';
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
  const { t } = useTranslation(['paper2ppt', 'common']);

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
          
          {/* 模型配置 */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-gray-400 mb-1 flex items-center gap-1">
                <Cpu size={12} /> {t('upload.config.model')}
              </label>
              <select 
                value={model} 
                onChange={e => setModel(e.target.value)}
                className="w-full rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-purple-500"
              >
                <option value="gpt-4o">gpt-4o</option>
                <option value="gpt-5.1">gpt-5.1</option>
                <option value="gpt-5.2">gpt-5.2</option>
                <option value="gemini-3-pro-preview">gemini-3-pro-preview</option>
              </select>
            </div>
          </div>
          
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-gray-400 mb-1">{t('upload.config.genModel')}</label>
              <select
                value={genFigModel}
                onChange={e => setGenFigModel(e.target.value)}
                disabled={llmApiUrl === 'http://123.129.219.111:3000/v1' || language === 'zh'}
                className="w-full rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-purple-500 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <option value="gemini-2.5-flash-image">Gemini 2.5 (Flash Image)</option>
                <option value="gemini-3-pro-image-preview">Gemini 3 Pro (中文推荐)</option>
              </select>
              {llmApiUrl === 'http://123.129.219.111:3000/v1' && (
                 <p className="text-[10px] text-gray-500 mt-1">此源仅支持 gemini-3-pro</p>
              )}
              {language === 'zh' && llmApiUrl !== 'http://123.129.219.111:3000/v1' && (
                 <p className="text-[10px] text-gray-500 mt-1">中文仅支持 Gemini 3 Pro</p>
              )}
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">{t('upload.config.pageCount')}</label>
              <input 
                type="number" 
                value={pageCount} 
                onChange={e => setPageCount(parseInt(e.target.value) || 6)}
                min={1}
                max={20}
                className="w-full rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-purple-500"
              />
              <div className="flex items-center justify-between mt-1 text-xs">
                <span className="text-gray-500">预计费用：</span>
                <PricingDisplay
                  service="paper2ppt"
                  endpoint="pagecontent"
                  pageCount={pageCount}
                  className="text-purple-400 font-semibold"
                />
              </div>
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
                onChange={e => {
                  const newLang = e.target.value as 'zh' | 'en';
                  setLanguage(newLang);
                  // 中文时强制使用 gemini-3-pro-image-preview
                  if (newLang === 'zh') {
                    setGenFigModel('gemini-3-pro-image-preview');
                  }
                }} 
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
