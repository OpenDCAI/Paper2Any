import React, { ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';
import { API_URL_OPTIONS } from '../../config/api';
import ManagedApiNotice from '../ManagedApiNotice';
import {
  UploadCloud, Settings2, Loader2, AlertCircle, ArrowRight,
  FileText, Key, Globe, Cpu, Image as ImageIcon, Ruler
} from 'lucide-react';
import { PosterConfig } from './types';

interface UploadStepProps {
  selectedFile: File | null;
  logoFile: File | null;
  affLogoFile: File | null;
  isDragOver: boolean;
  setIsDragOver: (isDragOver: boolean) => void;

  config: PosterConfig;
  setConfig: (config: PosterConfig) => void;

  isUploading: boolean;
  isValidating: boolean;
  progress: number;
  progressStatus: string;
  error: string | null;
  showApiConfig: boolean;

  llmApiUrl: string;
  setLlmApiUrl: (url: string) => void;
  apiKey: string;
  setApiKey: (key: string) => void;

  handleFileChange: (e: ChangeEvent<HTMLInputElement>) => void;
  handleLogoChange: (e: ChangeEvent<HTMLInputElement>) => void;
  handleAffLogoChange: (e: ChangeEvent<HTMLInputElement>) => void;
  handleDrop: (e: React.DragEvent<HTMLDivElement>) => void;
  handleUploadAndGenerate: () => void;
}

const UploadStep: React.FC<UploadStepProps> = ({
  selectedFile,
  logoFile,
  affLogoFile,
  isDragOver, setIsDragOver,
  config, setConfig,
  isUploading, isValidating,
  progress, progressStatus,
  error,
  showApiConfig,
  llmApiUrl, setLlmApiUrl,
  apiKey, setApiKey,
  handleFileChange,
  handleLogoChange,
  handleAffLogoChange,
  handleDrop,
  handleUploadAndGenerate
}) => {
  const { t } = useTranslation(['paper2poster', 'common']);

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-10 text-center">
        <p className="text-xs uppercase tracking-[0.2em] text-green-300 mb-3 font-semibold">
          {t('upload.subtitle', 'PAPER → POSTER')}
        </p>
        <h1 className="text-4xl md:text-5xl font-bold mb-4">
          <span className="bg-gradient-to-r from-green-400 via-emerald-400 to-teal-400 bg-clip-text text-transparent">
            {t('upload.title', 'Paper2Poster')}
          </span>
        </h1>
        <p className="text-base text-gray-300 max-w-2xl mx-auto leading-relaxed">
          {t('upload.desc', '上传学术论文PDF，自动生成精美的会议海报')}<br />
          <span className="text-green-400">{t('upload.descHighlight', '支持自定义尺寸和Logo')}</span>
        </p>
      </div>

      {/* 文件上传区域 */}
      <div className="glass rounded-2xl p-8 mb-6 border border-white/10">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-xl bg-transparent border border-green-500/40 flex items-center justify-center">
            <UploadCloud className="w-5 h-5 text-green-400" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-white">{t('upload.fileSection', '上传文件')}</h2>
            <p className="text-sm text-gray-400">{t('upload.fileSectionDesc', '上传论文PDF和Logo图片')}</p>
          </div>
        </div>

        {/* PDF上传 */}
        <div
          className={`relative border-2 border-dashed rounded-xl p-8 text-center transition-all ${
            isDragOver
              ? 'border-green-400 bg-green-500/10'
              : 'border-white/20 hover:border-green-400/50 bg-white/5'
          }`}
          onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={handleDrop}
        >
          <input
            type="file"
            accept=".pdf"
            onChange={handleFileChange}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          />
          <FileText className="w-12 h-12 mx-auto mb-4 text-green-400" />
          <p className="text-white font-medium mb-2">
            {selectedFile ? selectedFile.name : t('upload.dropFile', '拖拽PDF文件到此处或点击上传')}
          </p>
          <p className="text-sm text-gray-400">{t('upload.fileLimit', '支持PDF格式，最大50MB')}</p>
        </div>

        {/* Logo上传 */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
          <div className="border border-white/10 rounded-xl p-4 bg-white/5">
            <label className="block text-sm font-medium text-gray-300 mb-2">
              <ImageIcon className="w-4 h-4 inline mr-2" />
              {t('upload.conferenceLogo', '会议Logo（可选）')}
            </label>
            <input
              type="file"
              accept="image/*"
              onChange={handleLogoChange}
              className="block w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-green-500/20 file:text-green-300 hover:file:bg-green-500/30"
            />
          </div>

          <div className="border border-white/10 rounded-xl p-4 bg-white/5">
            <label className="block text-sm font-medium text-gray-300 mb-2">
              <ImageIcon className="w-4 h-4 inline mr-2" />
              {t('upload.affLogo', '机构Logo（可选，用于配色）')}
            </label>
            <input
              type="file"
              accept="image/*"
              onChange={handleAffLogoChange}
              className="block w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-green-500/20 file:text-green-300 hover:file:bg-green-500/30"
            />
          </div>
        </div>
      </div>

      {/* 海报配置区域 */}
      <div className="glass rounded-2xl p-8 mb-6 border border-white/10">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-xl bg-transparent border border-green-500/40 flex items-center justify-center">
            <Cpu className="w-5 h-5 text-green-400" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-white">{t('upload.posterConfig', '海报配置')}</h2>
            <p className="text-sm text-gray-400">{t('upload.posterConfigDesc', '配置模型和尺寸')}</p>
          </div>
        </div>

        {/* 模型配置 */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="flex items-center gap-2 text-sm font-medium text-gray-300 mb-2">
              <Cpu className="w-4 h-4" />
              {t('upload.textModel', '文本模型')}
            </label>
            <select
              value={config.text_model}
              onChange={(e) => setConfig({ ...config, text_model: e.target.value })}
              className="w-full px-4 py-3 bg-black/40 border border-white/20 rounded-xl text-gray-100 focus:outline-none focus:ring-2 focus:ring-green-500 transition-colors"
            >
              <option value="gpt-4o">gpt-4o</option>
              <option value="gpt-5.1">gpt-5.1</option>
              <option value="gpt-5.2">gpt-5.2</option>
              <option value="gemini-3-pro-preview">gemini-3-pro-preview</option>
            </select>
          </div>

          <div>
            <label className="flex items-center gap-2 text-sm font-medium text-gray-300 mb-2">
              <Cpu className="w-4 h-4" />
              {t('upload.visionModel', '视觉模型')}
            </label>
            <select
              value={config.vision_model}
              onChange={(e) => setConfig({ ...config, vision_model: e.target.value })}
              className="w-full px-4 py-3 bg-black/40 border border-white/20 rounded-xl text-gray-100 focus:outline-none focus:ring-2 focus:ring-green-500 transition-colors"
            >
              <option value="gemini-3-pro-image-preview">Gemini 3 Pro (中文必选)</option>
              <option value="gemini-2.5-flash-image">Gemini 2.5 (Flash Image)</option>
            </select>
          </div>
        </div>

        {/* 海报尺寸 */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
          <div>
            <label className="flex items-center gap-2 text-sm font-medium text-gray-300 mb-2">
              <Ruler className="w-4 h-4" />
              {t('upload.posterWidth', '海报宽度（英寸）')}
            </label>
            <input
              type="number"
              value={config.poster_width}
              onChange={(e) => setConfig({ ...config, poster_width: parseFloat(e.target.value) })}
              min="20"
              max="100"
              step="1"
              className="w-full px-4 py-3 bg-black/40 border border-white/20 rounded-xl text-gray-100 focus:outline-none focus:ring-2 focus:ring-green-500 transition-colors"
            />
          </div>

          <div>
            <label className="flex items-center gap-2 text-sm font-medium text-gray-300 mb-2">
              <Ruler className="w-4 h-4" />
              {t('upload.posterHeight', '海报高度（英寸）')}
            </label>
            <input
              type="number"
              value={config.poster_height}
              onChange={(e) => setConfig({ ...config, poster_height: parseFloat(e.target.value) })}
              min="20"
              max="100"
              step="1"
              className="w-full px-4 py-3 bg-black/40 border border-white/20 rounded-xl text-gray-100 focus:outline-none focus:ring-2 focus:ring-green-500 transition-colors"
            />
          </div>
        </div>
      </div>

      {showApiConfig && (
        <div className="glass rounded-2xl p-8 mb-6 border border-white/10">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 rounded-xl bg-transparent border border-green-500/40 flex items-center justify-center">
              <Settings2 className="w-5 h-5 text-green-400" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white">{t('upload.apiConfig', 'API配置')}</h2>
              <p className="text-sm text-gray-400">{t('upload.apiConfigDesc', '配置LLM API')}</p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="flex items-center gap-2 text-sm font-medium text-gray-300 mb-2">
                <Globe className="w-4 h-4" />
                {t('upload.apiUrl', 'API URL')}
              </label>
              <select
                value={llmApiUrl}
                onChange={(e) => setLlmApiUrl(e.target.value)}
                className="w-full px-4 py-3 bg-black/40 border border-white/20 rounded-xl text-gray-100 focus:outline-none focus:ring-2 focus:ring-green-500 transition-colors"
              >
                {API_URL_OPTIONS.map((url: string) => (
                  <option key={url} value={url}>{url}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="flex items-center gap-2 text-sm font-medium text-gray-300 mb-2">
                <Key className="w-4 h-4" />
                {t('upload.apiKey', 'API Key')}
              </label>
              <input
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder={t('upload.apiKeyPlaceholder', '输入API Key')}
                className="w-full px-4 py-3 bg-black/40 border border-white/20 rounded-xl text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-green-500 transition-colors"
              />
            </div>
          </div>
        </div>
      )}

      {!showApiConfig && (
        <ManagedApiNotice className="mb-6" />
      )}

      {/* 错误提示 */}
      {error && (
        <div className="glass rounded-xl p-4 mb-6 border border-red-500/50 bg-red-500/10">
          <div className="flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <p className="text-red-300 text-sm">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* 进度条 */}
      {isUploading && (
        <div className="glass rounded-xl p-6 mb-6 border border-white/10">
          <div className="flex items-center gap-3 mb-3">
            <Loader2 className="w-5 h-5 text-green-400 animate-spin" />
            <span className="text-white font-medium">{progressStatus}</span>
          </div>
          <div className="w-full bg-white/10 rounded-full h-2 overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-green-500 to-emerald-500 transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="text-sm text-gray-400 mt-2">{progress.toFixed(0)}%</p>
        </div>
      )}

      {/* 提交按钮 */}
      <button
        onClick={handleUploadAndGenerate}
        disabled={isUploading || isValidating || !selectedFile}
        className="w-full py-4 px-6 bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 disabled:from-gray-600 disabled:to-gray-600 text-white font-semibold rounded-xl transition-all flex items-center justify-center gap-2 disabled:cursor-not-allowed"
      >
        {isValidating ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            {t('upload.validating', '验证中...')}
          </>
        ) : isUploading ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            {t('upload.generating', '生成中...')}
          </>
        ) : (
          <>
            {t('upload.startGenerate', '开始生成海报')}
            <ArrowRight className="w-5 h-5" />
          </>
        )}
      </button>

      {/* 提示信息 */}
      <div className="mt-6 p-4 bg-blue-500/10 border border-blue-500/30 rounded-xl">
        <p className="text-sm text-blue-300 mb-2">💡 {t('upload.tips', '提示：')}</p>
        <ul className="text-sm text-gray-300 space-y-1 ml-4">
          <li>• {t('upload.tip1', '海报宽高比建议在1.4-2.0之间')}</li>
          <li>• {t('upload.tip2', '推荐尺寸：54" × 36"（标准会议海报）')}</li>
          <li>• {t('upload.tip3', '生成过程可能需要几分钟')}</li>
          <li>• {t('upload.tip4', '输出包含PPTX（可编辑）和PNG文件')}</li>
        </ul>
      </div>
    </div>
  );
};

export default UploadStep;
