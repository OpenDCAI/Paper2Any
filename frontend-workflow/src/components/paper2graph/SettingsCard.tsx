import React from 'react';
import { Settings2, ChevronUp, ChevronDown, Loader2, Download, Info, CheckCircle2, AlertCircle, ImageIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import QRCodeTooltip from '../QRCodeTooltip';
import { GraphType, Language, StyleType, FigureComplex } from './types';
import { GENERATION_STAGES } from './constants';

interface SettingsCardProps {
  showAdvanced: boolean;
  setShowAdvanced: React.Dispatch<React.SetStateAction<boolean>>;
  llmApiUrl: string;
  setLlmApiUrl: (url: string) => void;
  setModel: (model: string) => void;
  apiKey: string;
  setApiKey: (key: string) => void;
  model: string;
  graphType: GraphType;
  figureComplex: FigureComplex;
  setFigureComplex: (complex: FigureComplex) => void;
  language: Language;
  setLanguage: (lang: Language) => void;
  style: StyleType;
  setStyle: (style: StyleType) => void;
  isLoading: boolean;
  handleSubmit: () => void;
  currentStage: number;
  stageProgress: number;
  downloadUrl: string | null;
  lastFilename: string;
  pptPath: string | null;
  svgPath: string | null;
  svgPreviewPath: string | null;
  isValidating: boolean;
  error: string | null;
  successMessage: string | null;
}

const SettingsCard: React.FC<SettingsCardProps> = ({
  showAdvanced,
  setShowAdvanced,
  llmApiUrl,
  setLlmApiUrl,
  setModel,
  apiKey,
  setApiKey,
  model,
  graphType,
  figureComplex,
  setFigureComplex,
  language,
  setLanguage,
  style,
  setStyle,
  isLoading,
  handleSubmit,
  currentStage,
  stageProgress,
  downloadUrl,
  lastFilename,
  pptPath,
  svgPath,
  svgPreviewPath,
  isValidating,
  error,
  successMessage,
}) => {
  const { t } = useTranslation('paper2graph');

  return (
    <div className="glass rounded-xl border border-white/10 p-5 flex flex-col gap-4 text-sm">
      <button
        type="button"
        onClick={() => setShowAdvanced(v => !v)}
        className="flex items-center justify-between gap-2 mb-1 w-full text-left"
      >
        <div className="flex items-center gap-2">
          <Settings2 size={16} className="text-primary-300" />
          <span className="text-white font-medium">{t('advanced.title')}</span>
        </div>
        {showAdvanced ? (
          <ChevronUp size={16} className="text-gray-400" />
        ) : (
          <ChevronDown size={16} className="text-gray-400" />
        )}
      </button>

      {showAdvanced && (
        <div className="space-y-3">
          <div>
            <label className="block text-xs text-gray-400 mb-1">{t('advanced.apiUrlLabel')}</label>
            <div className="flex items-center gap-2">
              <select
                value={llmApiUrl}
                onChange={e => {
                  const val = e.target.value;
                  setLlmApiUrl(val);
                  if (val === 'http://123.129.219.111:3000/v1') {
                    setModel('gemini-3-pro-image-preview');
                  }
                }}
                className="flex-1 rounded-lg border border-white/10 bg-black/30 px-3 py-2 text-xs text-gray-200 outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              >
                <option value="https://api.apiyi.com/v1">https://api.apiyi.com/v1</option>
                <option value="http://b.apiyi.com:16888/v1">http://b.apiyi.com:16888/v1</option>
                <option value="http://123.129.219.111:3000/v1">http://123.129.219.111:3000/v1</option>
              </select>
              <QRCodeTooltip>
                <a
                  href={llmApiUrl === 'http://123.129.219.111:3000/v1' ? "http://123.129.219.111:3000" : "https://api.apiyi.com/register/?aff_code=TbrD"}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="whitespace-nowrap text-[10px] text-primary-300 hover:text-primary-200 hover:underline px-2"
                >
                  {t('advanced.buyLink')}
                </a>
              </QRCodeTooltip>
            </div>
          </div>

          <div>
            <label className="block text-xs text-gray-400 mb-1">
              {t('advanced.apiKeyLabel')}
            </label>
            <input
              type="password"
              value={apiKey}
              onChange={e => setApiKey(e.target.value)}
              placeholder={t('advanced.apiKeyPlaceholder')}
              className="w-full rounded-lg border border-white/10 bg-black/30 px-3 py-2 text-xs text-gray-200 outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            />
          </div>

          <div>
            <label className="block text-xs text-gray-400 mb-1">{t('advanced.modelLabel')}</label>
            <select
              value={model}
              onChange={e => setModel(e.target.value)}
              disabled={llmApiUrl === 'http://123.129.219.111:3000/v1'}
              className="w-full rounded-lg border border-white/10 bg-black/30 px-3 py-2 text-xs text-gray-200 outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <option value="gemini-3-pro-image-preview">gemini-3-pro-image-preview</option>
              <option value="gemini-2.5-flash-image-preview">gemini-2.5-flash-image-preview</option>
            </select>
            {llmApiUrl === 'http://123.129.219.111:3000/v1' && (
               <p className="text-[10px] text-gray-500 mt-1">{t('advanced.modelOnlyHint')}</p>
            )}
          </div>

          {graphType === 'model_arch' ? (
            <>
              <div>
                <label className="block text-xs text-gray-400 mb-1">{t('advanced.figureComplexLabel')}</label>
                <select
                  value={figureComplex}
                  onChange={e => setFigureComplex(e.target.value as FigureComplex)}
                  className="w-full rounded-lg border border-white/10 bg-black/30 px-3 py-2 text-xs text-gray-200 outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                >
                  <option value="easy">{t('advanced.figureComplex.easy')}</option>
                  <option value="mid">{t('advanced.figureComplex.mid')}</option>
                  <option value="hard">{t('advanced.figureComplex.hard')}</option>
                </select>
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">{t('advanced.languageLabel')}</label>
                <select
                  value={language}
                  onChange={e => setLanguage(e.target.value as Language)}
                  className="w-full rounded-lg border border-white/10 bg-black/30 px-3 py-2 text-xs text-gray-200 outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                >
                  <option value="zh">{t('advanced.language.zh')}</option>
                  <option value="en">{t('advanced.language.en')}</option>
                </select>
              </div>
            </>
          ) : (
            <div>
              <label className="block text-xs text-gray-400 mb-1">{t('advanced.languageLabel')}</label>
              <select
                value={language}
                onChange={e => setLanguage(e.target.value as Language)}
                className="w-full rounded-lg border border-white/10 bg-black/30 px-3 py-2 text-xs text-gray-200 outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              >
                <option value="zh">{t('advanced.language.zh')}</option>
                <option value="en">{t('advanced.language.en')}</option>
              </select>
            </div>
          )}

          <div>
            <label className="block text-xs text-gray-400 mb-1">{t('advanced.styleLabel')}</label>
            <select
              value={style}
              onChange={e => setStyle(e.target.value as StyleType)}
              className="w-full rounded-lg border border-white/10 bg-black/30 px-3 py-2 text-xs text-gray-200 outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            >
              <option value="cartoon">{t('advanced.style.cartoon')}</option>
              {graphType !== 'exp_data' && <option value="realistic">{t('advanced.style.realistic')}</option>}
              {graphType === 'exp_data' && <option value="Low Poly 3D">{t('advanced.style.lowPoly')}</option>}
              {graphType === 'exp_data' && <option value="blocky LEGO aesthetic">{t('advanced.style.lego')}</option>}
            </select>
          </div>
        </div>
      )}

      <div className="mt-auto space-y-2 pt-2">
        <button
          type="button"
          onClick={handleSubmit}
          disabled={isLoading}
          className="w-full inline-flex items-center justify-center gap-2 rounded-lg bg-primary-500 hover:bg-primary-600 disabled:bg-primary-500/60 disabled:cursor-not-allowed text-white text-sm font-medium py-2.5 transition-colors glow"
        >
          {isLoading ? <Loader2 size={16} className="animate-spin" /> : <Download size={16} />}
          <span>{isLoading ? t('submit.buttonLoading') : t('submit.buttonIdle')}</span>
        </button>

        <div className="flex items-start gap-2 text-xs text-gray-400 bg-white/5 border border-white/10 rounded-lg px-3 py-2">
          <Info size={14} className="mt-0.5 text-gray-500 flex-shrink-0" />
          <p>{t('submit.hintText')}</p>
        </div>

        {/* 改进的生成进度显示 */}
        {isLoading && !error && !successMessage && (
          <div className="flex flex-col gap-3 mt-2 text-xs rounded-lg border border-primary-400/40 bg-primary-500/10 px-3 py-3">
            <div className="flex items-center gap-2 text-primary-200">
              <Loader2 size={14} className="animate-spin" />
              <span className="font-medium">{GENERATION_STAGES[currentStage].message}</span>
            </div>
            
            {/* 阶段指示器 */}
            <div className="flex gap-1">
              {GENERATION_STAGES.map((stage, index) => (
                <div
                  key={stage.id}
                  className={`flex-1 h-1.5 rounded-full transition-all duration-500 ${
                    index < currentStage
                      ? 'bg-primary-400'
                      : index === currentStage
                      ? 'bg-gradient-to-r from-primary-400 to-primary-400/40'
                      : 'bg-primary-950/60'
                  }`}
                  style={{
                    width: index === currentStage ? `${stageProgress}%` : undefined,
                  }}
                />
              ))}
            </div>

            {/* 阶段详细信息 */}
            <div className="space-y-1.5 text-[11px] text-primary-200/80">
              <div className="flex items-center gap-1.5">
                <div className={`w-1.5 h-1.5 rounded-full ${currentStage >= 0 ? 'bg-primary-400 animate-pulse' : 'bg-primary-950/60'}`} />
                <span className={currentStage >= 0 ? 'text-primary-200 font-medium' : ''}>
                  {t('progress.stage1')}
                </span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className={`w-1.5 h-1.5 rounded-full ${currentStage >= 1 ? 'bg-primary-400 animate-pulse' : 'bg-primary-950/60'}`} />
                <span className={currentStage >= 1 ? 'text-primary-200 font-medium' : ''}>
                  {t('progress.stage2')}
                </span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className={`w-1.5 h-1.5 rounded-full ${currentStage >= 2 ? 'bg-primary-400 animate-pulse' : 'bg-primary-950/60'}`} />
                <span className={currentStage >= 2 ? 'text-primary-200 font-medium' : ''}>
                  {t('progress.stage3')}
                </span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className={`w-1.5 h-1.5 rounded-full ${currentStage >= 3 ? 'bg-primary-400 animate-pulse' : 'bg-primary-950/60'}`} />
                <span className={currentStage >= 3 ? 'text-primary-200 font-medium' : ''}>
                  {t('progress.stage4')}
                </span>
              </div>
            </div>

            <p className="text-[11px] text-primary-200/70 pt-1 border-t border-primary-400/20">
              {t('progress.eta')}
            </p>
          </div>
        )}

        {downloadUrl && (
          <button
            type="button"
            onClick={() => {
              if (!downloadUrl) return;
              const a = document.createElement('a');
              a.href = downloadUrl;
              a.download = lastFilename;
              document.body.appendChild(a);
              a.click();
              a.remove();
            }}
            className="w-full inline-flex items-center justify-center gap-2 rounded-lg border border-emerald-400/60 text-emerald-300 text-xs py-2 bg-emerald-500/10 hover:bg-emerald-500/20 transition-colors"
          >
            <CheckCircle2 size={14} />
            <span>{t('download.reDownload', { filename: lastFilename })}</span>
          </button>
        )}

        {graphType === 'tech_route' && (pptPath || svgPath || svgPreviewPath) && (
          <div className="mt-2 space-y-2">
            {svgPath && (
              <>
                <button
                  type="button"
                  onClick={() => {
                    if (!svgPath) return;
                    window.open(svgPath, '_blank');
                  }}
                  className="w-full inline-flex items-center justify-center gap-2 rounded-lg border border-sky-400/60 text-sky-300 text-xs py-2 bg-sky-500/10 hover:bg-sky-500/20 transition-colors"
                >
                  <ImageIcon size={14} />
                  <span>SVG 源文件下载</span>
                </button>
                <div className="text-[11px] text-gray-300 bg-black/30 border border-white/10 rounded-md px-2 py-1.5">
                  <div className="font-semibold text-gray-200">SVG 链接：</div>
                  <div className="mt-1 break-all text-sky-300 select-all cursor-text font-mono text-[10px] leading-tight p-1 bg-black/20 rounded">
                    {svgPath}
                  </div>
                </div>
              </>
            )}
          </div>
        )}

        {isValidating && (
          <div className="flex items-start gap-2 text-xs text-blue-300 bg-blue-500/10 border border-blue-500/40 rounded-lg px-3 py-2 mt-1 animate-pulse">
            <Loader2 size={14} className="mt-0.5 animate-spin" />
            <p>{t('validating.apiKey')}</p>
          </div>
        )}

        {error && (
          <div className="flex items-start gap-2 text-xs text-red-300 bg-red-500/10 border border-red-500/40 rounded-lg px-3 py-2 mt-1">
            <AlertCircle size={14} className="mt-0.5" />
            <p>{error}</p>
          </div>
        )}

        {successMessage && !error && (
          <div className="flex items-start gap-2 text-xs text-emerald-300 bg-emerald-500/10 border border-emerald-500/40 rounded-lg px-3 py-2 mt-1">
            <CheckCircle2 size={14} className="mt-0.5" />
            <p>{successMessage}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default SettingsCard;
