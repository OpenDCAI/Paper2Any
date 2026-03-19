import React, { useState, useRef, useEffect } from 'react';
import { Settings2, ChevronUp, ChevronDown, Loader2, Download, Info, CheckCircle2, AlertCircle, ImageIcon, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import QRCodeTooltip from '../QRCodeTooltip';
import { GraphType, Language, StyleType, FigureComplex } from './types';
import { GENERATION_STAGES, TECH_ROUTE_PALETTES, TECH_ROUTE_TEMPLATES } from './constants';
import { API_URL_OPTIONS, getPurchaseUrl } from '../../config/api';
import {
  DEFAULT_PAPER2FIGURE_MODELS,
  PAPER2FIGURE_EXP_DATA_MODELS,
  PAPER2FIGURE_MODEL_ARCH_MODELS,
  PAPER2FIGURE_TECH_ROUTE_MODELS,
  withModelOptions,
} from '../../config/models';

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
  resolution: '2K' | '4K';
  setResolution: (resolution: '2K' | '4K') => void;
  isLoading: boolean;
  handleSubmit: () => void;
  currentStage: number;
  stageProgress: number;
  downloadUrl: string | null;
  lastFilename: string;
  pptPath: string | null;
  svgPath: string | null;
  svgPreviewPath: string | null;
  svgBwPath: string | null;
  svgColorPath: string | null;
  techRoutePalette: string;
  setTechRoutePalette: (palette: string) => void;
  techRouteTemplate: string;
  setTechRouteTemplate: (templateId: string) => void;
  referenceImage: File | null;
  setReferenceImage: (file: File | null) => void;
  referenceImagePreview: string | null;
  setReferenceImagePreview: (url: string | null) => void;
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
  resolution,
  setResolution,
  isLoading,
  handleSubmit,
  currentStage,
  stageProgress,
  downloadUrl,
  lastFilename,
  pptPath,
  svgPath,
  svgPreviewPath,
  svgBwPath,
  svgColorPath,
  techRoutePalette,
  setTechRoutePalette,
  techRouteTemplate,
  setTechRouteTemplate,
  referenceImage,
  setReferenceImage,
  referenceImagePreview,
  setReferenceImagePreview,
  isValidating,
  error,
  successMessage,
}) => {
  const { t } = useTranslation('paper2graph');
  const selectedPalette = TECH_ROUTE_PALETTES.find(p => p.id === techRoutePalette) || TECH_ROUTE_PALETTES[0];
  const defaultModelForType = DEFAULT_PAPER2FIGURE_MODELS[graphType] || DEFAULT_PAPER2FIGURE_MODELS.model_arch;
  const baseModelOptions = graphType === 'tech_route'
    ? PAPER2FIGURE_TECH_ROUTE_MODELS
    : graphType === 'exp_data'
      ? PAPER2FIGURE_EXP_DATA_MODELS
      : PAPER2FIGURE_MODEL_ARCH_MODELS;
  const modelOptions = withModelOptions(baseModelOptions, model);

  // 配色方案下拉框状态
  const [paletteDropdownOpen, setPaletteDropdownOpen] = useState(false);
  const paletteDropdownRef = useRef<HTMLDivElement>(null);
  const [templatePreview, setTemplatePreview] = useState<{ src: string; label: string } | null>(null);

  // 点击外部关闭下拉框
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (paletteDropdownRef.current && !paletteDropdownRef.current.contains(event.target as Node)) {
        setPaletteDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div className="portal-panel-dark rounded-xl border border-primary-100 p-5 flex flex-col gap-4 text-sm">
      <button
        type="button"
        onClick={() => setShowAdvanced(v => !v)}
        className="flex items-center justify-between gap-2 mb-1 w-full text-left"
      >
        <div className="flex items-center gap-2">
          <Settings2 size={16} className="text-primary-700" />
          <span className="font-medium text-[var(--text-primary)]">{t('advanced.title')}</span>
        </div>
        {showAdvanced ? (
          <ChevronUp size={16} className="text-[#8b726b]" />
        ) : (
          <ChevronDown size={16} className="text-[#8b726b]" />
        )}
      </button>

      {showAdvanced && (
        <div className="space-y-3">
          <div>
            <label className="mb-1 block text-xs text-[var(--text-secondary)]">{t('advanced.apiUrlLabel')}</label>
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
                className="portal-input-soft flex-1 rounded-lg px-3 py-2 text-xs outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              >
                {API_URL_OPTIONS.map((url: string) => (
                  <option key={url} value={url}>{url}</option>
                ))}
              </select>
              <QRCodeTooltip>
                <a
                  href={getPurchaseUrl(llmApiUrl)}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="whitespace-nowrap px-2 text-[10px] text-primary-700 hover:text-primary-800 hover:underline"
                >
                  {t('advanced.buyLink')}
                </a>
              </QRCodeTooltip>
            </div>
          </div>

          <div>
            <label className="mb-1 block text-xs text-[var(--text-secondary)]">
              {t('advanced.apiKeyLabel')}
            </label>
            <input
              type="password"
              value={apiKey}
              onChange={e => setApiKey(e.target.value)}
              placeholder={t('advanced.apiKeyPlaceholder')}
              className="portal-input-soft w-full rounded-lg px-3 py-2 text-xs outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            />
          </div>

          <div>
            <label className="mb-1 block text-xs text-[var(--text-secondary)]">{t('advanced.modelLabel')}</label>
            <select
              value={model}
              onChange={e => setModel(e.target.value)}
              disabled={llmApiUrl === 'http://123.129.219.111:3000/v1'}
              className="portal-input-soft w-full rounded-lg px-3 py-2 text-xs outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {graphType === 'tech_route' ? (
                <>
                  {modelOptions.map((option) => (
                    <option key={option} value={option}>
                      {option === defaultModelForType ? `${option}（默认）` : option}
                    </option>
                  ))}
                </>
              ) : (
                <>
                  {modelOptions.map((option) => (
                    <option key={option} value={option}>
                      {option === defaultModelForType ? `${option}（默认）` : option}
                    </option>
                  ))}
                </>
              )}
            </select>
            {llmApiUrl === 'http://123.129.219.111:3000/v1' && (
               <p className="mt-1 text-[10px] text-[#8b726b]">{t('advanced.modelOnlyHint')}</p>
            )}
          </div>

          {graphType === 'model_arch' ? (
            <>
              <div>
                <label className="mb-1 block text-xs text-[var(--text-secondary)]">{t('advanced.figureComplexLabel')}</label>
                <select
                  value={figureComplex}
                  onChange={e => setFigureComplex(e.target.value as FigureComplex)}
                  className="portal-input-soft w-full rounded-lg px-3 py-2 text-xs outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                >
                  <option value="easy">{t('advanced.figureComplex.easy')}</option>
                  <option value="mid">{t('advanced.figureComplex.mid')}</option>
                  <option value="hard">{t('advanced.figureComplex.hard')}</option>
                </select>
              </div>
              <div>
                <label className="mb-1 block text-xs text-[var(--text-secondary)]">{t('advanced.languageLabel')}</label>
                <select
                  value={language}
                  onChange={e => setLanguage(e.target.value as Language)}
                  className="portal-input-soft w-full rounded-lg px-3 py-2 text-xs outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                >
                  <option value="zh">{t('advanced.language.zh')}</option>
                  <option value="en">{t('advanced.language.en')}</option>
                </select>
              </div>
              <div>
                <label className="mb-1 block text-xs text-[var(--text-secondary)]">{t('advanced.resolutionLabel')}</label>
                <select
                  value={resolution}
                  onChange={e => setResolution(e.target.value as '2K' | '4K')}
                  className="portal-input-soft w-full rounded-lg px-3 py-2 text-xs outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                >
                  <option value="2K">{t('advanced.resolution.2k')}</option>
                  <option value="4K">{t('advanced.resolution.4k')}</option>
                </select>
              </div>
            </>
          ) : (
            <div>
              <label className="mb-1 block text-xs text-[var(--text-secondary)]">{t('advanced.languageLabel')}</label>
              <select
                value={language}
                onChange={e => setLanguage(e.target.value as Language)}
                className="portal-input-soft w-full rounded-lg px-3 py-2 text-xs outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              >
                <option value="zh">{t('advanced.language.zh')}</option>
                <option value="en">{t('advanced.language.en')}</option>
              </select>
            </div>
          )}

          {/* 技术路线图不显示风格选择 */}
          {graphType !== 'tech_route' && (
            <div>
              <label className="mb-1 block text-xs text-[var(--text-secondary)]">{t('advanced.styleLabel')}</label>
              <select
                value={style}
                onChange={e => setStyle(e.target.value as StyleType)}
                className="portal-input-soft w-full rounded-lg px-3 py-2 text-xs outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              >
                <option value="cartoon">{t('advanced.style.cartoon')}</option>
                {graphType !== 'exp_data' && <option value="realistic">{t('advanced.style.realistic')}</option>}
                {graphType !== 'exp_data' && <option value="3d">{t('advanced.style.3d')}</option>}
                {graphType !== 'exp_data' && <option value="flat_2.5d">{t('advanced.style.flat_2.5d')}</option>}
                {graphType !== 'exp_data' && <option value="line_art">{t('advanced.style.line_art')}</option>}
                {graphType !== 'exp_data' && <option value="low_poly">{t('advanced.style.low_poly')}</option>}
                {graphType !== 'exp_data' && <option value="neon_glow">{t('advanced.style.neon_glow')}</option>}
                {graphType === 'exp_data' && <option value="Low Poly 3D">{t('advanced.style.lowPoly')}</option>}
                {graphType === 'exp_data' && <option value="blocky LEGO aesthetic">{t('advanced.style.lego')}</option>}
              </select>
            </div>
          )}

          {graphType === 'tech_route' && (
            <div className="space-y-2">
              <label className="block text-xs text-[var(--text-secondary)]">{t('techRoute.templateLabel')}</label>
              <div className="grid grid-cols-2 gap-2">
                {TECH_ROUTE_TEMPLATES.map((tpl) => {
                  const isActive = techRouteTemplate === tpl.id;
                  return (
                    <button
                      key={tpl.id || 'auto'}
                      type="button"
                      onClick={() => setTechRouteTemplate(tpl.id)}
                      className={`rounded-lg border text-left transition-all ${
                        isActive
                          ? 'border-primary-400/80 bg-primary-500/10'
                          : 'border-primary-100 bg-white/72 hover:bg-white'
                      }`}
                    >
                      <div className="p-2">
                        <div className="relative flex h-20 items-center justify-center overflow-hidden rounded-md border border-primary-100 bg-[#faf4ee] group">
                          {tpl.preview ? (
                            <img
                              src={tpl.preview}
                              alt={t(tpl.labelKey)}
                              className="w-full h-full object-cover"
                            />
                          ) : (
                            <div className="px-2 text-center text-[10px] text-[var(--text-secondary)]">
                              {t('techRoute.templateAuto')}
                            </div>
                          )}
                          {tpl.preview && (
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                setTemplatePreview({ src: tpl.preview, label: t(tpl.labelKey) });
                              }}
                              className="absolute bottom-1 right-1 rounded-full border border-primary-200 bg-white/92 px-1.5 py-0.5 text-[9px] text-primary-800 opacity-0 transition-opacity group-hover:opacity-100"
                            >
                              {t('techRoute.templateZoom')}
                            </button>
                          )}
                          {isActive && (
                            <span className="absolute top-1 right-1 text-[9px] px-1.5 py-0.5 rounded-full bg-primary-500/80 text-white">
                              {t('techRoute.templateSelected')}
                            </span>
                          )}
                        </div>
                        <div className="mt-1 text-[10px] text-[var(--text-primary)]">{t(tpl.labelKey)}</div>
                      </div>
                    </button>
                  );
                })}
              </div>
              <p className="text-[10px] text-[#8b726b]">{t('techRoute.templateHint')}</p>
            </div>
          )}

          {graphType === 'tech_route' && (
            <div ref={paletteDropdownRef} className="relative">
              <label className="mb-1 block text-xs text-[var(--text-secondary)]">{t('techRoute.paletteLabel')}</label>
              {/* 自定义下拉框触发器 */}
              <button
                type="button"
                onClick={() => setPaletteDropdownOpen(!paletteDropdownOpen)}
                className="portal-input-soft flex w-full items-center justify-between rounded-lg px-3 py-2 text-xs outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              >
                <div className="flex items-center gap-2">
                  <span>{selectedPalette.label}</span>
                  {selectedPalette.colors.length > 0 && (
                    <div className="flex items-center gap-1">
                      {selectedPalette.colors.map(color => (
                        <span
                          key={color}
                          className="w-3 h-3 rounded-full border border-white/20"
                          style={{ backgroundColor: color }}
                        />
                      ))}
                    </div>
                  )}
                </div>
                <ChevronDown size={14} className={`transition-transform ${paletteDropdownOpen ? 'rotate-180' : ''}`} />
              </button>
              {/* 下拉选项列表 */}
              {paletteDropdownOpen && (
                <div className="absolute z-50 mt-1 max-h-48 w-full overflow-y-auto rounded-lg border border-primary-100 bg-[#fffaf6] shadow-lg">
                  {TECH_ROUTE_PALETTES.map(palette => (
                    <button
                      key={palette.id || 'none'}
                      type="button"
                      onClick={() => {
                        setTechRoutePalette(palette.id);
                        setPaletteDropdownOpen(false);
                      }}
                      className={`flex w-full items-center gap-2 px-3 py-2 text-left text-xs transition-colors hover:bg-primary-50 ${
                        techRoutePalette === palette.id ? 'bg-primary-100 text-primary-800' : 'text-[var(--text-primary)]'
                      }`}
                    >
                      <span className="flex-shrink-0">{palette.label}</span>
                      {palette.colors.length > 0 && (
                        <div className="flex items-center gap-1 ml-auto">
                          {palette.colors.map(color => (
                            <span
                              key={color}
                              className="w-3 h-3 rounded-full border border-white/20"
                              style={{ backgroundColor: color }}
                              title={color}
                            />
                          ))}
                        </div>
                      )}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* 技术路线图参考图上传 */}
          {graphType === 'tech_route' && (
            <div className="mt-3">
              <label className="mb-1 block text-xs text-[var(--text-secondary)]">参考图（可选）</label>
              <div className="rounded-lg border border-dashed border-primary-200 bg-white/60 p-3">
                {referenceImagePreview ? (
                  <div className="relative">
                    <img
                      src={referenceImagePreview}
                      alt="参考图预览"
                      className="max-h-32 rounded mx-auto"
                    />
                    <button
                      type="button"
                      onClick={() => {
                        setReferenceImage(null);
                        setReferenceImagePreview(null);
                      }}
                      className="absolute top-1 right-1 bg-red-500 hover:bg-red-600 rounded-full p-1 transition-colors"
                    >
                      <X size={12} className="text-white" />
                    </button>
                  </div>
                ) : (
                  <label className="flex flex-col items-center cursor-pointer py-2">
                    <ImageIcon size={24} className="mb-1 text-[#8b726b]" />
                    <span className="text-xs text-[var(--text-secondary)]">点击上传参考图</span>
                    <input
                      type="file"
                      accept="image/png,image/jpeg,image/jpg,image/webp"
                      className="hidden"
                      onChange={(e) => {
                        const file = e.target.files?.[0];
                        if (file) {
                          // 验证文件类型，拒绝 SVG
                          const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp'];
                          if (!allowedTypes.includes(file.type)) {
                            alert('仅支持 PNG、JPG、WebP 格式的图片，不支持 SVG 格式');
                            e.target.value = ''; // 清空文件选择
                            return;
                          }
                          setReferenceImage(file);
                          setReferenceImagePreview(URL.createObjectURL(file));
                        }
                      }}
                    />
                  </label>
                )}
              </div>
              <p className="mt-1 text-[10px] text-[#8b726b]">
                上传参考图后，AI将分析其布局风格生成类似的技术路线图
              </p>
            </div>
          )}
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

        <div className="portal-card-soft flex items-start gap-2 rounded-lg px-3 py-2 text-xs text-[var(--text-secondary)]">
          <Info size={14} className="mt-0.5 flex-shrink-0 text-[#8b726b]" />
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

        {graphType === 'tech_route' && (pptPath || svgPath || svgPreviewPath || svgBwPath || svgColorPath) && (
          <div className="mt-2 space-y-2">
            {(svgBwPath || svgPath) && (
              <>
                <button
                  type="button"
                  onClick={async () => {
                    const bwPath = svgBwPath || svgPath;
                    if (!bwPath) return;
                    try {
                      const response = await fetch(bwPath);
                      const blob = await response.blob();
                      const blobUrl = URL.createObjectURL(blob);
                      const a = document.createElement('a');
                      a.href = blobUrl;
                      a.download = bwPath.split('/').pop() || 'technical_route_bw.svg';
                      document.body.appendChild(a);
                      a.click();
                      a.remove();
                      URL.revokeObjectURL(blobUrl);
                    } catch {
                      window.open(bwPath, '_blank');
                    }
                  }}
                  className="w-full inline-flex items-center justify-center gap-2 rounded-lg border border-primary-400/60 bg-primary-500/10 py-2 text-xs text-primary-800 transition-colors hover:bg-primary-500/20"
                >
                  <ImageIcon size={14} />
                  <span>黑白 SVG 源文件下载</span>
                </button>
                <div className="rounded-md border border-primary-100 bg-white/72 px-2 py-1.5 text-[11px] text-[var(--text-secondary)]">
                  <div className="font-semibold text-[var(--text-primary)]">黑白 SVG 链接：</div>
                  <div className="mt-1 cursor-text break-all rounded bg-white/88 p-1 font-mono text-[10px] leading-tight text-primary-800 select-all">
                    {svgBwPath || svgPath}
                  </div>
                </div>
              </>
            )}

            {svgColorPath && (
              <>
                <button
                  type="button"
                  onClick={async () => {
                    try {
                      const response = await fetch(svgColorPath);
                      const blob = await response.blob();
                      const blobUrl = URL.createObjectURL(blob);
                      const a = document.createElement('a');
                      a.href = blobUrl;
                      a.download = svgColorPath.split('/').pop() || 'technical_route_color.svg';
                      document.body.appendChild(a);
                      a.click();
                      a.remove();
                      URL.revokeObjectURL(blobUrl);
                    } catch {
                      window.open(svgColorPath, '_blank');
                    }
                  }}
                  className="w-full inline-flex items-center justify-center gap-2 rounded-lg border border-amber-400/60 bg-amber-500/10 py-2 text-xs text-amber-800 transition-colors hover:bg-amber-500/20"
                >
                  <ImageIcon size={14} />
                  <span>彩色 SVG 源文件下载</span>
                </button>
                <div className="rounded-md border border-amber-100 bg-white/72 px-2 py-1.5 text-[11px] text-[var(--text-secondary)]">
                  <div className="font-semibold text-[var(--text-primary)]">彩色 SVG 链接：</div>
                  <div className="mt-1 cursor-text break-all rounded bg-[#fff8ef] p-1 font-mono text-[10px] leading-tight text-amber-800 select-all">
                    {svgColorPath}
                  </div>
                </div>
              </>
            )}
          </div>
        )}

        {isValidating && (
          <div className="flex items-start gap-2 text-xs text-[#f0d7d1] bg-primary-500/10 border border-primary-500/40 rounded-lg px-3 py-2 mt-1 animate-pulse">
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

      {templatePreview && (
        <div
          className="fixed inset-0 z-[999] flex items-center justify-center bg-[rgba(54,24,33,0.32)] p-6 backdrop-blur-sm"
          onClick={() => setTemplatePreview(null)}
        >
          <div
            className="portal-card-strong max-w-4xl w-full rounded-xl p-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-3">
              <div className="text-sm text-[var(--text-primary)]">{templatePreview.label}</div>
              <button
                type="button"
                onClick={() => setTemplatePreview(null)}
                className="rounded bg-white/82 px-2 py-1 text-xs text-primary-800 hover:bg-white"
              >
                {t('techRoute.templateClose')}
              </button>
            </div>
            <div className="w-full max-h-[70vh] overflow-auto rounded-lg border border-primary-100 bg-[#faf4ee]">
              <img
                src={templatePreview.src}
                alt={templatePreview.label}
                className="w-full h-auto object-contain"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default SettingsCard;
