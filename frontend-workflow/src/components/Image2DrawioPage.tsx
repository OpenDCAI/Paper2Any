import { useState, useEffect, useRef, useCallback, ChangeEvent, DragEvent } from 'react';
import { useTranslation } from 'react-i18next';
import {
  UploadCloud,
  FileImage,
  Loader2,
  AlertCircle,
  Download,
  Copy,
  Wand2,
  ExternalLink,
  Key,
  Image as ImageIcon,
  Sparkles,
} from 'lucide-react';
import { API_KEY, API_URL_OPTIONS, DEFAULT_LLM_API_URL } from '../config/api';
import { useAuthStore } from '../stores/authStore';
import { getApiSettings, saveApiSettings } from '../services/apiSettingsService';
import { checkQuota, recordUsage } from '../services/quotaService';

const MAX_FILE_SIZE = 15 * 1024 * 1024; // 15MB
const STORAGE_KEY = 'image2drawio_settings';
const DRAWIO_ORIGINS = new Set(['https://embed.diagrams.net', 'https://app.diagrams.net']);
const DRAWIO_EXPORT_TIMEOUT_MS = 5000;
const DRAWIO_ANIMATE_STEP_MS = 60;
const DRAWIO_ANIMATE_MAX_CELLS = 240;
const DRAWIO_ANIMATE_LARGE_BATCH = 5;
const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

const panelClass =
  'rounded-2xl bg-white/5 border border-white/10 p-4 backdrop-blur-xl shadow-[0_20px_60px_rgba(0,0,0,0.25)]';
const inputClass =
  'w-full rounded-xl bg-white/5 border border-white/10 px-3 py-2 text-sm text-white placeholder-slate-500 outline-none transition focus:border-amber-400/60 focus:ring-2 focus:ring-amber-500/20';

const Image2DrawioPage = () => {
  const { t } = useTranslation(['image2drawio', 'common']);
  const { user, refreshQuota } = useAuthStore();

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState('');
  const [xmlContent, setXmlContent] = useState('');
  const [filePath, setFilePath] = useState('');
  const [copySuccess, setCopySuccess] = useState('');
  const [drawioReady, setDrawioReady] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [exportFormat, setExportFormat] = useState<'drawio' | 'png' | 'svg'>('drawio');
  const [exportFilename, setExportFilename] = useState('diagram');

  const [apiUrl, setApiUrl] = useState(DEFAULT_LLM_API_URL);
  const [apiKey, setApiKey] = useState('');
  const [genFigModel, setGenFigModel] = useState('gemini-3-pro-image-preview');
  const [vlmModel, setVlmModel] = useState('qwen-vl-ocr-2025-11-20');

  const iframeRef = useRef<HTMLIFrameElement>(null);
  const lastLoadedXmlRef = useRef('');
  const isAnimatingRef = useRef(false);
  const animationTokenRef = useRef(0);
  const pendingExportRef = useRef<{
    resolve: ((data: string) => void) | null;
    reject: ((error: Error) => void) | null;
    format: 'xml' | 'png' | 'svg' | null;
  }>({ resolve: null, reject: null, format: null });

  useEffect(() => {
    const userApiSettings = getApiSettings(user?.id || null);
    if (userApiSettings) {
      if (userApiSettings.apiUrl) setApiUrl(userApiSettings.apiUrl);
      if (userApiSettings.apiKey) setApiKey(userApiSettings.apiKey);
    }
  }, [user?.id]);

  useEffect(() => {
    if (!user?.id) return;
    if (apiUrl && apiKey) {
      saveApiSettings(user.id, { apiUrl, apiKey });
    }
  }, [apiUrl, apiKey, user?.id]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      const saved = JSON.parse(raw) as Partial<{
        apiUrl: string;
        apiKey: string;
        genFigModel: string;
        vlmModel: string;
        xmlContent: string;
        exportFormat: 'drawio' | 'png' | 'svg';
        exportFilename: string;
      }>;
      if (saved.apiUrl) setApiUrl(saved.apiUrl);
      if (saved.apiKey) setApiKey(saved.apiKey);
      if (saved.genFigModel) setGenFigModel(saved.genFigModel);
      if (saved.vlmModel) setVlmModel(saved.vlmModel);
      if (saved.xmlContent) setXmlContent(saved.xmlContent);
      if (saved.exportFormat) setExportFormat(saved.exportFormat);
      if (saved.exportFilename) setExportFilename(saved.exportFilename);
    } catch (e) {
      console.warn('[image2drawio] restore settings failed:', e);
    }
  }, []);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const payload = {
      apiUrl,
      apiKey,
      genFigModel,
      vlmModel,
      xmlContent,
      exportFormat,
      exportFilename,
    };
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
    } catch (e) {
      console.warn('[image2drawio] persist settings failed:', e);
    }
  }, [apiUrl, apiKey, genFigModel, vlmModel, xmlContent, exportFormat, exportFilename]);

  const validateImageFile = (file: File): boolean => {
    const validTypes = ['image/jpeg', 'image/png', 'image/jpg'];
    if (!validTypes.includes(file.type)) {
      setError(t('errors.imgOnly'));
      return false;
    }
    if (file.size > MAX_FILE_SIZE) {
      setError(t('errors.sizeLimit'));
      return false;
    }
    return true;
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !validateImageFile(file)) return;
    setSelectedFile(file);
    setError(null);
    setStatusMessage('');
    setXmlContent('');
    setFilePath('');
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (!file || !validateImageFile(file)) return;
    setSelectedFile(file);
    setError(null);
    setStatusMessage('');
    setXmlContent('');
    setFilePath('');
  };

  const postToDrawio = useCallback((payload: Record<string, unknown>) => {
    const frame = iframeRef.current?.contentWindow;
    if (!frame) return;
    frame.postMessage(JSON.stringify(payload), '*');
  }, []);

  const requestDrawioFit = useCallback(() => {
    postToDrawio({ action: 'zoom', zoom: 'fit' });
  }, [postToDrawio]);

  const handleGenerate = async () => {
    if (!selectedFile) {
      setError(t('errors.selectFile'));
      return;
    }

    const quota = await checkQuota(user?.id || null, user?.is_anonymous || false);
    if (quota.remaining <= 0) {
      setError(t('errors.quotaFull'));
      return;
    }

    setIsProcessing(true);
    setError(null);
    setStatusMessage(t('status.uploading'));

    try {
      const formData = new FormData();
      formData.append('image_file', selectedFile);
      formData.append('chat_api_url', apiUrl.trim());
      formData.append('api_key', apiKey.trim());
      formData.append('gen_fig_model', genFigModel);
      formData.append('vlm_model', vlmModel);
      formData.append('email', user?.id || user?.email || '');

      setStatusMessage(t('status.processing'));
      const res = await fetch(`${API_BASE}/api/v1/image2drawio/generate`, {
        method: 'POST',
        headers: {
          'X-API-Key': API_KEY,
        },
        body: formData,
      });
      const data = await res.json();

      if (!data?.success || !data?.xml_content) {
        throw new Error(data?.error || t('errors.apiFail'));
      }

      setXmlContent(data.xml_content);
      setFilePath(data.file_path || '');
      setStatusMessage(t('status.complete'));

      await recordUsage(user?.id || null, 'image2drawio');
      if (refreshQuota) refreshQuota();
    } catch (err) {
      const message = err instanceof Error ? err.message : t('errors.apiFail');
      setError(message);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleCopyXml = async () => {
    if (!xmlContent) return;
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(xmlContent);
      } else {
        const textArea = document.createElement('textarea');
        textArea.value = xmlContent;
        textArea.style.position = 'fixed';
        textArea.style.left = '-9999px';
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
      }
      setCopySuccess(t('actions.copied'));
      setTimeout(() => setCopySuccess(''), 2000);
    } catch (err) {
      console.error('copy xml failed', err);
    }
  };

  const handleDownload = () => {
    if (!xmlContent) return;
    const blob = new Blob([xmlContent], { type: 'application/xml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${selectedFile?.name?.split('.')[0] || 'diagram'}.drawio`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 100);
  };

  const requestDrawioExport = useCallback(
    (format: 'xml' | 'png' | 'svg') => {
      if (!drawioReady) {
        return Promise.reject(new Error('Draw.io not ready'));
      }

      return new Promise<string>((resolve, reject) => {
        pendingExportRef.current = { resolve, reject, format };
        postToDrawio({ action: 'export', format });
        window.setTimeout(() => {
          if (pendingExportRef.current.resolve === resolve) {
            pendingExportRef.current = { resolve: null, reject: null, format: null };
            reject(new Error('Export timeout'));
          }
        }, DRAWIO_EXPORT_TIMEOUT_MS);
      });
    },
    [drawioReady, postToDrawio],
  );

  const parseXmlForAnimation = useCallback((xml: string) => {
    try {
      const parser = new DOMParser();
      const doc = parser.parseFromString(xml, 'text/xml');
      if (doc.querySelector('parsererror')) return null;
      const root =
        doc.querySelector('mxGraphModel > root') ||
        doc.querySelector('root');
      if (!root) return null;

      const rootCells = Array.from(root.children).filter(
        node => node.nodeName === 'mxCell'
      ) as Element[];
      if (!rootCells.length) return null;

      const baseCells = rootCells.filter(cell => {
        const id = cell.getAttribute('id');
        return id === '0' || id === '1';
      });
      const normalCells = rootCells.filter(cell => {
        const id = cell.getAttribute('id');
        return id !== '0' && id !== '1';
      });
      const nonEdges = normalCells.filter(cell => cell.getAttribute('edge') !== '1');
      const edges = normalCells.filter(cell => cell.getAttribute('edge') === '1');
      const orderedCells = [...nonEdges, ...edges];

      return { doc, baseCells, orderedCells };
    } catch {
      return null;
    }
  }, []);

  const buildXmlWithCells = useCallback((sourceDoc: Document, cells: Element[]) => {
    const docClone = sourceDoc.cloneNode(true) as Document;
    const root =
      docClone.querySelector('mxGraphModel > root') ||
      docClone.querySelector('root');
    if (!root) return '';
    while (root.firstChild) {
      root.removeChild(root.firstChild);
    }
    for (const cell of cells) {
      root.appendChild(docClone.importNode(cell, true));
    }
    return new XMLSerializer().serializeToString(docClone);
  }, []);

  const animateDrawioLoad = useCallback(
    async (xml: string) => {
      const parsed = parseXmlForAnimation(xml);
      if (!parsed) {
        postToDrawio({ action: 'load', xml, autosave: 1 });
        lastLoadedXmlRef.current = xml;
        setTimeout(() => requestDrawioFit(), 120);
        return;
      }

      const { doc, baseCells, orderedCells } = parsed;
      const total = orderedCells.length;
      const batchSize =
        total > DRAWIO_ANIMATE_MAX_CELLS ? DRAWIO_ANIMATE_LARGE_BATCH : 1;
      const token = ++animationTokenRef.current;
      isAnimatingRef.current = true;

      for (let i = 0; i < total; i += batchSize) {
        if (animationTokenRef.current !== token) return;
        const subset = orderedCells.slice(0, Math.min(i + batchSize, total));
        const autosave = i + batchSize >= total ? 1 : 0;
        const partialXml = buildXmlWithCells(doc, [...baseCells, ...subset]);
        if (!partialXml) break;
        postToDrawio({ action: 'load', xml: partialXml, autosave });
        setTimeout(() => requestDrawioFit(), 80);
        await new Promise(resolve => setTimeout(resolve, DRAWIO_ANIMATE_STEP_MS));
      }

      if (animationTokenRef.current === token) {
        lastLoadedXmlRef.current = xml;
        isAnimatingRef.current = false;
        setTimeout(() => requestDrawioFit(), 120);
      }
    },
    [buildXmlWithCells, parseXmlForAnimation, postToDrawio, requestDrawioFit]
  );

  const downloadExportData = useCallback((data: string, format: 'png' | 'svg', filename: string) => {
    let url = '';
    let shouldRevoke = false;
    const trimmed = data.trim();

    if (trimmed.startsWith('data:')) {
      url = trimmed;
    } else if (format === 'png') {
      url = `data:image/png;base64,${trimmed}`;
    } else if (trimmed.startsWith('<svg')) {
      const blob = new Blob([trimmed], { type: 'image/svg+xml' });
      url = URL.createObjectURL(blob);
      shouldRevoke = true;
    } else {
      url = `data:image/svg+xml;base64,${trimmed}`;
    }

    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    if (shouldRevoke) {
      setTimeout(() => URL.revokeObjectURL(url), 100);
    }
  }, []);

  const handleExport = useCallback(async () => {
    if (!xmlContent || isExporting) return;
    setIsExporting(true);
    const safeName = exportFilename.trim() || 'diagram';

    if (exportFormat === 'drawio') {
      handleDownload();
      setIsExporting(false);
      return;
    }

    try {
      const exportData = await requestDrawioExport(exportFormat);
      if (exportData) {
        downloadExportData(exportData, exportFormat, `${safeName}.${exportFormat}`);
      }
    } catch (err) {
      console.error('导出失败:', err);
    } finally {
      setIsExporting(false);
    }
  }, [
    xmlContent,
    isExporting,
    exportFormat,
    exportFilename,
    handleDownload,
    requestDrawioExport,
    downloadExportData,
  ]);

  useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      if (!DRAWIO_ORIGINS.has(event.origin) || typeof event.data !== 'string') return;
      let message: { event?: string; xml?: string; data?: string } = {};
      try {
        message = JSON.parse(event.data) as { event?: string; xml?: string; data?: string };
      } catch {
        return;
      }

      if (message.event === 'init' || message.event === 'ready') {
        setDrawioReady(true);
        postToDrawio({
          action: 'configure',
          config: {
            sidebar: false,
            format: false,
            layers: false,
            menubar: false,
            toolbar: false,
            status: false,
          },
        });
        return;
      }

      if ((message.event === 'save' || message.event === 'autosave') && typeof message.xml === 'string') {
        if (isAnimatingRef.current) return;
        lastLoadedXmlRef.current = message.xml;
        setXmlContent(message.xml);
        return;
      }

      if (message.event === 'export' && pendingExportRef.current.resolve && typeof message.data === 'string') {
        const resolver = pendingExportRef.current.resolve;
        pendingExportRef.current = { resolve: null, reject: null, format: null };
        resolver(message.data);
      }
    };

    window.addEventListener('message', handleMessage);
    return () => window.removeEventListener('message', handleMessage);
  }, []);

  useEffect(() => {
    if (!drawioReady || !xmlContent) return;
    if (xmlContent === lastLoadedXmlRef.current) return;
    animateDrawioLoad(xmlContent);
  }, [drawioReady, xmlContent, animateDrawioLoad]);

  return (
    <div className="relative w-full h-full overflow-y-auto bg-[#0c0f12] text-slate-100">
      <div className="pointer-events-none absolute -top-24 right-[-10%] h-72 w-72 rounded-full bg-amber-500/10 blur-[120px]" />
      <div className="pointer-events-none absolute bottom-[-30%] left-[-5%] h-80 w-80 rounded-full bg-lime-500/10 blur-[140px]" />

      <div className="relative mx-auto w-full max-w-[1400px] px-6 pt-8 pb-8">
        <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
          <div className="space-y-2">
            <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-300">
              <span className={`h-1.5 w-1.5 rounded-full ${drawioReady ? 'bg-emerald-400' : 'bg-slate-500'}`} />
              image2drawio
            </div>
            <h1 className="text-2xl font-semibold text-white">{t('title')}</h1>
            <p className="text-sm text-slate-400">{t('subtitle')}</p>
          </div>
        </div>

        <div className="grid gap-6 lg:grid-cols-[360px_minmax(0,1fr)] mt-6" style={{ minHeight: '720px' }}>
          <div className="flex flex-col gap-4">
            <div className={panelClass}>
              <h3 className="text-sm font-semibold text-slate-200 mb-3 flex items-center gap-2">
                <UploadCloud className="text-amber-300" size={18} />
                {t('upload.title')}
              </h3>
              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  setIsDragOver(true);
                }}
                onDragLeave={(e) => {
                  e.preventDefault();
                  setIsDragOver(false);
                }}
                onDrop={handleDrop}
                className={`flex flex-col items-center justify-center gap-3 rounded-xl border border-dashed px-4 py-6 text-center transition-all ${
                  isDragOver ? 'border-amber-400/80 bg-amber-400/10' : 'border-white/10 bg-white/5'
                }`}
              >
                {selectedFile ? (
                  <>
                    <FileImage className="h-10 w-10 text-amber-300" />
                    <div className="text-sm text-slate-200">{selectedFile.name}</div>
                    <div className="text-xs text-slate-500">{(selectedFile.size / (1024 * 1024)).toFixed(2)} MB</div>
                    <button
                      onClick={() => setSelectedFile(null)}
                      className="text-xs text-amber-300 hover:text-amber-200"
                    >
                      {t('actions.clear')}
                    </button>
                  </>
                ) : (
                  <>
                    <UploadCloud className="h-10 w-10 text-amber-300" />
                    <p className="text-sm text-slate-300">{t('upload.drag')}</p>
                    <label className="px-4 py-2 rounded-full bg-gradient-to-r from-amber-500 to-lime-500 text-white text-xs font-semibold cursor-pointer hover:from-amber-400 hover:to-lime-400 transition-all">
                      {t('upload.button')}
                      <input type="file" accept="image/png,image/jpeg,image/jpg" className="hidden" onChange={handleFileChange} />
                    </label>
                    <p className="text-xs text-slate-500">{t('upload.hint')}</p>
                  </>
                )}
              </div>
            </div>

            <div className={panelClass}>
              <h3 className="text-sm font-semibold text-slate-200 mb-3 flex items-center gap-2">
                <Sparkles className="text-amber-300" size={18} />
                {t('config.title')}
              </h3>
                <div className="space-y-3">
                  <label className="block text-xs text-slate-400">{t('config.apiUrl')}</label>
                  <select
                    value={apiUrl}
                    onChange={(e) => setApiUrl(e.target.value)}
                    className={inputClass}
                  >
                    {API_URL_OPTIONS.map((url) => (
                      <option key={url} value={url}>{url}</option>
                    ))}
                  </select>

                  <label className="block text-xs text-slate-400 flex items-center gap-1">
                    <Key size={12} /> {t('config.apiKey')}
                  </label>
                  <input
                    type="password"
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    placeholder="sk-..."
                    className={inputClass}
                  />

                <label className="block text-xs text-slate-400 flex items-center gap-1 mb-1">
                  <ImageIcon size={12} /> {t('config.genModel')}
                </label>
                <select
                  value={genFigModel}
                  onChange={(e) => setGenFigModel(e.target.value)}
                  className={inputClass}
                >
                  <option value="gemini-3-pro-image-preview">gemini-3-pro-image-preview</option>
                </select>
              </div>
            </div>

            {error && (
              <div className="flex items-center gap-2 text-sm text-red-300 bg-red-500/10 border border-red-500/40 rounded-lg px-4 py-3">
                <AlertCircle size={16} />
                <span>{error}</span>
              </div>
            )}

            {statusMessage && !error && (
              <div className="flex items-center gap-2 text-sm text-amber-200 bg-amber-500/10 border border-amber-500/30 rounded-lg px-4 py-3">
                {isProcessing ? <Loader2 size={16} className="animate-spin" /> : <Wand2 size={16} />}
                <span>{statusMessage}</span>
              </div>
            )}

            <button
              onClick={handleGenerate}
              disabled={isProcessing}
              className="w-full py-3 rounded-xl bg-gradient-to-r from-amber-500 to-lime-500 text-white text-sm font-semibold hover:from-amber-400 hover:to-lime-400 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-[0_10px_30px_rgba(217,119,6,0.25)]"
            >
              {isProcessing ? t('actions.processing') : t('actions.generate')}
            </button>
          </div>

          <div className="flex flex-col gap-4">
            <div className="flex flex-col h-full rounded-3xl bg-white/5 border border-white/10 p-4 md:p-6 backdrop-blur-xl shadow-[0_25px_70px_rgba(0,0,0,0.35)]">
              <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                  <Wand2 className="text-amber-300" size={18} />
                  {t('preview.title')}
                </h3>
                {xmlContent && (
                  <div className="flex flex-wrap items-center gap-2">
                    <div className="flex items-center rounded-full bg-white/5 border border-white/10 p-1">
                      {(['drawio', 'svg', 'png'] as const).map(format => (
                        <button
                          key={format}
                          onClick={() => setExportFormat(format)}
                          className={`px-3 py-1 text-xs rounded-full transition ${
                            exportFormat === format
                              ? 'bg-white/20 text-white'
                              : 'text-slate-400 hover:text-white'
                          }`}
                        >
                          {format.toUpperCase()}
                        </button>
                      ))}
                    </div>
                    <div className="flex items-center rounded-xl bg-white/5 border border-white/10 px-3 py-2">
                      <input
                        type="text"
                        value={exportFilename}
                        onChange={e => setExportFilename(e.target.value)}
                        className="w-24 bg-transparent text-xs text-white placeholder-slate-500 outline-none"
                        placeholder="diagram"
                      />
                      <span className="ml-2 text-xs text-slate-400">.{exportFormat}</span>
                    </div>
                    <button
                      onClick={handleExport}
                      disabled={isExporting || isProcessing}
                      className="flex items-center gap-2 px-4 py-2 rounded-xl bg-white/10 text-white text-xs font-semibold hover:bg-white/20 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                    >
                      {isExporting ? <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" /> : <Download size={14} />}
                      {t('actions.download')}
                    </button>
                    {filePath && (
                      <button
                        onClick={() => {
                          const relative = filePath.includes('outputs/')
                            ? filePath.split('outputs/')[1]
                            : filePath;
                          window.open(`${API_BASE}/outputs/${relative}`, '_blank');
                        }}
                        className="flex items-center gap-2 px-4 py-2 rounded-xl bg-white/10 text-white text-xs font-semibold hover:bg-white/20 transition-all"
                      >
                        <ExternalLink size={14} />
                        {t('actions.open')}
                      </button>
                    )}
                  </div>
                )}
              </div>
              <div className={`mt-4 flex-1 bg-[#0b0f17] rounded-2xl border border-white/10 min-h-[420px] lg:min-h-[720px] overflow-hidden ${xmlContent ? 'relative block' : 'flex items-center justify-center'}`}>
                {xmlContent ? (
                  <iframe
                    ref={iframeRef}
                    src="https://embed.diagrams.net/?embed=1&spin=1&proto=json&autosave=1&saveAndExit=0&noSaveBtn=1&noExitBtn=1&sidebar=0&layers=0&toolbar=0&menubar=0&status=0&format=0"
                    className="absolute inset-0 w-full h-full border-0"
                    title="draw.io editor"
                  />
                ) : (
                  <div className="text-center animate-fade-in">
                    <FileImage className="w-12 h-12 mx-auto text-slate-500 mb-3" />
                    <p className="text-sm text-slate-400">{t('preview.placeholder')}</p>
                  </div>
                )}
              </div>
            </div>

            <div className={panelClass}>
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                  <Wand2 className="text-amber-300" size={18} />
                  {t('xml.title')}
                </h3>
                <button
                  onClick={handleCopyXml}
                  disabled={!xmlContent}
                  className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 text-xs text-slate-200 hover:bg-white/10 disabled:opacity-50"
                >
                  <Copy size={12} />
                  {copySuccess || t('actions.copy')}
                </button>
              </div>
              <textarea
                value={xmlContent}
                readOnly
                placeholder={t('xml.placeholder')}
                className="w-full h-48 rounded-xl bg-[#0b0f17] border border-white/10 px-3 py-2 text-xs text-slate-200 outline-none"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Image2DrawioPage;
