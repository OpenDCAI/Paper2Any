import React, { useState, useRef, useCallback, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { Wand2, Upload, FileText, Send, Download } from 'lucide-react';
import type { DiagramType, DiagramStyle, ChatMessage } from './types';
import { API_KEY, API_URL_OPTIONS } from '../../config/api';
import { useAuthStore } from '../../stores/authStore';
import { getApiSettings, saveApiSettings } from '../../services/apiSettingsService';
import { verifyLlmConnection } from '../../services/llmService';
import Banner from './Banner';
import ExamplesSection from './ExamplesSection';

const DRAWIO_ORIGINS = new Set(['https://embed.diagrams.net', 'https://app.diagrams.net']);
const STORAGE_KEY = 'paper2drawio_settings';
const DRAWIO_EXPORT_TIMEOUT_MS = 5000;
const DRAWIO_ANIMATE_STEP_MS = 60;
const DRAWIO_ANIMATE_MAX_CELLS = 240;
const DRAWIO_ANIMATE_LARGE_BATCH = 5;

const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

export default function Paper2DrawioPage() {
  const { t } = useTranslation('paper2drawio');
  const { user } = useAuthStore();

  // 状态
  const [generationMode, setGenerationMode] = useState<'ai' | 'paper2drawio'>('ai');
  const [modePicked, setModePicked] = useState(false);
  const [uploadMode, setUploadMode] = useState<'file' | 'text'>('text');
  const [textContent, setTextContent] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [diagramType, setDiagramType] = useState<DiagramType>('auto');
  const [diagramStyle, setDiagramStyle] = useState<DiagramStyle>('default');
  const [xmlContent, setXmlContent] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isExporting, setIsExporting] = useState(false);
  const [exportFormat, setExportFormat] = useState<'drawio' | 'png' | 'svg'>('drawio');
  const [exportFilename, setExportFilename] = useState('diagram');
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [drawioReady, setDrawioReady] = useState(false);

  // GitHub Stars
  const [stars, setStars] = useState<{dataflow: number | null, agent: number | null, dataflex: number | null}>({
    dataflow: null,
    agent: null,
    dataflex: null,
  });
  const [showBanner, setShowBanner] = useState(true);

  // API 配置
  const [apiUrl, setApiUrl] = useState(import.meta.env.VITE_DEFAULT_LLM_API_URL || 'https://api.apiyi.com/v1');
  const [apiKey, setApiKey] = useState('');
  const [model, setModel] = useState('claude-sonnet-4-5-20250929');
  const [drawioLanguage, setDrawioLanguage] = useState<'zh' | 'en'>('zh');
  const [enableVlmValidation, setEnableVlmValidation] = useState(false);
  const [p2dImageModel, setP2dImageModel] = useState('gemini-3-pro-image-preview');
  const [p2dLanguage, setP2dLanguage] = useState<'zh' | 'en'>('zh');
  const [p2dStyle, setP2dStyle] = useState<'cartoon' | 'realistic'>('cartoon');
  const [p2dFigureComplex, setP2dFigureComplex] = useState<'easy' | 'mid' | 'hard'>('easy');

  const iframeRef = useRef<HTMLIFrameElement>(null);
  const chatListRef = useRef<HTMLDivElement>(null);
  const lastLoadedXmlRef = useRef('');
  const isAnimatingRef = useRef(false);
  const animationTokenRef = useRef(0);
  const pendingExportRef = useRef<{
    resolve: ((data: string) => void) | null;
    reject: ((error: Error) => void) | null;
    format: 'xml' | 'png' | 'svg' | null;
  }>({ resolve: null, reject: null, format: null });
  const panelClass = 'rounded-2xl bg-white/5 border border-white/10 p-4 backdrop-blur-xl shadow-[0_20px_60px_rgba(0,0,0,0.25)] transition-all duration-300';
  const inputClass = 'w-full rounded-xl bg-white/5 border border-white/10 px-3 py-2 text-sm text-white placeholder-slate-500 outline-none transition focus:border-sky-400/60 focus:ring-2 focus:ring-sky-500/20';
  const modeButtonActive = 'bg-gradient-to-r from-sky-500 to-cyan-500 text-white shadow-[0_0_30px_rgba(14,165,233,0.6),0_0_60px_rgba(6,182,212,0.4)] border border-sky-400/50 scale-105';
  const modeButtonIdle = 'bg-white/5 text-slate-300 hover:bg-gradient-to-r hover:from-sky-500/20 hover:to-cyan-500/20 hover:text-white hover:shadow-[0_0_20px_rgba(14,165,233,0.3)] hover:border-sky-400/30 hover:scale-105 border border-white/10';

  // 自动滚动到底部
  useEffect(() => {
    if (chatListRef.current) {
      // 使用 setTimeout 确保 DOM 更新完成后再滚动
      setTimeout(() => {
        if (chatListRef.current) {
          chatListRef.current.scrollTop = chatListRef.current.scrollHeight;
        }
      }, 100);
    }
  }, [chatHistory, isLoading]);

  // 获取 GitHub Stars
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
        const saved = JSON.parse(raw) as {
          uploadMode?: 'file' | 'text';
          textContent?: string;
          diagramType?: DiagramType;
          diagramStyle?: DiagramStyle;
          apiUrl?: string;
          apiKey?: string;
          model?: string;
          drawioLanguage?: 'zh' | 'en';
          enableVlmValidation?: boolean;
          xmlContent?: string;
          chatHistory?: ChatMessage[];
          chatInput?: string;
          exportFormat?: 'drawio' | 'png' | 'svg';
          exportFilename?: string;
          generationMode?: 'ai' | 'paper2drawio';
          p2dImageModel?: string;
          p2dLanguage?: 'zh' | 'en';
          p2dStyle?: 'cartoon' | 'realistic';
          p2dFigureComplex?: 'easy' | 'mid' | 'hard';
        };

        if (saved.uploadMode) setUploadMode(saved.uploadMode);
        if (saved.textContent) setTextContent(saved.textContent);
        if (saved.diagramType) setDiagramType(saved.diagramType);
        if (saved.diagramStyle) setDiagramStyle(saved.diagramStyle);
        if (saved.model) setModel(saved.model);
        if (saved.drawioLanguage) setDrawioLanguage(saved.drawioLanguage);
        if (typeof saved.enableVlmValidation === 'boolean') setEnableVlmValidation(saved.enableVlmValidation);
        if (saved.xmlContent) setXmlContent(saved.xmlContent);
        if (saved.chatHistory) setChatHistory(saved.chatHistory);
        if (saved.chatInput) setChatInput(saved.chatInput);
        if (saved.exportFormat) setExportFormat(saved.exportFormat);
        if (saved.exportFilename) setExportFilename(saved.exportFilename);
        if (saved.generationMode) setGenerationMode(saved.generationMode);
        if (saved.p2dImageModel) setP2dImageModel(saved.p2dImageModel);
        if (saved.p2dLanguage) setP2dLanguage(saved.p2dLanguage);
        if (saved.p2dStyle) setP2dStyle(saved.p2dStyle);
        if (saved.p2dFigureComplex) setP2dFigureComplex(saved.p2dFigureComplex);

        const userApiSettings = getApiSettings(user?.id || null);
        if (userApiSettings) {
          if (userApiSettings.apiUrl) setApiUrl(userApiSettings.apiUrl);
          if (userApiSettings.apiKey) setApiKey(userApiSettings.apiKey);
        } else {
          if (saved.apiUrl) setApiUrl(saved.apiUrl);
          if (saved.apiKey) setApiKey(saved.apiKey);
        }
      }
    } catch (e) {
      console.error('Failed to restore paper2drawio config', e);
    }
  }, [user?.id]);

  // 将配置写入 localStorage
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const data = {
      uploadMode,
      textContent,
      diagramType,
      diagramStyle,
      apiUrl,
      apiKey,
      model,
      drawioLanguage,
      enableVlmValidation,
      xmlContent,
      chatHistory,
      chatInput,
      exportFormat,
      exportFilename,
      generationMode,
      p2dImageModel,
      p2dLanguage,
      p2dStyle,
      p2dFigureComplex,
    };
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
      if (user?.id && apiUrl && apiKey) {
        saveApiSettings(user.id, { apiUrl, apiKey });
      }
    } catch (e) {
      console.error('Failed to persist paper2drawio config', e);
    }
  }, [
    uploadMode,
    textContent,
    diagramType,
    diagramStyle,
    apiUrl,
    apiKey,
    model,
    drawioLanguage,
    enableVlmValidation,
    xmlContent,
    chatHistory,
    chatInput,
    exportFormat,
    exportFilename,
    generationMode,
    p2dImageModel,
    p2dLanguage,
    p2dStyle,
    p2dFigureComplex,
    user?.id,
  ]);

  // 生成图表
  const handleGenerate = useCallback(async () => {
    if (!textContent && !file) return;

    // Step 0: Verify LLM Connection first
    try {
      setIsValidating(true);
      setError(null);
      await verifyLlmConnection(apiUrl, apiKey, model);
      setIsValidating(false);
    } catch (err) {
      setIsValidating(false);
      const errorMsg = err instanceof Error ? err.message : '验证 LLM 连接失败';
      setError(errorMsg);
      return;
    }

    setIsLoading(true);

    try {
      if (generationMode === 'paper2drawio') {
        const formData = new FormData();
        formData.append('img_gen_model_name', p2dImageModel);
        formData.append('chat_api_url', apiUrl);
        formData.append('api_key', apiKey);
        formData.append('input_type', uploadMode);
        formData.append('graph_type', 'model_arch');
        formData.append('style', p2dStyle);
        formData.append('figure_complex', p2dFigureComplex);
        formData.append('language', p2dLanguage);
        formData.append('output_format', 'drawio');

        if (uploadMode === 'text') {
          formData.append('text', textContent);
        } else if (file) {
          formData.append('file', file);
          formData.append('file_kind', 'pdf');
        }

        const res = await fetch(`${API_BASE}/api/v1/paper2figure/generate-json`, {
          method: 'POST',
          headers: { 'X-API-Key': API_KEY },
          body: formData,
        });

        const data = await res.json();
        if (data.success && data.drawio_filename) {
          let drawioUrl = data.drawio_filename as string;
          if (typeof window !== 'undefined' && window.location.protocol === 'https:' && drawioUrl.startsWith('http:')) {
            drawioUrl = drawioUrl.replace(/^http:/, 'https:');
          }
          const xml = await fetch(drawioUrl).then(r => r.text());
          if (xml && xml.includes('<mxfile')) {
            setXmlContent(xml);
          }
        }
        return;
      }

      const formData = new FormData();
      formData.append('chat_api_url', apiUrl);
      formData.append('api_key', apiKey);
      formData.append('model', model);
      formData.append('input_type', uploadMode === 'file' ? 'PDF' : 'TEXT');
      formData.append('diagram_type', diagramType);
      formData.append('diagram_style', diagramStyle);
      formData.append('language', drawioLanguage);
      formData.append('enable_vlm_validation', enableVlmValidation ? 'true' : 'false');

      if (uploadMode === 'text') {
        formData.append('text_content', textContent);
      } else if (file) {
        formData.append('file', file);
      }

      const res = await fetch(`${API_BASE}/api/v1/paper2drawio/generate`, {
        method: 'POST',
        headers: { 'X-API-Key': API_KEY },
        body: formData,
      });

      const data = await res.json();
      if (data.success && data.xml_content) {
        setXmlContent(data.xml_content);
      }
    } catch (err) {
      console.error('生成失败:', err);
    } finally {
      setIsLoading(false);
    }
  }, [
    textContent,
    file,
    uploadMode,
    apiUrl,
    apiKey,
    model,
    diagramType,
    diagramStyle,
    generationMode,
    p2dImageModel,
    p2dLanguage,
    p2dStyle,
    p2dFigureComplex,
    drawioLanguage,
    enableVlmValidation,
  ]);

  const handleSelectMode = useCallback((mode: 'ai' | 'paper2drawio') => {
    setGenerationMode(mode);
    setModePicked(true);
  }, []);

  const postToDrawio = useCallback((payload: Record<string, unknown>) => {
    const frame = iframeRef.current?.contentWindow;
    if (!frame) return;
    frame.postMessage(JSON.stringify(payload), '*');
  }, []);

  const requestDrawioFit = useCallback(() => {
    postToDrawio({ action: 'zoom', zoom: 'fit' });
  }, [postToDrawio]);

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

  const syncXmlFromDrawio = useCallback(async () => {
    if (!drawioReady) return xmlContent;
    try {
      const exported = await requestDrawioExport('xml');
      if (exported && exported.includes('<mxfile')) {
        lastLoadedXmlRef.current = exported;
        setXmlContent(exported);
        return exported;
      }
    } catch (e) {
      console.warn('Failed to sync XML from draw.io:', e);
    }
    return xmlContent;
  }, [drawioReady, xmlContent, requestDrawioExport]);

  const downloadXmlFile = useCallback((xml: string, filename: string) => {
    const blob = new Blob([xml], { type: 'application/xml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 100);
  }, []);

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

  // 发送聊天消息
  const handleSendChat = useCallback(async () => {
    if (!chatInput.trim() || !xmlContent) return;

    const newMessage: ChatMessage = { role: 'user', content: chatInput };
    const nextHistory = [...chatHistory, newMessage];
    setChatHistory(nextHistory);
    setChatInput('');
    setIsLoading(true);

    try {
      const latestXml = await syncXmlFromDrawio();
      const res = await fetch(`${API_BASE}/api/v1/paper2drawio/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': API_KEY,
        },
        body: JSON.stringify({
          current_xml: latestXml || xmlContent,
          message: chatInput,
          chat_history: nextHistory,
          chat_api_url: apiUrl,
          api_key: apiKey,
          model: model,
        }),
      });

      const data = await res.json();
      if (data.success && data.xml_content) {
        setXmlContent(data.xml_content);
        setChatHistory(prev => [...prev, { role: 'assistant', content: t('diagramUpdated') }]);
      }
    } catch (err) {
      console.error('编辑失败:', err);
    } finally {
      setIsLoading(false);
    }
  }, [chatInput, xmlContent, chatHistory, apiUrl, apiKey, model, syncXmlFromDrawio]);

  // 导出图表
  const handleExport = useCallback(async () => {
    if (!xmlContent || isExporting) return;
    setIsExporting(true);

    const trimmedName = exportFilename.trim();
    const safeName = (trimmedName || 'diagram').replace(/[\\/:*?"<>|]/g, '_');

    if (exportFormat === 'drawio') {
      let latestXml = '';
      try {
        latestXml = await syncXmlFromDrawio();
      } catch (err) {
        console.warn('Export via draw.io failed, falling back to server:', err);
      }

      if (latestXml && latestXml.includes('<mxfile')) {
        downloadXmlFile(latestXml, `${safeName}.drawio`);
        setIsExporting(false);
        return;
      }

      try {
        const res = await fetch(`${API_BASE}/api/v1/paper2drawio/export`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-API-Key': API_KEY,
          },
          body: JSON.stringify({
            xml_content: xmlContent,
            format: 'drawio',
            filename: safeName,
          }),
        });

        const data = await res.json();
        if (data.success && data.file_path) {
          window.open(`${API_BASE}/outputs/${data.file_path.split('outputs/')[1]}`, '_blank');
        }
      } catch (err) {
        console.error('导出失败:', err);
      } finally {
        setIsExporting(false);
      }
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
    syncXmlFromDrawio,
    downloadXmlFile,
    downloadExportData,
    requestDrawioExport,
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
    <div className="relative w-full h-full overflow-y-auto bg-[#0b0d12] text-slate-100">
      <Banner show={showBanner} onClose={() => setShowBanner(false)} stars={stars} />
      <div className="pointer-events-none absolute -top-40 right-[-10%] h-72 w-72 rounded-full bg-sky-500/10 blur-[120px]" />
      <div className="pointer-events-none absolute bottom-[-25%] left-[-5%] h-80 w-80 rounded-full bg-cyan-500/10 blur-[140px]" />
      <div className="relative mx-auto w-full max-w-[1400px] px-6 pt-8 pb-8">
        <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between animate-fade-in shrink-0">
          <div className="space-y-2">
            <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-300">
              <span className={`h-1.5 w-1.5 rounded-full ${drawioReady ? 'bg-emerald-400' : 'bg-slate-500'}`} />
              paper2diagram
            </div>
            <h1 className="text-2xl font-semibold text-white">
              {t('title')}
            </h1>
            <p className="text-sm text-slate-400">
              {t('subtitle')}
            </p>
          </div>
        </div>

        <div className="grid gap-6 lg:grid-cols-[340px_minmax(0,1fr)] mt-6" style={{ minHeight: '720px' }}>
          {/* 左侧：输入区域 */}
          <div className="flex flex-col gap-4 animate-slide-in" style={{ animationDelay: '40ms' }}>
            <div className={panelClass}>
              <h3 className="text-sm font-semibold text-slate-200 mb-3 flex items-center gap-2">
                <Wand2 className="text-sky-300" size={18} />
                选择功能
              </h3>
              <div className="flex gap-2 mb-3">
                <button
                  onClick={() => handleSelectMode('ai')}
                  className={`flex-1 px-3 py-2 rounded-xl text-sm font-medium transition-all ${
                    modePicked && generationMode === 'ai' ? modeButtonActive : modeButtonIdle
                  }`}
                >
                  AI 驱动 DrawIO
                </button>
                <button
                  onClick={() => handleSelectMode('paper2drawio')}
                  className={`flex-1 px-3 py-2 rounded-xl text-sm font-medium transition-all ${
                    modePicked && generationMode === 'paper2drawio' ? modeButtonActive : modeButtonIdle
                  }`}
                >
                  DrawIO 版本科研绘图生成
                </button>
              </div>
              <div className="space-y-2 text-xs text-slate-400">
                <div className="relative group rounded-xl border border-white/10 bg-white/5 p-3 hover:border-sky-400/40 hover:bg-white/10 transition-all cursor-pointer">
                  <p className="text-slate-200 font-semibold mb-1">Demo · AI 驱动</p>
                  <p>输入文本或论文 PDF，直接生成可编辑流程图、架构图等通用 DrawIO 图。</p>
                  {/* Hover 预览框 - 显示在上方 */}
                  <div className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-[320px] opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-300 pointer-events-none" style={{ zIndex: 9999 }}>
                    <div className="rounded-lg border border-sky-400/60 bg-slate-900/98 backdrop-blur-xl shadow-2xl overflow-hidden">
                      <div className="px-2.5 py-1.5 border-b border-white/10 flex items-center justify-between">
                        <p className="text-[11px] font-semibold text-sky-300">AI 驱动演示</p>
                        <span className="text-[9px] text-slate-400">悬停查看</span>
                      </div>
                      <div className="p-1.5">
                        <img
                          src="/demos/drawio-1.gif"
                          alt="AI驱动DrawIO演示"
                          className="w-full rounded"
                          onError={(e) => {
                            e.currentTarget.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="320" height="213"%3E%3Crect width="320" height="213" fill="%23334155"/%3E%3Ctext x="50%25" y="50%25" dominant-baseline="middle" text-anchor="middle" fill="%2394a3b8" font-size="12"%3EGIF 加载中...%3C/text%3E%3C/svg%3E';
                          }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
                <div className="relative group rounded-xl border border-white/10 bg-white/5 p-3 hover:border-sky-400/40 hover:bg-white/10 transition-all cursor-pointer">
                  <p className="text-slate-200 font-semibold mb-1">Demo · 科研绘图</p>
                  <p>先生成模型结构图图片，再自动转为可编辑 DrawIO 图元。</p>
                  {/* Hover 预览框 - 显示在中央 */}
                  <div className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-[320px] opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-300 pointer-events-none" style={{ zIndex: 9999 }}>
                    <div className="rounded-lg border border-sky-400/60 bg-slate-900/98 backdrop-blur-xl shadow-2xl overflow-hidden">
                      <div className="px-2.5 py-1.5 border-b border-white/10 flex items-center justify-between">
                        <p className="text-[11px] font-semibold text-sky-300">科研绘图演示</p>
                        <span className="text-[9px] text-slate-400">悬停查看</span>
                      </div>
                      <div className="p-1.5">
                        <img
                          src="/demos/drawio-2.gif"
                          alt="科研绘图DrawIO演示"
                          className="w-full rounded"
                          onError={(e) => {
                            e.currentTarget.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="320" height="213"%3E%3Crect width="320" height="213" fill="%23334155"/%3E%3Ctext x="50%25" y="50%25" dominant-baseline="middle" text-anchor="middle" fill="%2394a3b8" font-size="12"%3EGIF 加载中...%3C/text%3E%3C/svg%3E';
                          }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {modePicked && (
              <>
            {/* API 配置 */}
            <div className={panelClass}>
              <h3 className="text-sm font-semibold text-slate-200 mb-3 flex items-center gap-2">
                <Wand2 className="text-sky-300" size={18} />
                {t('apiConfig')}
              </h3>
              <div className="space-y-2">
                {generationMode === 'paper2drawio' ? (
                  <select
                    value={apiUrl}
                    onChange={e => setApiUrl(e.target.value)}
                    className={inputClass}
                  >
                    {API_URL_OPTIONS.map((url: string) => (
                      <option key={url} value={url} className="bg-slate-900">{url}</option>
                    ))}
                  </select>
                ) : (
                  <input
                    type="text"
                    placeholder={t('apiUrl')}
                    value={apiUrl}
                    onChange={e => setApiUrl(e.target.value)}
                    className={inputClass}
                  />
                )}
                <input
                  type="password"
                  placeholder={t('apiKey')}
                  value={apiKey}
                  onChange={e => setApiKey(e.target.value)}
                  className={inputClass}
                />
                <input
                  type="text"
                  placeholder={t('model')}
                  value={model}
                  onChange={e => setModel(e.target.value)}
                  className={inputClass}
                />
              </div>
            </div>

            {/* 输入模式切换 */}
            <div className={panelClass}>
              <h3 className="text-sm font-semibold text-slate-200 mb-3 flex items-center gap-2">
                <Wand2 className="text-sky-300" size={18} />
                输入内容
              </h3>
              <div className="flex gap-2 mb-3">
                <button
                  onClick={() => setUploadMode('text')}
                  className={`flex-1 px-3 py-2 rounded-xl text-sm font-medium transition-all ${
                    uploadMode === 'text'
                      ? 'bg-white/10 text-white shadow-[0_12px_30px_rgba(59,130,246,0.2)]'
                      : 'bg-white/5 text-slate-300 hover:bg-white/10'
                  }`}
                >
                  <FileText className="inline-block mr-2" size={16} />
                  {t('textInput')}
                </button>
                <button
                  onClick={() => setUploadMode('file')}
                  className={`flex-1 px-3 py-2 rounded-xl text-sm font-medium transition-all ${
                    uploadMode === 'file'
                      ? 'bg-white/10 text-white shadow-[0_12px_30px_rgba(59,130,246,0.2)]'
                      : 'bg-white/5 text-slate-300 hover:bg-white/10'
                  }`}
                >
                  <Upload className="inline-block mr-2" size={16} />
                  {t('uploadPdf')}
                </button>
              </div>

              {uploadMode === 'text' ? (
                <textarea
                  placeholder={t('textPlaceholder')}
                  value={textContent}
                  onChange={e => setTextContent(e.target.value)}
                  className="w-full h-40 rounded-xl bg-white/5 border border-white/10 px-3 py-2 text-sm text-white placeholder-slate-500 outline-none transition focus:border-sky-400/60 focus:ring-2 focus:ring-sky-500/20 resize-none"
                />
              ) : (
                <div className="border border-dashed border-white/20 rounded-xl p-6 text-center hover:border-white/40 transition-colors">
                  <input
                    type="file"
                    accept=".pdf"
                    onChange={e => setFile(e.target.files?.[0] || null)}
                    className="hidden"
                    id="pdf-upload"
                  />
                  <label htmlFor="pdf-upload" className="cursor-pointer">
                    <Upload className="w-10 h-10 mx-auto text-slate-400 mb-2" />
                    <p className="text-sm text-slate-200 font-medium mb-1">
                      {file ? file.name : t('uploadPlaceholder')}
                    </p>
                    <p className="text-xs text-slate-500">
                      {t('uploadHint')}
                    </p>
                  </label>
                </div>
              )}
            </div>

            {generationMode === 'paper2drawio' && (
              <div className={panelClass}>
                <h3 className="text-sm font-semibold text-slate-200 mb-3 flex items-center gap-2">
                  <Wand2 className="text-sky-300" size={18} />
                  {t('modelParams.title')}
                </h3>
                <div className="space-y-2">
                  <select
                    value={p2dImageModel}
                    onChange={e => setP2dImageModel(e.target.value)}
                    className={inputClass}
                  >
                    <option value="gemini-3-pro-image-preview" className="bg-slate-900">gemini-3-pro-image-preview</option>
                    <option value="gemini-2.5-flash-image-preview" className="bg-slate-900">gemini-2.5-flash-image-preview</option>
                  </select>
                  <select
                    value={p2dLanguage}
                    onChange={e => setP2dLanguage(e.target.value as 'zh' | 'en')}
                    className={inputClass}
                  >
                    <option value="zh" className="bg-slate-900">{t('modelParams.language.zh')}</option>
                    <option value="en" className="bg-slate-900">{t('modelParams.language.en')}</option>
                  </select>
                  <select
                    value={p2dStyle}
                    onChange={e => setP2dStyle(e.target.value as 'cartoon' | 'realistic')}
                    className={inputClass}
                  >
                    <option value="cartoon" className="bg-slate-900">{t('modelParams.style.cartoon')}</option>
                    <option value="realistic" className="bg-slate-900">{t('modelParams.style.realistic')}</option>
                  </select>
                  <select
                    value={p2dFigureComplex}
                    onChange={e => setP2dFigureComplex(e.target.value as 'easy' | 'mid' | 'hard')}
                    className={inputClass}
                  >
                    <option value="easy" className="bg-slate-900">{t('modelParams.complexity.easy')}</option>
                    <option value="mid" className="bg-slate-900">{t('modelParams.complexity.mid')}</option>
                    <option value="hard" className="bg-slate-900">{t('modelParams.complexity.hard')}</option>
                  </select>
                </div>
              </div>
            )}

            {/* 图表类型选择 */}
            {generationMode === 'ai' && (
              <div className={panelClass}>
              <h3 className="text-sm font-semibold text-slate-200 mb-3 flex items-center gap-2">
                <Wand2 className="text-sky-300" size={18} />
                {t('diagramType')}
              </h3>
              <select
                value={diagramType}
                onChange={e => setDiagramType(e.target.value as DiagramType)}
                className={inputClass}
              >
                <option value="auto" className="bg-slate-900">{t('auto')}</option>
                <option value="flowchart" className="bg-slate-900">{t('flowchart')}</option>
                <option value="architecture" className="bg-slate-900">{t('architecture')}</option>
                <option value="sequence" className="bg-slate-900">{t('sequence')}</option>
                <option value="mindmap" className="bg-slate-900">{t('mindmap')}</option>
                <option value="er" className="bg-slate-900">{t('er')}</option>
              </select>
              </div>
            )}

            {generationMode === 'ai' && (
              <div className={panelClass}>
                <h3 className="text-sm font-semibold text-slate-200 mb-3 flex items-center gap-2">
                  <Wand2 className="text-sky-300" size={18} />
                  {t('diagramOptions')}
                </h3>
                <div className="space-y-2">
                  <div className="text-xs text-slate-400">{t('diagramLanguage')}</div>
                  <select
                    value={drawioLanguage}
                    onChange={e => setDrawioLanguage(e.target.value as 'zh' | 'en')}
                    className={inputClass}
                  >
                    <option value="zh" className="bg-slate-900">{t('modelParams.language.zh')}</option>
                    <option value="en" className="bg-slate-900">{t('modelParams.language.en')}</option>
                  </select>
                  <label className="flex items-center gap-3 text-sm text-slate-300">
                    <input
                      type="checkbox"
                      checked={enableVlmValidation}
                      onChange={e => setEnableVlmValidation(e.target.checked)}
                      className="h-4 w-4 rounded border border-white/20 bg-transparent text-sky-400 focus:ring-2 focus:ring-sky-500/30"
                    />
                    {t('vlmValidation')}
                  </label>
                  <p className="text-xs text-slate-500">{t('vlmValidationHint')}</p>
                </div>
              </div>
            )}

            {/* 错误信息显示 */}
            {error && (
              <div className="rounded-xl bg-red-500/10 border border-red-500/30 p-3 text-sm text-red-300">
                {error}
              </div>
            )}

            {/* 生成按钮 */}
            <button
              onClick={handleGenerate}
              disabled={isLoading || isValidating || (!textContent && !file)}
              className="w-full py-3 rounded-2xl font-semibold text-white bg-gradient-to-r from-sky-500 to-cyan-500 hover:from-sky-400 hover:to-cyan-400 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-[0_18px_45px_rgba(14,165,233,0.35)] flex items-center justify-center gap-2"
            >
              {isValidating ? (
                <>
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  验证连接中...
                </>
              ) : isLoading ? (
                <>
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  {t('generating')}
                </>
              ) : (
                <>
                  <Wand2 size={18} />
                  {t('generate')}
                </>
              )}
            </button>
              </>
            )}

            {/* 对话编辑 - 仅 AI 驱动模式支持 */}
            {xmlContent && generationMode === 'ai' && (
              <div className={panelClass}>
                <h3 className="text-sm font-semibold text-slate-200 mb-3 flex items-center gap-2">
                  <Send className="text-sky-300" size={18} />
                  {t('chatEdit')}
                </h3>
                <div className="bg-white/5 rounded-xl border border-white/10 flex flex-col" style={{ height: '280px' }}>
                  {/* 聊天内容区 */}
                  <div ref={chatListRef} className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-thin scrollbar-thumb-white/20 scrollbar-track-transparent">
                    {chatHistory.map((msg, i) => (
                      <div
                        key={i}
                        className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                      >
                        <div
                          className={`max-w-[85%] px-4 py-2.5 text-sm ${
                            msg.role === 'user'
                              ? 'bg-[#007AFF] text-white rounded-[20px] rounded-br-sm'
                              : 'bg-[#262628] text-slate-100 rounded-[20px] rounded-bl-sm'
                          }`}
                          style={{
                            boxShadow: msg.role === 'user'
                              ? '0 2px 8px rgba(0,122,255,0.25)'
                              : '0 2px 8px rgba(0,0,0,0.1)'
                          }}
                        >
                          <p className="leading-relaxed whitespace-pre-wrap break-words">{msg.content}</p>
                        </div>
                      </div>
                    ))}
                    {isLoading && (
                      <div className="flex justify-start animate-fade-in">
                        <div className="bg-[#262628] px-4 py-3 rounded-[20px] rounded-bl-sm flex gap-1.5 items-center">
                          <div className="w-2 h-2 rounded-full bg-slate-400 animate-bounce" style={{ animationDelay: '0ms' }} />
                          <div className="w-2 h-2 rounded-full bg-slate-400 animate-bounce" style={{ animationDelay: '150ms' }} />
                          <div className="w-2 h-2 rounded-full bg-slate-400 animate-bounce" style={{ animationDelay: '300ms' }} />
                        </div>
                      </div>
                    )}
                  </div>
                  
                  {/* 底部输入区 - iOS 胶囊风格 */}
                  <div className="p-3 bg-white/5 border-t border-white/10 backdrop-blur-md">
                    <div className="relative flex items-center">
                      <input
                        type="text"
                        placeholder={t('chatPlaceholder')}
                        value={chatInput}
                        onChange={e => setChatInput(e.target.value)}
                        onKeyPress={e => e.key === 'Enter' && handleSendChat()}
                        className="w-full pl-4 pr-12 py-3 rounded-full bg-black/20 border border-white/10 text-sm text-white placeholder-slate-500 outline-none transition focus:border-sky-500/50 focus:bg-black/30"
                      />
                      <button
                        onClick={handleSendChat}
                        disabled={isLoading || !chatInput.trim()}
                        className="absolute right-1.5 p-1.5 rounded-full bg-[#007AFF] text-white hover:bg-[#006ee6] disabled:opacity-0 disabled:pointer-events-none transition-all duration-200 shadow-lg"
                      >
                        <Send size={16} className="translate-x-[-1px] translate-y-[1px]" />
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* 右侧：预览区域 */}
          <div className="flex flex-col h-full rounded-3xl bg-white/5 border border-white/10 p-4 md:p-6 backdrop-blur-xl shadow-[0_25px_70px_rgba(0,0,0,0.35)] animate-fade-in" style={{ animationDelay: '80ms' }}>
            <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
              <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                <Wand2 className="text-sky-300" size={18} />
                {t('preview')}
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
                    disabled={isExporting || isLoading}
                    className="flex items-center gap-2 px-4 py-2 rounded-xl bg-white/10 text-white text-xs font-semibold hover:bg-white/20 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                  >
                    {isExporting ? <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" /> : <Download size={14} />}
                    {t('export')}
                  </button>
                </div>
              )}
            </div>
            <div className={`mt-4 flex-1 bg-[#0b0f17] rounded-2xl border border-white/10 min-h-[420px] lg:min-h-[720px] overflow-hidden ${xmlContent ? 'relative block' : 'flex items-center justify-center'}`}>
              {xmlContent ? (
                <iframe
                  ref={iframeRef}
                  src={`https://embed.diagrams.net/?embed=1&spin=1&proto=json&autosave=1&saveAndExit=0&noSaveBtn=1&noExitBtn=1&sidebar=0&layers=0&toolbar=0&menubar=0&status=0&format=0`}
                  className="absolute inset-0 w-full h-full border-0"
                  title="draw.io editor"
                />
              ) : (
                <div className="text-center animate-fade-in">
                  <Wand2 className="w-12 h-12 mx-auto text-slate-500 mb-3" />
                  <p className="text-sm text-slate-400">{t('previewPlaceholder')}</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
