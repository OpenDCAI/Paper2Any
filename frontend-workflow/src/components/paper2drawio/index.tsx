import React, { useState, useRef, useCallback, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { Wand2, Upload, FileText, Send, Download } from 'lucide-react';
import type { DiagramType, DiagramStyle, ChatMessage } from './types';
import { API_KEY } from '../../config/api';
import { useAuthStore } from '../../stores/authStore';
import { getApiSettings, saveApiSettings } from '../../services/apiSettingsService';
import Banner from './Banner';
import ExamplesSection from './ExamplesSection';

const DRAWIO_ORIGINS = new Set(['https://embed.diagrams.net', 'https://app.diagrams.net']);
const STORAGE_KEY = 'paper2drawio_settings';
const DRAWIO_EXPORT_TIMEOUT_MS = 5000;

const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

export default function Paper2DrawioPage() {
  const { t } = useTranslation('paper2drawio');
  const { user } = useAuthStore();

  // 状态
  const [uploadMode, setUploadMode] = useState<'file' | 'text'>('text');
  const [textContent, setTextContent] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [diagramType, setDiagramType] = useState<DiagramType>('auto');
  const [diagramStyle, setDiagramStyle] = useState<DiagramStyle>('default');
  const [xmlContent, setXmlContent] = useState('');
  const [isLoading, setIsLoading] = useState(false);
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
  const [model, setModel] = useState('gpt-4o');

  const iframeRef = useRef<HTMLIFrameElement>(null);
  const chatListRef = useRef<HTMLDivElement>(null);
  const lastLoadedXmlRef = useRef('');
  const pendingExportRef = useRef<{
    resolve: ((data: string) => void) | null;
    reject: ((error: Error) => void) | null;
    format: 'xml' | 'png' | 'svg' | null;
  }>({ resolve: null, reject: null, format: null });
  const panelClass = 'rounded-2xl bg-white/5 border border-white/10 p-4 backdrop-blur-xl shadow-[0_20px_60px_rgba(0,0,0,0.25)] transition-all duration-300';
  const inputClass = 'w-full rounded-xl bg-white/5 border border-white/10 px-3 py-2 text-sm text-white placeholder-slate-500 outline-none transition focus:border-sky-400/60 focus:ring-2 focus:ring-sky-500/20';

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
          xmlContent?: string;
          chatHistory?: ChatMessage[];
          chatInput?: string;
          exportFormat?: 'drawio' | 'png' | 'svg';
          exportFilename?: string;
        };

        if (saved.uploadMode) setUploadMode(saved.uploadMode);
        if (saved.textContent) setTextContent(saved.textContent);
        if (saved.diagramType) setDiagramType(saved.diagramType);
        if (saved.diagramStyle) setDiagramStyle(saved.diagramStyle);
        if (saved.model) setModel(saved.model);
        if (saved.xmlContent) setXmlContent(saved.xmlContent);
        if (saved.chatHistory) setChatHistory(saved.chatHistory);
        if (saved.chatInput) setChatInput(saved.chatInput);
        if (saved.exportFormat) setExportFormat(saved.exportFormat);
        if (saved.exportFilename) setExportFilename(saved.exportFilename);

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
      xmlContent,
      chatHistory,
      chatInput,
      exportFormat,
      exportFilename,
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
    xmlContent,
    chatHistory,
    chatInput,
    exportFormat,
    exportFilename,
    user?.id,
  ]);

  // 生成图表
  const handleGenerate = useCallback(async () => {
    if (!textContent && !file) return;
    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append('chat_api_url', apiUrl);
      formData.append('api_key', apiKey);
      formData.append('model', model);
      formData.append('input_type', uploadMode === 'file' ? 'PDF' : 'TEXT');
      formData.append('diagram_type', diagramType);
      formData.append('diagram_style', diagramStyle);
      formData.append('language', 'zh');

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
  }, [textContent, file, uploadMode, apiUrl, apiKey, model, diagramType, diagramStyle]);

  const postToDrawio = useCallback((payload: Record<string, unknown>) => {
    const frame = iframeRef.current?.contentWindow;
    if (!frame) return;
    frame.postMessage(JSON.stringify(payload), '*');
  }, []);

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
        return;
      }

      if ((message.event === 'save' || message.event === 'autosave') && typeof message.xml === 'string') {
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
    postToDrawio({ action: 'load', xml: xmlContent, autosave: 1 });
    lastLoadedXmlRef.current = xmlContent;
  }, [drawioReady, xmlContent, postToDrawio]);

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
            {/* API 配置 */}
            <div className={panelClass}>
              <h3 className="text-sm font-semibold text-slate-200 mb-3 flex items-center gap-2">
                <Wand2 className="text-sky-300" size={18} />
                {t('apiConfig')}
              </h3>
              <div className="space-y-2">
                <input
                  type="text"
                  placeholder={t('apiUrl')}
                  value={apiUrl}
                  onChange={e => setApiUrl(e.target.value)}
                  className={inputClass}
                />
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

            {/* 图表类型选择 */}
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

            {/* 生成按钮 */}
            <button
              onClick={handleGenerate}
              disabled={isLoading || (!textContent && !file)}
              className="w-full py-3 rounded-2xl font-semibold text-white bg-gradient-to-r from-sky-500 to-cyan-500 hover:from-sky-400 hover:to-cyan-400 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-[0_18px_45px_rgba(14,165,233,0.35)] flex items-center justify-center gap-2"
            >
              {isLoading ? (
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

            {/* 对话编辑 */}
            {xmlContent && (
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
                  src={`https://embed.diagrams.net/?embed=1&spin=1&proto=json&autosave=1&saveAndExit=0&noSaveBtn=1&noExitBtn=1`}
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

        {/* 示例区域 */}
        <ExamplesSection />
      </div>
    </div>
  );
}
