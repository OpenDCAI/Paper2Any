import { useState, useEffect, ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';
import {
  UploadCloud, Download, Loader2, CheckCircle2,
  AlertCircle, Github, Star, X, FileText, ArrowRight, Key, Globe, ToggleLeft, ToggleRight, Sparkles, Image, MessageSquare, Copy, Info
} from 'lucide-react';
import { uploadAndSaveFile } from '../services/fileService';
import { API_KEY, API_URL_OPTIONS, DEFAULT_LLM_API_URL, getPurchaseUrl } from '../config/api';
import { DEFAULT_PDF2PPT_GEN_FIG_MODEL, PDF2PPT_GEN_FIG_MODELS, withModelOptions } from '../config/models';
import { checkQuota, recordUsage } from '../services/quotaService';
import { verifyLlmConnection } from '../services/llmService';
import { useAuthStore } from '../stores/authStore';
import { getApiSettings, saveApiSettings } from '../services/apiSettingsService';
import QRCodeTooltip from './QRCodeTooltip';

const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB
const STORAGE_KEY = 'pdf2ppt-storage';

// ============== 主组件 ==============
const Pdf2PptPage = () => {
  const { t } = useTranslation(['pdf2ppt', 'common']);
  const { user, refreshQuota } = useAuthStore();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [isComplete, setIsComplete] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showBanner, setShowBanner] = useState(true);
  const [downloadBlob, setDownloadBlob] = useState<Blob | null>(null);
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState('');

  // GitHub Stars
  const [stars, setStars] = useState<{dataflow: number | null, agent: number | null, dataflex: number | null}>({
    dataflow: null,
    agent: null,
    dataflex: null,
  });
  const [copySuccess, setCopySuccess] = useState('');

  const shareText = `发现一个超好用的AI工具 DataFlow-Agent！🚀
支持论文转PPT、PDF转PPT、PPT美化等功能，科研打工人的福音！

🔗 在线体验：https://dcai-paper2any.nas.cpolar.cn/
⭐ GitHub Agent：https://github.com/OpenDCAI/Paper2Any
🌟 GitHub Core：https://github.com/OpenDCAI/DataFlow

转发本文案+截图，联系微信群管理员即可获取免费Key！🎁
#AI工具 #PPT制作 #科研效率 #开源项目`;

  const handleCopyShareText = async () => {
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(shareText);
      } else {
        const textArea = document.createElement("textarea");
        textArea.value = shareText;
        textArea.style.position = "fixed";
        textArea.style.left = "-9999px";
        textArea.style.top = "0";
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        try {
          document.execCommand('copy');
        } catch (err) {
          console.error('Fallback: Oops, unable to copy', err);
          throw err;
        } finally {
          document.body.removeChild(textArea);
        }
      }
      setCopySuccess('文案已复制！快去分享吧');
      setTimeout(() => setCopySuccess(''), 2000);
    } catch (err) {
      console.error('复制失败', err);
      setCopySuccess('复制失败，请手动复制');
    }
  };

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
  
  // 配置
  const [useAiEdit, setUseAiEdit] = useState(false);
  const [llmApiUrl, setLlmApiUrl] = useState(DEFAULT_LLM_API_URL);
  const [apiKey, setApiKey] = useState('');
  const [genFigModel, setGenFigModel] = useState(DEFAULT_PDF2PPT_GEN_FIG_MODEL);
  const genFigModelOptions = withModelOptions(PDF2PPT_GEN_FIG_MODELS, genFigModel);

  // 从 localStorage 恢复配置
  useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const saved = JSON.parse(raw);
        
        if (saved.useAiEdit !== undefined) setUseAiEdit(saved.useAiEdit);
        if (saved.genFigModel) setGenFigModel(saved.genFigModel);

        // API settings: prioritize user-specific settings from apiSettingsService
        const userApiSettings = getApiSettings(user?.id || null);
        if (userApiSettings) {
          if (userApiSettings.apiUrl) setLlmApiUrl(userApiSettings.apiUrl);
          if (userApiSettings.apiKey) setApiKey(userApiSettings.apiKey);
        } else {
          if (saved.llmApiUrl) setLlmApiUrl(saved.llmApiUrl);
          if (saved.apiKey) setApiKey(saved.apiKey);
        }
      }
    } catch (e) {
      console.error('Failed to restore pdf2ppt config', e);
    }
  }, [user?.id]);

  // 将配置写入 localStorage
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const data = {
      useAiEdit,
      llmApiUrl,
      apiKey,
      genFigModel,
    };
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
      if (user?.id && llmApiUrl && apiKey) {
        saveApiSettings(user.id, { apiUrl: llmApiUrl, apiKey });
      }
    } catch (e) {
      console.error('Failed to persist pdf2ppt config', e);
    }
  }, [useAiEdit, llmApiUrl, apiKey, genFigModel, user?.id]);

  const validateDocFile = (file: File): boolean => {
    const ext = file.name.split('.').pop()?.toLowerCase();
    if (ext !== 'pdf') {
      setError(t('errors.pdfOnly'));
      return false;
    }
    return true;
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !validateDocFile(file)) return;
    if (file.size > MAX_FILE_SIZE) {
      setError(t('errors.sizeLimit'));
      return;
    }
    setSelectedFile(file);
    setError(null);
    setIsComplete(false);
    setDownloadBlob(null);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (!file || !validateDocFile(file)) return;
    if (file.size > MAX_FILE_SIZE) {
      setError(t('errors.sizeLimit'));
      return;
    }
    setSelectedFile(file);
    setError(null);
    setIsComplete(false);
    setDownloadBlob(null);
  };

  const handleConvert = async () => {
    if (!selectedFile) {
      setError(t('errors.selectFile'));
      return;
    }

    // Check quota before proceeding
    const quota = await checkQuota(user?.id || null, user?.is_anonymous || false);
    if (quota.remaining <= 0) {
      setError(t('errors.quotaFull'));
      return;
    }

    if (useAiEdit) {
      if (!apiKey.trim()) {
        setError(t('errors.enterKey'));
        return;
      }
      if (!llmApiUrl.trim()) {
        setError(t('errors.enterUrl'));
        return;
      }

      // Step 0: Verify LLM Connection if AI Edit is enabled
      try {
        setIsValidating(true);
        setError(null);
        await verifyLlmConnection(llmApiUrl, apiKey, import.meta.env.VITE_DEFAULT_LLM_MODEL || 'deepseek-v3.2'); 
        setIsValidating(false);
      } catch (err) {
        setIsValidating(false);
        const message = err instanceof Error ? err.message : t('errors.apiFail');
        setError(message);
        return;
      }
    }
    
    setIsProcessing(true);
    setError(null);
    setProgress(0);
    setStatusMessage(t('status.uploading'));
    
    // 模拟进度
    const progressInterval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 90) {
          clearInterval(progressInterval);
          return 90;
        }
        const messages = [
          t('status.analyzing'),
          t('status.extracting'),
          t('status.extractingIcon'),
          t('status.generating'),
          t('status.exporting'),
        ];
        const msgIndex = Math.floor(prev / 20);
        if (msgIndex < messages.length) {
          setStatusMessage(messages[msgIndex]);
        }
        return prev + Math.random() * 5;
      });
    }, 3000);
    
    try {
      const formData = new FormData();
      formData.append('pdf_file', selectedFile);
      formData.append('email', user?.id || user?.email || '');
      
      if (useAiEdit) {
        formData.append('use_ai_edit', 'true');
        formData.append('chat_api_url', llmApiUrl.trim());
        formData.append('api_key', apiKey.trim());
        formData.append('gen_fig_model', genFigModel);
      } else {
        formData.append('use_ai_edit', 'false');
      }
      
      const res = await fetch('/api/v1/pdf2ppt/generate', {
        method: 'POST',
        headers: { 'X-API-Key': API_KEY },
        body: formData,
      });
      
      clearInterval(progressInterval);
      
      if (!res.ok) {
        let msg = t('errors.serverBusy');
        if (res.status === 403) {
          msg = '邀请码不正确或已失效';
        } else if (res.status === 429) {
          msg = '请求过于频繁，请稍后再试';
        } else {
          try {
            const errBody = await res.json();
            if (errBody?.error) msg = errBody.error;
          } catch { /* ignore parse error */ }
        }
        throw new Error(msg);
      }

      // 获取文件 blob
      const blob = await res.blob();
      if (!blob || blob.size === 0) {
        throw new Error('生成失败：未能获取到有效的文件，请检查 API Key 余额后重试');
      }
      setDownloadBlob(blob);
      setProgress(100);
      setStatusMessage(t('status.complete'));
      setIsComplete(true);

      // 校验通过后才扣积分
      await recordUsage(user?.id || null, 'pdf2ppt', { isAnonymous: user?.is_anonymous || false });
      refreshQuota();
      const outputName = selectedFile?.name.replace('.pdf', '.pptx') || 'pdf2ppt_output.pptx';
      console.log('[Pdf2PptPage] Uploading file to storage:', outputName);
      await uploadAndSaveFile(blob, outputName, 'pdf2ppt');
      console.log('[Pdf2PptPage] File uploaded successfully');
      
    } catch (err) {
      clearInterval(progressInterval);
      const message = err instanceof Error ? err.message : t('errors.serverBusy');
      setError(message);
      setProgress(0);
      setStatusMessage('');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDownload = () => {
    if (!downloadBlob) return;
    const url = URL.createObjectURL(downloadBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = selectedFile?.name.replace('.pdf', '.pptx') || 'converted.pptx';
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleReset = () => {
    setSelectedFile(null);
    setIsComplete(false);
    setDownloadBlob(null);
    setError(null);
    setProgress(0);
    setStatusMessage('');
  };

  return (
    <div className="w-full h-screen flex flex-col bg-[#050512] overflow-hidden">
      {showBanner && (
        <div className="w-full bg-gradient-to-r from-purple-600 via-pink-600 to-orange-500 relative overflow-hidden flex-shrink-0">
          <div className="absolute inset-0 bg-black opacity-20"></div>
          <div className="absolute inset-0 animate-pulse">
            <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-r from-transparent via-white to-transparent opacity-10 animate-shimmer"></div>
          </div>
          
          <div className="relative max-w-7xl mx-auto px-4 py-3 flex flex-col sm:flex-row items-center justify-between gap-3">
            <div className="flex items-center gap-3 flex-wrap justify-center sm:justify-start">
              <a
                href="https://github.com/OpenDCAI"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 bg-white/20 backdrop-blur-sm rounded-full px-3 py-1 hover:bg-white/30 transition-colors"
              >
                <Star size={16} className="text-yellow-300 fill-yellow-300 animate-pulse" />
                <span className="text-xs font-bold text-white">{t('app.githubProject', { ns: 'common' })}</span>
              </a>
              
              <span className="text-sm font-medium text-white">
                {t('app.exploreMore', { ns: 'common' })}
              </span>
            </div>

            <div className="flex items-center gap-2 flex-wrap justify-center">
              <a
                href="https://github.com/OpenDCAI/DataFlow"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-1.5 bg-white/95 hover:bg-white text-gray-900 rounded-full text-xs font-semibold transition-all hover:scale-105 shadow-lg"
              >
                <Github size={14} />
                <span>DataFlow</span>
                <span className="bg-gray-200 text-gray-800 px-1.5 py-0.5 rounded-full text-[10px] flex items-center gap-0.5"><Star size={8} fill="currentColor" /> {stars.dataflow || 'Star'}</span>
                <span className="bg-purple-600 text-white px-2 py-0.5 rounded-full text-[10px]">HOT</span>
              </a>

              <a
                href="https://github.com/OpenDCAI/Paper2Any"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-1.5 bg-white/95 hover:bg-white text-gray-900 rounded-full text-xs font-semibold transition-all hover:scale-105 shadow-lg"
              >
                <Github size={14} />
                <span>Paper2Any</span>
                <span className="bg-gray-200 text-gray-800 px-1.5 py-0.5 rounded-full text-[10px] flex items-center gap-0.5"><Star size={8} fill="currentColor" /> {stars.agent || 'Star'}</span>
                <span className="bg-pink-600 text-white px-2 py-0.5 rounded-full text-[10px]">NEW</span>
              </a>

              <a
                href="https://github.com/OpenDCAI/DataFlex"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-1.5 bg-white/95 hover:bg-white text-gray-900 rounded-full text-xs font-semibold transition-all hover:scale-105 shadow-lg"
              >
                <Github size={14} />
                <span>DataFlex</span>
                <span className="bg-gray-200 text-gray-800 px-1.5 py-0.5 rounded-full text-[10px] flex items-center gap-0.5"><Star size={8} fill="currentColor" /> {stars.dataflex || 'Star'}</span>
                <span className="bg-sky-600 text-white px-2 py-0.5 rounded-full text-[10px]">NEW</span>
              </a>

              <button
                onClick={() => setShowBanner(false)}
                className="p-1 hover:bg-white/20 rounded-full transition-colors"
                aria-label="关闭"
              >
                <X size={16} className="text-white" />
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="flex-1 overflow-auto">
        <div className="max-w-6xl w-full mx-auto px-6 py-8">
          <div className="max-w-2xl mx-auto">
          {/* 标题 */}
          <div className="text-center mb-8">
            <p className="text-xs uppercase tracking-[0.2em] text-purple-300 mb-3 font-semibold">{t('subtitle')}</p>
            <h1 className="text-4xl md:text-5xl font-bold mb-4">
              <span className="bg-gradient-to-r from-violet-400 via-purple-400 to-fuchsia-400 bg-clip-text text-transparent">
                {t('title')}
              </span>
            </h1>
            <p className="text-base text-gray-300 max-w-xl mx-auto leading-relaxed">
              {t('desc')}<br />
              <span className="text-purple-400">{t('descHighlight')}</span>
            </p>
          </div>

          {/* 主卡片 */}
          <div className="glass rounded-2xl border border-white/10 p-8">
            {!isComplete ? (
              <>
                {/* 上传区域 */}
                <div 
                  className={`border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center text-center gap-4 transition-all mb-6 ${
                    isDragOver ? 'border-purple-500 bg-purple-500/10' : 'border-white/20 hover:border-purple-400'
                  }`} 
                  onDragOver={e => { e.preventDefault(); setIsDragOver(true); }} 
                  onDragLeave={e => { e.preventDefault(); setIsDragOver(false); }} 
                  onDrop={handleDrop}
                >
                  <div className="w-16 h-16 rounded-full bg-gradient-to-br from-violet-500/20 to-fuchsia-500/20 flex items-center justify-center">
                    {selectedFile ? (
                      <FileText size={32} className="text-purple-400" />
                    ) : (
                      <UploadCloud size={32} className="text-purple-400" />
                    )}
                  </div>
                  
                  {selectedFile ? (
                    <div className="px-4 py-2 bg-purple-500/20 border border-purple-500/40 rounded-lg">
                      <p className="text-sm text-purple-300">{t('dropzone.fileInfo', { name: selectedFile.name })}</p>
                      <p className="text-xs text-gray-400 mt-1">
                        {t('dropzone.fileSize', { size: (selectedFile.size / 1024 / 1024).toFixed(2) })}
                      </p>
                    </div>
                  ) : (
                    <>
                      <div>
                        <p className="text-white font-medium mb-1">{t('dropzone.dragText')}</p>
                        <p className="text-sm text-gray-400">{t('dropzone.clickText')}</p>
                      </div>
                      <label className="px-6 py-2.5 rounded-full bg-gradient-to-r from-violet-600 to-fuchsia-600 text-white text-sm font-medium cursor-pointer hover:from-violet-700 hover:to-fuchsia-700 transition-all">
                        {t('dropzone.button')}
                        <input type="file" accept=".pdf" className="hidden" onChange={handleFileChange} />
                      </label>
                    </>
                  )}
                </div>

                {/* 必填配置：邀请码 */}
                {/* <div className="mb-6">
                    <label className="block text-xs text-gray-400 mb-1.5 flex items-center gap-1">
                      <Key size={12} /> 邀请码 <span className="text-red-400">*</span>
                    </label>
                    <input 
                      type="text" 
                      value={inviteCode} 
                      onChange={e => setInviteCode(e.target.value)}
                      placeholder="xxx-xxx"
                      className="w-full rounded-lg border border-white/20 bg-black/40 px-3 py-2.5 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-purple-500"
                    />
                </div> */}

                {/* AI 增强选项开关 */}
                <div className="mb-4 flex items-center justify-between p-3 rounded-xl border border-white/10 bg-white/5">
                  <div className="flex items-center gap-2">
                    <div className="p-1.5 rounded-lg bg-gradient-to-br from-violet-500/20 to-fuchsia-500/20">
                      <Sparkles size={16} className="text-purple-400" />
                    </div>
                    <div>
                      <p className="text-sm font-medium text-white">{t('config.aiEdit')}</p>
                      <p className="text-xs text-gray-400">{t('config.aiEditDesc')}</p>
                    </div>
                  </div>
                  <button 
                    onClick={() => setUseAiEdit(!useAiEdit)}
                    className="focus:outline-none transition-colors"
                  >
                    {useAiEdit ? (
                      <ToggleRight size={32} className="text-purple-500" />
                    ) : (
                      <ToggleLeft size={32} className="text-gray-500" />
                    )}
                  </button>
                </div>

                {/* AI 增强配置面板 - 仅开启时显示 */}
                {useAiEdit && (
                  <div className="space-y-4 mb-6 p-4 rounded-xl border border-purple-500/20 bg-purple-500/5 animate-in fade-in slide-in-from-top-2">
                    <div>
                      <label className="block text-xs text-gray-400 mb-1.5 flex items-center gap-1">
                        <Globe size={12} /> {t('config.apiUrl')} <span className="text-red-400">*</span>
                      </label>
                      <div className="flex items-center gap-2">
                        <select 
                          value={llmApiUrl} 
                          onChange={e => setLlmApiUrl(e.target.value)}
                          className="flex-1 rounded-lg border border-white/20 bg-black/40 px-3 py-2.5 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-purple-500"
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
                          className="whitespace-nowrap text-[10px] text-purple-300 hover:text-purple-200 hover:underline px-1"
                        >
                          {t('config.buyLink')}
                        </a>
                        </QRCodeTooltip>
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-xs text-gray-400 mb-1.5 flex items-center gap-1">
                        <Key size={12} /> {t('config.apiKey')} <span className="text-red-400">*</span>
                      </label>
                      <input 
                        type="password" 
                        value={apiKey} 
                        onChange={e => setApiKey(e.target.value)}
                        placeholder="sk-..."
                        className="w-full rounded-lg border border-white/20 bg-black/40 px-3 py-2.5 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-purple-500"
                      />
                    </div>
                      <div>
                        <label className="block text-xs text-gray-400 mb-1.5 flex items-center gap-1">
                          <Image size={12} /> {t('config.genModel')}
                        </label>
                        <div className="relative">
                          <select 
                            value={genFigModel} 
                            onChange={e => setGenFigModel(e.target.value)}
                            className="w-full appearance-none rounded-lg border border-white/20 bg-black/40 px-3 py-2.5 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-purple-500"
                          >
                            {genFigModelOptions.map((option) => (
                              <option key={option} value={option}>
                                {option === 'gemini-3-pro-image-preview' ? 'Gemini 3 Pro' : option}
                              </option>
                            ))}
                          </select>
                          <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none">
                            <svg width="10" height="6" viewBox="0 0 10 6" fill="none" xmlns="http://www.w3.org/2000/svg">
                              <path d="M1 1L5 5L9 1" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                            </svg>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* 验证状态 */}
                {isValidating && (
                  <div className="mb-6 flex items-center gap-2 text-sm text-purple-300 bg-purple-500/10 border border-purple-500/40 rounded-lg px-4 py-3 animate-pulse">
                    <Loader2 size={16} className="animate-spin" />
                    <p>{t('config.validating')}</p>
                  </div>
                )}

                {/* 进度条 */}
                {isProcessing && (
                  <div className="mb-6">
                    <div className="flex justify-between text-sm text-gray-400 mb-2">
                      <span>{statusMessage}</span>
                      <span>{Math.round(progress)}%</span>
                    </div>
                    <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-gradient-to-r from-violet-500 to-fuchsia-500 transition-all duration-500"
                        style={{ width: `${progress}%` }}
                      />
                    </div>
                  </div>
                )}

                {/* 转换按钮 */}
                <button 
                  onClick={handleConvert} 
                  disabled={!selectedFile || isProcessing} 
                  className="w-full py-4 rounded-xl bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-700 hover:to-fuchsia-700 disabled:from-gray-600 disabled:to-gray-700 text-white font-semibold flex items-center justify-center gap-2 transition-all text-lg"
                >
                  {isProcessing ? (
                    <><Loader2 size={20} className="animate-spin" /> {t('action.converting')}</>
                  ) : (
                    <><ArrowRight size={20} /> {t('action.convert')}</>
                  )}
                </button>

                <div className="flex items-start gap-2 text-xs text-gray-500 mt-3 px-1">
                  <Info size={14} className="mt-0.5 text-gray-400 flex-shrink-0" />
                  <p>处理时间取决于文件大小，请耐心等待。</p>
                </div>
              </>
            ) : (
              /* 完成状态 */
              <div className="text-center py-8">
                <div className="w-24 h-24 rounded-full bg-gradient-to-br from-emerald-500 to-teal-500 flex items-center justify-center mx-auto mb-6">
                  <CheckCircle2 size={48} className="text-white" />
                </div>
                <h2 className="text-2xl font-bold text-white mb-2">{t('complete.title')}</h2>
                <p className="text-gray-400 mb-8">{t('complete.desc')}</p>
                
                <div className="space-y-4">
                  <button 
                    onClick={handleDownload} 
                    className="w-full py-4 rounded-xl bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-600 hover:to-teal-600 text-white font-semibold flex items-center justify-center gap-2 transition-all text-lg"
                  >
                    <Download size={20} /> {t('action.download')}
                  </button>
                  
                  <button 
                    onClick={handleReset} 
                    className="w-full py-3 rounded-xl border border-white/20 text-gray-300 hover:bg-white/10 transition-all"
                  >
                    {t('action.reset')}
                  </button>
                </div>

                {/* 分享与交流群区域 */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-8 text-left">
                  {/* 获取免费 Key */}
                  <div className="glass rounded-xl border border-white/10 p-5 flex flex-col items-center text-center hover:bg-white/5 transition-colors">
                    <div className="w-12 h-12 rounded-full bg-yellow-500/20 text-yellow-300 flex items-center justify-center mb-3">
                      <Star size={24} />
                    </div>
                    <h4 className="text-white font-semibold mb-2">获取免费 API Key</h4>
                    <p className="text-xs text-gray-400 mb-4 leading-relaxed">
                      点击下方平台图标复制推广文案<br/>
                      分享至朋友圈/小红书/推特，截图联系微信群管理员领 Key！
                    </p>
                    
                    {/* 分享按钮组 */}
                    <div className="flex items-center justify-center gap-4 mb-5 w-full">
                      <button onClick={handleCopyShareText} className="flex flex-col items-center gap-1 group">
                        <div className="w-10 h-10 rounded-full bg-[#00C300]/20 text-[#00C300] flex items-center justify-center border border-[#00C300]/30 group-hover:scale-110 transition-transform">
                          <MessageSquare size={18} />
                        </div>
                        <span className="text-[10px] text-gray-400">微信</span>
                      </button>
                      <button onClick={handleCopyShareText} className="flex flex-col items-center gap-1 group">
                        <div className="w-10 h-10 rounded-full bg-[#FF2442]/20 text-[#FF2442] flex items-center justify-center border border-[#FF2442]/30 group-hover:scale-110 transition-transform">
                          <span className="font-bold text-xs">小红书</span>
                        </div>
                        <span className="text-[10px] text-gray-400">小红书</span>
                      </button>
                      <button onClick={handleCopyShareText} className="flex flex-col items-center gap-1 group">
                        <div className="w-10 h-10 rounded-full bg-white/10 text-white flex items-center justify-center border border-white/20 group-hover:scale-110 transition-transform">
                          <span className="font-bold text-lg">𝕏</span>
                        </div>
                        <span className="text-[10px] text-gray-400">Twitter</span>
                      </button>
                      <button onClick={handleCopyShareText} className="flex flex-col items-center gap-1 group">
                        <div className="w-10 h-10 rounded-full bg-purple-500/20 text-purple-300 flex items-center justify-center border border-purple-500/30 group-hover:scale-110 transition-transform">
                          <Copy size={18} />
                        </div>
                        <span className="text-[10px] text-gray-400">复制</span>
                      </button>
                    </div>

                    {copySuccess && (
                      <div className="mb-4 px-3 py-1 bg-green-500/20 text-green-300 text-xs rounded-full animate-in fade-in zoom-in">
                        ✨ {copySuccess}
                      </div>
                    )}

            <div className="w-full space-y-2">
               <a href="https://github.com/OpenDCAI/Paper2Any" target="_blank" rel="noopener noreferrer" className="block w-full py-1.5 px-3 rounded bg-white/5 hover:bg-white/10 text-xs text-purple-300 truncate transition-colors border border-white/5 text-center">
                 ✨如果本项目对你有帮助，可以点个star嘛～
               </a>
               <div className="flex gap-2">
                 <a href="https://github.com/OpenDCAI/Paper2Any" target="_blank" rel="noopener noreferrer" className="flex-1 inline-flex items-center justify-center gap-1 px-2 py-1.5 bg-white/95 hover:bg-white text-gray-900 rounded-full text-[10px] font-semibold transition-all hover:scale-105 shadow-lg">
                   <Github size={10} />
                   <span>Agent</span>
                   <span className="bg-gray-200 text-gray-800 px-1 py-0.5 rounded-full text-[9px] flex items-center gap-0.5"><Star size={7} fill="currentColor" /> {stars.agent || 'Star'}</span>
                 </a>
                 <a href="https://github.com/OpenDCAI/DataFlow" target="_blank" rel="noopener noreferrer" className="flex-1 inline-flex items-center justify-center gap-1 px-2 py-1.5 bg-white/95 hover:bg-white text-gray-900 rounded-full text-[10px] font-semibold transition-all hover:scale-105 shadow-lg">
                   <Github size={10} />
                   <span>Core</span>
                   <span className="bg-gray-200 text-gray-800 px-1 py-0.5 rounded-full text-[9px] flex items-center gap-0.5"><Star size={7} fill="currentColor" /> {stars.dataflow || 'Star'}</span>
                 </a>
               </div>
            </div>
                  </div>

                  {/* 交流群 */}
                  <div className="glass rounded-xl border border-white/10 p-5 flex flex-col items-center text-center hover:bg-white/5 transition-colors">
                    <div className="w-12 h-12 rounded-full bg-green-500/20 text-green-300 flex items-center justify-center mb-3">
                      <MessageSquare size={24} />
                    </div>
                    <h4 className="text-white font-semibold mb-2">加入交流群</h4>
                    <p className="text-xs text-gray-400 mb-4">
                      效果满意？遇到问题？<br/>欢迎扫码加入交流群反馈与讨论
                    </p>
                    <div className="w-32 h-32 bg-white p-1 rounded-lg mb-2">
                      <img src="/wechat.png" alt="交流群二维码" className="w-full h-full object-contain" />
                    </div>
                    <p className="text-[10px] text-gray-500">扫码加入微信交流群</p>
                  </div>
                </div>
              </div>
            )}

            {error && (
              <div className="mt-4 flex items-center gap-2 text-sm text-red-300 bg-red-500/10 border border-red-500/40 rounded-lg px-4 py-3">
                <AlertCircle size={16} /> {error}
              </div>
            )}
          </div>

          {/* 说明文字 */}
          <p className="text-center text-xs text-gray-500 mt-6">
            {t('footer.support')}
          </p>
          </div>

          {/* 示例区 */}
          <div className="space-y-4 mt-16 max-w-4xl mx-auto">
            <div className="flex items-center justify-between flex-wrap gap-2">
              <div className="flex items-center gap-3">
                <h3 className="text-sm font-medium text-gray-200">{t('demo.title')}</h3>
                <a
                  href="https://wcny4qa9krto.feishu.cn/wiki/VXKiwYndwiWAVmkFU6kcqsTenWh"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="group relative inline-flex items-center gap-2 px-3 py-1 rounded-full bg-black/50 border border-white/10 text-xs font-medium text-white overflow-hidden transition-all hover:border-white/30 hover:shadow-[0_0_15px_rgba(168,85,247,0.5)]"
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 via-purple-500/20 to-pink-500/20 opacity-0 group-hover:opacity-100 transition-opacity" />
                  <Sparkles size={12} className="text-yellow-300 animate-pulse" />
                  <span className="bg-gradient-to-r from-blue-300 via-purple-300 to-pink-300 bg-clip-text text-transparent group-hover:from-blue-200 group-hover:via-purple-200 group-hover:to-pink-200">
                    {t('demo.more')}
                  </span>
                </a>
              </div>
              <span className="text-[11px] text-gray-500">
                {t('demo.desc')}
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <DemoCard
                title={t('demo.card1.title')}
                desc={t('demo.card1.desc')}
                inputImg="/pdf2ppt/input_1.png"
                outputImg="/pdf2ppt/output_1.png"
              />
              <DemoCard
                title={t('demo.card2.title')}
                desc={t('demo.card2.desc')}
                inputImg="/pdf2ppt/input_2.png"
                outputImg="/pdf2ppt/output_2.png"
              />
            </div>
          </div>
        </div>

      </div>

      <style>{`
        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
        .animate-shimmer {
          animation: shimmer 3s infinite;
        }
        .glass { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(10px); }
        .demo-input-placeholder {
          min-height: 120px;
        }
        .demo-output-placeholder {
          min-height: 120px;
        }
      `}</style>
    </div>
  );
};

interface DemoCardProps {
  title: string;
  desc: string;
  inputImg?: string;
  outputImg?: string;
}

const DemoCard = ({ title, desc, inputImg, outputImg }: DemoCardProps) => {
  return (
    <div className="glass rounded-lg border border-white/10 p-4 flex flex-col gap-3 hover:bg-white/5 transition-colors">
      <div className="flex gap-3">
        {/* 左侧：输入示例图片 */}
        <div className="flex-1 rounded-md bg-white/5 border border-dashed border-white/10 flex items-center justify-center demo-input-placeholder overflow-hidden relative group">
          {inputImg ? (
            <>
              <img
                src={inputImg}
                alt="输入示例图"
                className="w-full h-full object-cover"
              />
              <div className="absolute inset-0 bg-black/50 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                <span className="text-xs text-white font-medium">Input (PDF)</span>
              </div>
            </>
          ) : (
            <span className="text-[10px] text-gray-400">输入示例图</span>
          )}
        </div>
        
        {/* 中间箭头 */}
        <div className="flex items-center justify-center text-gray-500">
          <ArrowRight size={16} />
        </div>

        {/* 右侧：输出 PPTX 示例图片 */}
        <div className="flex-1 rounded-md bg-violet-500/10 border border-dashed border-violet-300/40 flex items-center justify-center demo-output-placeholder overflow-hidden relative group">
          {outputImg ? (
            <>
              <img
                src={outputImg}
                alt="PPTX 示例图"
                className="w-full h-full object-cover"
              />
              <div className="absolute inset-0 bg-black/50 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                <span className="text-xs text-white font-medium">Output (PPTX)</span>
              </div>
            </>
          ) : (
            <span className="text-[10px] text-violet-200">PPTX 示例图</span>
          )}
        </div>
      </div>
      <div>
        <p className="text-sm text-white font-medium mb-1">{title}</p>
        <p className="text-xs text-gray-400 leading-relaxed">{desc}</p>
      </div>
    </div>
  );
};

export default Pdf2PptPage;
