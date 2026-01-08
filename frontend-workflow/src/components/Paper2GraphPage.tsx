import { useState, useEffect, ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';
import { FileText, UploadCloud, Type, Settings2, Download, Loader2, CheckCircle2, AlertCircle, Image as ImageIcon, ChevronDown, ChevronUp, Github, Star, X, Info, Sparkles } from 'lucide-react';
import { uploadAndSaveFile } from '../services/fileService';
import { API_KEY } from '../config/api';
import { checkQuota, recordUsage, QuotaInfo } from '../services/quotaService';
import { verifyLlmConnection } from '../services/llmService';
import { useAuthStore } from '../stores/authStore';
import QRCodeTooltip from './QRCodeTooltip';

type UploadMode = 'file' | 'text' | 'image';
type FileKind = 'pdf' | 'image' | null;
type GraphType = 'model_arch' | 'tech_route' | 'exp_data';
type Language = 'zh' | 'en';
type StyleType = 'cartoon' | 'realistic';
type FigureComplex = 'easy' | 'mid' | 'hard';

const BACKEND_API = '/api/paper2figure/generate';
const JSON_API = '/api/paper2figure/generate_json';
const HISTORY_API = '/api/paper2figure/history_files';

const IMAGE_EXTENSIONS = ['png', 'jpg', 'jpeg', 'bmp', 'gif', 'webp', 'tiff'];

function detectFileKind(file: File): FileKind {
  const ext = file.name.split('.').pop()?.toLowerCase();
  if (!ext) return null;
  if (ext === 'pdf') return 'pdf';
  if (IMAGE_EXTENSIONS.includes(ext)) return 'image';
  return null;
}

// 生成阶段定义
type GenerationStage = {
  id: number;
  message: string;
  duration: number; // 该阶段持续时间（秒）
};

const GENERATION_STAGES: GenerationStage[] = [
  { id: 1, message: '正在分析论文内容...', duration: 30 },
  { id: 2, message: '正在生成科研绘图...', duration: 30 },
  { id: 3, message: '正在转为可编辑绘图...', duration: 30 },
  { id: 4, message: '正在合成 PPT...', duration: 30 },
];

const MAX_FILE_SIZE = 20 * 1024 * 1024; // 20MB
const STORAGE_KEY = 'paper2figure_config_v1';

const Paper2FigurePage = () => {
  const { t, i18n } = useTranslation('paper2graph');
  const { user, refreshQuota } = useAuthStore();
  const [uploadMode, setUploadMode] = useState<UploadMode>('file');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileKind, setFileKind] = useState<FileKind>(null);
  const [textContent, setTextContent] = useState('');
  const [graphType, setGraphType] = useState<GraphType>('model_arch');
  const [language, setLanguage] = useState<Language>('zh');
  const [style, setStyle] = useState<StyleType>('cartoon');
  const [figureComplex, setFigureComplex] = useState<FigureComplex>('easy');
  const [inviteCode, setInviteCode] = useState('');

  const [llmApiUrl, setLlmApiUrl] = useState('https://api.apiyi.com/v1');
  const [apiKey, setApiKey] = useState('');
  const [model, setModel] = useState('gemini-2.5-flash-image-preview');
  const [showAdvanced, setShowAdvanced] = useState(false);

  const [isLoading, setIsLoading] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [lastFilename, setLastFilename] = useState('paper2figure.pptx');
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [showBanner, setShowBanner] = useState(true);
  const [isDragOver, setIsDragOver] = useState(false);

  // 技术路线图 JSON 返回的资源路径
  const [pptPath, setPptPath] = useState<string | null>(null);
  const [svgPath, setSvgPath] = useState<string | null>(null);
  const [svgPreviewPath, setSvgPreviewPath] = useState<string | null>(null);

  // 新增：本次任务所有输出文件 URL 列表 + 是否展示输出面板
  const [allOutputFiles, setAllOutputFiles] = useState<string[]>([]);
  const [showOutputPanel, setShowOutputPanel] = useState(false);

  // GitHub Stars
  const [stars, setStars] = useState<{dataflow: number | null, agent: number | null, dataflex: number | null}>({
    dataflow: null,
    agent: null,
    dataflex: null,
  });

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

  // 根据邀请码拉取历史文件列表（所有 graph_type）
  const fetchHistoryFiles = async (code: string) => {
    const invite = code.trim();
    if (!invite) return;
    try {
      const res = await fetch(
        `${HISTORY_API}?invite_code=${encodeURIComponent(invite)}`
      );
      if (!res.ok) return;
      const data = await res.json();
      const urls: string[] = (data.files || []).map((f: any) =>
        typeof f === 'string' ? f : f.url,
      );
      setAllOutputFiles(urls);
    } catch (e) {
      console.error('fetch history files error', e);
    }
  };

  // 新增：生成阶段状态
  const [currentStage, setCurrentStage] = useState(0);
  const [stageProgress, setStageProgress] = useState(0);

  useEffect(() => {
    return () => {
      if (downloadUrl) {
        URL.revokeObjectURL(downloadUrl);
      }
    };
  }, [downloadUrl]);

  // 从 localStorage 恢复配置
  useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      const saved = JSON.parse(raw) as {
        uploadMode?: UploadMode;
        textContent?: string;
        graphType?: GraphType;
        language?: Language;
        style?: StyleType;
        figureComplex?: FigureComplex;
        inviteCode?: string;
        llmApiUrl?: string;
        apiKey?: string;
        model?: string;
      };

      if (saved.uploadMode) setUploadMode(saved.uploadMode);
      if (saved.textContent) setTextContent(saved.textContent);
      if (saved.graphType) setGraphType(saved.graphType);
      if (saved.language) setLanguage(saved.language);
      if (saved.style) setStyle(saved.style);
      if (saved.figureComplex) setFigureComplex(saved.figureComplex);
      if (saved.inviteCode) setInviteCode(saved.inviteCode);
      if (saved.llmApiUrl) setLlmApiUrl(saved.llmApiUrl);
      if (saved.apiKey) setApiKey(saved.apiKey);
      if (saved.model) setModel(saved.model);
    } catch (e) {
      console.error('Failed to restore paper2figure config', e);
    }
  }, []);

  // 将配置写入 localStorage
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const data = {
      uploadMode,
      textContent,
      graphType,
      language,
      style,
      figureComplex,
      inviteCode,
      llmApiUrl,
      apiKey,
      model,
    };
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    } catch (e) {
      console.error('Failed to persist paper2figure config', e);
    }
  }, [uploadMode, textContent, graphType, language, style, figureComplex, inviteCode, llmApiUrl, apiKey, model]);

  // 新增：管理生成阶段的定时器
  useEffect(() => {
    if (!isLoading) {
      setCurrentStage(0);
      setStageProgress(0);
      return;
    }

    let stageTimer: ReturnType<typeof setTimeout>;
    let progressTimer: ReturnType<typeof setInterval>;
    let currentStageIndex = 0;
    let elapsedTime = 0;

    const updateProgress = () => {
      elapsedTime += 0.5;
      const currentStageDuration = GENERATION_STAGES[currentStageIndex].duration;
      const progress = Math.min((elapsedTime % currentStageDuration) / currentStageDuration * 100, 100);
      setStageProgress(progress);
    };

    const advanceStage = () => {
      if (currentStageIndex < GENERATION_STAGES.length - 1) {
        currentStageIndex++;
        setCurrentStage(currentStageIndex);
        elapsedTime = 0;
        setStageProgress(0);
      }
    };

    // 每0.5秒更新进度条
    progressTimer = setInterval(updateProgress, 500);

    // 根据阶段时长切换阶段
    const scheduleNextStage = () => {
      const duration = GENERATION_STAGES[currentStageIndex].duration * 1000;
      stageTimer = setTimeout(() => {
        advanceStage();
        if (currentStageIndex < GENERATION_STAGES.length - 1) {
          scheduleNextStage();
        }
      }, duration);
    };

    scheduleNextStage();

    return () => {
      clearTimeout(stageTimer);
      clearInterval(progressTimer);
    };
  }, [isLoading]);

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) {
      setSelectedFile(null);
      setFileKind(null);
      return;
    }
    if (file.size > MAX_FILE_SIZE) {
      setError('文件大小超过 20MB 限制');
      return;
    }
    const kind = detectFileKind(file);
    setSelectedFile(file);
    setFileKind(kind);
    setError(null);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);

    const file = e.dataTransfer.files?.[0];
    if (!file) {
      setSelectedFile(null);
      setFileKind(null);
      return;
    }
    if (file.size > MAX_FILE_SIZE) {
      setError('文件大小超过 20MB 限制');
      return;
    }

    const kind = detectFileKind(file);
    setSelectedFile(file);
    setFileKind(kind);
    setError(null);
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.items && e.dataTransfer.items.length > 0) {
      setIsDragOver(true);
    }
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  };

  const handleSubmit = async () => {
    if (isLoading) return;
    setError(null);
    setSuccessMessage(null);
    setDownloadUrl(null);
    setPptPath(null);
    setSvgPath(null);
    setSvgPreviewPath(null);
    setCurrentStage(0);
    setStageProgress(0);
    setShowOutputPanel(true);

    // Check quota before proceeding
    const quota = await checkQuota(user?.id || null, user?.is_anonymous || false);
    if (quota.remaining <= 0) {
      setError(quota.isAuthenticated
        ? t('errors.quotaUserExhausted')
        : t('errors.quotaGuestExhausted'));
      return;
    }

    if (!llmApiUrl.trim() || !apiKey.trim()) {
      setError(t('errors.missingApiConfig'));
      return;
    }

    // 技术路线图 / 实验数据图 不支持 image 作为输入
    if ((graphType === 'tech_route' || graphType === 'exp_data') && uploadMode === 'image') {
      setError(t('errors.techRouteNoImage'));
      return;
    }

    // 实验数据图 仅支持 file (PDF)
    if (graphType === 'exp_data' && uploadMode !== 'file') {
      setError(t('errors.expDataFileOnly'));
      return;
    }

    const formData = new FormData();
    formData.append('img_gen_model_name', model);
    formData.append('chat_api_url', llmApiUrl.trim());
    formData.append('api_key', apiKey.trim());
    formData.append('input_type', uploadMode);
    formData.append('invite_code', inviteCode.trim());
    formData.append('graph_type', graphType);
    formData.append('style', style);

    // 使用全局 i18n 语言作为后端语言参数
    const backendLanguage = i18n.language && i18n.language.startsWith('zh') ? 'zh' : 'en';

    if (graphType === 'model_arch') {
      // 模型结构图：使用绘图难度，不再传语言
      formData.append('figure_complex', figureComplex);
    } else {
      // 其他图：使用语言配置，不传绘图难度
      formData.append('language', backendLanguage);
    }

    if (uploadMode === 'file' || uploadMode === 'image') {
      if (!selectedFile) {
        setError(t('errors.noFile'));
        return;
      }
      const kind = fileKind ?? detectFileKind(selectedFile);
      if (!kind) {
        setError(t('errors.unsupportedFile'));
        return;
      }
      formData.append('file', selectedFile);
      formData.append('file_kind', kind);
    } else if (uploadMode === 'text') {
      if (!textContent.trim()) {
        setError(t('errors.noText'));
        return;
      }
      formData.append('text', textContent.trim());
    }

    try {
      // Step 0: Verify LLM Connection first
      setIsValidating(true);
      setError(null);
      await verifyLlmConnection(llmApiUrl, apiKey, model);
      setIsValidating(false);

      setIsLoading(true);

      if (graphType === 'tech_route') {
        // 技术路线图：调用 JSON 接口，返回 PPT + SVG
        const res = await fetch(JSON_API, {
          method: 'POST',
          headers: { 'X-API-Key': API_KEY },
          body: formData,
        });

        if (!res.ok) {
          let msg = '服务器繁忙，请稍后再试';
          if (res.status === 403) {
            msg = t('errors.inviteInvalid');
          } else if (res.status === 429) {
            msg = t('errors.tooManyRequests');
          }
          throw new Error(msg);
        }

        type Paper2FigureJsonResp = {
          success: boolean;
          ppt_filename: string;
          svg_filename: string;
          svg_image_filename: string;
          all_output_files?: string[];
        };

        const data: Paper2FigureJsonResp = await res.json();

        if (!data.success) {
          throw new Error(t('errors.serverBusy'));
        }

        setPptPath(data.ppt_filename);
        setSvgPath(data.svg_filename);
        setSvgPreviewPath(data.svg_image_filename);
        setAllOutputFiles(data.all_output_files ?? []);
        setSuccessMessage(t('success.techRouteGenerated'));

        // Record usage
        await recordUsage(user?.id || null, 'paper2figure');
        refreshQuota();

        // Fetch PPT file and upload to Supabase Storage
        if (data.ppt_filename) {
          try {
            console.log('[Paper2GraphPage] Fetching tech_route file from:', data.ppt_filename);
            const pptRes = await fetch(data.ppt_filename);
            if (!pptRes.ok) {
              throw new Error(`HTTP ${pptRes.status}: ${pptRes.statusText}`);
            }
            const pptBlob = await pptRes.blob();
            const pptName = data.ppt_filename.split('/').pop() || 'tech_route.pptx';
            console.log('[Paper2GraphPage] Uploading tech_route file to storage:', pptName);
            const uploadResult = await uploadAndSaveFile(pptBlob, pptName, 'paper2figure');
            if (uploadResult) {
              console.log('[Paper2GraphPage] Tech_route file uploaded successfully:', uploadResult.file_name);
            } else {
              console.warn('[Paper2GraphPage] Tech_route file upload skipped or failed');
            }
          } catch (e) {
            console.error('[Paper2GraphPage] Failed to upload tech_route file:', e);
          }
        }
      } else {
        // 其他类型：保持原来的 PPTX blob 下载逻辑
        const res = await fetch(BACKEND_API, {
          method: 'POST',
          headers: { 'X-API-Key': API_KEY },
          body: formData,
        });

        if (!res.ok) {
          let msg = t('errors.serverBusy');
          if (res.status === 403) {
            msg = t('errors.inviteInvalid');
          } else if (res.status === 429) {
            msg = t('errors.tooManyRequests');
          }
          throw new Error(msg);
        }

        const disposition = res.headers.get('content-disposition') || '';
        let filename = 'paper2figure.pptx';
        const match = disposition.match(/filename="?([^";]+)"?/i);
        if (match?.[1]) {
          filename = decodeURIComponent(match[1]);
        }

        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        setDownloadUrl(url);
        setLastFilename(filename);
        setSuccessMessage(t('success.pptGenerated'));

        // Record usage and save file to Supabase Storage
        await recordUsage(user?.id || null, 'paper2figure');
        refreshQuota();

        console.log('[Paper2GraphPage] Uploading file to storage:', filename);
        const uploadResult = await uploadAndSaveFile(blob, filename, 'paper2figure');
        if (uploadResult) {
          console.log('[Paper2GraphPage] File uploaded successfully:', uploadResult.file_name);
        } else {
          console.warn('[Paper2GraphPage] File upload skipped or failed');
        }

        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        a.remove();
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : t('errors.serverBusy');
      setError(message);
    } finally {
      setIsLoading(false);
      setIsValidating(false);
    }
  };

  const showFileHint = () => {
    if (!selectedFile) return t('upload.fileHint');
    if (fileKind === 'pdf') return `PDF：${selectedFile.name}`;
    if (fileKind === 'image') return `Image：${selectedFile.name}`;
    return `Unknown file type: ${selectedFile.name}`;
  };

  return (
    <div className="w-full h-full flex flex-col bg-[#050512]">
      {/* GitHub 引流横幅 */}
      {showBanner && (
        <div className="w-full bg-gradient-to-r from-purple-600 via-pink-600 to-orange-500 relative overflow-hidden">
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
                <span className="text-xs font-bold text-white">GitHub开源项目</span>
              </a>
              
              <span className="text-sm font-medium text-white">
                🚀 探索更多 AI 数据处理工具
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

      {/* 主区域：居中简洁布局 */}
      <div className="flex-1 flex flex-col items-center justify-start px-6 pt-20 pb-10 overflow-auto">
        <div className="w-full max-w-5xl animate-fade-in">
          {/* 顶部标题区 */}
          <div className="mb-8 text-center">
            <p className="text-xs uppercase tracking-[0.2em] text-primary-300 mb-2">
              {t('hero.badge')}
            </p>
            <h1 className="text-3xl font-semibold text-white mb-2">
              {t('hero.title')}
            </h1>
            <p className="text-sm text-gray-400 max-w-2xl mx-auto">
              {t('hero.subtitle')}
            </p>
          </div>

          {/* 上半区：上传区 + 高级配置 */}
          <div className="grid grid-cols-1 lg:grid-cols-[2fr,minmax(260px,1fr)] gap-6 mb-10">
            {/* 上传卡片 */}
            <div className="glass rounded-xl border border-white/10 p-6 lg:p-8 relative overflow-hidden">
              {/* 装饰背景光 */}
              <div className="absolute top-0 left-1/2 -translate-x-1/2 w-2/3 h-1 bg-gradient-to-r from-transparent via-blue-500 to-transparent opacity-50 blur-sm"></div>

              <div className="relative">
                {/* 绘图类型选择 */}
                <div className="mb-6">
                  <label className="block text-xs font-medium text-gray-400 mb-2">{t('graphType.label')}</label>
                  <select
                    value={graphType}
                    onChange={e => {
                      const newType = e.target.value as GraphType;
                      setGraphType(newType);
                      // 实验数据图强制使用 file 模式
                      if (newType === 'exp_data') {
                        setUploadMode('file');
                      }
                    }}
                    className="w-full rounded-xl border border-white/10 bg-black/40 px-4 py-3 text-sm text-gray-200 outline-none focus:ring-2 focus:ring-blue-500 transition-all"
                  >
                    <option value="model_arch">{t('graphType.model_arch')}</option>
                    <option value="tech_route">{t('graphType.tech_route')}</option>
                    <option value="exp_data">{t('graphType.exp_data')}</option>
                  </select>
                </div>

                {/* 上传模式 Tab (炫酷卡片式 - 蓝色系) */}
                <div className="grid grid-cols-3 gap-3 mb-6 p-1.5 bg-black/40 rounded-2xl border border-white/5">
                  <button
                    type="button"
                    onClick={() => setUploadMode('file')}
                    className={`relative group flex flex-col items-center justify-center py-3 rounded-xl transition-all duration-300 overflow-hidden ${
                      uploadMode === 'file'
                        ? 'bg-gradient-to-br from-blue-600 to-cyan-500 text-white shadow-lg shadow-blue-500/30 scale-[1.02] ring-1 ring-white/20'
                        : 'bg-white/5 text-gray-400 hover:bg-white/10 hover:text-gray-200 hover:scale-[1.02]'
                    }`}
                  >
                     {uploadMode === 'file' && (
                        <div className="absolute inset-0 w-full h-full bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full animate-shimmer-fast"></div>
                     )}
                     <FileText size={22} className={`mb-1.5 transition-colors ${uploadMode === 'file' ? 'text-white' : 'text-gray-500 group-hover:text-blue-400'}`} />
                     <span className={`text-sm font-bold tracking-wide ${uploadMode === 'file' ? 'text-white' : 'text-gray-300'}`}>{t('uploadTabs.file')}</span>
                     <span className={`text-[10px] uppercase tracking-wider font-medium ${uploadMode === 'file' ? 'text-blue-100' : 'text-gray-600'}`}>{t('uploadTabs.fileSub')}</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => {
                      if (graphType === 'exp_data') {
                        setError(t('errors.expDataNoImage'));
                        return;
                      }
                      setUploadMode('text');
                    }}
                    className={`relative group flex flex-col items-center justify-center py-3 rounded-xl transition-all duration-300 overflow-hidden ${
                      graphType === 'exp_data'
                        ? 'opacity-40 cursor-not-allowed bg-white/5 text-gray-600'
                        : uploadMode === 'text'
                           ? 'bg-gradient-to-br from-blue-600 to-cyan-500 text-white shadow-lg shadow-blue-500/30 scale-[1.02] ring-1 ring-white/20'
                           : 'bg-white/5 text-gray-400 hover:bg-white/10 hover:text-gray-200 hover:scale-[1.02]'
                    }`}
                  >
                     {uploadMode === 'text' && (
                        <div className="absolute inset-0 w-full h-full bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full animate-shimmer-fast"></div>
                     )}
                     <Type size={22} className={`mb-1.5 transition-colors ${uploadMode === 'text' ? 'text-white' : 'text-gray-500 group-hover:text-blue-400'}`} />
                     <span className={`text-sm font-bold tracking-wide ${uploadMode === 'text' ? 'text-white' : 'text-gray-300'}`}>{t('uploadTabs.text')}</span>
                     <span className={`text-[10px] uppercase tracking-wider font-medium ${uploadMode === 'text' ? 'text-blue-100' : 'text-gray-600'}`}>{t('uploadTabs.textSub')}</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => {
                      if (graphType === 'tech_route') {
                        setError(t('errors.techRouteNoImage'));
                        return;
                      }
                      if (graphType === 'exp_data') {
                        setError(t('errors.expDataNoImage'));
                        return;
                      }
                      setUploadMode('image');
                    }}
                    className={`relative group flex flex-col items-center justify-center py-3 rounded-xl transition-all duration-300 overflow-hidden ${
                      graphType === 'tech_route' || graphType === 'exp_data'
                        ? 'opacity-40 cursor-not-allowed bg-white/5 text-gray-600'
                        : uploadMode === 'image'
                           ? 'bg-gradient-to-br from-blue-600 to-cyan-500 text-white shadow-lg shadow-blue-500/30 scale-[1.02] ring-1 ring-white/20'
                           : 'bg-white/5 text-gray-400 hover:bg-white/10 hover:text-gray-200 hover:scale-[1.02]'
                    }`}
                  >
                     {uploadMode === 'image' && (
                        <div className="absolute inset-0 w-full h-full bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full animate-shimmer-fast"></div>
                     )}
                     <ImageIcon size={22} className={`mb-1.5 transition-colors ${uploadMode === 'image' ? 'text-white' : 'text-gray-500 group-hover:text-blue-400'}`} />
                     <span className={`text-sm font-bold tracking-wide ${uploadMode === 'image' ? 'text-white' : 'text-gray-300'}`}>{t('uploadTabs.image')}</span>
                     <span className={`text-[10px] uppercase tracking-wider font-medium ${uploadMode === 'image' ? 'text-blue-100' : 'text-gray-600'}`}>{t('uploadTabs.imageSub')}</span>
                  </button>
                </div>

                {/* 不同模式内容区域 */}
                {(uploadMode === 'file' || uploadMode === 'image') && (
                  <div
                    className={`border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center text-center gap-4 transition-all h-[300px] ${
                      isDragOver ? 'border-blue-500 bg-blue-500/10' : 'border-white/20 hover:border-blue-400 bg-black/20'
                    }`}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                  >
                    <div className="w-16 h-16 rounded-full bg-gradient-to-br from-blue-500/20 to-cyan-500/20 flex items-center justify-center">
                      <UploadCloud size={32} className="text-blue-400" />
                    </div>
                    <div>
                      <p className="text-white font-medium mb-1">
                        {uploadMode === 'file' ? t('upload.fileDragTitleFile') : t('upload.fileDragTitleImage')}
                      </p>
                      <p className="text-sm text-gray-400">
                        {showFileHint()}
                      </p>
                    </div>
                    <label className="px-6 py-2.5 rounded-full bg-gradient-to-r from-blue-600 to-cyan-600 text-white text-sm font-medium cursor-pointer hover:from-blue-700 hover:to-cyan-700 transition-all shadow-lg shadow-blue-500/20">
                      {t('upload.selectFile')}
                      <input
                        type="file"
                        accept={
                          uploadMode === 'file'
                            ? graphType === 'model_arch'
                              ? '.pdf,image/*'
                              : '.pdf'
                            : 'image/*'
                        }
                        className="hidden"
                        onChange={handleFileChange}
                      />
                    </label>
                    {selectedFile && (
                        <div className="px-4 py-2 bg-blue-500/20 border border-blue-500/40 rounded-lg animate-fade-in">
                          <p className="text-sm text-blue-300 font-medium">✓ {selectedFile.name}</p>
                        </div>
                    )}
                  </div>
                )}

                {uploadMode === 'text' && (
                  <div className="space-y-3 h-[300px] flex flex-col">
                    <label className="block text-xs font-medium text-gray-400">
                      {t('upload.textLabel')}
                    </label>
                    <textarea
                      value={textContent}
                      onChange={e => setTextContent(e.target.value)}
                      placeholder={t('upload.textPlaceholder')}
                      className="flex-1 w-full rounded-xl border border-white/20 bg-black/40 px-4 py-3 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-blue-500 resize-none placeholder:text-gray-600"
                    />
                    <p className="text-[11px] text-gray-500 text-right">
                      {t('upload.textTip')}
                    </p>
                  </div>
                )}
              </div>
            </div>

            {/* 高级配置卡片（折叠） */}
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
                  {/* <div>
                    <label className="block text-xs text-gray-400 mb-1">邀请码</label>
                    <input
                      type="text"
                      value={inviteCode}
                      onChange={e => setInviteCode(e.target.value)}
                      placeholder="请输入邀请码"
                      className="w-full rounded-lg border border-white/10 bg-black/30 px-3 py-2 text-xs text-gray-200 outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                    />
                  </div> */}

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
                      <option value="gemini-2.5-flash-image-preview">gemini-2.5-flash-image-preview</option>
                      <option value="gemini-3-pro-image-preview">gemini-3-pro-image-preview</option>
                    </select>
                    {llmApiUrl === 'http://123.129.219.111:3000/v1' && (
                       <p className="text-[10px] text-gray-500 mt-1">{t('advanced.modelOnlyHint')}</p>
                    )}
                  </div>

                  {graphType === 'model_arch' ? (
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
                    {pptPath && (
                      <>
                        <button
                          type="button"
                          onClick={() => {
                            if (!pptPath) return;
                            window.open(pptPath, '_blank');
                          }}
                          className="w-full inline-flex items-center justify-center gap-2 rounded-lg border border-emerald-400/60 text-emerald-300 text-xs py-2 bg-emerald-500/10 hover:bg-emerald-500/20 transition-colors"
                        >
                          <CheckCircle2 size={14} />
                          <span>{t('techRoute.pptDownload', { filename: pptPath.split('/').pop() })}</span>
                        </button>

                        <div className="text-[11px] text-gray-300 bg-black/30 border border-white/10 rounded-md px-2 py-1.5">
                          <div>{t('techRoute.pptCopyHintTitle')}</div>
                          <div className="mt-1 break-all text-primary-200 underline decoration-dotted">
                            {pptPath}
                          </div>
                        </div>
                      </>
                    )}

                    {svgPath && (
                      <button
                        type="button"
                        onClick={() => {
                          if (!svgPath) return;
                          window.open(svgPath, '_blank');
                        }}
                        className="w-full inline-flex items-center justify-center gap-2 rounded-lg border border-sky-400/60 text-sky-300 text-xs py-2 bg-sky-500/10 hover:bg-sky-500/20 transition-colors"
                      >
                        <ImageIcon size={14} />
                        <span>{t('techRoute.svgDownload', { filename: svgPath.split('/').pop() })}</span>
                      </button>
                    )}

                    {svgPreviewPath && (
                      <div className="rounded-lg border border-white/10 bg-black/30 p-2">
                        <p className="text-[11px] text-gray-300 mb-1">{t('techRoute.svgPreviewTitle')}</p>
                        <div className="w-full max-h-64 overflow-auto bg-black/60 rounded-md flex items-center justify-center">
                          <img
                            src={svgPreviewPath}
                            alt="技术路线图预览"
                            className="max-w-full h-auto object-contain"
                          />
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* 新增：邀请码历史任务输出文件列表（所有 graphType 通用） */}
                {/* {showOutputPanel && (
                  <div className="mt-3 glass rounded-lg border border-white/10 p-3 text-xs text-gray-200">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium">邀请码所有任务输出文件列表</span>
                      {isLoading && (
                        <span className="flex items-center gap-1 text-primary-200">
                          <Loader2 size={12} className="animate-spin" />
                          正在生成中...
                        </span>
                      )}
                    </div>

                    {allOutputFiles.length === 0 ? (
                      <p className="text-[11px] text-gray-400">
                        任务正在执行或尚未产生可下载文件，请稍候。生成完成后，这里会显示本次任务下的 PPTX / PNG / SVG 文件。
                      </p>
                    ) : (
                      <ul className="space-y-1 max-h-60 overflow-auto">
                        {allOutputFiles.map((url: string, idx: number) => {
                          const name = url.split('/').pop() || `文件${idx + 1}`;
                          const ext = name.split('.').pop()?.toLowerCase() || '';
                          let icon: JSX.Element | null = null;
                          if (ext === 'pptx') icon = <FileText size={12} />;
                          else if (['png', 'jpg', 'jpeg', 'bmp', 'gif', 'webp', 'tiff', 'svg'].includes(ext)) {
                            icon = <ImageIcon size={12} />;

                            }

                          return (
                            <li key={url} className="flex items-center justify-between gap-2">
                              <button
                                type="button"
                                onClick={() => window.open(url, '_blank')}
                                className="flex-1 inline-flex items-center gap-2 text-left text-primary-200 hover:text-primary-100 hover:underline"
                              >
                                {icon}
                                <span className="truncate">{name}</span>
                              </button>
                              <button
                                type="button"
                                onClick={() => window.open(url, '_blank')}
                                className="px-2 py-1 rounded border border-primary-400/60 text-[11px] text-primary-200 hover:bg-primary-500/10"
                              >
                                打开 / 下载
                              </button>
                            </li>
                          );
                        })}
                      </ul>
                    )}
                  </div>
                )} */}

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
          </div>

            {/* 示例区：留出图片占位位 */}
            <div className="space-y-4 mb-2">
            <div className="flex items-center justify-between flex-wrap gap-2">
              <div className="flex items-center gap-3">
                <h3 className="text-sm font-medium text-gray-200">{t('examples.sectionTitle')}</h3>
                <a
                  href="https://wcny4qa9krto.feishu.cn/wiki/VXKiwYndwiWAVmkFU6kcqsTenWh"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="group relative inline-flex items-center gap-2 px-3 py-1 rounded-full bg-black/50 border border-white/10 text-xs font-medium text-white overflow-hidden transition-all hover:border-white/30 hover:shadow-[0_0_15px_rgba(168,85,247,0.5)]"
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 via-purple-500/20 to-pink-500/20 opacity-0 group-hover:opacity-100 transition-opacity" />
                  <Sparkles size={12} className="text-yellow-300 animate-pulse" />
                  <span className="bg-gradient-to-r from-blue-300 via-purple-300 to-pink-300 bg-clip-text text-transparent group-hover:from-blue-200 group-hover:via-purple-200 group-hover:to-pink-200">
                    {t('examples.feishuLink')}
                  </span>
                </a>
              </div>
              <span className="text-[11px] text-gray-500">
                {t('examples.sectionSubtitle')}
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
              <DemoCard
                title={t('examples.cards.paperPdfToFigureTitle')}
                desc={t('examples.cards.paperPdfToFigureDesc')}
                inputImg="/p2f_paper_pdf_img.png"
                outputImg="/p2f_paper_pdf_img_2.png"
              />
              <DemoCard
                title={t('examples.cards.figureScreenshotToPptTitle')}
                desc={t('examples.cards.figureScreenshotToPptDesc')}
                inputImg="/p2f_paper_model_img.png"
                outputImg="/p2f_paper_modle_img_2.png"
              />
              <DemoCard
                title={t('examples.cards.abstractTextToPptTitle')}
                desc={t('examples.cards.abstractTextToPptDesc')}
                inputImg="/p2f_paper_content.png"
                outputImg="/p2f_paper_content_2.png"
              />
              <DemoCard
                title={t('examples.cards.pdfToTechRouteTitle')}
                desc={t('examples.cards.pdfToTechRouteDesc')}
                inputImg="/p2t_paper_img.png"
                outputImg="/p2t_paper_img_2.png"
              />
              <DemoCard
                title={t('examples.cards.textToTechRouteTitle')}
                desc={t('examples.cards.textToTechRouteDesc')}
                inputImg="/p2t_paper_text.png"
                outputImg="/p2t_paper_text_2.png"
              />
              <DemoCard
                title={t('examples.cards.pdfToExpDataTitle')}
                desc={t('examples.cards.pdfToExpDataDesc')}
                inputImg="/p2e_paper_1.png"
                outputImg="/p2e_paper_2.png"
              />
              <DemoCard
                title={t('examples.cards.tableTextToExpDataTitle')}
                desc={t('examples.cards.tableTextToExpDataDesc')}
                inputImg="/p2f_exp_content_1.png"
                outputImg="/p2f_exp_content_2.png"
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
        .animate-shimmer-fast {
          animation: shimmer 1.5s infinite;
        }
        @keyframes fade-in {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fade-in {
          animation: fade-in 0.5s ease-out;
        }
        .gradient-border {
          background: linear-gradient(135deg, rgba(168, 85, 247, 0.4) 0%, rgba(236, 72, 153, 0.4) 100%);
          padding: 2px;
          border-radius: 0.75rem;
        }
        .glass {
          background: rgba(255, 255, 255, 0.03);
          backdrop-filter: blur(10px);
        }
        .glow {
          box-shadow: 0 0 20px rgba(168, 85, 247, 0.3);
        }
        .demo-input-placeholder {
          min-height: 80px;
        }
        .demo-output-placeholder {
          min-height: 80px;
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
    <div className="glass rounded-lg border border-white/10 p-3 flex flex-col gap-2 hover:bg-white/5 transition-colors">
      <div className="flex gap-2">
        {/* 左侧：输入示例图片 */}
        <div className="flex-1 rounded-md bg-white/5 border border-dashed border-white/10 flex items-center justify-center demo-input-placeholder overflow-hidden">
          {inputImg ? (
            <img
              src={inputImg}
              alt="输入示例图"
              className="w-full h-full object-cover"
            />
          ) : (
            <span className="text-[10px] text-gray-400">输入示例图（待替换）</span>
          )}
        </div>
        {/* 右侧：输出 PPTX 示例图片 */}
        <div className="flex-1 rounded-md bg-primary-500/10 border border-dashed border-primary-300/40 flex items-center justify-center demo-output-placeholder overflow-hidden">
          {outputImg ? (
            <img
              src={outputImg}
              alt="PPTX 示例图"
              className="w-full h-full object-cover"
            />
          ) : (
            <span className="text-[10px] text-primary-200">PPTX 示例图（待替换）</span>
          )}
        </div>
      </div>
      <div>
        <p className="text-[13px] text-white font-medium mb-1">{title}</p>
        <p className="text-[11px] text-gray-400 leading-snug">{desc}</p>
      </div>
    </div>
  );
};

export default Paper2FigurePage;
