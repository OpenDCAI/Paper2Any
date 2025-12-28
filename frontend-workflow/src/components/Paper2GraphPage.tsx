import { useState, useEffect, ChangeEvent } from 'react';
import { FileText, UploadCloud, Type, Settings2, Download, Loader2, CheckCircle2, AlertCircle, Image as ImageIcon, ChevronDown, ChevronUp, Github, Star, X } from 'lucide-react';
import { uploadAndSaveFile } from '../services/fileService';
import { API_KEY } from '../config/api';
import { checkQuota, recordUsage, QuotaInfo } from '../services/quotaService';
import { useAuthStore } from '../stores/authStore';

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

const STORAGE_KEY = 'paper2figure_config_v1';

const Paper2FigurePage = () => {
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
          fetch('https://api.github.com/repos/OpenDCAI/DataFlow-Agent'),
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
        ? '今日配额已用完（10次/天），请明天再试'
        : '今日配额已用完（5次/天），登录后可获得更多配额');
      return;
    }

    if (!llmApiUrl.trim() || !apiKey.trim()) {
      setError('请先配置模型 API URL 和 API Key');
      return;
    }

    // 技术路线图 / 实验数据图 不支持 image 作为输入
    if ((graphType === 'tech_route' || graphType === 'exp_data') && uploadMode === 'image') {
      setError('技术路线图和实验数据图仅支持 PDF 或文本输入，不支持图片');
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

    if (graphType === 'model_arch') {
      // 模型结构图：使用绘图难度，不再传语言
      formData.append('figure_complex', figureComplex);
    } else {
      // 其他图：使用语言配置，不传绘图难度
      formData.append('language', language);
    }

    if (uploadMode === 'file' || uploadMode === 'image') {
      if (!selectedFile) {
        setError('请先选择要上传的文件或图片');
        return;
      }
      const kind = fileKind ?? detectFileKind(selectedFile);
      if (!kind) {
        setError('仅支持 PDF 和常见图片格式，请检查文件类型');
        return;
      }
      formData.append('file', selectedFile);
      formData.append('file_kind', kind);
    } else if (uploadMode === 'text') {
      if (!textContent.trim()) {
        setError('请输入要转换为 PPTX 的文本内容');
        return;
      }
      formData.append('text', textContent.trim());
    }

    try {
      setIsLoading(true);

      if (graphType === 'tech_route') {
        // 技术路线图：调用 JSON 接口，返回 PPT + SVG
        const res = await fetch(JSON_API, {
          method: 'POST',
          headers: { 'X-API-Key': API_KEY },
          body: formData,
        });

        if (!res.ok) {
          let msg = '生成技术路线图失败';
          if (res.status === 403) {
            msg = '邀请码不正确或已失效';
          } else {
            try {
              const text = await res.text();
              if (text) msg = text;
            } catch {
              // ignore
            }
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
          throw new Error('生成技术路线图失败');
        }

        setPptPath(data.ppt_filename);
        setSvgPath(data.svg_filename);
        setSvgPreviewPath(data.svg_image_filename);
        setAllOutputFiles(data.all_output_files ?? []);
        setSuccessMessage('技术路线图已生成，可下载 PPT / SVG 或直接预览 PNG');

        // Record usage
        await recordUsage(user?.id || null, 'paper2figure');
        refreshQuota();

        // Fetch PPT file and upload to Supabase Storage
        if (data.ppt_filename) {
          try {
            const pptRes = await fetch(data.ppt_filename);
            if (pptRes.ok) {
              const pptBlob = await pptRes.blob();
              const pptName = data.ppt_filename.split('/').pop() || 'tech_route.pptx';
              uploadAndSaveFile(pptBlob, pptName, 'paper2figure');
            }
          } catch (e) {
            console.warn('[Paper2GraphPage] Failed to upload tech_route file:', e);
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
          let msg = '生成 PPTX 失败';
          if (res.status === 403) {
            msg = '邀请码不正确或已失效';
          } else {
            try {
              const text = await res.text();
              if (text) msg = text;
            } catch {
              // ignore
            }
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
        setSuccessMessage('PPTX 已生成，正在下载...');

        // Record usage and save file to Supabase Storage
        await recordUsage(user?.id || null, 'paper2figure');
        refreshQuota();
        uploadAndSaveFile(blob, filename, 'paper2figure');

        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        a.remove();
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : '生成 PPTX 失败';
      setError(message);
    } finally {
      setIsLoading(false);
    }
  };

  const showFileHint = () => {
    if (!selectedFile) return '支持 PDF、PNG、JPG 等格式';
    if (fileKind === 'pdf') return `已选择 PDF：${selectedFile.name}`;
    if (fileKind === 'image') return `已选择图片：${selectedFile.name}`;
    return `文件类型暂不识别：${selectedFile.name}`;
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
                href="https://github.com/OpenDCAI/DataFlow-Agent"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-1.5 bg-white/95 hover:bg-white text-gray-900 rounded-full text-xs font-semibold transition-all hover:scale-105 shadow-lg"
              >
                <Github size={14} />
                <span>DataFlow-Agent</span>
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
              PAPER → EDITABLE PPTX
            </p>
            <h1 className="text-3xl font-semibold text-white mb-2">
              一键根据论文内容绘制（可编辑）科研绘图
            </h1>
            <p className="text-sm text-gray-400 max-w-2xl mx-auto">
              上传论文 PDF / 图片，或直接粘贴文字，一键生成可编辑的 科研绘图PPTX，方便你继续修改、增删和排版。
            </p>
          </div>

          {/* 上半区：上传区 + 高级配置 */}
          <div className="grid grid-cols-1 lg:grid-cols-[2fr,minmax(260px,1fr)] gap-6 mb-10">
            {/* 上传卡片 */}
            <div className="gradient-border">
              <div className="relative rounded-xl bg-white/95 text-gray-900 p-6 lg:p-8 overflow-hidden">
                <div className="absolute -right-10 -top-10 w-40 h-40 bg-primary-100 rounded-full opacity-60 blur-3xl pointer-events-none" />
                <div className="relative">
                  <p className="text-xs font-medium text-primary-600 mb-2">选择你的输入方式</p>
                  <h2 className="text-xl font-semibold mb-1">从 Paper 出发，生成 PPTX</h2>
                  <p className="text-xs text-gray-500 mb-4">
                    支持上传 PDF / 图片，或直接粘贴文字内容，我们会帮你生成结构清晰、可编辑的 PPTX。
                  </p>

                  {/* 绘图类型选择 */}
                  <div className="mb-4">
                    <label className="block text-xs font-medium text-gray-600 mb-1">绘图类型</label>
                    <select
                      value={graphType}
                      onChange={e => setGraphType(e.target.value as GraphType)}
                      className="w-full rounded-lg border border-gray-300 bg-white/80 px-3 py-2 text-xs outline-none focus:ring-2 focus:ring-primary-400 focus:border-primary-400"
                    >
                      <option value="model_arch">模型架构图</option>
                      <option value="tech_route">技术路线图</option>
                      <option value="exp_data">实验数据图</option>
                    </select>
                  </div>

                  {/* 上传模式 Tab */}
                  <div className="inline-flex items-center rounded-full bg-gray-100 p-1 text-xs mb-5">
                    <button
                      type="button"
                      onClick={() => setUploadMode('file')}
                      className={`flex items-center gap-1 px-3 py-1.5 rounded-full ${
                        uploadMode === 'file'
                          ? 'bg-white shadow text-gray-900'
                          : 'text-gray-500 hover:text-gray-800'
                      }`}
                    >
                      <UploadCloud size={14} />
                      文件（PDF）
                    </button>
                    <button
                      type="button"
                      onClick={() => setUploadMode('text')}
                      className={`flex items-center gap-1 px-3 py-1.5 rounded-full ${
                        uploadMode === 'text'
                          ? 'bg-white shadow text-gray-900'
                          : 'text-gray-500 hover:text-gray-800'
                      }`}
                    >
                      <Type size={14} />
                      文本
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        if (graphType === 'tech_route' || graphType === 'exp_data') {
                          // 不允许切换为 image，给出友好提示
                          setError('技术路线图和实验数据图仅支持 PDF 或文本输入，不支持图片');
                          return;
                        }
                        setUploadMode('image');
                      }}
                      className={`flex items-center gap-1 px-3 py-1.5 rounded-full ${
                        uploadMode === 'image'
                          ? 'bg-white shadow text-gray-900'
                          : 'text-gray-500 hover:text-gray-800'
                      } ${graphType === 'tech_route' || graphType === 'exp_data' ? 'opacity-40 cursor-not-allowed' : ''}`}
                    >
                      <ImageIcon size={14} />
                      图片
                    </button>
                  </div>

                  {/* 不同模式内容区域 */}
                  {(uploadMode === 'file' || uploadMode === 'image') && (
                    <div
                      className={`border border-dashed rounded-xl p-5 flex flex-col items-center justify-center text-center gap-3 bg-white/60 transition-colors ${
                        isDragOver ? 'border-primary-500 bg-primary-50' : 'border-gray-300'
                      }`}
                      onDragOver={handleDragOver}
                      onDragLeave={handleDragLeave}
                      onDrop={handleDrop}
                    >
                      <div className="flex items-center justify-center gap-2 text-gray-600 text-sm">
                        <FileText size={20} />
                        <span className="font-medium">
                          拖拽 {uploadMode === 'file' ? 'PDF' : '图片'} 到此处，或点击选择文件
                        </span>
                      </div>
                      <label className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary-500 text-white text-xs font-medium cursor-pointer hover:bg-primary-600 transition-colors">
                        选择文件
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
                      <p className="text-[11px] text-gray-500">
                        {showFileHint()}，单个文件建议小于 20MB。
                      </p>
                    </div>
                  )}

                  {uploadMode === 'text' && (
                    <div className="space-y-3">
                      <label className="block text-xs font-medium text-gray-600">
                        粘贴论文摘要、章节内容或任意需要做成 PPT 的文字
                      </label>
                      <textarea
                        value={textContent}
                        onChange={e => setTextContent(e.target.value)}
                        rows={8}
                        placeholder="在这里粘贴论文的摘要、章节内容，或任意需要转换为 PPTX 的文本（支持中英文）..."
                        className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-primary-400 focus:border-primary-400 bg-white/80 resize-none"
                      />
                      <p className="text-[11px] text-gray-500">
                        建议控制在 5,000 字以内，过长内容可以分段多次生成 PPTX。
                      </p>
                    </div>
                  )}
                </div>
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
                  <span className="text-white font-medium">模型配置（高级设置）</span>
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
                    <label className="block text-xs text-gray-400 mb-1">模型 API URL</label>
                    <input
                      type="text"
                      value={llmApiUrl}
                      onChange={e => setLlmApiUrl(e.target.value)}
                      placeholder="例如：https://api.apiyi.com/v1"
                      className="w-full rounded-lg border border-white/10 bg-black/30 px-3 py-2 text-xs text-gray-200 outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                    />
                  </div>

                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <label className="block text-xs text-gray-400">
                        API Key
                      </label>
                      <a
                        href="https://api.apiyi.com/"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-[10px] text-primary-300 hover:text-primary-200 hover:underline"
                      >
                        点击购买
                      </a>
                    </div>
                    <input
                      type="password"
                      value={apiKey}
                      onChange={e => setApiKey(e.target.value)}
                      placeholder="用于调用 OpenAI / 兼容模型的 API Key"
                      className="w-full rounded-lg border border-white/10 bg-black/30 px-3 py-2 text-xs text-gray-200 outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                    />
                  </div>

                  <div>
                    <label className="block text-xs text-gray-400 mb-1">模型选择</label>
                    <select
                      value={model}
                      onChange={e => setModel(e.target.value)}
                      className="w-full rounded-lg border border-white/10 bg-black/30 px-3 py-2 text-xs text-gray-200 outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                    >
                      <option value="gemini-2.5-flash-image-preview">gemini-2.5-flash-image-preview</option>
                      <option value="gemini-3-pro-image-preview">gemini-3-pro-image-preview</option>
                    </select>
                  </div>

                  {graphType === 'model_arch' ? (
                    <div>
                      <label className="block text-xs text-gray-400 mb-1">绘图难度</label>
                      <select
                        value={figureComplex}
                        onChange={e => setFigureComplex(e.target.value as FigureComplex)}
                        className="w-full rounded-lg border border-white/10 bg-black/30 px-3 py-2 text-xs text-gray-200 outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                      >
                        <option value="easy">简单</option>
                        <option value="mid">中等</option>
                        <option value="hard">复杂</option>
                      </select>
                    </div>
                  ) : (
                    <div>
                      <label className="block text-xs text-gray-400 mb-1">语言</label>
                      <select
                        value={language}
                        onChange={e => setLanguage(e.target.value as Language)}
                        className="w-full rounded-lg border border-white/10 bg-black/30 px-3 py-2 text-xs text-gray-200 outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                      >
                        <option value="zh">中文</option>
                        <option value="en">英文</option>
                      </select>
                    </div>
                  )}

                  <div>
                    <label className="block text-xs text-gray-400 mb-1">风格</label>
                    <select
                      value={style}
                      onChange={e => setStyle(e.target.value as StyleType)}
                      className="w-full rounded-lg border border-white/10 bg-black/30 px-3 py-2 text-xs text-gray-200 outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                    >
                      <option value="cartoon">卡通</option>
                      {graphType !== 'exp_data' && <option value="realistic">写实</option>}
                      {graphType === 'exp_data' && <option value="Low Poly 3D">低多边形</option>}
                      {graphType === 'exp_data' && <option value="blocky LEGO aesthetic">乐高风</option>}
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
                  <span>{isLoading ? '生成中...' : '生成可编辑 PPTX'}</span>
                </button>

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
                          分析论文内容
                        </span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <div className={`w-1.5 h-1.5 rounded-full ${currentStage >= 1 ? 'bg-primary-400 animate-pulse' : 'bg-primary-950/60'}`} />
                        <span className={currentStage >= 1 ? 'text-primary-200 font-medium' : ''}>
                          生成科研绘图
                        </span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <div className={`w-1.5 h-1.5 rounded-full ${currentStage >= 2 ? 'bg-primary-400 animate-pulse' : 'bg-primary-950/60'}`} />
                        <span className={currentStage >= 2 ? 'text-primary-200 font-medium' : ''}>
                          转为可编辑绘图
                        </span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <div className={`w-1.5 h-1.5 rounded-full ${currentStage >= 3 ? 'bg-primary-400 animate-pulse' : 'bg-primary-950/60'}`} />
                        <span className={currentStage >= 3 ? 'text-primary-200 font-medium' : ''}>
                          合成 PPT
                        </span>
                      </div>
                    </div>

                    <p className="text-[11px] text-primary-200/70 pt-1 border-t border-primary-400/20">
                      预计需要 2-5 分钟，请耐心等待...
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
                    <span>重新下载：{lastFilename}</span>
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
                          <span>下载技术路线图 PPT：{pptPath.split('/').pop()}</span>
                        </button>

                        <div className="text-[11px] text-gray-300 bg-black/30 border border-white/10 rounded-md px-2 py-1.5">
                          <div>如果下载失败，请复制下面链接到浏览器地址栏打开：</div>
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
                        <span>下载 SVG 源文件：{svgPath.split('/').pop()}</span>
                      </button>
                    )}

                    {svgPreviewPath && (
                      <div className="rounded-lg border border-white/10 bg-black/30 p-2">
                        <p className="text-[11px] text-gray-300 mb-1">SVG 预览（PNG 渲染图）</p>
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
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-medium text-gray-200">示例：从 Paper 到 PPTX</h3>
              <span className="text-[11px] text-gray-500">
                下方示例展示从 PDF / 图片 / 文本 到可编辑 PPTX 的效果。
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
              <DemoCard
                title="论文 PDF → 符合论文主题的 科研绘图（PPT）"
                desc="上传英文论文 PDF，自动提炼研究背景、方法、实验设计和结论，生成结构清晰、符合学术风格的汇报 PPTX。"
                inputImg="/p2f_paper_pdf_img.png"
                outputImg="/p2f_paper_pdf_img_2.png"
              />
              <DemoCard
                title="科研配图 / 示意图截图 → 可编辑 PPTX"
                desc="上传科研配图或示意图截图，自动识别段落层级与要点，自动排版为可编辑的英文 PPTX。"
                inputImg="/p2f_paper_model_img.png"
                outputImg="/p2f_paper_modle_img_2.png"
              />
              <DemoCard
                title="论文摘要文本 → 科研绘图 PPTX"
                desc="粘贴论文摘要或章节内容，一键生成包含标题层级、关键要点与图示占位的 PPTX 大纲，方便后续细化与美化。"
                inputImg="/p2f_paper_content.png"
                outputImg="/p2f_paper_content_2.png"
              />
              <DemoCard
                title="论文 PDF → 符合论文主题的 技术路线图 PPT + SVG"
                desc="根据论文方法部分，自动梳理技术路线与模块依赖关系，生成清晰的技术路线图 PPTX 与 SVG 示意图。"
                inputImg="/p2t_paper_img.png"
                outputImg="/p2t_paper_img_2.png"
              />
              <DemoCard
                title="论文摘要文本 → 符合论文主题的 技术路线图 PPT + SVG"
                desc="从整篇技术方案 PDF 中提取关键步骤与时间轴，自动生成技术路线时间线 PPTX 与 SVG。"
                inputImg="/p2t_paper_text.png"
                outputImg="/p2t_paper_text_2.png"
              />
              <DemoCard
                title="论文 PDF → 自动提取实验数据 绘制成 PPT"
                desc="从论文实验部分 PDF 中提取表格与结果描述，自动生成对比柱状图 / 折线图 PPTX，便于直观展示结果。"
                inputImg="/p2e_paper_1.png"
                outputImg="/p2e_paper_2.png"
              />
              <DemoCard
                title="论文实验表格文本 → 自动整理实验数据 绘制成 PPT"
                desc="从文本形式的实验结果描述中抽取指标与对照组，一键生成适合汇报的实验结果 PPTX。"
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
