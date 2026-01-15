import React, { useState, useEffect, ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';
import { useAuthStore } from '../../stores/authStore';
import { uploadAndSaveFile } from '../../services/fileService';
import { API_KEY } from '../../config/api';
import { checkQuota, recordUsage } from '../../services/quotaService';
import { verifyLlmConnection } from '../../services/llmService';

import {
  UploadMode,
  FileKind,
  GraphType,
  Language,
  StyleType,
  FigureComplex,
} from './types';
import {
  BACKEND_API,
  JSON_API,
  HISTORY_API,
  IMAGE_EXTENSIONS,
  GENERATION_STAGES,
  MAX_FILE_SIZE,
  STORAGE_KEY,
} from './constants';

import Banner from './Banner';
import Header from './Header';
import UploadCard from './UploadCard';
import SettingsCard from './SettingsCard';
import PreviewSection from './PreviewSection';
import ExamplesSection from './ExamplesSection';

function detectFileKind(file: File): FileKind {
  const ext = file.name.split('.').pop()?.toLowerCase();
  if (!ext) return null;
  if (ext === 'pdf') return 'pdf';
  if (IMAGE_EXTENSIONS.includes(ext)) return 'image';
  return null;
}

const Paper2FigurePage = () => {
  const { t, i18n } = useTranslation('paper2graph');
  const { user, refreshQuota } = useAuthStore();
  
  // State from original file
  const [uploadMode, setUploadMode] = useState<UploadMode>('file');
  const [graphStep, setGraphStep] = useState<'input' | 'preview' | 'done'>('input');
  const [previewImgUrl, setPreviewImgUrl] = useState<string | null>(null);
  const [pptUrl, setPptUrl] = useState<string | null>(null);
  const [editPrompt, setEditPrompt] = useState('');
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
  const [model, setModel] = useState('gemini-3-pro-image-preview');
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
  // const [showOutputPanel, setShowOutputPanel] = useState(false);

  // GitHub Stars
  const [stars, setStars] = useState<{dataflow: number | null, agent: number | null, dataflex: number | null}>({
    dataflow: null,
    agent: null,
    dataflex: null,
  });

  // 新增：生成阶段状态
  const [currentStage, setCurrentStage] = useState(0);
  const [stageProgress, setStageProgress] = useState(0);

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
  // const fetchHistoryFiles = async (code: string) => {
  //   const invite = code.trim();
  //   if (!invite) return;
  //   try {
  //     const res = await fetch(
  //       `${HISTORY_API}?invite_code=${encodeURIComponent(invite)}`
  //     );
  //     if (!res.ok) return;
  //     const data = await res.json();
  //     const urls: string[] = (data.files || []).map((f: any) =>
  //       typeof f === 'string' ? f : f.url,
  //     );
  //     setAllOutputFiles(urls);
  //   } catch (e) {
  //     console.error('fetch history files error', e);
  //   }
  // };

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
    // 当前 UploadMode 仅支持 'file' | 'text'，此分支保留作为防御性检查

    if (graphType === 'model_arch') {
      // model_arch 统一走 JSON API，先生成图和 PPT，再由前端决定交互
      if (isLoading) return;
      setError(null);
      setSuccessMessage(null);
      setDownloadUrl(null);
      setPptPath(null);
      setSvgPath(null);
      setSvgPreviewPath(null);
      setCurrentStage(0);
      setStageProgress(0);
      // setShowOutputPanel(true);

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

      const formData = new FormData();
      formData.append('img_gen_model_name', model);
      formData.append('chat_api_url', llmApiUrl.trim());
      formData.append('api_key', apiKey.trim());
      formData.append('input_type', uploadMode);
      formData.append('invite_code', inviteCode.trim());
      formData.append('graph_type', graphType);
      formData.append('style', style);
      formData.append('figure_complex', figureComplex);
      formData.append('language', language);

      if (uploadMode === 'file') {
        if (!selectedFile) {
          setError(t('errors.noFile'));
          return;
        }
        const kind: FileKind = 'pdf';
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
        setIsValidating(true);
        setError(null);
        await verifyLlmConnection(llmApiUrl, apiKey, model);
        setIsValidating(false);

        setIsLoading(true);
        const res = await fetch(JSON_API, {
          method: 'POST',
          headers: { 'X-API-Key': API_KEY },
          body: formData,
        });

        if (!res.ok) {
          let msg = t('errors.serverBusy');
          if (res.status === 403) msg = t('errors.inviteInvalid');
          else if (res.status === 429) msg = t('errors.tooManyRequests');
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

        setAllOutputFiles(data.all_output_files ?? []);
        setSuccessMessage(t('success.previewGenerated', '模型结构图预览已生成，请确认并转为 PPT'));
        await recordUsage(user?.id || null, 'paper2figure');
        refreshQuota();

        // 选一张主图做预览：优先 fig_*.png，其次最大 png
        let mainImg: string | null = null;
        const files = data.all_output_files ?? [];
        const pngs = files.filter(f => /\.(png|jpg|jpeg|webp)$/i.test(f));
        const figPngs = pngs.filter(f => /fig_/i.test(f));
        if (figPngs.length > 0) {
          mainImg = figPngs[0];
        } else if (pngs.length > 0) {
          mainImg = pngs[0];
        }

        let pptUrlCandidate: string | null = null;
        if (data.ppt_filename) {
          pptUrlCandidate = data.ppt_filename;
        } else {
          const pptx = files.find(f => /\.pptx$/i.test(f));
          if (pptx) pptUrlCandidate = pptx;
        }

        setPreviewImgUrl(mainImg);
        // Step 1 结束，暂不设置 pptUrl，因为 PPT 还没生成
        setPptUrl(null); 
        setGraphStep('preview');
      } catch (err) {
        const message = err instanceof Error ? err.message : t('errors.serverBusy');
        setError(message);
      } finally {
        setIsLoading(false);
        setIsValidating(false);
      }
      return;
    }

    if (isLoading) return;
    setError(null);
    setSuccessMessage(null);
    setDownloadUrl(null);
    setPptPath(null);
    setSvgPath(null);
    setSvgPreviewPath(null);
    setCurrentStage(0);
    setStageProgress(0);
    // setShowOutputPanel(true);

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

    // 当前 UploadMode 仅支持 'file' | 'text'，无需图片输入
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

    // 其他图（tech_route / exp_data）：使用语言配置，不传绘图难度
    formData.append('language', backendLanguage);

    if (uploadMode === 'file') {
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

  return (
    <div className="w-full h-full flex flex-col bg-[#050512]">
      <Banner show={showBanner} onClose={() => setShowBanner(false)} stars={stars} />

      <div className="flex-1 flex flex-col items-center justify-start px-6 pt-20 pb-10 overflow-auto">
        <div className="w-full max-w-5xl animate-fade-in">
          <Header />

          <div className="grid grid-cols-1 lg:grid-cols-[2fr,minmax(260px,1fr)] gap-6 mb-10">
            <UploadCard
              graphType={graphType}
              setGraphType={setGraphType}
              uploadMode={uploadMode}
              setUploadMode={setUploadMode}
              selectedFile={selectedFile}
              fileKind={fileKind}
              isDragOver={isDragOver}
              handleDragOver={handleDragOver}
              handleDragLeave={handleDragLeave}
              handleDrop={handleDrop}
              handleFileChange={handleFileChange}
              textContent={textContent}
              setTextContent={setTextContent}
            />

            <SettingsCard
              showAdvanced={showAdvanced}
              setShowAdvanced={setShowAdvanced}
              llmApiUrl={llmApiUrl}
              setLlmApiUrl={setLlmApiUrl}
              setModel={setModel}
              apiKey={apiKey}
              setApiKey={setApiKey}
              model={model}
              graphType={graphType}
              figureComplex={figureComplex}
              setFigureComplex={setFigureComplex}
              language={language}
              setLanguage={setLanguage}
              style={style}
              setStyle={setStyle}
              isLoading={isLoading}
              handleSubmit={handleSubmit}
              currentStage={currentStage}
              stageProgress={stageProgress}
              downloadUrl={downloadUrl}
              lastFilename={lastFilename}
              pptPath={pptPath}
              svgPath={svgPath}
              svgPreviewPath={svgPreviewPath}
              isValidating={isValidating}
              error={error}
              successMessage={successMessage}
            />
          </div>

          <PreviewSection
            graphType={graphType}
            graphStep={graphStep}
            previewImgUrl={previewImgUrl}
            setPreviewImgUrl={setPreviewImgUrl}
            pptUrl={pptUrl}
            setPptUrl={setPptUrl}
            setGraphStep={setGraphStep}
            editPrompt={editPrompt}
            setEditPrompt={setEditPrompt}
            isLoading={isLoading}
            setIsLoading={setIsLoading}
            setError={setError}
            model={model}
            llmApiUrl={llmApiUrl}
            apiKey={apiKey}
            inviteCode={inviteCode}
            figureComplex={figureComplex}
            language={language}
          />

          <ExamplesSection />
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

export default Paper2FigurePage;
