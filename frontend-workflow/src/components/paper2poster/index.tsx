import React, { useState, useEffect, ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';
import { API_KEY } from '../../config/api';
import { checkQuota, recordUsage } from '../../services/quotaService';
import { verifyLlmConnection } from '../../services/llmService';
import { useAuthStore } from '../../stores/authStore';
import { getApiSettings, saveApiSettings } from '../../services/apiSettingsService';
import { uploadAndSaveFile } from '../../services/fileService';

import { Step, PosterConfig, GenerateResult } from './types';
import { MAX_FILE_SIZE, STORAGE_KEY, DEFAULT_CONFIG } from './constants';

import Banner from './Banner';
import StepIndicator from './StepIndicator';
import UploadStep from './UploadStep';
import GenerateStep from './GenerateStep';
import CompleteStep from './CompleteStep';

const Paper2PosterPage = () => {
  const { user, refreshQuota } = useAuthStore();

  // Step 状态
  const [currentStep, setCurrentStep] = useState<Step>('upload');

  // 文件状态
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [logoFile, setLogoFile] = useState<File | null>(null);
  const [affLogoFile, setAffLogoFile] = useState<File | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);

  // 配置状态
  const [config, setConfig] = useState<PosterConfig>(DEFAULT_CONFIG);

  // API 配置状态
  const [llmApiUrl, setLlmApiUrl] = useState(import.meta.env.VITE_DEFAULT_LLM_API_URL || 'https://api.apiyi.com/v1');
  const [apiKey, setApiKey] = useState('');

  // 生成状态
  const [isUploading, setIsUploading] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressStatus, setProgressStatus] = useState('');
  const [error, setError] = useState<string | null>(null);

  // 结果状态
  const [result, setResult] = useState<GenerateResult>({
    status: 'pending',
  });

  // GitHub Stars
  const [stars, setStars] = useState<{dataflow: number | null, agent: number | null, dataflex: number | null}>({
    dataflow: null,
    agent: null,
    dataflex: null,
  });
  const [copySuccess, setCopySuccess] = useState('');
  const [showBanner, setShowBanner] = useState(true);

  const shareText = `发现一个超好用的AI工具 DataFlow-Agent！🚀
支持论文转PPT、PDF转PPT、论文转海报等功能，科研打工人的福音！

🔗 在线体验：https://dcai-paper2any.nas.cpolar.cn/
⭐ GitHub Agent：https://github.com/OpenDCAI/Paper2Any
🌟 GitHub Core：https://github.com/OpenDCAI/DataFlow

转发本文案+截图，联系微信群管理员即可获取免费Key！🎁
#AI工具 #学术海报 #科研效率 #开源项目`;

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
        const saved = JSON.parse(raw);
        if (saved.config) setConfig(saved.config);

        // API settings: prioritize user-specific settings
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
      console.error('Failed to restore paper2poster config', e);
    }
  }, [user?.id]);

  // 将配置写入 localStorage
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const data = {
      config,
      llmApiUrl,
      apiKey,
    };
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
      if (user?.id && llmApiUrl && apiKey) {
        saveApiSettings(user.id, { apiUrl: llmApiUrl, apiKey });
      }
    } catch (e) {
      console.error('Failed to persist paper2poster config', e);
    }
  }, [config, llmApiUrl, apiKey, user?.id]);

  // 文件处理函数
  const validatePdfFile = (file: File): boolean => {
    const ext = file.name.split('.').pop()?.toLowerCase();
    if (ext !== 'pdf') {
      setError('仅支持 PDF 格式');
      return false;
    }
    return true;
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !validatePdfFile(file)) return;
    if (file.size > MAX_FILE_SIZE) {
      setError('文件大小超过 50MB 限制');
      return;
    }
    setSelectedFile(file);
    setError(null);
  };

  const handleLogoChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setLogoFile(file);
  };

  const handleAffLogoChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setAffLogoFile(file);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (!file || !validatePdfFile(file)) return;
    if (file.size > MAX_FILE_SIZE) {
      setError('文件大小超过 50MB 限制');
      return;
    }
    setSelectedFile(file);
    setError(null);
  };

  // 上传并生成海报
  const handleUploadAndGenerate = async () => {
    if (!selectedFile) {
      setError('请先选择 PDF 文件');
      return;
    }
    if (!apiKey.trim()) {
      setError('请输入 API Key');
      return;
    }

    // Check quota
    const quota = await checkQuota(user?.id || null, user?.is_anonymous || false);
    if (quota.remaining <= 0) {
      setError(quota.isAuthenticated
        ? '今日配额已用完（10次/天），请明天再试'
        : '今日配额已用完（5次/天），登录后可获得更多配额');
      return;
    }

    try {
      // Verify LLM Connection
      setIsValidating(true);
      setError(null);
      await verifyLlmConnection(llmApiUrl, apiKey, 'gpt-4o');
      setIsValidating(false);
    } catch (err) {
      setIsValidating(false);
      const message = err instanceof Error ? err.message : 'API 验证失败';
      setError(message);
      return;
    }

    setIsUploading(true);
    setError(null);
    setProgress(0);
    setProgressStatus('正在初始化...');
    setCurrentStep('generate');
    setResult({ status: 'processing', progress: 0 });

    // 模拟进度
    const progressInterval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 90) return 90;
        const messages = [
          '正在上传文件...',
          '正在解析PDF...',
          '正在提取内容...',
          '正在生成布局...',
          '正在渲染海报...'
        ];
        const msgIndex = Math.floor(prev / 20);
        if (msgIndex < messages.length) {
          setProgressStatus(messages[msgIndex]);
          setResult(r => ({ ...r, progress: prev }));
        }
        return prev + (Math.random() * 0.6 + 0.2);
      });
    }, 1000);

    try {
      const formData = new FormData();
      formData.append('paper_file', selectedFile);
      formData.append('chat_api_url', llmApiUrl.trim());
      formData.append('api_key', apiKey.trim());
      formData.append('model', config.text_model);
      formData.append('vision_model', config.vision_model);
      formData.append('poster_width', config.poster_width.toString());
      formData.append('poster_height', config.poster_height.toString());

      if (logoFile) {
        formData.append('logo_file', logoFile);
      }
      if (affLogoFile) {
        formData.append('aff_logo_file', affLogoFile);
      }

      const res = await fetch('/api/v1/paper2poster/generate', {
        method: 'POST',
        headers: { 'X-API-Key': API_KEY },
        body: formData,
      });

      if (!res.ok) {
        let detail = `HTTP error! status: ${res.status}`;
        try {
          const errData = await res.json();
          detail = errData?.detail || errData?.message || detail;
        } catch {
          // ignore json parse error and keep default message
        }
        throw new Error(detail);
      }

      const data = await res.json();

      clearInterval(progressInterval);
      setProgress(100);
      setProgressStatus('生成完成！');

      // 设置结果
      setResult({
        status: 'done',
        pptxUrl: data.pptx_url,
        pngUrl: data.png_url,
      });

      // Record usage
      await recordUsage(user?.id || null, 'paper2poster', { isAnonymous: user?.is_anonymous || false });
      refreshQuota();

      // Upload to storage
      if (data.pptx_url) {
        try {
          const fileRes = await fetch(data.pptx_url);
          if (fileRes.ok) {
            const fileBlob = await fileRes.blob();
            const fileName = data.pptx_url.split('/').pop() || 'poster.pptx';
            await uploadAndSaveFile(fileBlob, fileName, 'paper2poster');
          }
        } catch (e) {
          console.error('Failed to upload file:', e);
        }
      }

      setTimeout(() => {
        setCurrentStep('complete');
      }, 500);

    } catch (err) {
      clearInterval(progressInterval);
      setProgress(0);
      const message = err instanceof Error ? err.message : '生成失败，请稍后再试';
      setError(message);
      setResult({ status: 'pending' });
      setCurrentStep('upload');
    } finally {
      setIsUploading(false);
    }
  };

  // 重置
  const handleReset = () => {
    setCurrentStep('upload');
    setSelectedFile(null);
    setLogoFile(null);
    setAffLogoFile(null);
    setResult({ status: 'pending' });
    setError(null);
    setProgress(0);
    setProgressStatus('');
  };

  return (
    <div className="w-full h-screen flex flex-col bg-[#050512] overflow-hidden">
      <Banner show={showBanner} onClose={() => setShowBanner(false)} stars={stars} />

      <div className="flex-1 overflow-auto">
        <div className="max-w-7xl mx-auto px-6 py-8 pb-24">
          <StepIndicator currentStep={currentStep} />

          {currentStep === 'upload' && (
            <UploadStep
              selectedFile={selectedFile}
              logoFile={logoFile}
              affLogoFile={affLogoFile}
              isDragOver={isDragOver}
              setIsDragOver={setIsDragOver}
              config={config}
              setConfig={setConfig}
              isUploading={isUploading}
              isValidating={isValidating}
              progress={progress}
              progressStatus={progressStatus}
              error={error}
              llmApiUrl={llmApiUrl}
              setLlmApiUrl={setLlmApiUrl}
              apiKey={apiKey}
              setApiKey={setApiKey}
              handleFileChange={handleFileChange}
              handleLogoChange={handleLogoChange}
              handleAffLogoChange={handleAffLogoChange}
              handleDrop={handleDrop}
              handleUploadAndGenerate={handleUploadAndGenerate}
            />
          )}

          {currentStep === 'generate' && (
            <GenerateStep result={result} error={error} />
          )}

          {currentStep === 'complete' && (
            <CompleteStep
              result={result}
              handleReset={handleReset}
              handleCopyShareText={handleCopyShareText}
              copySuccess={copySuccess}
              stars={stars}
            />
          )}
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
      `}</style>
    </div>
  );
};

export default Paper2PosterPage;
