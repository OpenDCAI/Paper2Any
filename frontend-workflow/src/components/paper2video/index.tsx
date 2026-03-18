import React, { useState, useEffect, ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';
import { API_KEY } from '../../config/api';
import { useAuthStore } from '../../stores/authStore';
import { getApiSettings, saveApiSettings } from '../../services/apiSettingsService';
import { checkQuota, recordUsage } from '../../services/quotaService';
import { Step, ScriptPage } from './types';
import {
  MAX_FILE_SIZE,
  STORAGE_KEY,
  TTS_MODEL_DEFAULT,
  TALKING_MODEL_DEFAULT,
  VIDEO_GENERATION_COST,
} from './constants';
import Banner from '../paper2ppt/Banner';
import StepIndicator from './StepIndicator';
import UploadStep from './UploadStep';
import ScriptStep from './ScriptStep';
import CompleteStep from './CompleteStep';

// 开发时 Vite 会代理 /outputs 到后端，用相对路径即可；生产或跨域时需配置 VITE_API_BASE_URL
const BACKEND_ORIGIN = import.meta.env.VITE_API_BASE_URL || '';

function convertToHttpUrl(path: string): string {
  if (path.startsWith('http://') || path.startsWith('https://')) return path;
  const outputsIndex = path.indexOf('/outputs/');
  if (outputsIndex !== -1) {
    const relativePath = path.substring(outputsIndex);
    // 未配置 BACKEND_ORIGIN 时用相对路径，请求会发到当前页面同源（开发时由 Vite 代理到后端）
    if (!BACKEND_ORIGIN) return relativePath;
    return `${BACKEND_ORIGIN.replace(/\/$/, '')}${relativePath}`;
  }
  return path;
}

const EXAMPLE_BASE = '/paper2video/example';

const Paper2VideoPage = () => {
  const { user, refreshQuota } = useAuthStore();
  const { t } = useTranslation(['paper2video', 'common']);

  const [currentStep, setCurrentStep] = useState<Step>('upload');

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [useAvatar, setUseAvatar] = useState<'none' | 'yes'>('none');
  const [avatarFile, setAvatarFile] = useState<File | null>(null);
  const [avatarPreview, setAvatarPreview] = useState<string | null>(null);
  const [avatarPreset, setAvatarPreset] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressStatus, setProgressStatus] = useState('');

  const [scriptPages, setScriptPages] = useState<ScriptPage[]>([]);
  const [resultPath, setResultPath] = useState<string | null>(null);
  const [stateSnapshot, setStateSnapshot] = useState<string | null>(null);

  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [isGeneratingVideo, setIsGeneratingVideo] = useState(false);

  const [error, setError] = useState<string | null>(null);
  const [showBanner, setShowBanner] = useState(true);

  const [stars, setStars] = useState<{ dataflow: number | null; agent: number | null; dataflex: number | null }>({
    dataflow: null,
    agent: null,
    dataflex: null,
  });

  const [apiKey, setApiKey] = useState('');
  const [scriptApiUrl, setScriptApiUrl] = useState(
    import.meta.env.VITE_DEFAULT_LLM_API_URL || 'https://api.apiyi.com/v1'
  );
  const [scriptModel, setScriptModel] = useState('gpt-4o');
  const [ttsModel, setTtsModel] = useState<string>(TTS_MODEL_DEFAULT);
  const [ttsVoiceName, setTtsVoiceName] = useState<string>('longanyang');
  const [language, setLanguage] = useState<'zh' | 'en'>('zh');

  useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const saved = JSON.parse(raw);
        if (saved.apiKey) setApiKey(saved.apiKey);
        if (saved.scriptApiUrl) setScriptApiUrl(saved.scriptApiUrl);
        if (saved.scriptModel) setScriptModel(saved.scriptModel);
        if (saved.ttsModel) setTtsModel(saved.ttsModel);
        if (saved.ttsVoiceName) setTtsVoiceName(saved.ttsVoiceName);
        if (saved.language) setLanguage(saved.language);
      }
      const userApi = getApiSettings(user?.id || null);
      if (userApi) {
        if (userApi.apiUrl) setScriptApiUrl(userApi.apiUrl);
        if (userApi.apiKey) setApiKey(userApi.apiKey);
      }
    } catch (e) {
      console.error('Failed to restore paper2video config', e);
    }
  }, [user?.id]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const data = { apiKey, scriptApiUrl, scriptModel, ttsModel, ttsVoiceName, language };
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
      if (user?.id && scriptApiUrl && apiKey) {
        saveApiSettings(user.id, { apiUrl: scriptApiUrl, apiKey });
      }
    } catch (e) {
      console.error('Failed to persist paper2video config', e);
    }
  }, [apiKey, scriptApiUrl, scriptModel, ttsModel, ttsVoiceName, language, user?.id]);

  useEffect(() => {
    const fetchStars = async () => {
      try {
        const [res1, res2, res3] = await Promise.all([
          fetch('https://api.github.com/repos/OpenDCAI/DataFlow'),
          fetch('https://api.github.com/repos/OpenDCAI/Paper2Any'),
          fetch('https://api.github.com/repos/OpenDCAI/DataFlex'),
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

  const validateDocument = (file: File): boolean => {
    const ext = file.name.split('.').pop()?.toLowerCase();
    if (ext !== 'pdf' && ext !== 'pptx') {
      setError('仅支持 PDF 或 PPTX 格式');
      return false;
    }
    return true;
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !validateDocument(file)) return;
    if (file.size > MAX_FILE_SIZE) {
      setError('文件大小超过 50MB 限制');
      return;
    }
    setSelectedFile(file);
    setError(null);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (!file || !validateDocument(file)) return;
    if (file.size > MAX_FILE_SIZE) {
      setError('文件大小超过 50MB 限制');
      return;
    }
    setSelectedFile(file);
    setError(null);
  };

  const handleAvatarChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const ext = file.name.split('.').pop()?.toLowerCase();
    if (!['jpg', 'jpeg', 'png'].includes(ext || '')) {
      setError('头像仅支持 JPG/PNG 格式');
      return;
    }
    if (avatarPreview) URL.revokeObjectURL(avatarPreview);
    setAvatarPreset(null);
    setAvatarFile(file);
    setAvatarPreview(URL.createObjectURL(file));
    setError(null);
  };

  const handleSelectAvatarPreset = (presetId: string) => {
    if (avatarPreview) URL.revokeObjectURL(avatarPreview);
    setAvatarFile(null);
    setAvatarPreview(null);
    setAvatarPreset(presetId || null);
    setError(null);
  };

  const handleRemoveAvatar = () => {
    if (avatarPreview) URL.revokeObjectURL(avatarPreview);
    setAvatarFile(null);
    setAvatarPreview(null);
    setAvatarPreset(null);
  };

  const handleStartParse = async () => {
    if (!selectedFile) {
      setError('请先选择 PDF 或 PPTX 文件');
      return;
    }
    if (!apiKey.trim()) {
      setError('请输入 API Key');
      return;
    }

    setIsUploading(true);
    setError(null);
    setProgress(0);
    setProgressStatus('正在初始化...');

    const progressInterval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 90) return 90;
        const messages = ['正在解析内容...', '正在生成脚本...'];
        const msgIndex = Math.floor(prev / 30);
        if (msgIndex < messages.length) setProgressStatus(messages[msgIndex]);
        return prev + (Math.random() * 0.5 + 0.2);
      });
    }, 800);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('email', user?.id || user?.email || '');
      formData.append('api_key', apiKey.trim());
      formData.append('chat_api_url', scriptApiUrl.trim());
      formData.append('model', scriptModel);
      formData.append('tts_model', ttsModel);
      formData.append('tts_voice_name', ttsVoiceName.trim() || 'longanyang');
      formData.append('language', language);
      formData.append('talking_model', TALKING_MODEL_DEFAULT);
      if (useAvatar === 'yes') {
        if (avatarFile) formData.append('avatar', avatarFile);
        else if (avatarPreset) formData.append('avatar_preset', avatarPreset);
      }

      const res = await fetch('/api/v1/paper2video/generate-subtitle', {
        method: 'POST',
        headers: { 'X-API-Key': API_KEY },
        body: formData,
      });

      if (!res.ok) {
        let msg = '服务器繁忙，请稍后再试';
        if (res.status === 403) msg = '邀请码不正确或已失效';
        else if (res.status === 429) msg = '请求过于频繁，请稍后再试';
        throw new Error(msg);
      }

      const data = await res.json();
      if (!data.success) throw new Error(data.message || '解析失败');

      const path = data.result_path || '';
      if (!path) throw new Error('后端未返回 result_path');

      clearInterval(progressInterval);
      setProgress(100);
      setProgressStatus('解析完成！');

      const pages: ScriptPage[] = (data.script_pages || data.pages || []).map((item: { page_num?: number; image_url?: string; script_text?: string; scriptText?: string }, index: number) => ({
        pageNum: item.page_num ?? index + 1,
        imageUrl: item.image_url ? convertToHttpUrl(item.image_url) : '',
        scriptText: item.script_text ?? item.scriptText ?? '',
      }));

      if (pages.length === 0) {
        throw new Error('未返回脚本页面数据，请检查后端 generate_subtitle 返回 script_pages');
      }

      setResultPath(path);
      setScriptPages(pages);
      setStateSnapshot(data.state_snapshot != null ? JSON.stringify(data.state_snapshot) : null);
      setTimeout(() => setCurrentStep('script'), 400);
    } catch (err) {
      clearInterval(progressInterval);
      setProgress(0);
      setError(err instanceof Error ? err.message : '服务器繁忙，请稍后再试');
      console.error(err);
    } finally {
      setIsUploading(false);
    }
  };

  const handleConfirmScript = async () => {
    if (!resultPath) {
      setError('缺少 result_path');
      return;
    }

    const quota = await checkQuota(user?.id || null, user?.is_anonymous || false);
    if (quota.remaining < VIDEO_GENERATION_COST) {
      setError(
        quota.isAuthenticated
          ? t('errors.quotaUserInsufficient', { count: VIDEO_GENERATION_COST })
          : t('errors.quotaGuestInsufficient', { count: VIDEO_GENERATION_COST })
      );
      return;
    }

    setIsGeneratingVideo(true);
    setError(null);
    setCurrentStep('complete');
    setVideoUrl(null);

    try {
      const formData = new FormData();
      formData.append('result_path', resultPath);
      formData.append('script_pages', JSON.stringify(scriptPages.map((p) => ({ page_num: p.pageNum, script_text: p.scriptText }))));
      formData.append('email', user?.id || user?.email || '');
      if (stateSnapshot) formData.append('state_snapshot', stateSnapshot);

      const res = await fetch('/api/v1/paper2video/generate-video', {
        method: 'POST',
        headers: { 'X-API-Key': API_KEY },
        body: formData,
      });

      if (!res.ok) {
        let msg = '视频生成请求失败';
        if (res.status === 429) msg = '请求过于频繁，请稍后再试';
        throw new Error(msg);
      }

      const data = await res.json();
      if (!data.success) throw new Error(data.message || '视频生成失败');

      const url = data.video_url || (data.video_path ? convertToHttpUrl(data.video_path) : null);
      if (url) setVideoUrl(url);
      else setError(data.message || '后端未返回视频地址');

      const usageRecorded = await recordUsage(user?.id || null, 'paper2video', {
        amount: VIDEO_GENERATION_COST,
        isAnonymous: user?.is_anonymous || false,
      });
      if (usageRecorded) {
        refreshQuota();
      } else {
        setError(t('complete.usageRecordFailed', { count: VIDEO_GENERATION_COST }));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '视频生成失败');
      console.error(err);
    } finally {
      setIsGeneratingVideo(false);
    }
  };

  const handleDownload = () => {
    if (!videoUrl) return;
    const a = document.createElement('a');
    a.href = videoUrl;
    a.download = `paper2video_${Date.now()}.mp4`;
    a.target = '_blank';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const handleReset = () => {
    setCurrentStep('upload');
    setSelectedFile(null);
    setUseAvatar('none');
    setAvatarFile(null);
    setAvatarPreview(null);
    setAvatarPreset(null);
    setScriptPages([]);
    setResultPath(null);
    setStateSnapshot(null);
    setVideoUrl(null);
    setTtsModel(TTS_MODEL_DEFAULT);
    setTtsVoiceName('longanyang');
    setError(null);
    setProgress(0);
    setProgressStatus('');
  };

  return (
    <div className="w-full h-full min-h-0 flex flex-col bg-[#050512] overflow-hidden">
      <Banner show={showBanner} onClose={() => setShowBanner(false)} stars={stars} />
      <div className="flex-1 min-h-0 overflow-y-auto overflow-x-hidden">
        <div className="max-w-7xl mx-auto px-6 py-8 pb-24">
          <StepIndicator currentStep={currentStep} />

          {currentStep === 'upload' && (
            <UploadStep
              selectedFile={selectedFile}
              isDragOver={isDragOver}
              setIsDragOver={setIsDragOver}
              useAvatar={useAvatar}
              setUseAvatar={setUseAvatar}
              avatarFile={avatarFile}
              avatarPreview={avatarPreview}
              avatarPreset={avatarPreset}
              isUploading={isUploading}
              progress={progress}
              progressStatus={progressStatus}
              error={error}
              apiKey={apiKey}
              setApiKey={setApiKey}
              scriptApiUrl={scriptApiUrl}
              setScriptApiUrl={setScriptApiUrl}
              scriptModel={scriptModel}
              setScriptModel={setScriptModel}
              ttsModel={ttsModel}
              setTtsModel={setTtsModel}
              ttsVoiceName={ttsVoiceName}
              setTtsVoiceName={setTtsVoiceName}
              language={language}
              setLanguage={setLanguage}
              handleFileChange={handleFileChange}
              handleDrop={handleDrop}
              handleAvatarChange={handleAvatarChange}
              handleSelectAvatarPreset={handleSelectAvatarPreset}
              handleRemoveAvatar={handleRemoveAvatar}
              handleStartParse={handleStartParse}
            />
          )}

          {currentStep === 'script' && (
            <ScriptStep
              scriptPages={scriptPages}
              setScriptPages={setScriptPages}
              handleConfirmScript={handleConfirmScript}
              setCurrentStep={setCurrentStep}
              error={error}
              isGenerating={false}
            />
          )}

          {currentStep === 'complete' && (
            <CompleteStep
              videoUrl={videoUrl}
              isGenerating={isGeneratingVideo}
              handleDownload={handleDownload}
              handleReset={handleReset}
              error={error}
            />
          )}

          {/* 示例效果：预加载展示 public/paper2video/example 下的示例 PDF 与视频 */}
          <section className="mt-16 pt-10 border-t border-white/10">
            <h3 className="text-lg font-semibold text-white mb-1">{t('paper2video:example.title')}</h3>
            <p className="text-sm text-gray-400 mb-6">{t('paper2video:example.subtitle')}</p>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div className="glass rounded-xl border border-white/10 p-4 space-y-3">
                <h4 className="text-teal-300 font-medium">{t('paper2video:example.example1Title')}</h4>
                <div className="rounded-lg overflow-hidden border border-white/10 bg-black/40">
                  <iframe
                    title="示例一 PDF"
                    src={`${EXAMPLE_BASE}/dataflow.pdf#view=FitH`}
                    className="w-full h-[280px]"
                  />
                </div>
                <div className="rounded-lg overflow-hidden border border-white/10 bg-black/40">
                  <video
                    src={`${EXAMPLE_BASE}/dataflow.mp4`}
                    controls
                    className="w-full max-h-[280px]"
                    preload="metadata"
                  />
                </div>
              </div>
              <div className="glass rounded-xl border border-white/10 p-4 space-y-3">
                <h4 className="text-teal-300 font-medium">{t('paper2video:example.example2Title')}</h4>
                <div className="rounded-lg overflow-hidden border border-white/10 bg-black/40">
                  <iframe
                    title="示例二 PDF"
                    src={`${EXAMPLE_BASE}/poetry.pdf#view=FitH`}
                    className="w-full h-[280px]"
                  />
                </div>
                <div className="rounded-lg overflow-hidden border border-white/10 bg-black/40">
                  <video
                    src={`${EXAMPLE_BASE}/poetry.mp4`}
                    controls
                    className="w-full max-h-[280px]"
                    preload="metadata"
                  />
                </div>
              </div>
            </div>
          </section>
        </div>
      </div>

      <style>{`
        .glass { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(10px); }
      `}</style>
    </div>
  );
};

export default Paper2VideoPage;
