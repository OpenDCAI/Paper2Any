import React, { useState, ChangeEvent, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { API_KEY } from '../config/api';
import { getApiSettings } from '../services/apiSettingsService';
import { useAuthStore } from '../stores/authStore';
import {
  UploadCloud, Settings2, Loader2, FileText, Type, Lightbulb
} from 'lucide-react';
import type { UploadMode } from './paper2ppt/types';
import type { Step, SlideOutline, GenerateResult } from './paper2ppt/types';
import { MAX_FILE_SIZE } from './paper2ppt/constants';
import StepIndicator from './paper2ppt/StepIndicator';
import OutlineStep from './paper2ppt/OutlineStep';
import GenerateStep from './paper2ppt/GenerateStep';
import CompleteStep from './paper2ppt/CompleteStep';

const Paper2PptBeamerPage: React.FC = () => {
  const { t } = useTranslation(['paper2ppt', 'common']);
  const { user } = useAuthStore();

  const [currentStep, setCurrentStep] = useState<Step>('upload');
  const [uploadMode, setUploadMode] = useState<UploadMode>('file');
  const [textContent, setTextContent] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [pageCount, setPageCount] = useState(6);
  const [language, setLanguage] = useState<'zh' | 'en'>('en');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [resultPath, setResultPath] = useState<string | null>(null);
  const [outlineData, setOutlineData] = useState<SlideOutline[]>([]);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editContent, setEditContent] = useState<{
    title: string;
    layout_description: string;
    key_points: string[];
  }>({ title: '', layout_description: '', key_points: [] });
  const [outlineFeedback, setOutlineFeedback] = useState('');
  const [isRefiningOutline, setIsRefiningOutline] = useState(false);

  const [generateResults, setGenerateResults] = useState<GenerateResult[]>([]);
  const [currentSlideIndex, setCurrentSlideIndex] = useState(0);
  const [isGenerating, setIsGenerating] = useState(false);
  const [slidePrompt, setSlidePrompt] = useState('');
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);

  const [stars, setStars] = useState<{ dataflow: number | null; agent: number | null; dataflex: number | null }>({
    dataflow: null,
    agent: null,
    dataflex: null,
  });
  const [copySuccess, setCopySuccess] = useState('');

  const apiSettings = getApiSettings(user?.id || null);
  const chatApiUrl = apiSettings?.apiUrl || '';
  const apiKey = apiSettings?.apiKey || '';

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const ext = file.name.split('.').pop()?.toLowerCase();
    if (ext !== 'pdf') {
      setError('仅支持 PDF 格式');
      return;
    }
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
    if (!file) return;
    const ext = file.name.split('.').pop()?.toLowerCase();
    if (ext !== 'pdf') {
      setError('仅支持 PDF 格式');
      return;
    }
    if (file.size > MAX_FILE_SIZE) {
      setError('文件大小超过 50MB 限制');
      return;
    }
    setSelectedFile(file);
    setError(null);
  };

  // ---------- Step 1: 仅调用 page-content，进入大纲步骤 ----------
  const handleStartParse = async () => {
    if (uploadMode === 'file' && !selectedFile) {
      setError('请上传 PDF 文件');
      return;
    }
    if ((uploadMode === 'text' || uploadMode === 'topic') && !textContent.trim()) {
      setError(uploadMode === 'topic' ? '请输入主题' : '请输入文本内容');
      return;
    }

    setError(null);
    setIsSubmitting(true);

    try {
      const formData = new FormData();
      if (uploadMode === 'file' && selectedFile) {
        formData.append('file', selectedFile);
        formData.append('input_type', 'pdf');
      } else {
        formData.append('text', textContent.trim());
        formData.append('input_type', uploadMode);
      }
      formData.append('email', user?.id || user?.email || '');
      formData.append('chat_api_url', chatApiUrl.trim());
      formData.append('api_key', apiKey.trim());
      formData.append('model', 'gpt-4o');
      formData.append('language', language);
      formData.append('style', '');
      formData.append('gen_fig_model', 'gemini-3-pro-image-preview');
      formData.append('page_count', String(pageCount));
      formData.append('use_long_paper', 'false');
      formData.append('ppt_mode', 'beamer');

      const res = await fetch('/api/v1/paper2ppt/page-content', {
        method: 'POST',
        headers: { 'X-API-Key': API_KEY },
        body: formData,
      });

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody?.error || errBody?.detail || '解析失败');
      }

      const data = await res.json();
      if (!data.success) throw new Error(data.error || '解析失败');

      const path = data.result_path;
      const pagecontent = data.pagecontent;
      if (!path || !pagecontent?.length) {
        throw new Error('未返回 result_path 或 pagecontent');
      }

      setResultPath(path);
      const slides: SlideOutline[] = pagecontent.map((item: any, index: number) => ({
        id: String(index + 1),
        pageNum: index + 1,
        title: item.title || `第 ${index + 1} 页`,
        layout_description: item.layout_description || '',
        key_points: item.key_points || [],
        asset_ref: item.asset_ref ?? null,
      }));
      setOutlineData(slides);
      setCurrentStep('outline');
    } catch (err) {
      setError(err instanceof Error ? err.message : '请求失败');
    } finally {
      setIsSubmitting(false);
    }
  };

  // ---------- Outline 编辑与确认：确认后调用 generate (beamer)，进入逐页预览 ----------
  const handleEditStart = (slide: SlideOutline) => {
    setEditingId(slide.id);
    setEditContent({
      title: slide.title,
      layout_description: slide.layout_description,
      key_points: [...slide.key_points],
    });
  };

  const handleEditSave = () => {
    if (!editingId) return;
    setOutlineData((prev) =>
      prev.map((s) =>
        s.id === editingId
          ? {
              ...s,
              title: editContent.title,
              layout_description: editContent.layout_description,
              key_points: editContent.key_points,
            }
          : s
      )
    );
    setEditingId(null);
  };

  const handleEditCancel = () => setEditingId(null);

  const handleKeyPointChange = (index: number, value: string) => {
    setEditContent((prev) => {
      const next = [...prev.key_points];
      next[index] = value;
      return { ...prev, key_points: next };
    });
  };

  const handleAddKeyPoint = () => {
    setEditContent((prev) => ({ ...prev, key_points: [...prev.key_points, ''] }));
  };

  const handleRemoveKeyPoint = (index: number) => {
    setEditContent((prev) => ({
      ...prev,
      key_points: prev.key_points.filter((_, i) => i !== index),
    }));
  };

  const handleDeleteSlide = (id: string) => {
    setOutlineData((prev) =>
      prev.filter((s) => s.id !== id).map((s, i) => ({ ...s, pageNum: i + 1 }))
    );
  };

  const handleAddSlide = (index: number) => {
    setOutlineData((prev) => {
      const newSlide: SlideOutline = {
        id: String(Date.now()),
        pageNum: 0,
        title: '新页面',
        layout_description: '左右图文',
        key_points: [''],
        asset_ref: null,
      };
      const next = [...prev];
      next.splice(index + 1, 0, newSlide);
      return next.map((s, i) => ({
        ...s,
        pageNum: i + 1,
        title: s.title === '新页面' ? `第 ${i + 1} 页` : s.title,
      }));
    });
  };

  const handleMoveSlide = (index: number, direction: 'up' | 'down') => {
    const next = [...outlineData];
    const target = direction === 'up' ? index - 1 : index + 1;
    if (target < 0 || target >= next.length) return;
    [next[index], next[target]] = [next[target], next[index]];
    setOutlineData(next.map((s, i) => ({ ...s, pageNum: i + 1 })));
  };

  const handleConfirmOutline = async () => {
    if (!resultPath) {
      setError('缺少 result_path');
      return;
    }
    setError(null);
    setIsGenerating(true);
    setIsRefiningOutline(true); // 禁用大纲确认按钮，防止重复提交

    const pagecontent = outlineData.map((s) => ({
      title: s.title,
      layout_description: s.layout_description,
      key_points: s.key_points,
      asset_ref: s.asset_ref,
    }));

    try {
      const form = new FormData();
      form.append('img_gen_model_name', 'gemini-3-pro-image-preview');
      form.append('chat_api_url', chatApiUrl.trim());
      form.append('api_key', apiKey.trim());
      form.append('model', 'gpt-4o');
      form.append('language', language);
      form.append('style', '');
      form.append('aspect_ratio', '16:9');
      form.append('email', user?.id || user?.email || '');
      form.append('result_path', resultPath);
      form.append('get_down', 'false');
      form.append('all_edited_down', 'true');
      form.append('ppt_mode', 'beamer');
      form.append('pagecontent', JSON.stringify(pagecontent));

      const res = await fetch('/api/v1/paper2ppt/generate', {
        method: 'POST',
        headers: { 'X-API-Key': API_KEY },
        body: form,
      });

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody?.error || errBody?.detail || '生成失败');
      }

      const data = await res.json();
      if (!data.success) throw new Error(data.error || '生成失败');

      const pdfUrl =
        data.ppt_pdf_path ||
        (data.all_output_files &&
          data.all_output_files.find(
            (url: string) => url.endsWith('.pdf') && !url.includes('input')
          ));
      if (pdfUrl) setDownloadUrl(pdfUrl);

      const results: GenerateResult[] = outlineData.map((slide, index) => {
        const pageNumStr = String(index).padStart(3, '0');
        let afterImage = '';
        if (data.all_output_files && Array.isArray(data.all_output_files)) {
          const url = data.all_output_files.find((u: string) =>
            u.includes(`ppt_pages/page_${pageNumStr}.png`)
          );
          if (url) afterImage = url;
        }
        return {
          slideId: slide.id,
          beforeImage: '',
          afterImage,
          status: 'done' as const,
          versionHistory: [],
          currentVersionIndex: -1,
        };
      });

      setGenerateResults(results);
      setCurrentSlideIndex(0);
      setCurrentStep('generate');
    } catch (err) {
      setError(err instanceof Error ? err.message : '生成失败');
    } finally {
      setIsGenerating(false);
      setIsRefiningOutline(false);
    }
  };

  const handleConfirmSlide = () => {
    setError(null);
    if (currentSlideIndex < outlineData.length - 1) {
      setCurrentSlideIndex((i) => i + 1);
      setSlidePrompt('');
    } else {
      setCurrentStep('complete');
    }
  };

  const handleRegenerateSlide = () => {}; // Beamer 不支持逐页重新生成
  const handleRevertToVersion = () => {}; // Beamer 无版本历史

  const handleReset = () => {
    setCurrentStep('upload');
    setResultPath(null);
    setOutlineData([]);
    setGenerateResults([]);
    setDownloadUrl(null);
    setError(null);
    setCurrentSlideIndex(0);
    setEditingId(null);
    setEditContent({ title: '', layout_description: '', key_points: [] });
    setOutlineFeedback('');
    setSelectedFile(null);
    setTextContent('');
  };

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
        const textArea = document.createElement('textarea');
        textArea.value = shareText;
        textArea.style.position = 'fixed';
        textArea.style.left = '-9999px';
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
      }
      setCopySuccess('文案已复制！快去分享吧');
      setTimeout(() => setCopySuccess(''), 2000);
    } catch {
      setCopySuccess('复制失败，请手动复制');
    }
  };

  useEffect(() => {
    const fetchStars = async () => {
      try {
        const [res1, res2, res3] = await Promise.all([
          fetch('https://api.github.com/repos/OpenDCAI/DataFlow'),
          fetch('https://api.github.com/repos/OpenDCAI/Paper2Any'),
          fetch('https://api.github.com/repos/OpenDCAI/DataFlex'),
        ]);
        const [data1, data2, data3] = await Promise.all([res1.json(), res2.json(), res3.json()]);
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

  const handleDownloadPdf = () => {
    if (downloadUrl) window.open(downloadUrl, '_blank');
  };

  // ---------- 完成页：与 paper2ppt 一致布局，仅保留「下载 PDF」与「处理新的论文」，无「下载 PPTX」----------
  if (currentStep === 'complete') {
    return (
      <div className="w-full h-screen flex flex-col bg-[#050512] overflow-hidden">
        <div className="flex-1 overflow-auto">
          <div className="max-w-7xl mx-auto px-6 py-8 pb-24">
            <StepIndicator currentStep="complete" />
            <CompleteStep
              outlineData={outlineData}
              generateResults={generateResults}
              downloadUrl={null}
              pdfPreviewUrl={downloadUrl}
              isGeneratingFinal={false}
              handleGenerateFinal={() => {}}
              handleDownloadPptx={() => {}}
              handleDownloadPdf={handleDownloadPdf}
              handleReset={handleReset}
              error={error}
              handleCopyShareText={handleCopyShareText}
              copySuccess={copySuccess}
              stars={stars}
              pdfOnly
            />
          </div>
        </div>
        <style>{`.glass { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(10px); }`}</style>
      </div>
    );
  }

  // ---------- 上传步骤 ----------
  if (currentStep === 'upload') {
    return (
      <div className="w-full h-screen flex flex-col bg-[#050512] overflow-hidden">
        <div className="flex-1 overflow-auto">
          <div className="max-w-6xl mx-auto px-6 py-8">
            <StepIndicator currentStep="upload" />
            <div className="mb-10 text-center">
              <p className="text-xs uppercase tracking-[0.2em] text-purple-300 mb-3 font-semibold">
                Beamer · PDF
              </p>
              <h1 className="text-4xl md:text-5xl font-bold mb-4">
                <span className="bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
                  Paper2PPT Beamer
                </span>
              </h1>
              <p className="text-base text-gray-300 max-w-2xl mx-auto">
                上传 PDF、长文本或 Topic，解析后可在第二步编辑大纲，再生成 LaTeX Beamer 逐页预览与 PDF。
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="glass rounded-xl border border-white/10 p-6">
                <div className="grid grid-cols-3 gap-3 mb-6 p-1.5 bg-black/40 rounded-2xl border border-white/5">
                  {[
                    { id: 'file' as const, label: t('upload.tabs.file'), icon: FileText },
                    { id: 'text' as const, label: t('upload.tabs.text'), icon: Type },
                    { id: 'topic' as const, label: t('upload.tabs.topic'), icon: Lightbulb },
                  ].map((item) => (
                    <button
                      key={item.id}
                      onClick={() => setUploadMode(item.id)}
                      className={`flex flex-col items-center py-3 rounded-xl transition-all ${
                        uploadMode === item.id
                          ? 'bg-gradient-to-br from-indigo-600 to-purple-600 text-white'
                          : 'bg-white/5 text-gray-400 hover:bg-white/10'
                      }`}
                    >
                      <item.icon size={22} className="mb-1.5" />
                      <span className="text-sm font-bold">{item.label}</span>
                    </button>
                  ))}
                </div>

                {uploadMode === 'file' ? (
                  <div
                    className={`border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center text-center gap-4 h-[280px] ${
                      isDragOver ? 'border-purple-500 bg-purple-500/10' : 'border-white/20'
                    }`}
                    onDragOver={(e) => {
                      e.preventDefault();
                      setIsDragOver(true);
                    }}
                    onDragLeave={(e) => {
                      e.preventDefault();
                      setIsDragOver(false);
                    }}
                    onDrop={handleDrop}
                  >
                    <UploadCloud size={32} className="text-purple-400" />
                    <p className="text-white font-medium">{t('upload.dropzone.dragText')}</p>
                    <p className="text-sm text-gray-400">{t('upload.dropzone.supportText')}</p>
                    <label className="px-6 py-2.5 rounded-full bg-gradient-to-r from-indigo-600 to-purple-600 text-white text-sm font-medium cursor-pointer">
                      {t('upload.dropzone.button')}
                      <input
                        type="file"
                        accept=".pdf"
                        className="hidden"
                        onChange={handleFileChange}
                      />
                    </label>
                    {selectedFile && (
                      <p className="text-sm text-purple-300">✓ {selectedFile.name}</p>
                    )}
                  </div>
                ) : (
                  <textarea
                    value={textContent}
                    onChange={(e) => setTextContent(e.target.value)}
                    placeholder={
                      uploadMode === 'text'
                        ? t('upload.textInput.placeholderText')
                        : t('upload.textInput.placeholderTopic')
                    }
                    className="w-full h-[280px] rounded-xl border border-white/20 bg-black/40 px-4 py-3 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-purple-500 resize-none"
                  />
                )}
              </div>

              <div className="glass rounded-xl border border-white/10 p-6 space-y-4">
                <h3 className="text-white font-semibold flex items-center gap-2">
                  <Settings2 size={18} className="text-purple-400" /> {t('upload.config.title')}
                </h3>
                <div>
                  <label className="block text-xs text-gray-400 mb-1">
                    {t('upload.config.language')}
                  </label>
                  <select
                    value={language}
                    onChange={(e) => setLanguage(e.target.value as 'zh' | 'en')}
                    className="w-full rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-purple-500"
                  >
                    <option value="zh">中文</option>
                    <option value="en">English</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs text-gray-400 mb-1">
                    {t('upload.config.pageCount')}
                  </label>
                  <input
                    type="number"
                    value={pageCount}
                    onChange={(e) => setPageCount(parseInt(e.target.value) || 6)}
                    min={1}
                    max={20}
                    className="w-full rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-purple-500"
                  />
                </div>
                <button
                  onClick={handleStartParse}
                  disabled={isSubmitting}
                  className="w-full py-3 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 text-white font-semibold flex items-center justify-center gap-2 disabled:opacity-60"
                >
                  {isSubmitting ? (
                    <>
                      <Loader2 size={18} className="animate-spin" /> 解析中...
                    </>
                  ) : (
                    t('upload.config.startButton.parse')
                  )}
                </button>
                {error && (
                  <div className="text-sm text-red-300 bg-red-500/10 border border-red-500/40 rounded-lg px-3 py-2">
                    {error}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
        <style>{`.glass { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(10px); }`}</style>
      </div>
    );
  }

  // ---------- 大纲步骤 ----------
  if (currentStep === 'outline') {
    return (
      <div className="w-full h-screen flex flex-col bg-[#050512] overflow-hidden">
        <div className="flex-1 overflow-auto">
          <div className="max-w-7xl mx-auto px-6 py-8 pb-24">
            <StepIndicator currentStep="outline" />
            <OutlineStep
              outlineData={outlineData}
              editingId={editingId}
              editContent={editContent}
              setEditContent={setEditContent}
              handleEditStart={handleEditStart}
              handleEditSave={handleEditSave}
              handleEditCancel={handleEditCancel}
              handleKeyPointChange={handleKeyPointChange}
              handleAddKeyPoint={handleAddKeyPoint}
              handleRemoveKeyPoint={handleRemoveKeyPoint}
              handleDeleteSlide={handleDeleteSlide}
              handleAddSlide={handleAddSlide}
              handleMoveSlide={handleMoveSlide}
              handleConfirmOutline={handleConfirmOutline}
              handleRefineOutline={() => {}}
              setCurrentStep={setCurrentStep}
              error={error}
              outlineFeedback={outlineFeedback}
              setOutlineFeedback={setOutlineFeedback}
              isRefiningOutline={isRefiningOutline}
              hideRefine
            />
          </div>
        </div>
        <style>{`.glass { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(10px); }`}</style>
      </div>
    );
  }

  // ---------- 逐页预览步骤（generate） ----------
  return (
    <div className="w-full h-screen flex flex-col bg-[#050512] overflow-hidden">
      <div className="flex-1 overflow-auto">
        <div className="max-w-7xl mx-auto px-6 py-8 pb-24">
          <StepIndicator currentStep="generate" />
          <GenerateStep
            outlineData={outlineData}
            currentSlideIndex={currentSlideIndex}
            setCurrentSlideIndex={setCurrentSlideIndex}
            generateResults={generateResults}
            isGenerating={isGenerating}
            slidePrompt={slidePrompt}
            setSlidePrompt={setSlidePrompt}
            handleRegenerateSlide={handleRegenerateSlide}
            handleConfirmSlide={handleConfirmSlide}
            setCurrentStep={setCurrentStep}
            error={error}
            handleRevertToVersion={handleRevertToVersion}
            pptMode="beamer"
          />
        </div>
      </div>
      <style>{`.glass { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(10px); }`}</style>
    </div>
  );
};

export default Paper2PptBeamerPage;
