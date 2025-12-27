import { useState, useEffect, ChangeEvent } from 'react';
import {
  UploadCloud, Download, Loader2, CheckCircle2,
  AlertCircle, Github, Star, X, FileText, ArrowRight, Key, Globe, ToggleLeft, ToggleRight, Sparkles, Image, MessageSquare, Copy
} from 'lucide-react';
import { saveFileRecord } from '../services/fileService';
import { API_KEY } from '../config/api';
import { checkQuota, recordUsage } from '../services/quotaService';
import { useAuthStore } from '../stores/authStore';

// ============== 主组件 ==============
const Pdf2PptPage = () => {
  const { user, refreshQuota } = useAuthStore();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isComplete, setIsComplete] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showBanner, setShowBanner] = useState(true);
  const [downloadBlob, setDownloadBlob] = useState<Blob | null>(null);
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState('');

  // GitHub Stars
  const [stars, setStars] = useState<{dataflow: number | null, agent: number | null}>({ dataflow: null, agent: null });
  const [copySuccess, setCopySuccess] = useState('');

  const shareText = `发现一个超好用的AI工具 DataFlow-Agent！🚀
支持论文转PPT、PDF转PPT、PPT美化等功能，科研打工人的福音！

🔗 在线体验：https://dcai-paper2any.nas.cpolar.cn/
⭐ GitHub Agent：https://github.com/OpenDCAI/DataFlow-Agent
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
        const [res1, res2] = await Promise.all([
          fetch('https://api.github.com/repos/OpenDCAI/DataFlow'),
          fetch('https://api.github.com/repos/OpenDCAI/DataFlow-Agent')
        ]);
        const data1 = await res1.json();
        const data2 = await res2.json();
        setStars({
          dataflow: data1.stargazers_count,
          agent: data2.stargazers_count
        });
      } catch (e) {
        console.error('Failed to fetch stars', e);
      }
    };
    fetchStars();
  }, []);
  
  // 配置
  const [inviteCode, setInviteCode] = useState('');
  const [useAiEdit, setUseAiEdit] = useState(false);
  const [llmApiUrl, setLlmApiUrl] = useState('https://api.apiyi.com/v1');
  const [apiKey, setApiKey] = useState('');
  const [genFigModel, setGenFigModel] = useState('gemini-2.5-flash-image');

  const validateDocFile = (file: File): boolean => {
    const ext = file.name.split('.').pop()?.toLowerCase();
    if (ext !== 'pdf') {
      setError('仅支持 PDF 格式');
      return false;
    }
    return true;
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !validateDocFile(file)) return;
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
    setSelectedFile(file);
    setError(null);
    setIsComplete(false);
    setDownloadBlob(null);
  };

  const handleConvert = async () => {
    if (!selectedFile) {
      setError('请先选择 PDF 文件');
      return;
    }

    // Check quota before proceeding
    const quota = await checkQuota(user?.id || null, user?.is_anonymous || false);
    if (quota.remaining <= 0) {
      setError(quota.isAuthenticated
        ? '今日配额已用完（10次/天），请明天再试'
        : '今日配额已用完（5次/天），登录后可获得更多配额');
      return;
    }

    if (useAiEdit) {
      if (!apiKey.trim()) {
        setError('开启 AI 增强时必须输入 API Key');
        return;
      }
      if (!llmApiUrl.trim()) {
        setError('开启 AI 增强时必须输入 API URL');
        return;
      }
    }
    
    setIsProcessing(true);
    setError(null);
    setProgress(0);
    setStatusMessage('正在上传文件...');
    
    // 模拟进度
    const progressInterval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 90) {
          clearInterval(progressInterval);
          return 90;
        }
        const messages = [
          '正在分析 PPT 结构...',
          '正在提取关键内容...',
          '正在提取 Icon ...',
          '正在生成 PPT 页面...',
          '正在导出文件...',
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
      formData.append('invite_code', inviteCode.trim());
      
      if (useAiEdit) {
        formData.append('use_ai_edit', 'true');
        formData.append('chat_api_url', llmApiUrl.trim());
        formData.append('api_key', apiKey.trim());
        formData.append('gen_fig_model', genFigModel);
      } else {
        formData.append('use_ai_edit', 'false');
      }
      
      const res = await fetch('/api/pdf2ppt/generate', {
        method: 'POST',
        headers: { 'X-API-Key': API_KEY },
        body: formData,
      });
      
      clearInterval(progressInterval);
      
      if (!res.ok) {
        let msg = '转换失败';
        if (res.status === 403) {
          msg = '邀请码不正确或已失效';
        } else {
          try {
            const errorData = await res.json();
            msg = errorData.detail || errorData.message || msg;
          } catch {
            const text = await res.text();
            if (text) msg = text;
          }
        }
        throw new Error(msg);
      }
      
      // 获取文件 blob
      const blob = await res.blob();
      setDownloadBlob(blob);
      setProgress(100);
      setStatusMessage('转换完成！');
      setIsComplete(true);

      // Record usage and save file record
      await recordUsage(user?.id || null, 'pdf2ppt');
      refreshQuota();
      saveFileRecord('pdf2ppt_output.pptx', 'pdf2ppt', blob.size);
      
    } catch (err) {
      clearInterval(progressInterval);
      const message = err instanceof Error ? err.message : '转换失败，请重试';
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
            <p className="text-xs uppercase tracking-[0.2em] text-purple-300 mb-3 font-semibold">PDF → PPTX</p>
            <h1 className="text-4xl md:text-5xl font-bold mb-4">
              <span className="bg-gradient-to-r from-violet-400 via-purple-400 to-fuchsia-400 bg-clip-text text-transparent">
                PDF2PPT
              </span>
            </h1>
            <p className="text-base text-gray-300 max-w-xl mx-auto leading-relaxed">
              上传 PDF 版本 PPT，提取PPT元素以及文字可编辑。<br />
              <span className="text-purple-400">一键转换，快速生成！</span>
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
                      <p className="text-sm text-purple-300">✓ {selectedFile.name}</p>
                      <p className="text-xs text-gray-400 mt-1">
                        {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                      </p>
                    </div>
                  ) : (
                    <>
                      <div>
                        <p className="text-white font-medium mb-1">拖拽 PDF 文件到此处</p>
                        <p className="text-sm text-gray-400">或点击下方按钮选择文件</p>
                      </div>
                      <label className="px-6 py-2.5 rounded-full bg-gradient-to-r from-violet-600 to-fuchsia-600 text-white text-sm font-medium cursor-pointer hover:from-violet-700 hover:to-fuchsia-700 transition-all">
                        选择文件
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
                      <p className="text-sm font-medium text-white">AI 背景增强</p>
                      <p className="text-xs text-gray-400">使用 Gemini 模型清除文字并修复背景</p>
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
                        <Globe size={12} /> API URL <span className="text-red-400">*</span>
                      </label>
                      <input 
                        type="text" 
                        value={llmApiUrl} 
                        onChange={e => setLlmApiUrl(e.target.value)}
                        placeholder="https://api.openai.com/v1"
                        className="w-full rounded-lg border border-white/20 bg-black/40 px-3 py-2.5 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-purple-500"
                      />
                    </div>
                    
                    <div className="grid grid-cols-2 gap-4">
                    <div>
                      <div className="flex items-center justify-between mb-1.5">
                        <label className="block text-xs text-gray-400 mb-0 flex items-center gap-1">
                          <Key size={12} /> API Key <span className="text-red-400">*</span>
                        </label>
                        <a
                          href="https://api.apiyi.com/"
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-[10px] text-purple-300 hover:text-purple-200 hover:underline"
                        >
                          点击购买
                        </a>
                      </div>
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
                          <Image size={12} /> 生成模型
                        </label>
                        <div className="relative">
                          <select 
                            value={genFigModel} 
                            onChange={e => setGenFigModel(e.target.value)}
                            className="w-full appearance-none rounded-lg border border-white/20 bg-black/40 px-3 py-2.5 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-purple-500"
                          >
                            <option value="gemini-2.5-flash-image">Gemini 2.5 Flash</option>
                            <option value="gemini-3-pro-image-preview">Gemini 3 Pro</option>
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
                    <><Loader2 size={20} className="animate-spin" /> 正在转换中...</>
                  ) : (
                    <><ArrowRight size={20} /> 开始转换</>
                  )}
                </button>
              </>
            ) : (
              /* 完成状态 */
              <div className="text-center py-8">
                <div className="w-24 h-24 rounded-full bg-gradient-to-br from-emerald-500 to-teal-500 flex items-center justify-center mx-auto mb-6">
                  <CheckCircle2 size={48} className="text-white" />
                </div>
                <h2 className="text-2xl font-bold text-white mb-2">转换完成！</h2>
                <p className="text-gray-400 mb-8">您的 PPT 文件已准备好下载</p>
                
                <div className="space-y-4">
                  <button 
                    onClick={handleDownload} 
                    className="w-full py-4 rounded-xl bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-600 hover:to-teal-600 text-white font-semibold flex items-center justify-center gap-2 transition-all text-lg"
                  >
                    <Download size={20} /> 下载 PPT
                  </button>
                  
                  <button 
                    onClick={handleReset} 
                    className="w-full py-3 rounded-xl border border-white/20 text-gray-300 hover:bg-white/10 transition-all"
                  >
                    转换新的文件
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
                       <a href="https://dcai-paper2any.nas.cpolar.cn/" target="_blank" rel="noopener noreferrer" className="block w-full py-1.5 px-3 rounded bg-white/5 hover:bg-white/10 text-xs text-purple-300 truncate transition-colors border border-white/5 text-center">
                         🌐 如果项目对你有帮助的话，可以点个star嘛~
                       </a>
                       <div className="flex gap-2">
                         <a href="https://github.com/OpenDCAI/DataFlow-Agent" target="_blank" rel="noopener noreferrer" className="flex-1 py-1.5 px-3 rounded bg-white/5 hover:bg-white/10 text-xs text-gray-300 truncate transition-colors border border-white/5 flex items-center justify-center gap-1">
                           <Github size={12} /> Agent
                         </a>
                         <a href="https://github.com/OpenDCAI/DataFlow" target="_blank" rel="noopener noreferrer" className="flex-1 py-1.5 px-3 rounded bg-white/5 hover:bg-white/10 text-xs text-gray-300 truncate transition-colors border border-white/5 flex items-center justify-center gap-1">
                           <Github size={12} /> Core
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
            支持的文件格式：PDF | 最大文件大小：50MB
          </p>
          </div>

          {/* 示例区 */}
          <div className="space-y-4 mt-16 max-w-4xl mx-auto">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-medium text-gray-200">示例：从 PDF 到 可编辑 PPTX（文字 + 元素ICON）</h3>
              <span className="text-[11px] text-gray-500">
                下方示例展示不同模式下的转换效果。
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <DemoCard
                title="基础转换（白色背景）"
                desc="快速将 PDF 转换为可编辑的 PPT，保留原始排版和内容，适合标准文档转换。"
                inputImg="/pdf2ppt/input_1.png"
                outputImg="/pdf2ppt/output_1.png"
              />
              <DemoCard
                title="AI 增强模式（AI重塑背景）"
                desc="利用 AI 清除原有背景，智能重塑页面风格，提升视觉效果，打造专业演示文稿。"
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
