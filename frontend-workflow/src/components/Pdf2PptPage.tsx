import { useState, ChangeEvent } from 'react';
import { 
  UploadCloud, Download, Loader2, CheckCircle2, 
  AlertCircle, Github, Star, X, FileText, ArrowRight, Key, Globe
} from 'lucide-react';

// ============== 主组件 ==============
const Pdf2PptPage = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isComplete, setIsComplete] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showBanner, setShowBanner] = useState(true);
  const [downloadBlob, setDownloadBlob] = useState<Blob | null>(null);
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState('');
  
  // 三个必填配置
  const [inviteCode, setInviteCode] = useState('');
  const [llmApiUrl, setLlmApiUrl] = useState('https://api.apiyi.com/v1');
  const [apiKey, setApiKey] = useState('');

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
    if (!inviteCode.trim()) {
      setError('请输入邀请码');
      return;
    }
    if (!apiKey.trim()) {
      setError('请输入 API Key');
      return;
    }
    if (!llmApiUrl.trim()) {
      setError('请输入 API URL');
      return;
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
          '正在分析论文结构...',
          '正在提取关键内容...',
          '正在生成 PPT 页面...',
          '正在美化样式...',
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
      formData.append('chat_api_url', llmApiUrl.trim());
      formData.append('api_key', apiKey.trim());
      formData.append('invite_code', inviteCode.trim());
      
      const res = await fetch('/api/pdf2ppt/generate', {
        method: 'POST',
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
        <div className="w-full bg-gradient-to-r from-violet-600 via-purple-600 to-fuchsia-500 relative flex-shrink-0">
          <div className="absolute inset-0 bg-black opacity-20"></div>
          <div className="relative max-w-7xl mx-auto px-4 py-2.5 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Star size={14} className="text-yellow-300 fill-yellow-300" />
              <span className="text-sm text-white">⚡ PDF2PPT - 一键将 PDF 转换为 PPT</span>
            </div>
            <div className="flex items-center gap-2">
              <a 
                href="https://github.com/OpenDCAI/DataFlow-Agent" 
                target="_blank" 
                rel="noopener noreferrer" 
                className="px-3 py-1 bg-white/90 text-gray-900 rounded-full text-xs font-medium flex items-center gap-1"
              >
                <Github size={12} /> GitHub
              </a>
              <button onClick={() => setShowBanner(false)} className="p-1 hover:bg-white/20 rounded-full">
                <X size={14} className="text-white" />
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="flex-1 overflow-auto flex items-center justify-center">
        <div className="max-w-2xl w-full mx-auto px-6 py-8">
          {/* 标题 */}
          <div className="text-center mb-8">
            <p className="text-xs uppercase tracking-[0.2em] text-purple-300 mb-3 font-semibold">PDF → PPTX</p>
            <h1 className="text-4xl md:text-5xl font-bold mb-4">
              <span className="bg-gradient-to-r from-violet-400 via-purple-400 to-fuchsia-400 bg-clip-text text-transparent">
                PDF2PPT
              </span>
            </h1>
            <p className="text-base text-gray-300 max-w-xl mx-auto leading-relaxed">
              上传 PDF 文件，AI 自动分析内容并生成精美 PPT。<br />
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

                {/* 三个必填配置 */}
                <div className="space-y-4 mb-6">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-xs text-gray-400 mb-1.5 flex items-center gap-1">
                        <Key size={12} /> 邀请码 <span className="text-red-400">*</span>
                      </label>
                      <input 
                        type="text" 
                        value={inviteCode} 
                        onChange={e => setInviteCode(e.target.value)}
                        placeholder="ABC123"
                        className="w-full rounded-lg border border-white/20 bg-black/40 px-3 py-2.5 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-purple-500"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-gray-400 mb-1.5 flex items-center gap-1">
                        <Key size={12} /> API Key <span className="text-red-400">*</span>
                      </label>
                      <input 
                        type="password" 
                        value={apiKey} 
                        onChange={e => setApiKey(e.target.value)}
                        placeholder="sk-..."
                        className="w-full rounded-lg border border-white/20 bg-black/40 px-3 py-2.5 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-purple-500"
                      />
                    </div>
                  </div>
                  
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
                </div>

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
      </div>

      <style>{`.glass { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(10px); }`}</style>
    </div>
  );
};

export default Pdf2PptPage;
