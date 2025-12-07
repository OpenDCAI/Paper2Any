import { useState, useEffect, ChangeEvent } from 'react';
import { FileText, UploadCloud, Settings2, Download, Loader2, CheckCircle2, AlertCircle, ChevronDown, ChevronUp, Github, Star, X } from 'lucide-react';

const BACKEND_API = '/api/paper2ppt/generate';

const Paper2PptPage = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const [llmApiUrl, setLlmApiUrl] = useState('https://api.openai.com/v1/chat/completions');
  const [apiKey, setApiKey] = useState('');
  const [model, setModel] = useState('gpt-4o');
  const [showAdvanced, setShowAdvanced] = useState(false);

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [lastFilename, setLastFilename] = useState('paper2ppt.pptx');
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [showBanner, setShowBanner] = useState(true);

  useEffect(() => {
    return () => {
      if (downloadUrl) {
        URL.revokeObjectURL(downloadUrl);
      }
    };
  }, [downloadUrl]);

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) {
      setSelectedFile(null);
      return;
    }
    const ext = file.name.split('.').pop()?.toLowerCase();
    if (ext !== 'pdf') {
      setError('仅支持 PDF 格式');
      setSelectedFile(null);
      return;
    }
    setSelectedFile(file);
    setError(null);
  };

  const handleSubmit = async () => {
    if (isLoading) return;
    setError(null);
    setSuccessMessage(null);
    setDownloadUrl(null);

    if (!llmApiUrl.trim() || !apiKey.trim()) {
      setError('请先配置模型 API URL 和 API Key');
      return;
    }

    if (!selectedFile) {
      setError('请先选择要上传的 PDF 文件');
      return;
    }

    const formData = new FormData();
    formData.append('model_name', model);
    formData.append('chat_api_url', llmApiUrl.trim());
    formData.append('api_key', apiKey.trim());
    formData.append('input_type', 'file');
    formData.append('file', selectedFile);
    formData.append('file_kind', 'pdf');

    try {
      setIsLoading(true);
      const res = await fetch(BACKEND_API, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        let msg = '生成 PPTX 失败';
        try {
          const text = await res.text();
          if (text) msg = text;
        } catch {
          // ignore
        }
        throw new Error(msg);
      }

      const disposition = res.headers.get('content-disposition') || '';
      let filename = 'paper2ppt.pptx';
      const match = disposition.match(/filename="?([^";]+)"?/i);
      if (match?.[1]) {
        filename = decodeURIComponent(match[1]);
      }

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      setDownloadUrl(url);
      setLastFilename(filename);
      setSuccessMessage('PPTX 已生成，正在下载...');

      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (err) {
      const message = err instanceof Error ? err.message : '生成 PPTX 失败';
      setError(message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    // 修改点：min-h-screen 改为 h-screen，并添加 overflow-hidden
    // 这样强制外层容器高度固定，内部的 overflow-auto 才会生效
    <div className="w-full h-screen flex flex-col bg-[#050512] overflow-hidden">
      {/* GitHub 引流横幅 */}
      {showBanner && (
        <div className="w-full bg-gradient-to-r from-purple-600 via-pink-600 to-orange-500 relative overflow-hidden flex-shrink-0">
          <div className="absolute inset-0 bg-black opacity-20"></div>
          <div className="absolute inset-0 animate-pulse">
            <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-r from-transparent via-white to-transparent opacity-10 animate-shimmer"></div>
          </div>
          
          <div className="relative max-w-7xl mx-auto px-4 py-3 flex flex-col sm:flex-row items-center justify-between gap-3">
            <div className="flex items-center gap-3 flex-wrap justify-center sm:justify-start">
              <div className="flex items-center gap-2 bg-white/20 backdrop-blur-sm rounded-full px-3 py-1">
                <Star size={16} className="text-yellow-300 fill-yellow-300 animate-pulse" />
                <span className="text-xs font-bold text-white">开源项目</span>
              </div>
              
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

      {/* 主区域 - 这里保持 overflow-auto，现在它会正常工作了 */}
      <div className="flex-1 w-full overflow-auto">
        <div className="max-w-6xl mx-auto px-6 py-12">
          <div className="animate-fade-in">
            {/* 顶部标题区 */}
            <div className="mb-12 text-center">
              <p className="text-xs uppercase tracking-[0.2em] text-purple-300 mb-3 font-semibold">
                PAPER → PPTX
              </p>
              <h1 className="text-4xl md:text-5xl font-bold text-white mb-4">
                Paper2PPT
              </h1>
              <p className="text-base text-gray-300 max-w-2xl mx-auto leading-relaxed">
                上传论文或报告的 PDF 文件，一键生成结构化的 PPTX 演示文稿，适合学术汇报、答辩与分享。
              </p>
            </div>

            {/* 上传区 + 配置区 */}
            <div className="grid grid-cols-1 lg:grid-cols-[2fr,1fr] gap-6 mb-12">
              {/* 上传卡片 */}
              <div className="gradient-border">
                <div className="relative rounded-xl bg-white/95 text-gray-900 p-8 overflow-hidden">
                  <div className="absolute -right-10 -top-10 w-40 h-40 bg-purple-100 rounded-full opacity-60 blur-3xl pointer-events-none" />
                  <div className="relative">
                    <div className="flex items-center gap-2 mb-3">
                      <FileText size={20} className="text-purple-600" />
                      <p className="text-sm font-semibold text-purple-600">上传 PDF 文档</p>
                    </div>
                    <h2 className="text-2xl font-bold mb-2">从论文到 PPTX</h2>
                    <p className="text-sm text-gray-600 mb-6 leading-relaxed">
                      上传学术论文、研究报告或技术文档的 PDF 文件，系统会自动提取关键内容，生成结构化的 PPTX 演示文稿。
                    </p>

                    {/* PDF 上传区域 */}
                    <div className="border-2 border-dashed border-gray-300 rounded-xl p-10 flex flex-col items-center justify-center text-center gap-4 bg-gradient-to-br from-white to-purple-50/30 hover:border-purple-400 transition-all group">
                      <div className="flex items-center justify-center w-20 h-20 rounded-full bg-gradient-to-br from-purple-50 to-purple-100 group-hover:from-purple-100 group-hover:to-purple-200 transition-all">
                        <UploadCloud size={36} className="text-purple-600" />
                      </div>
                      
                      <div>
                        <p className="text-base font-semibold text-gray-800 mb-1">
                          拖拽 PDF 文件到此处
                        </p>
                        <p className="text-sm text-gray-500">
                          或点击下方按钮选择文件
                        </p>
                      </div>

                      <label className="inline-flex items-center gap-2 px-8 py-3 rounded-full bg-gradient-to-r from-purple-600 to-pink-600 text-white text-sm font-semibold cursor-pointer hover:from-purple-700 hover:to-pink-700 transition-all shadow-lg hover:shadow-xl hover:scale-105">
                        <FileText size={18} />
                        选择 PDF 文件
                        <input
                          type="file"
                          accept=".pdf"
                          className="hidden"
                          onChange={handleFileChange}
                        />
                      </label>

                      {selectedFile && (
                        <div className="mt-2 px-5 py-2.5 bg-emerald-50 border-2 border-emerald-300 rounded-lg">
                          <p className="text-sm text-emerald-700 font-semibold">
                            ✓ 已选择：{selectedFile.name}
                          </p>
                        </div>
                      )}

                      <p className="text-xs text-gray-500 mt-2">
                        支持 PDF 格式，单个文件建议小于 20MB
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* 高级配置卡片 */}
              <div className="glass rounded-xl border border-white/10 p-6 flex flex-col gap-4">
                <button
                  type="button"
                  onClick={() => setShowAdvanced(v => !v)}
                  className="flex items-center justify-between gap-2 w-full text-left group"
                >
                  <div className="flex items-center gap-2">
                    <Settings2 size={18} className="text-purple-400" />
                    <span className="text-white font-semibold">模型配置</span>
                  </div>
                  {showAdvanced ? (
                    <ChevronUp size={18} className="text-gray-400 group-hover:text-white transition-colors" />
                  ) : (
                    <ChevronDown size={18} className="text-gray-400 group-hover:text-white transition-colors" />
                  )}
                </button>

                {showAdvanced && (
                  <div className="space-y-4 pt-2">
                    <div>
                      <label className="block text-sm text-gray-300 mb-2 font-medium">模型 API URL</label>
                      <input
                        type="text"
                        value={llmApiUrl}
                        onChange={e => setLlmApiUrl(e.target.value)}
                        placeholder="https://api.openai.com/v1/chat/completions"
                        className="w-full rounded-lg border border-white/20 bg-black/40 px-4 py-2.5 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500 placeholder:text-gray-500"
                      />
                    </div>

                    <div>
                      <label className="block text-sm text-gray-300 mb-2 font-medium">API Key</label>
                      <input
                        type="password"
                        value={apiKey}
                        onChange={e => setApiKey(e.target.value)}
                        placeholder="sk-..."
                        className="w-full rounded-lg border border-white/20 bg-black/40 px-4 py-2.5 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500 placeholder:text-gray-500"
                      />
                    </div>

                    <div>
                      <label className="block text-sm text-gray-300 mb-2 font-medium">模型名称</label>
                      <input
                        type="text"
                        value={model}
                        onChange={e => setModel(e.target.value)}
                        placeholder="gpt-4o"
                        className="w-full rounded-lg border border-white/20 bg-black/40 px-4 py-2.5 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500 placeholder:text-gray-500"
                      />
                    </div>
                  </div>
                )}

                <div className="mt-auto space-y-3 pt-4 border-t border-white/10">
                  <button
                    type="button"
                    onClick={handleSubmit}
                    disabled={isLoading}
                    className="w-full inline-flex items-center justify-center gap-2 rounded-lg bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:from-gray-600 disabled:to-gray-700 disabled:cursor-not-allowed text-white font-semibold py-3 transition-all glow shadow-lg hover:shadow-xl"
                  >
                    {isLoading ? <Loader2 size={18} className="animate-spin" /> : <Download size={18} />}
                    <span>生成 PPTX</span>
                  </button>

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
                      className="w-full inline-flex items-center justify-center gap-2 rounded-lg border-2 border-emerald-400/60 text-emerald-300 text-sm font-medium py-2.5 bg-emerald-500/10 hover:bg-emerald-500/20 transition-colors"
                    >
                      <CheckCircle2 size={16} />
                      <span>重新下载</span>
                    </button>
                  )}

                  {error && (
                    <div className="flex items-start gap-2 text-sm text-red-300 bg-red-500/10 border border-red-500/40 rounded-lg px-4 py-3">
                      <AlertCircle size={16} className="mt-0.5 flex-shrink-0" />
                      <p>{error}</p>
                    </div>
                  )}

                  {successMessage && !error && (
                    <div className="flex items-start gap-2 text-sm text-emerald-300 bg-emerald-500/10 border border-emerald-500/40 rounded-lg px-4 py-3">
                      <CheckCircle2 size={16} className="mt-0.5 flex-shrink-0" />
                      <p>{successMessage}</p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* 示例区 */}
            <div className="space-y-6">
              <div className="text-center">
                <h3 className="text-lg font-semibold text-white mb-2">应用示例</h3>
                <p className="text-sm text-gray-400">查看从 Paper 到 PPTX 的转换效果</p>
              </div>

              <div className="glass rounded-xl border border-white/10 p-6 hover:border-purple-500/30 transition-all">
                <div className="flex flex-col md:flex-row gap-6 items-center">
                  {/* 左侧：输入示例 */}
                  <div className="flex-1 w-full">
                    <div className="rounded-lg bg-white/5 border-2 border-dashed border-white/20 flex items-center justify-center text-sm text-gray-400 aspect-[4/3] mb-3">
                      输入：论文 PDF
                    </div>
                    <p className="text-sm text-gray-300 text-center">上传学术论文、研究报告 PDF</p>
                  </div>

                  {/* 中间：箭头 */}
                  <div className="flex items-center justify-center">
                    <div className="text-3xl text-purple-400 font-bold">→</div>
                  </div>

                  {/* 右侧：输出示例 */}
                  <div className="flex-1 w-full">
                    <div className="rounded-lg bg-gradient-to-br from-purple-500/20 to-pink-500/20 border-2 border-dashed border-purple-400/40 flex items-center justify-center text-sm text-purple-200 aspect-[4/3] mb-3">
                      输出：结构化 PPTX
                    </div>
                    <p className="text-sm text-gray-300 text-center">自动提取关键内容生成演示文稿</p>
                  </div>
                </div>

                <div className="mt-6 pt-6 border-t border-white/10">
                  <h4 className="text-base text-white font-semibold mb-3">Paper → PPTX 转换流程</h4>
                  <p className="text-sm text-gray-300 leading-relaxed">
                    系统会自动分析论文结构，提取研究背景、方法、实验结果和结论等关键内容，生成适合学术汇报、答辩演示的 PPTX 文件。生成的文稿包含清晰的标题层级、要点归纳和逻辑结构，可直接用于演示或进一步编辑。
                  </p>
                </div>
              </div>
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
          box-shadow: 0 0 20px rgba(168, 85, 247, 0.4);
        }
      `}</style>
    </div>
  );
};

export default Paper2PptPage;