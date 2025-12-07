import { useState, useEffect, ChangeEvent } from 'react';
import { FileText, UploadCloud, Type, Settings2, Download, Loader2, CheckCircle2, AlertCircle, Image as ImageIcon, ChevronDown, ChevronUp, Github, Star, X } from 'lucide-react';

type UploadMode = 'file' | 'text' | 'image';
type FileKind = 'pdf' | 'image' | null;

const BACKEND_API = '/api/paper2figure/generate';

const IMAGE_EXTENSIONS = ['png', 'jpg', 'jpeg', 'bmp', 'gif', 'webp', 'tiff'];

function detectFileKind(file: File): FileKind {
  const ext = file.name.split('.').pop()?.toLowerCase();
  if (!ext) return null;
  if (ext === 'pdf') return 'pdf';
  if (IMAGE_EXTENSIONS.includes(ext)) return 'image';
  return null;
}

const Paper2FigurePage = () => {
  const [uploadMode, setUploadMode] = useState<UploadMode>('file');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileKind, setFileKind] = useState<FileKind>(null);
  const [textContent, setTextContent] = useState('');
  const [inviteCode, setInviteCode] = useState('');

  const [llmApiUrl, setLlmApiUrl] = useState('http://123.129.219.111:3000/v1');
  const [apiKey, setApiKey] = useState('');
  const [model, setModel] = useState('NanoBanana');
  const [showAdvanced, setShowAdvanced] = useState(false);

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [lastFilename, setLastFilename] = useState('paper2figure.pptx');
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
      setFileKind(null);
      return;
    }
    const kind = detectFileKind(file);
    setSelectedFile(file);
    setFileKind(kind);
    setError(null);
  };

  const handleSubmit = async () => {
    if (isLoading) return;
    setError(null);
    setSuccessMessage(null);
    setDownloadUrl(null);

    if (!inviteCode.trim()) {
      setError('请先输入邀请码');
      return;
    }

    if (!llmApiUrl.trim() || !apiKey.trim()) {
      setError('请先配置模型 API URL 和 API Key');
      return;
    }

    const formData = new FormData();
    formData.append('img_gen_model_name', model);
    formData.append('chat_api_url', llmApiUrl.trim());
    formData.append('api_key', apiKey.trim());
    formData.append('input_type', uploadMode);
    formData.append('invite_code', inviteCode.trim());

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
      const res = await fetch(BACKEND_API, {
        method: 'POST',
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

      {/* 主区域：居中简洁布局 */}
      <div className="flex-1 flex flex-col items-center justify-center px-6 py-10 overflow-auto">
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
              上传论文 PDF / 图片，或直接粘贴文字，一键生成可编辑的 PPTX，方便你继续修改、增删和排版。
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
                      文件（PDF / 图片）
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
                      onClick={() => setUploadMode('image')}
                      className={`flex items-center gap-1 px-3 py-1.5 rounded-full ${
                        uploadMode === 'image'
                          ? 'bg-white shadow text-gray-900'
                          : 'text-gray-500 hover:text-gray-800'
                      }`}
                    >
                      <ImageIcon size={14} />
                      图片
                    </button>
                  </div>

                  {/* 不同模式内容区域 */}
                  {(uploadMode === 'file' || uploadMode === 'image') && (
                    <div className="border border-dashed border-gray-300 rounded-xl p-5 flex flex-col items-center justify-center text-center gap-3 bg-white/60">
                      <div className="flex items-center justify-center gap-2 text-gray-600 text-sm">
                        <FileText size={20} />
                        <span className="font-medium">
                          拖拽 {uploadMode === 'file' ? 'PDF / 图片' : '图片'} 到此处，或点击选择文件
                        </span>
                      </div>
                      <label className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary-500 text-white text-xs font-medium cursor-pointer hover:bg-primary-600 transition-colors">
                        选择文件
                        <input
                          type="file"
                          accept={uploadMode === 'file' ? '.pdf,image/*' : 'image/*'}
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
                        placeholder="在这里粘贴论文的摘要、章节内容，或任意需要转换为 PPTX 的文字（支持中英文）..."
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
                      <div>
                        <label className="block text-xs text-gray-400 mb-1">邀请码</label>
                        <input
                          type="text"
                          value={inviteCode}
                          onChange={e => setInviteCode(e.target.value)}
                          placeholder="请输入邀请码"
                          className="w-full rounded-lg border border-white/10 bg-black/30 px-3 py-2 text-xs text-gray-200 outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                        />
                      </div>

                      <div>
                        <label className="block text-xs text-gray-400 mb-1">模型 API URL</label>
                    <input
                      type="text"
                      value={llmApiUrl}
                      onChange={e => setLlmApiUrl(e.target.value)}
                      placeholder="例如：https://api.openai.com/v1/chat/completions"
                      className="w-full rounded-lg border border-white/10 bg-black/30 px-3 py-2 text-xs text-gray-200 outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                    />
                  </div>

                  <div>
                    <label className="block text-xs text-gray-400 mb-1">API Key</label>
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
                      <option value="gemini-2.5-flash-image-preview">NanoBanana</option>
                      <option value="gemini-3-pro-image-preview">NanoBanana Pro</option>
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
                  <span>生成可编辑 PPTX</span>
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
                    className="w-full inline-flex items-center justify-center gap-2 rounded-lg border border-emerald-400/60 text-emerald-300 text-xs py-2 bg-emerald-500/10 hover:bg-emerald-500/20 transition-colors"
                  >
                    <CheckCircle2 size={14} />
                    <span>重新下载：{lastFilename}</span>
                  </button>
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
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-medium text-gray-200">示例：从 Paper 到 PPTX</h3>
              <span className="text-[11px] text-gray-500">
                下方示例展示从 PDF / 图片 / 文本 到可编辑 PPTX 的效果，你可以替换为自己的示例图片。
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
              <DemoCard
                title="论文 PDF → 符合论文主题的 科研绘图（PPT）"
                desc="上传英文论文 PDF，自动提炼研究背景、方法、实验设计和结论，生成结构清晰、符合学术风格的汇报 PPTX。"
              />
              <DemoCard
                title="生图模型结果 → 可编辑 PPTX"
                desc="上传由 Gemini 等模型生成的科研配图或示意图截图，智能识别段落层级与要点，自动排版为可编辑的中英文 PPTX。"
              />
              <DemoCard
                title="摘要文本 → 科研绘图"
                desc="粘贴论文摘要或章节内容，一键生成包含标题层级、关键要点与图示占位的 PPTX 大纲，方便后续细化与美化。"
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
}

const DemoCard = ({ title, desc }: DemoCardProps) => {
  return (
    <div className="glass rounded-lg border border-white/10 p-3 flex flex-col gap-2 hover:bg-white/5 transition-colors">
      <div className="flex gap-2">
        {/* 左侧：输入示例图片占位，你可以替换为真实 img */}
        <div className="flex-1 rounded-md bg-white/5 border border-dashed border-white/10 flex items-center justify-center text-[10px] text-gray-400 demo-input-placeholder">
          输入示例图（待替换）
        </div>
        {/* 右侧：输出 PPTX 示例图片占位，你可以替换为真实 img */}
        <div className="flex-1 rounded-md bg-primary-500/10 border border-dashed border-primary-300/40 flex items-center justify-center text-[10px] text-primary-200 demo-output-placeholder">
          PPTX 示例图（待替换）
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