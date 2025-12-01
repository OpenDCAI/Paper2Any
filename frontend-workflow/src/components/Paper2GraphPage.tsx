import { useState, useEffect, ChangeEvent } from 'react';
import {
  FileText,
  Image as ImageIcon,
  UploadCloud,
  Link2,
  Type,
  Settings2,
  Download,
  Loader2,
  CheckCircle2,
  AlertCircle,
} from 'lucide-react';

type UploadMode = 'file' | 'url' | 'text';
type FileKind = 'pdf' | 'image' | null;

const BACKEND_API = '/api/paper2graph/generate';

const IMAGE_EXTENSIONS = ['png', 'jpg', 'jpeg', 'bmp', 'gif', 'webp', 'tiff'];

function detectFileKind(file: File): FileKind {
  const ext = file.name.split('.').pop()?.toLowerCase();
  if (!ext) return null;
  if (ext === 'pdf') return 'pdf';
  if (IMAGE_EXTENSIONS.includes(ext)) return 'image';
  return null;
}

const Paper2GraphPage = () => {
  const [uploadMode, setUploadMode] = useState<UploadMode>('file');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileKind, setFileKind] = useState<FileKind>(null);
  const [sourceUrl, setSourceUrl] = useState('');
  const [textContent, setTextContent] = useState('');

  const [llmApiUrl, setLlmApiUrl] = useState('https://api.openai.com/v1/chat/completions');
  const [apiKey, setApiKey] = useState('');
  const [model, setModel] = useState('NanoBanana');

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [lastFilename, setLastFilename] = useState('paper2graph.pptx');
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

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

    if (!llmApiUrl.trim() || !apiKey.trim()) {
      setError('请先配置模型 API URL 和 API Key');
      return;
    }

    const formData = new FormData();
    formData.append('model_name', model);
    formData.append('chat_api_url', llmApiUrl.trim());
    formData.append('api_key', apiKey.trim());
    formData.append('input_type', uploadMode);

    if (uploadMode === 'file') {
      if (!selectedFile) {
        setError('请先选择要上传的文件');
        return;
      }
      const kind = fileKind ?? detectFileKind(selectedFile);
      if (!kind) {
        setError('仅支持 PDF 和常见图片格式，请检查文件类型');
        return;
      }
      formData.append('file', selectedFile);
      formData.append('file_kind', kind);
    } else if (uploadMode === 'url') {
      if (!sourceUrl.trim()) {
        setError('请输入文档 URL');
        return;
      }
      formData.append('source_url', sourceUrl.trim());
    } else if (uploadMode === 'text') {
      if (!textContent.trim()) {
        setError('请输入要解析的文本内容');
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
        try {
          const text = await res.text();
          if (text) msg = text;
        } catch {
          // ignore
        }
        throw new Error(msg);
      }

      const disposition = res.headers.get('content-disposition') || '';
      let filename = 'paper2graph.pptx';
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
    <div className="w-full h-full flex bg-[#050512]">
      {/* 左侧导航栏（仿 MinerU） */}
      <aside className="w-60 border-r border-white/5 glass-dark flex flex-col">
        <div className="px-5 pt-6 pb-4 border-b border-white/5">
          <p className="text-xs uppercase tracking-wide text-gray-400 mb-1">Paper2Graph</p>
          <h2 className="text-lg font-semibold text-white">解析文档</h2>
        </div>

        <nav className="flex-1 px-2 py-4 space-y-1 text-sm">
          <button className="w-full flex items-center gap-2 px-3 py-2 rounded-lg bg-white/10 text-white">
            <FileText size={16} />
            <span>解析文档</span>
          </button>
          <button className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-gray-400 hover:bg-white/5">
            <UploadCloud size={16} />
            <span>文件</span>
          </button>
          <button className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-gray-400 hover:bg-white/5">
            <StarIcon />
            <span>收藏</span>
          </button>
        </nav>

        <div className="px-4 py-4 border-t border-white/5 text-xs text-gray-500">
          支持论文、报告、教科书等多种文档，一键生成知识图谱 PPT。
        </div>
      </aside>

      {/* 右侧主区域 */}
      <div className="flex-1 flex flex-col items-center justify-center px-8 py-10 overflow-auto">
        <div className="w-full max-w-5xl animate-fade-in">
          {/* 顶部标题区 */}
          <div className="mb-6 flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-semibold text-white mb-1">
                文档解析 & Paper2Graph
              </h1>
              <p className="text-sm text-gray-400">
                支持文本 / PDF / 图片，多模态解析论文内容，自动生成结构化 PPTX。
              </p>
            </div>
            <div className="hidden md:flex items-center gap-2 text-xs text-gray-400">
              <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span>后端 FastAPI 服务已接入（假定）</span>
            </div>
          </div>

          {/* 上半区：上传区 + 配置区 */}
          <div className="grid grid-cols-1 lg:grid-cols-[2fr,minmax(260px,1fr)] gap-6 mb-8">
            {/* 上传卡片（仿 MinerU 中央白卡） */}
            <div className="gradient-border">
              <div className="relative rounded-xl bg-white/95 text-gray-900 p-6 lg:p-8 overflow-hidden">
                <div className="absolute -right-10 -top-10 w-40 h-40 bg-primary-100 rounded-full opacity-60 blur-3xl pointer-events-none" />
                <div className="relative">
                  <p className="text-xs font-medium text-primary-600 mb-2">文档解析</p>
                  <h2 className="text-xl font-semibold mb-1">选择你的输入方式</h2>
                  <p className="text-xs text-gray-500 mb-4">
                    支持本地文件上传、URL 远程文档解析，以及直接粘贴文本内容。
                  </p>

                  {/* 上传模式 Tab */}
                  <div className="inline-flex items-center rounded-full bg-gray-100 p-1 text-xs mb-4">
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
                      本地上传
                    </button>
                    <button
                      type="button"
                      onClick={() => setUploadMode('url')}
                      className={`flex items-center gap-1 px-3 py-1.5 rounded-full ${
                        uploadMode === 'url'
                          ? 'bg-white shadow text-gray-900'
                          : 'text-gray-500 hover:text-gray-800'
                      }`}
                    >
                      <Link2 size={14} />
                      URL 上传
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
                      粘贴文本
                    </button>
                  </div>

                  {/* 不同模式内容区域 */}
                  {uploadMode === 'file' && (
                    <div className="border border-dashed border-gray-300 rounded-xl p-5 flex flex-col items-center justify-center text-center gap-3 bg-white/60">
                      <div className="flex items-center justify-center gap-2 text-gray-600 text-sm">
                        <FileText size={20} />
                        <span className="font-medium">拖拽文件到此处，或点击选择文件</span>
                      </div>
                      <label className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary-500 text-white text-xs font-medium cursor-pointer hover:bg-primary-600 transition-colors">
                        选择文件
                        <input
                          type="file"
                          accept=".pdf,image/*"
                          className="hidden"
                          onChange={handleFileChange}
                        />
                      </label>
                      <p className="text-[11px] text-gray-500">{showFileHint()}，单个文件建议小于 20MB。</p>
                    </div>
                  )}

                  {uploadMode === 'url' && (
                    <div className="space-y-3">
                      <label className="block text-xs font-medium text-gray-600">
                        文档 URL
                      </label>
                      <input
                        type="url"
                        value={sourceUrl}
                        onChange={e => setSourceUrl(e.target.value)}
                        placeholder="例如：https://arxiv.org/abs/xxxx.xxxxx 或 PDF 直链"
                        className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-primary-400 focus:border-primary-400 bg-white/80"
                      />
                      <p className="text-[11px] text-gray-500">
                        支持公开可访问的 PDF / HTML 页面，后端将自动抓取并解析。
                      </p>
                    </div>
                  )}

                  {uploadMode === 'text' && (
                    <div className="space-y-3">
                      <label className="block text-xs font-medium text-gray-600">
                        粘贴论文摘要或章节内容
                      </label>
                      <textarea
                        value={textContent}
                        onChange={e => setTextContent(e.target.value)}
                        rows={8}
                        placeholder="在这里粘贴论文的摘要、章节或任意要解析的文本..."
                        className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-primary-400 focus:border-primary-400 bg-white/80 resize-none"
                      />
                      <p className="text-[11px] text-gray-500">
                        建议控制在 5,000 字以内，过长内容可以分段多次解析。
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* 配置卡片 */}
            <div className="glass rounded-xl border border-white/10 p-5 flex flex-col gap-4 text-sm">
              <div className="flex items-center gap-2 mb-1">
                <Settings2 size={16} className="text-primary-300" />
                <span className="text-white font-medium">解析配置</span>
              </div>

              <div className="space-y-3">
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
                    placeholder="用于 DataFlow Agent 调用 OpenAI / 兼容模型的 API Key"
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
                    <option value="NanoBanana">NanoBanana</option>
                    <option value="NanoBanana Pro">NanoBanana Pro</option>
                  </select>
                </div>
              </div>

              <div className="mt-2 space-y-2">
                <button
                  type="button"
                  onClick={handleSubmit}
                  disabled={isLoading}
                  className="w-full inline-flex items-center justify-center gap-2 rounded-lg bg-primary-500 hover:bg-primary-600 disabled:bg-primary-500/60 disabled:cursor-not-allowed text-white text-sm font-medium py-2.5 transition-colors glow"
                >
                  {isLoading ? <Loader2 size={16} className="animate-spin" /> : <Download size={16} />}
                  <span>生成 Paper2Graph PPTX</span>
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

          {/* 示例区 */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-medium text-gray-200">示例</h3>
              <span className="text-[11px] text-gray-500">以下示例仅为展示样式，可根据需要接入真实数据。</span>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
              <ExampleCard
                title="英文论文 PDF"
                tag="科研汇报"
                desc="从论文中抽取研究背景、方法、实验与结论，自动生成可编辑汇报 PPT。"
              />
              <ExampleCard
                title="技术报告 / 白皮书"
                tag="技术总结"
                desc="根据章节结构拆分要点，生成适合路演或评审的结构化 PPT。"
              />
              <ExampleCard
                title="项目需求文档"
                tag="产品设计"
                desc="提取角色、需求与约束，生成需求评审用 PPT 大纲与页面草稿。"
              />
              <ExampleCard
                title="数据分析报告"
                tag="可视化"
                desc="识别关键图表与指标说明，生成带占位图的可编辑数据分析 PPT。"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const StarIcon = () => (
  <svg
    className="w-4 h-4"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <polygon points="12 2 15 8.5 22 9.3 17 14 18.5 21 12 17.8 5.5 21 7 14 2 9.3 9 8.5 12 2" />
  </svg>
);

interface ExampleCardProps {
  title: string;
  tag: string;
  desc: string;
}

const ExampleCard = ({ title, tag, desc }: ExampleCardProps) => {
  return (
    <div className="glass rounded-lg border border-white/10 p-3 flex flex-col gap-1 hover:bg-white/5 transition-colors">
      <div className="flex items-center justify-between gap-2 mb-1">
        <p className="text-[13px] text-white font-medium truncate">{title}</p>
        <span className="text-[10px] px-2 py-0.5 rounded-full bg-primary-500/20 text-primary-200 whitespace-nowrap">
          {tag}
        </span>
      </div>
      <p className="text-[11px] text-gray-400 leading-snug">{desc}</p>
    </div>
  );
};

export default Paper2GraphPage;
