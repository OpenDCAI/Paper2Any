import { useEffect, useMemo, useState } from 'react';
import { FileBarChart, Loader2, FileText, Maximize2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { API_URL_OPTIONS } from '../../../config/api';
import { KnowledgeFile } from '../types';
import { getApiSettings } from '../../../services/apiSettingsService';
import { backendFetch } from '../../../services/backendClient';
import { useAuthStore } from '../../../stores/authStore';
import { MarkdownViewerModal } from './MarkdownViewerModal';

interface ReportToolProps {
  files: KnowledgeFile[];
  selectedIds: Set<string>;
  onGenerateSuccess: (file: KnowledgeFile) => void;
}

export const ReportTool = ({ files = [], selectedIds, onGenerateSuccess }: ReportToolProps) => {
  const { user } = useAuthStore();
  const [apiUrl, setApiUrl] = useState('https://api.apiyi.com/v1');
  const [apiKey, setApiKey] = useState('');
  const [model, setModel] = useState('gpt-5.1');
  const [language, setLanguage] = useState<'zh' | 'en'>('zh');
  const [style, setStyle] = useState<'insight' | 'analysis'>('insight');
  const [length, setLength] = useState<'short' | 'standard' | 'long'>('standard');
  const [loading, setLoading] = useState(false);
  const [reportMarkdown, setReportMarkdown] = useState('');
  const [showReportModal, setShowReportModal] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    const settings = getApiSettings(user?.id || null);
    if (settings) {
      setApiUrl(settings.apiUrl || apiUrl);
      setApiKey(settings.apiKey || apiKey);
    }
  }, [user?.id]);

  const selectedFiles = useMemo(() => files.filter(f => selectedIds.has(f.id)), [files, selectedIds]);
  const validDocs = selectedFiles.filter(f => {
    const name = f.name.toLowerCase();
    return name.endsWith('.pdf') || name.endsWith('.pptx') || name.endsWith('.ppt') || name.endsWith('.docx') || name.endsWith('.doc');
  });

  const handleGenerate = async () => {
    if (!user?.id || !user?.email) {
      alert('请先登录后再生成报告。');
      return;
    }
    if (!apiKey.trim()) {
      alert('请输入 API Key');
      return;
    }
    if (validDocs.length === 0) {
      alert('请选择至少一个 PDF/PPTX/DOCX 文档生成报告。');
      return;
    }

    setLoading(true);
    setError('');
    setReportMarkdown('');
    try {
      const filePaths = validDocs.map(f => f.url).filter(Boolean);
      const res = await backendFetch('/api/v1/kb/generate-report', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          file_paths: filePaths,
          api_url: apiUrl,
          api_key: apiKey,
          model,
          language,
          report_style: style,
          length,
          email: user.email,
          user_id: user.id
        })
      });

      if (!res.ok) {
        if (res.status === 401) {
          throw new Error('API Key 无效，请确认前端 VITE_API_KEY 与后端一致（默认 df-internal-2024-workflow-key）。');
        }
        if (res.status === 404) {
          throw new Error('接口未找到：请确认后端已重启并加载 /api/v1/kb/generate-report。');
        }
        throw new Error(await res.text());
      }
      const data = await res.json();
      setReportMarkdown(data.report_markdown || '');

      if (data.report_path) {
        onGenerateSuccess({
          id: data.output_file_id || `rp_${Date.now()}`,
          name: `report_${Date.now()}.md`,
          type: 'doc',
          size: '未知',
          uploadTime: new Date().toLocaleString(),
          url: data.report_path,
          desc: `Report from ${validDocs.length} document(s)`
        });
      }
    } catch (err: any) {
      setError(err?.message || '报告生成失败');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex-1 overflow-y-auto p-6 bg-[#0a0a1a] h-full">
      <div className="mb-6 bg-gradient-to-br from-fuchsia-900/20 to-purple-900/20 border border-fuchsia-500/20 rounded-xl p-4 flex items-start gap-3">
        <FileBarChart className="text-fuchsia-400 mt-1 flex-shrink-0" size={18} />
        <div>
          <h4 className="text-sm font-medium text-fuchsia-300 mb-1">报告生成</h4>
          <p className="text-xs text-fuchsia-200/70">
            选择多个文档生成洞察型分析报告，支持不同风格与长度。
          </p>
        </div>
      </div>

      <div className="space-y-5">
        <div>
          <label className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2 block">
            选中素材 ({validDocs.length} 个文档)
          </label>
          <div className="bg-white/5 border border-white/10 rounded-lg p-3 text-xs text-gray-300 flex items-center gap-2">
            <FileText size={14} className="text-fuchsia-400" />
            <span className="truncate">{validDocs.length ? validDocs.map(f => f.name).join(', ') : '未选择文档'}</span>
          </div>
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium text-gray-300">API URL</label>
          <select
            value={apiUrl}
            onChange={e => setApiUrl(e.target.value)}
            className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-gray-200 outline-none focus:border-fuchsia-500"
          >
            {[apiUrl, ...API_URL_OPTIONS].filter((v, i, a) => a.indexOf(v) === i).map((url: string) => (
              <option key={url} value={url}>{url}</option>
            ))}
          </select>
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium text-gray-300">API Key</label>
          <input
            type="password"
            value={apiKey}
            onChange={e => setApiKey(e.target.value)}
            placeholder="sk-..."
            className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-gray-200 outline-none focus:border-fuchsia-500 font-mono"
          />
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium text-gray-300">模型</label>
          <input
            type="text"
            value={model}
            onChange={e => setModel(e.target.value)}
            className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-gray-200 outline-none focus:border-fuchsia-500"
          />
        </div>

        <div className="grid grid-cols-2 gap-3">
          <button
            onClick={() => setLanguage('zh')}
            className={`py-2.5 rounded-lg border text-sm transition-all ${
              language === 'zh'
                ? 'bg-fuchsia-500/20 border-fuchsia-500 text-fuchsia-300'
                : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'
            }`}
          >
            中文
          </button>
          <button
            onClick={() => setLanguage('en')}
            className={`py-2.5 rounded-lg border text-sm transition-all ${
              language === 'en'
                ? 'bg-fuchsia-500/20 border-fuchsia-500 text-fuchsia-300'
                : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'
            }`}
          >
            English
          </button>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-300">报告风格</label>
            <select
              value={style}
              onChange={e => setStyle(e.target.value as 'insight' | 'analysis')}
              className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-gray-200 outline-none focus:border-fuchsia-500"
            >
              <option value="insight">洞察型</option>
              <option value="analysis">分析型</option>
            </select>
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-300">长度</label>
            <select
              value={length}
              onChange={e => setLength(e.target.value as 'short' | 'standard' | 'long')}
              className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-gray-200 outline-none focus:border-fuchsia-500"
            >
              <option value="short">简短</option>
              <option value="standard">标准</option>
              <option value="long">详细</option>
            </select>
          </div>
        </div>
      </div>

      <div className="mt-6">
        <button
          onClick={handleGenerate}
          disabled={loading}
          className="w-full bg-gradient-to-r from-fuchsia-600 to-purple-600 hover:from-fuchsia-500 hover:to-purple-500 text-white py-3 rounded-xl font-medium flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-fuchsia-500/20"
        >
          {loading ? <Loader2 size={18} className="animate-spin" /> : <FileBarChart size={18} />}
          {loading ? '生成中...' : '开始生成报告'}
        </button>
      </div>

      {error && <div className="mt-4 text-sm text-red-400">{error}</div>}

      {reportMarkdown && (
        <div className="mt-6 space-y-3">
          <div className="flex items-center justify-between">
            <div className="text-sm text-fuchsia-300 font-medium">报告预览</div>
            <button
              onClick={() => setShowReportModal(true)}
              className="flex items-center gap-1 text-xs text-fuchsia-200 hover:text-fuchsia-100"
            >
              <Maximize2 size={14} />
              放大阅读
            </button>
          </div>
          <div className="bg-white/5 border border-white/10 rounded-xl p-4 text-sm text-gray-200 leading-relaxed">
            <ReactMarkdown>{reportMarkdown}</ReactMarkdown>
          </div>
        </div>
      )}

      <MarkdownViewerModal
        open={showReportModal}
        onClose={() => setShowReportModal(false)}
        markdown={reportMarkdown}
        title="研究报告"
      />
    </div>
  );
};
