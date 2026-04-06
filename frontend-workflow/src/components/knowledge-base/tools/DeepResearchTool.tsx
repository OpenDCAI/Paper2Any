import { useEffect, useMemo, useState } from 'react';
import { FlaskConical, Loader2, Globe, Search, FileText, ExternalLink, Maximize2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { API_URL_OPTIONS } from '../../../config/api';
import { KnowledgeFile } from '../types';
import { getApiSettings } from '../../../services/apiSettingsService';
import { backendFetch } from '../../../services/backendClient';
import { useAuthStore } from '../../../stores/authStore';
import { MarkdownViewerModal } from './MarkdownViewerModal';

interface DeepResearchToolProps {
  files: KnowledgeFile[];
  selectedIds: Set<string>;
  onGenerateSuccess: (file: KnowledgeFile) => void;
}

interface SearchResultItem {
  title: string;
  url: string;
  snippet?: string;
  source?: string;
}

interface SummaryItem {
  query: string;
  summary: any;
}

export const DeepResearchTool = ({ files = [], selectedIds, onGenerateSuccess }: DeepResearchToolProps) => {
  const { user } = useAuthStore();
  const [mode, setMode] = useState<'llm' | 'web'>('llm');
  const [topic, setTopic] = useState('');
  const [apiUrl, setApiUrl] = useState('https://api.apiyi.com/v1');
  const [apiKey, setApiKey] = useState('');
  const [model, setModel] = useState('claude-sonnet-4-5-20250929-all');
  const [language, setLanguage] = useState<'zh' | 'en'>('zh');
  const [searchApiKey, setSearchApiKey] = useState('');
  const [searchEngine, setSearchEngine] = useState<'google' | 'baidu'>('google');
  const [searchNum, setSearchNum] = useState(10);
  const [searchProvider, setSearchProvider] = useState<'serpapi' | 'google_cse' | 'brave'>('serpapi');
  const [googleCseId, setGoogleCseId] = useState('');
  const [braveSummarizer, setBraveSummarizer] = useState(true);
  const [loading, setLoading] = useState(false);
  const [reportMarkdown, setReportMarkdown] = useState('');
  const [showReportModal, setShowReportModal] = useState(false);
  const [searchResults, setSearchResults] = useState<SearchResultItem[]>([]);
  const [summaries, setSummaries] = useState<SummaryItem[]>([]);
  const [error, setError] = useState('');

  useEffect(() => {
    const settings = getApiSettings(user?.id || null);
    if (settings) {
      setApiUrl(settings.apiUrl || apiUrl);
      setApiKey(settings.apiKey || apiKey);
    }
  }, [user?.id]);

  const selectedFiles = useMemo(() => files.filter(f => selectedIds.has(f.id)), [files, selectedIds]);

  const renderSummaryContent = (summary: any) => {
    if (!summary) return null;
    if (typeof summary === 'string') {
      return <div className="text-xs text-gray-300 whitespace-pre-wrap">{summary}</div>;
    }
    if (summary.summary && typeof summary.summary === 'string') {
      return <div className="text-xs text-gray-300 whitespace-pre-wrap">{summary.summary}</div>;
    }
    if (summary.summary && Array.isArray(summary.summary)) {
      return (
        <div className="text-xs text-gray-300 space-y-1">
          {summary.summary.map((line: string, idx: number) => (
            <div key={`${line}-${idx}`} className="whitespace-pre-wrap">{line}</div>
          ))}
        </div>
      );
    }
    return (
      <pre className="text-[11px] text-gray-400 whitespace-pre-wrap">
        {JSON.stringify(summary, null, 2)}
      </pre>
    );
  };

  const handleRun = async () => {
    if (!user?.id || !user?.email) {
      alert('请先登录后再进行深度研究。');
      return;
    }
    if (!topic.trim() && selectedFiles.length === 0) {
      alert('请输入研究主题，或选择至少一个文件作为上下文。');
      return;
    }
    if (!apiKey.trim()) {
      alert('请输入 API Key');
      return;
    }
    if (mode === 'web' && !searchApiKey.trim()) {
      alert('Web 模式需要 Search API Key');
      return;
    }
    if (mode === 'web' && searchProvider === 'google_cse' && !googleCseId.trim()) {
      alert('Google CSE 需要填写 cx (Search Engine ID)');
      return;
    }

    setLoading(true);
    setError('');
    setReportMarkdown('');
    setSearchResults([]);
    setSummaries([]);
    try {
      const filePaths = selectedFiles.map(f => f.url).filter(Boolean);
      const res = await backendFetch('/api/v1/kb/deep-research', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          mode,
          topic: topic.trim(),
          file_paths: filePaths,
          api_url: apiUrl,
          api_key: apiKey,
          model,
          language,
          email: user.email,
          user_id: user.id,
          search_provider: searchProvider,
          search_api_key: searchApiKey,
          search_engine: searchEngine,
          search_num: searchNum,
          google_cse_id: googleCseId,
          brave_summarizer: braveSummarizer
        })
      });

      if (!res.ok) {
        if (res.status === 401) {
          throw new Error('API Key 无效，请确认前端 VITE_API_KEY 与后端一致（默认 df-internal-2024-workflow-key）。');
        }
        if (res.status === 404) {
          throw new Error('接口未找到：请确认后端已重启并加载 /api/v1/kb/deep-research。');
        }
        throw new Error(await res.text());
      }
      const data = await res.json();
      setReportMarkdown(data.report_markdown || '');
      setSearchResults(data.search_results || []);
      setSummaries(data.summaries || []);

      if (data.report_path) {
        onGenerateSuccess({
          id: data.output_file_id || `dr_${Date.now()}`,
          name: `deep_research_${Date.now()}.md`,
          type: 'doc',
          size: '未知',
          uploadTime: new Date().toLocaleString(),
          url: data.report_path,
          desc: `Deep research: ${topic || 'untitled'}`
        });
      }
    } catch (err: any) {
      setError(err?.message || '深度研究失败');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex-1 overflow-y-auto p-6 bg-[#0a0a1a] h-full">
      <div className="mb-6 bg-gradient-to-br from-emerald-900/20 to-teal-900/20 border border-emerald-500/20 rounded-xl p-4 flex items-start gap-3">
        <FlaskConical className="text-emerald-400 mt-1 flex-shrink-0" size={18} />
        <div>
          <h4 className="text-sm font-medium text-emerald-300 mb-1">深度研究</h4>
          <p className="text-xs text-emerald-200/70">
            支持联网模型研究（模型自带检索能力），也可调用 Web 检索后生成带来源的研究摘要。
          </p>
        </div>
      </div>

      <div className="space-y-5">
        <div className="grid grid-cols-2 gap-3">
          <button
            onClick={() => setMode('llm')}
            className={`py-2 rounded-lg border text-sm transition-all ${
              mode === 'llm'
                ? 'bg-emerald-500/20 border-emerald-500 text-emerald-300'
                : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'
            }`}
          >
            联网模型研究
          </button>
          <button
            onClick={() => setMode('web')}
            className={`py-2 rounded-lg border text-sm transition-all ${
              mode === 'web'
                ? 'bg-emerald-500/20 border-emerald-500 text-emerald-300'
                : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'
            }`}
          >
            Web 检索 + 总结
          </button>
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium text-gray-300">研究主题</label>
          <input
            type="text"
            value={topic}
            onChange={e => setTopic(e.target.value)}
            placeholder="例如：AI Agent 在科研写作中的应用趋势"
            className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-gray-200 outline-none focus:border-emerald-500"
          />
        </div>

        <div>
          <label className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2 block">
            选中素材 ({selectedFiles.length} 个文件)
          </label>
          <div className="bg-white/5 border border-white/10 rounded-lg p-3 text-xs text-gray-300 flex items-center gap-2">
            <FileText size={14} className="text-emerald-400" />
            <span className="truncate">{selectedFiles.length ? selectedFiles.map(f => f.name).join(', ') : '未选择'}</span>
          </div>
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium text-gray-300">API URL</label>
          <select
            value={apiUrl}
            onChange={e => setApiUrl(e.target.value)}
            className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-gray-200 outline-none focus:border-emerald-500"
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
            className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-gray-200 outline-none focus:border-emerald-500 font-mono"
          />
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium text-gray-300">模型</label>
          <input
            type="text"
            value={model}
            onChange={e => setModel(e.target.value)}
            className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-gray-200 outline-none focus:border-emerald-500"
          />
        </div>

        <div className="grid grid-cols-2 gap-3">
          <button
            onClick={() => setLanguage('zh')}
            className={`py-2.5 rounded-lg border text-sm transition-all ${
              language === 'zh'
                ? 'bg-emerald-500/20 border-emerald-500 text-emerald-300'
                : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'
            }`}
          >
            中文
          </button>
          <button
            onClick={() => setLanguage('en')}
            className={`py-2.5 rounded-lg border text-sm transition-all ${
              language === 'en'
                ? 'bg-emerald-500/20 border-emerald-500 text-emerald-300'
                : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'
            }`}
          >
            English
          </button>
        </div>

        {mode === 'web' && (
          <div className="space-y-4 rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-4">
            <div className="flex items-center gap-2 text-xs text-emerald-200">
              <Globe size={14} />
              Web 检索设置
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-300">搜索源</label>
              <select
                value={searchProvider}
                onChange={e => {
                  const next = e.target.value as 'serpapi' | 'google_cse' | 'brave';
                  setSearchProvider(next);
                  if (next === 'google_cse' && searchNum > 10) {
                    setSearchNum(10);
                  }
                }}
                className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-gray-200 outline-none focus:border-emerald-500"
              >
                <option value="serpapi">SerpAPI</option>
                <option value="google_cse">Google CSE</option>
                <option value="brave">Brave Search</option>
              </select>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-300">Search API Key</label>
              <input
                type="password"
                value={searchApiKey}
                onChange={e => setSearchApiKey(e.target.value)}
                placeholder="search_api_key"
                className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-gray-200 outline-none focus:border-emerald-500 font-mono"
              />
            </div>
            {searchProvider === 'google_cse' && (
              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-300">Google CSE ID (cx)</label>
                <input
                  type="text"
                  value={googleCseId}
                  onChange={e => setGoogleCseId(e.target.value)}
                  placeholder="YOUR_CSE_ID"
                  className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-gray-200 outline-none focus:border-emerald-500 font-mono"
                />
              </div>
            )}
            {searchProvider === 'brave' && (
              <label className="flex items-center gap-3 text-sm text-gray-300">
                <input
                  type="checkbox"
                  checked={braveSummarizer}
                  onChange={e => setBraveSummarizer(e.target.checked)}
                  className="w-4 h-4 accent-emerald-500"
                />
                启用 Brave Summarizer
              </label>
            )}
            {searchProvider === 'serpapi' ? (
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-300">搜索引擎</label>
                  <select
                    value={searchEngine}
                    onChange={e => setSearchEngine(e.target.value as 'google' | 'baidu')}
                    className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-gray-200 outline-none focus:border-emerald-500"
                  >
                    <option value="google">Google</option>
                    <option value="baidu">Baidu</option>
                  </select>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-300">Top N</label>
                  <input
                    type="number"
                    min={3}
                    max={20}
                    value={searchNum}
                    onChange={e => setSearchNum(Math.max(3, Math.min(20, Number(e.target.value))))}
                    className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-gray-200 outline-none focus:border-emerald-500"
                  />
                </div>
              </div>
            ) : (
              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-300">Top N</label>
                <input
                  type="number"
                  min={3}
                  max={searchProvider === 'google_cse' ? 10 : 20}
                  value={searchNum}
                  onChange={e => {
                    const max = searchProvider === 'google_cse' ? 10 : 20;
                    setSearchNum(Math.max(3, Math.min(max, Number(e.target.value))));
                  }}
                  className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-gray-200 outline-none focus:border-emerald-500"
                />
              </div>
            )}
          </div>
        )}
      </div>

      <div className="mt-6">
        <button
          onClick={handleRun}
          disabled={loading}
          className="w-full bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white py-3 rounded-xl font-medium flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-emerald-500/20"
        >
          {loading ? <Loader2 size={18} className="animate-spin" /> : <Search size={18} />}
          {loading ? '研究中...' : '开始深度研究'}
        </button>
      </div>

      {error && <div className="mt-4 text-sm text-red-400">{error}</div>}

      {reportMarkdown && (
        <div className="mt-6 space-y-4">
          <div className="flex items-center justify-between">
            <div className="text-sm text-emerald-300 font-medium">研究报告</div>
            <button
              onClick={() => setShowReportModal(true)}
              className="flex items-center gap-1 text-xs text-emerald-200 hover:text-emerald-100"
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

      {searchResults.length > 0 && (
        <div className="mt-6 space-y-3">
          <div className="text-sm text-emerald-300 font-medium">检索结果</div>
          {searchResults.map((item, idx) => (
            <div key={`${item.url}-${idx}`} className="bg-white/5 border border-white/10 rounded-xl p-3">
              <div className="flex items-center justify-between mb-1">
                <div className="text-sm text-gray-200 truncate">{item.title}</div>
                <a
                  href={item.url}
                  target="_blank"
                  rel="noreferrer"
                  className="text-xs text-emerald-300 hover:text-emerald-200 flex items-center gap-1"
                >
                  <ExternalLink size={12} /> 打开
                </a>
              </div>
              {item.snippet && <div className="text-xs text-gray-500">{item.snippet}</div>}
              {item.source && <div className="text-[10px] text-gray-600 mt-1">{item.source}</div>}
            </div>
          ))}
        </div>
      )}

      {summaries.length > 0 && (
        <div className="mt-6 space-y-3">
          <div className="text-sm text-emerald-300 font-medium">Brave Summaries</div>
          {summaries.map((item, idx) => (
            <div key={`${item.query}-${idx}`} className="bg-white/5 border border-white/10 rounded-xl p-3 space-y-2">
              <div className="text-xs text-gray-400">Query: {item.query}</div>
              {renderSummaryContent(item.summary)}
            </div>
          ))}
        </div>
      )}

      <MarkdownViewerModal
        open={showReportModal}
        onClose={() => setShowReportModal(false)}
        markdown={reportMarkdown}
        title="深度研究报告"
      />
    </div>
  );
};
