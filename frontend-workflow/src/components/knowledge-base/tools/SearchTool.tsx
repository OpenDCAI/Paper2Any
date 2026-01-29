import { useState } from 'react';
import { Search, Loader2, FileText, Image as ImageIcon, Video as VideoIcon, ExternalLink } from 'lucide-react';
import { API_KEY, API_URL_OPTIONS } from '../../../config/api';
import { useAuthStore } from '../../../stores/authStore';

interface SearchResult {
  score: number;
  content: string;
  type: string;
  source_file: {
    id?: string;
    file_type?: string;
    original_path?: string;
    url?: string;
  };
  media?: {
    path?: string;
    url?: string;
  } | null;
  metadata?: Record<string, any>;
}

interface SearchToolProps {
  files?: any[];
  selectedIds?: Set<string>;
}

export const SearchTool = ({ files = [], selectedIds = new Set() }: SearchToolProps) => {
  const { user } = useAuthStore();
  const [query, setQuery] = useState('');
  const [topK, setTopK] = useState(5);
  const [apiKey, setApiKey] = useState('');
  const [apiUrl, setApiUrl] = useState('https://api.apiyi.com/v1/embeddings');
  const [modelName, setModelName] = useState('text-embedding-3-small');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [error, setError] = useState('');

  const getMediaKind = (url?: string) => {
    if (!url) return null;
    const lower = url.toLowerCase();
    if (lower.endsWith('.mp4')) return 'video';
    if (lower.endsWith('.png') || lower.endsWith('.jpg') || lower.endsWith('.jpeg')) return 'image';
    return 'file';
  };

  const handleSearch = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setError('');
    setResults([]);
    try {
      const selectedFileIds = files
        .filter((f: any) => selectedIds.has(f.id))
        .map((f: any) => f.kbFileId)
        .filter(Boolean);

      if (selectedIds.size > 0 && selectedFileIds.length === 0) {
        setError('所选文件尚未入库（kb_file_id 为空），请先向量入库后再检索。');
        setLoading(false);
        return;
      }

      const res = await fetch('/api/v1/kb/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': API_KEY
        },
        body: JSON.stringify({
          query: query.trim(),
          top_k: topK,
          email: user?.email || null,
          api_url: apiUrl,
          api_key: apiKey,
          model_name: modelName,
          file_ids: selectedFileIds.length > 0 ? selectedFileIds : null
        })
      });
      if (!res.ok) {
        throw new Error(await res.text());
      }
      const data = await res.json();
      setResults(data.results || []);
    } catch (err: any) {
      setError(err?.message || '检索失败');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex-1 overflow-y-auto p-6 bg-[#0a0a1a] h-full">
      <div className="mb-6 bg-gradient-to-br from-blue-900/20 to-purple-900/20 border border-blue-500/20 rounded-xl p-4 flex items-start gap-3">
        <Search className="text-blue-400 mt-1 flex-shrink-0" size={18} />
        <div>
          <h4 className="text-sm font-medium text-blue-300 mb-1">语义检索</h4>
          <p className="text-xs text-blue-200/70">
            输入 query 与 topK，返回知识库中的相关文本或多模态描述。
          </p>
        </div>
      </div>

      <div className="space-y-4">
        <div className="space-y-2">
          <label className="text-sm font-medium text-gray-300">查询内容</label>
          <input
            type="text"
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="例如：模型架构的关键贡献"
            className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-gray-200 outline-none focus:border-blue-500"
          />
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-300">TopK</label>
            <input
              type="number"
              min={1}
              max={50}
              value={topK}
              onChange={e => setTopK(Math.max(1, Math.min(50, Number(e.target.value))))}
              className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-gray-200 outline-none focus:border-blue-500"
            />
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-300">Embedding Model</label>
            <input
              type="text"
              value={modelName}
              onChange={e => setModelName(e.target.value)}
              className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-gray-200 outline-none focus:border-blue-500"
            />
          </div>
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium text-gray-300">API URL</label>
          <select
            value={apiUrl}
            onChange={e => setApiUrl(e.target.value)}
            className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-gray-200 outline-none focus:border-blue-500"
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
            className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-gray-200 outline-none focus:border-blue-500 font-mono"
          />
        </div>
      </div>

      <div className="mt-6">
        <button
          onClick={handleSearch}
          disabled={loading || !query.trim()}
          className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 text-white py-3 rounded-xl font-medium flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-blue-500/20"
        >
          {loading ? <Loader2 size={18} className="animate-spin" /> : <Search size={18} />}
          {loading ? '检索中...' : '开始检索'}
        </button>
      </div>

      {error && (
        <div className="mt-4 text-sm text-red-400">{error}</div>
      )}

      {results.length > 0 && (
        <div className="mt-6 space-y-4">
          {results.map((item, idx) => {
            const mediaKind = getMediaKind(item.media?.url || item.source_file?.url);
            const displayUrl = item.media?.url || item.source_file?.url;
            return (
              <div key={`${item.source_file?.id || idx}-${idx}`} className="bg-white/5 border border-white/10 rounded-xl p-4">
                <div className="flex items-center justify-between mb-2">
                  <div className="text-xs text-gray-400">Score: {item.score?.toFixed(4)}</div>
                  {displayUrl && (
                    <a
                      href={displayUrl}
                      target="_blank"
                      rel="noreferrer"
                      className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1"
                    >
                      <ExternalLink size={12} /> 打开素材
                    </a>
                  )}
                </div>

                <div className="text-sm text-gray-200 whitespace-pre-wrap mb-3">
                  {item.content}
                </div>

                <div className="flex items-center gap-2 text-xs text-gray-500 mb-3">
                  <FileText size={12} />
                  <span>{item.source_file?.original_path?.split('/').pop() || item.source_file?.id || 'unknown'}</span>
                </div>

                {mediaKind === 'image' && item.media?.url && (
                  <div className="w-full bg-black/30 rounded-lg overflow-hidden border border-white/10">
                    <img src={item.media.url} alt="media" className="w-full h-48 object-contain" />
                  </div>
                )}

                {mediaKind === 'video' && item.media?.url && (
                  <div className="w-full bg-black/30 rounded-lg overflow-hidden border border-white/10">
                    <video src={item.media.url} controls className="w-full h-48 object-contain" />
                  </div>
                )}

                {!item.media?.url && mediaKind && (
                  <div className="flex items-center gap-2 text-xs text-gray-500">
                    {mediaKind === 'image' ? <ImageIcon size={14} /> : <VideoIcon size={14} />}
                    <span>多模态描述结果</span>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
