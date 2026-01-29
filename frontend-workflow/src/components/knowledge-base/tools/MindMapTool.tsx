import { useState } from 'react';
import { BrainCircuit, Loader2, CheckCircle2, X } from 'lucide-react';
import { API_KEY, API_URL_OPTIONS } from '../../../config/api';
import { KnowledgeFile } from '../types';
import { MermaidPreview } from './MermaidPreview';

interface MindMapToolProps {
  files: KnowledgeFile[];
  selectedIds: Set<string>;
  onGenerateSuccess: (file: KnowledgeFile) => void;
}

export const MindMapTool = ({ files = [], selectedIds, onGenerateSuccess }: MindMapToolProps) => {
  const [mindmapGenerating, setMindmapGenerating] = useState(false);
  const [generatedMermaidCode, setGeneratedMermaidCode] = useState('');
  const [showPreview, setShowPreview] = useState(false);
  const [mindmapParams, setMindmapParams] = useState({
    api_key: '',
    api_url: 'https://api.apiyi.com/v1',
    model: 'gpt-4o',
    mindmap_style: 'default',
    max_depth: 3,
    language: 'zh'
  });

  const handleGenerateMindMap = async () => {
    if (selectedIds.size === 0) {
      alert('请至少选择一个文件进行思维导图生成。');
      return;
    }

    if (!mindmapParams.api_key) {
      alert('请输入 API Key');
      return;
    }

    // Get selected file paths
    const selectedFiles = (files || []).filter(f => selectedIds.has(f.id));
    const filePaths = selectedFiles.map(f => f.url).filter(url => url);

    if (filePaths.length === 0) {
      alert('无法获取文件路径，请重新上传文件。');
      return;
    }

    setMindmapGenerating(true);
    setShowPreview(false);
    try {
      const res = await fetch('/api/v1/kb/generate-mindmap', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': API_KEY
        },
        body: JSON.stringify({
          file_paths: filePaths,
          user_id: 'user_id_placeholder',
          email: 'user@example.com',
          api_url: mindmapParams.api_url,
          api_key: mindmapParams.api_key,
          model: mindmapParams.model,
          mindmap_style: mindmapParams.mindmap_style,
          max_depth: mindmapParams.max_depth,
          language: mindmapParams.language
        })
      });

      if (!res.ok) {
        const errorText = await res.text();
        throw new Error('生成失败: ' + errorText);
      }

      const data = await res.json();

      if (data.success && data.mermaid_code) {
        setGeneratedMermaidCode(data.mermaid_code);
        setShowPreview(true);
        alert('思维导图生成成功！');

        onGenerateSuccess({
          id: data.output_file_id || 'o' + Date.now(),
          name: `mindmap_${Date.now()}.mmd`,
          type: 'doc',
          size: '未知',
          uploadTime: new Date().toLocaleString(),
          url: data.result_path,
          desc: `MindMap from ${selectedFiles.length} file(s)`
        });
      } else {
        throw new Error('生成失败');
      }

    } catch (e: any) {
      alert('Error: ' + e.message);
    } finally {
      setMindmapGenerating(false);
    }
  };

  const selectedFileNames = (files || [])
    .filter(f => selectedIds.has(f.id))
    .map(f => f.name)
    .join(', ');

  return (
    <div className="flex-1 overflow-y-auto p-6 bg-[#0a0a1a] h-full">
      <div className="mb-6 bg-gradient-to-br from-cyan-900/20 to-blue-900/20 border border-cyan-500/20 rounded-xl p-4 flex items-start gap-3">
        <BrainCircuit className="text-cyan-400 mt-1 flex-shrink-0" size={18} />
        <div>
          <h4 className="text-sm font-medium text-cyan-300 mb-1">思维导图生成</h4>
          <p className="text-xs text-cyan-200/70">
            支持选择多个文件。AI 将分析文档内容并生成 Mermaid 格式的思维导图。
          </p>
        </div>
      </div>

      <div className="space-y-6">
        {/* Context Info */}
        <div>
          <label className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2 block">
            当前选中素材 ({selectedIds.size} 个文件)
          </label>
          <div className="bg-white/5 border border-white/10 rounded-lg p-3 text-sm text-gray-300 flex items-center justify-between">
            <span className="truncate">{selectedIds.size > 0 ? selectedFileNames : '未选择'}</span>
            {selectedIds.size > 0 ? <CheckCircle2 size={16} className="text-green-500" /> : <X size={16} className="text-red-500" />}
          </div>
        </div>

        {/* Configuration */}
        <div className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-300">API Key</label>
            <input
              type="password"
              value={mindmapParams.api_key}
              onChange={e => setMindmapParams({...mindmapParams, api_key: e.target.value})}
              placeholder="sk-..."
              className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-gray-200 outline-none focus:border-cyan-500 font-mono"
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-300">API URL</label>
            <select
              value={mindmapParams.api_url}
              onChange={e => setMindmapParams({...mindmapParams, api_url: e.target.value})}
              className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-gray-200 outline-none focus:border-cyan-500"
            >
              {API_URL_OPTIONS.map((url: string) => (
                <option key={url} value={url}>{url}</option>
              ))}
            </select>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-300">LLM Model</label>
            <select
              value={mindmapParams.model}
              onChange={e => setMindmapParams({...mindmapParams, model: e.target.value})}
              className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-gray-200 outline-none focus:border-cyan-500"
            >
              <option value="gpt-4o">gpt-4o</option>
              <option value="gpt-5.1">gpt-5.1</option>
              <option value="gpt-5.2">gpt-5.2</option>
              <option value="gemini-3-pro-preview">gemini-3-pro-preview</option>
            </select>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-300">思维导图风格</label>
            <select
              value={mindmapParams.mindmap_style}
              onChange={e => setMindmapParams({...mindmapParams, mindmap_style: e.target.value})}
              className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-gray-200 outline-none focus:border-cyan-500"
            >
              <option value="default">默认风格</option>
              <option value="flowchart">流程图风格</option>
              <option value="tree">树形结构</option>
            </select>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-300">最大深度: <span className="text-cyan-400">{mindmapParams.max_depth}</span></label>
            <input
              type="range"
              min="2" max="5"
              value={mindmapParams.max_depth}
              onChange={e => setMindmapParams({...mindmapParams, max_depth: parseInt(e.target.value)})}
              className="w-full accent-cyan-500 h-1 bg-white/10 rounded-lg appearance-none cursor-pointer"
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-300">目标语言</label>
            <div className="grid grid-cols-2 gap-3">
              <button
                onClick={() => setMindmapParams({...mindmapParams, language: 'zh'})}
                className={`py-2.5 rounded-lg border text-sm transition-all ${
                  mindmapParams.language === 'zh'
                    ? 'bg-cyan-500/20 border-cyan-500 text-cyan-300'
                    : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'
                }`}
              >
                中文
              </button>
              <button
                onClick={() => setMindmapParams({...mindmapParams, language: 'en'})}
                className={`py-2.5 rounded-lg border text-sm transition-all ${
                  mindmapParams.language === 'en'
                    ? 'bg-cyan-500/20 border-cyan-500 text-cyan-300'
                    : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'
                }`}
              >
                English
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-8 pb-8">
        <button
          onClick={handleGenerateMindMap}
          disabled={mindmapGenerating || selectedIds.size === 0}
          className="w-full bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white py-3.5 rounded-xl font-medium flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-cyan-500/20 transition-all transform active:scale-95"
        >
          {mindmapGenerating ? <Loader2 size={18} className="animate-spin" /> : <BrainCircuit size={18} />}
          {mindmapGenerating ? '正在生成思维导图...' : '开始生成思维导图'}
        </button>
      </div>

      {/* Preview Section */}
      {showPreview && generatedMermaidCode && (
        <MermaidPreview mermaidCode={generatedMermaidCode} />
      )}
    </div>
  );
};
