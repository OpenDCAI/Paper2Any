import { useState } from 'react';
import { Presentation, Loader2, CheckCircle2, X } from 'lucide-react';
import { API_KEY, API_URL_OPTIONS } from '../../../config/api';
import { KnowledgeFile } from '../types';

interface PptToolProps {
  files: KnowledgeFile[];
  selectedIds: Set<string>;
  onGenerateSuccess: (file: KnowledgeFile) => void;
}

export const PptTool = ({ files, selectedIds, onGenerateSuccess }: PptToolProps) => {
  const [pptGenerating, setPptGenerating] = useState(false);
  const [pptParams, setPptParams] = useState({
    api_key: '',
    api_url: 'https://api.apiyi.com/v1',
    style_preset: 'modern',
    language: 'zh',
    page_count: 10,
    model: 'gpt-5.1',
    gen_fig_model: 'gemini-2.5-flash-image'
  });
  const [query, setQuery] = useState('');
  const [needEmbedding, setNeedEmbedding] = useState(false);

  const handleGeneratePPT = async () => {
    const selectedFiles = files.filter(f => selectedIds.has(f.id));
    const docFiles = selectedFiles.filter(f => f.type === 'doc');
    const imageFiles = selectedFiles.filter(f => f.type === 'image');
    const validDocFiles = docFiles.filter(f => {
      const name = f.name.toLowerCase();
      return name.endsWith('.pdf') || name.endsWith('.pptx') || name.endsWith('.ppt') || name.endsWith('.docx') || name.endsWith('.doc');
    });
    const invalidDocFiles = docFiles.filter(f => !validDocFiles.includes(f));

    if (validDocFiles.length === 0) {
      alert('请至少选择 1 个 PDF/PPTX/DOCX 文档进行生成。');
      return;
    }
    if (invalidDocFiles.length > 0) {
      alert('当前仅支持 PDF/PPTX/DOCX 文档。');
      return;
    }

    if (!pptParams.api_key) {
      alert('请输入 API Key');
      return;
    }

    const docPaths = validDocFiles.map(f => f.url).filter(Boolean) as string[];
    const imageItems = imageFiles
      .map(f => ({ path: f.url, description: f.desc || '' }))
      .filter(item => Boolean(item.path));

    if (docPaths.length !== validDocFiles.length) {
      alert('无法获取文档路径，请重新上传文件。');
      return;
    }

    setPptGenerating(true);
    try {
      
      const getStyleDescription = (preset: string): string => {
        const styles: Record<string, string> = {
          modern: '现代简约风格，使用干净的线条和充足的留白',
          business: '商务专业风格，稳重大气，适合企业演示',
          academic: '学术报告风格，清晰的层次结构，适合论文汇报',
          creative: '创意设计风格，活泼生动，色彩丰富',
        };
        return styles[preset] || styles.modern;
      };

      const res = await fetch('/api/v1/kb/generate-ppt', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': API_KEY
        },
        body: JSON.stringify({
          file_paths: docPaths,
          image_items: imageItems,
          query: query.trim(),
          need_embedding: needEmbedding,
          user_id: 'user_id_placeholder',
          email: 'user@example.com',
          api_url: pptParams.api_url,
          api_key: pptParams.api_key,
          style: getStyleDescription(pptParams.style_preset),
          language: pptParams.language,
          page_count: pptParams.page_count,
          model: pptParams.model,
          gen_fig_model: pptParams.gen_fig_model
        })
      });

      if (!res.ok) {
        const errorText = await res.text();
        throw new Error('生成失败: ' + errorText);
      }

      const data = await res.json();

      if (data.success) {
        alert('PPT 生成成功！');

        onGenerateSuccess({
          id: data.output_file_id || 'o' + Date.now(),
          name: `kb_ppt_${Date.now()}.pptx`,
          type: 'doc',
          size: '未知',
          uploadTime: new Date().toLocaleString(),
          url: data.pdf_path || data.pptx_path,
          desc: `Generated PPT from ${validDocFiles.length} doc(s)${imageFiles.length ? ` + ${imageFiles.length} image(s)` : ''}`
        });
      } else {
        throw new Error('生成失败');
      }

    } catch (e: any) {
      alert('Error: ' + e.message);
    } finally {
      setPptGenerating(false);
    }
  };

  return (
    <div className="flex-1 overflow-y-auto p-6 bg-[#0a0a1a] h-full">
      <div className="mb-6 bg-gradient-to-br from-purple-900/20 to-pink-900/20 border border-purple-500/20 rounded-xl p-4 flex items-start gap-3">
        <Presentation className="text-purple-400 mt-1 flex-shrink-0" size={18} />
        <div>
          <h4 className="text-sm font-medium text-purple-300 mb-1">PPT 生成助手</h4>
          <p className="text-xs text-purple-200/70">
            支持选择多个 PDF/PPTX/DOCX 文档，并可附带图片素材。AI 将自动分析文档结构并生成演示文稿。
          </p>
        </div>
      </div>

      <div className="space-y-6">
        {/* Context Info */}
        <div>
          <label className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2 block">当前选中素材</label>
          <div className="bg-white/5 border border-white/10 rounded-lg p-3 text-sm text-gray-300 flex items-center justify-between">
            <span className="truncate">{selectedIds.size > 0 ? `${selectedIds.size} 个文件` : '未选择'}</span>
            {selectedIds.size > 0 ? <CheckCircle2 size={16} className="text-green-500" /> : <X size={16} className="text-red-500" />}
          </div>
        </div>

        {/* Configuration */}
        <div className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-300">API Key</label>
            <input 
              type="password" 
              value={pptParams.api_key}
              onChange={e => setPptParams({...pptParams, api_key: e.target.value})}
              placeholder="sk-..."
              className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-gray-200 outline-none focus:border-purple-500 font-mono"
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-300">主题 / Query（可为空）</label>
            <input
              type="text"
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder="例如：模型贡献与实验结果"
              className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-gray-200 outline-none focus:border-purple-500"
            />
          </div>

          <label className="flex items-center gap-3 text-sm text-gray-300">
            <input
              type="checkbox"
              checked={needEmbedding}
              onChange={e => setNeedEmbedding(e.target.checked)}
              className="w-4 h-4 accent-purple-500"
            />
            需要向量入库并基于检索生成大纲
          </label>

          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-300">API URL</label>
            <select 
              value={pptParams.api_url} 
              onChange={e => {
                const val = e.target.value;
                setPptParams(prev => ({
                  ...prev, 
                  api_url: val,
                  // Auto-switch gen model if using specific endpoint
                  gen_fig_model: val.includes('123.129.219.111') ? 'gemini-3-pro-image-preview' : prev.gen_fig_model
                }));
              }}
              className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-gray-200 outline-none focus:border-purple-500"
            >
              {API_URL_OPTIONS.map((url: string) => (
                <option key={url} value={url}>{url}</option>
              ))}
            </select>
          </div>

          <div className="grid grid-cols-2 gap-3">
             <div className="space-y-2">
              <label className="text-sm font-medium text-gray-300">Model</label>
              <select 
                value={pptParams.model} 
                onChange={e => setPptParams({...pptParams, model: e.target.value})}
                className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-gray-200 outline-none focus:border-purple-500"
              >
                <option value="gpt-4o">gpt-4o</option>
                <option value="gpt-5.1">gpt-5.1</option>
                <option value="gpt-5.2">gpt-5.2</option>
                <option value="gemini-3-pro-preview">gemini-3-pro-preview</option>
              </select>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-300">Image Model</label>
              <select
                value={pptParams.gen_fig_model}
                onChange={e => setPptParams({...pptParams, gen_fig_model: e.target.value})}
                disabled={pptParams.api_url === 'http://123.129.219.111:3000/v1'}
                className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-gray-200 outline-none focus:border-purple-500 disabled:opacity-50"
              >
                <option value="gemini-2.5-flash-image">Gemini 2.5</option>
                <option value="gemini-3-pro-image-preview">Gemini 3 Pro</option>
              </select>
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-300">风格偏好</label>
            <select
              value={pptParams.style_preset}
              onChange={e => setPptParams({...pptParams, style_preset: e.target.value})}
              className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-gray-200 outline-none focus:border-purple-500"
            >
              <option value="modern">现代简约风格</option>
              <option value="business">商务专业风格</option>
              <option value="academic">学术报告风格</option>
              <option value="creative">创意设计风格</option>
            </select>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-300">生成页数: <span className="text-purple-400">{pptParams.page_count}</span></label>
            <input 
              type="range" 
              min="5" max="20" 
              value={pptParams.page_count}
              onChange={e => setPptParams({...pptParams, page_count: parseInt(e.target.value)})}
              className="w-full accent-purple-500 h-1 bg-white/10 rounded-lg appearance-none cursor-pointer"
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-300">目标语言</label>
            <div className="grid grid-cols-2 gap-3">
              <button
                onClick={() => setPptParams({...pptParams, language: 'zh'})}
                className={`py-2.5 rounded-lg border text-sm transition-all ${
                  pptParams.language === 'zh'
                    ? 'bg-purple-500/20 border-purple-500 text-purple-300'
                    : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'
                }`}
              >
                中文
              </button>
              <button
                onClick={() => setPptParams({...pptParams, language: 'en'})}
                className={`py-2.5 rounded-lg border text-sm transition-all ${
                  pptParams.language === 'en'
                    ? 'bg-purple-500/20 border-purple-500 text-purple-300'
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
          onClick={handleGeneratePPT}
          disabled={pptGenerating || selectedIds.size === 0}
          className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white py-3.5 rounded-xl font-medium flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-purple-500/20 transition-all transform active:scale-95"
        >
          {pptGenerating ? <Loader2 size={18} className="animate-spin" /> : <Presentation size={18} />}
          {pptGenerating ? '正在生成演示文稿...' : '开始生成 PPT'}
        </button>
      </div>
    </div>
  );
};
