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

  const handleGeneratePPT = async () => {
    if (selectedIds.size !== 1) {
      alert('演示版本：请选择且仅选择一个 PDF 文档进行生成。');
      return;
    }
    const fileId = Array.from(selectedIds)[0];
    const file = files.find(f => f.id === fileId);
    
    if (!file || file.type !== 'doc' || (!file.name.endsWith('.pdf') && !file.name.endsWith('.PDF'))) {
       alert('演示版本：仅支持 PDF 文件生成 PPT。');
       return;
    }

    if (!pptParams.api_key) {
      alert('请输入 API Key');
      return;
    }

    setPptGenerating(true);
    try {
      const formData = new FormData();
      if (file.file) {
        formData.append('pdf_file', file.file);
      } else {
        alert('无法获取原始文件（可能是示例文件），请上传一个新的 PDF 测试。');
        setPptGenerating(false);
        return;
      }
      
      const getStyleDescription = (preset: string): string => {
        const styles: Record<string, string> = {
          modern: '现代简约风格，使用干净的线条和充足的留白',
          business: '商务专业风格，稳重大气，适合企业演示',
          academic: '学术报告风格，清晰的层次结构，适合论文汇报',
          creative: '创意设计风格，活泼生动，色彩丰富',
        };
        return styles[preset] || styles.modern;
      };

      formData.append('chat_api_url', pptParams.api_url);
      formData.append('api_key', pptParams.api_key);
      formData.append('language', pptParams.language);
      formData.append('style', getStyleDescription(pptParams.style_preset));
      formData.append('page_count', pptParams.page_count.toString());
      formData.append('use_ai_edit', 'true');
      formData.append('model', pptParams.model);
      formData.append('img_gen_model_name', pptParams.gen_fig_model);

      const res = await fetch('/api/v1/pdf2ppt/generate', {
        method: 'POST',
        headers: { 'X-API-Key': API_KEY },
        body: formData,
      });

      if (!res.ok) {
        throw new Error('生成失败: ' + res.statusText);
      }

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = file.name.replace('.pdf', '.pptx');
      document.body.appendChild(a);
      a.click();
      a.remove();
      
      onGenerateSuccess({
        id: 'o' + Date.now(),
        name: file.name.replace('.pdf', '.pptx'),
        type: 'doc',
        size: '2MB', 
        uploadTime: new Date().toLocaleString(),
        desc: 'Generated from ' + file.name
      });

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
            仅支持选择 1 个 PDF 文档。AI 将自动分析文档结构并生成演示文稿。
          </p>
        </div>
      </div>

      <div className="space-y-6">
        {/* Context Info */}
        <div>
          <label className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2 block">当前选中素材</label>
          <div className="bg-white/5 border border-white/10 rounded-lg p-3 text-sm text-gray-300 flex items-center justify-between">
            <span className="truncate">{selectedIds.size === 1 ? files.find(f => f.id === Array.from(selectedIds)[0])?.name : '未选择或选择了多个'}</span>
            {selectedIds.size === 1 ? <CheckCircle2 size={16} className="text-green-500" /> : <X size={16} className="text-red-500" />}
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
          disabled={pptGenerating || selectedIds.size !== 1}
          className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white py-3.5 rounded-xl font-medium flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-purple-500/20 transition-all transform active:scale-95"
        >
          {pptGenerating ? <Loader2 size={18} className="animate-spin" /> : <Presentation size={18} />}
          {pptGenerating ? '正在生成演示文稿...' : '开始生成 PPT'}
        </button>
      </div>
    </div>
  );
};
