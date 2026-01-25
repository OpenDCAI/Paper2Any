import { useState } from 'react';
import { KnowledgeFile } from './types';
import { FileText, Image, Video, Link as LinkIcon, Trash2, Search, Filter, X, Eye, Database, Loader2, CheckCircle } from 'lucide-react';
import { supabase } from '../../lib/supabase';
import { API_URL_OPTIONS } from '../../config/api';

interface LibraryViewProps {
  files: KnowledgeFile[];
  selectedIds: Set<string>;
  onToggleSelect: (id: string) => void;
  onGoToUpload: () => void;
  onRefresh: () => Promise<void>;
  onPreview: (file: KnowledgeFile) => void;
  onDelete: (file: KnowledgeFile) => void;
}

export const LibraryView = ({ files, selectedIds, onToggleSelect, onGoToUpload, onRefresh, onPreview, onDelete }: LibraryViewProps) => {
  const [filterType, setFilterType] = useState<'all' | 'embedded'>('all');
  const [isEmbedding, setIsEmbedding] = useState(false);
  
  // Embedding Config Modal
  const [showEmbedConfig, setShowEmbedConfig] = useState(false);
  const [embedConfig, setEmbedConfig] = useState({
      api_url: 'https://api.apiyi.com/v1/embeddings',
      api_key: '',
      model_name: 'text-embedding-3-small',
      image_model: 'gemini-2.5-flash',
      video_model: 'gemini-2.5-flash'
  });

  const handleDelete = async (file: KnowledgeFile, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm(`Delete ${file.name}?`)) return;

    try {
      const { error } = await supabase
        .from('knowledge_base_files')
        .delete()
        .eq('id', file.id);

      if (error) throw error;
      
      onRefresh();
    } catch (err) {
      console.error('Delete error:', err);
      alert('Delete failed');
    }
  };

  const handleBulkDelete = async () => {
    if (selectedIds.size === 0) return;
    if (!confirm(`Delete ${selectedIds.size} selected files?`)) return;

    try {
      const { error } = await supabase
        .from('knowledge_base_files')
        .delete()
        .in('id', Array.from(selectedIds));

      if (error) throw error;
      
      onRefresh();
    } catch (err) {
      console.error('Bulk delete error:', err);
      alert('Delete failed');
    }
  };

  const startEmbeddingProcess = async () => {
    if (selectedIds.size === 0) return;
    setShowEmbedConfig(false);
    setIsEmbedding(true);
    try {
        const fileIds = Array.from(selectedIds);
        
        // Prepare data for backend
        const filesToProcess = files
            .filter(f => selectedIds.has(f.id))
            .map(f => ({
                path: f.url,
                description: f.desc
            }));

        // Call Real API
        const res = await fetch('/api/v1/kb/embedding', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-API-Key': 'df-internal-2024-workflow-key'
          },
          body: JSON.stringify({ 
              files: filesToProcess,
              api_url: embedConfig.api_url,
              api_key: embedConfig.api_key,
              model_name: embedConfig.model_name,
              image_model: embedConfig.image_model,
              video_model: embedConfig.video_model
          })
        });
        
        if (!res.ok) throw new Error("Embedding failed");
        
        // Update DB locally to reflect change
        const { error } = await supabase
            .from('knowledge_base_files')
            .update({ is_embedded: true })
            .in('id', fileIds);

        if (error) throw error;

        await onRefresh();
        // Switch to embedded view to show results
        setFilterType('embedded');
        alert("Files successfully embedded!");
        
    } catch (err) {
        console.error("Embedding error:", err);
        alert("Failed to start embedding process");
    } finally {
        setIsEmbedding(false);
    }
  };

  const getIcon = (type: string) => {
    switch (type) {
      case 'doc': return <FileText size={20} className="text-blue-400" />;
      case 'image': return <Image size={20} className="text-purple-400" />;
      case 'video': return <Video size={20} className="text-pink-400" />;
      case 'link': return <LinkIcon size={20} className="text-green-400" />;
      default: return <FileText size={20} className="text-gray-400" />;
    }
  };

  const filteredFiles = files.filter(file => {
      if (filterType === 'embedded') return file.isEmbedded;
      return true;
  });

  return (
    <div className="h-full flex flex-col relative">
      {/* Tabs */}
      <div className="flex items-center gap-6 mb-6 border-b border-white/10 pb-1">
          <button 
            onClick={() => setFilterType('all')}
            className={`pb-3 text-sm font-medium transition-all relative ${
                filterType === 'all' ? 'text-white' : 'text-gray-400 hover:text-gray-300'
            }`}
          >
              全部文件
              {filterType === 'all' && <div className="absolute bottom-0 left-0 w-full h-0.5 bg-purple-500 rounded-full" />}
          </button>
          <button 
            onClick={() => setFilterType('embedded')}
            className={`pb-3 text-sm font-medium transition-all relative ${
                filterType === 'embedded' ? 'text-white' : 'text-gray-400 hover:text-gray-300'
            }`}
          >
              向量入库文件
              {filterType === 'embedded' && <div className="absolute bottom-0 left-0 w-full h-0.5 bg-purple-500 rounded-full" />}
          </button>
      </div>

      {/* Toolbar */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-4 flex-1">
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" size={16} />
            <input 
              type="text" 
              placeholder="Search files..." 
              className="w-full pl-10 pr-4 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-gray-200 outline-none focus:border-purple-500/50"
            />
          </div>
          <button className="p-2 text-gray-400 hover:text-white bg-white/5 rounded-lg border border-white/10">
            <Filter size={18} />
          </button>
          
        </div>
        <button 
          onClick={onGoToUpload}
          className="px-4 py-2 bg-purple-600 hover:bg-purple-500 text-white rounded-lg text-sm font-medium transition-colors"
        >
          + Upload
        </button>
      </div>

      {/* Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 overflow-y-auto pb-20 flex-1">
        {filteredFiles.map(file => (
          <div 
            key={file.id}
            onClick={() => onPreview(file)}
            className={`group relative p-4 rounded-xl border transition-all cursor-pointer ${
              selectedIds.has(file.id) 
                ? 'bg-purple-500/10 border-purple-500/50' 
                : 'bg-white/5 border-white/10 hover:border-white/20 hover:bg-white/10'
            }`}
          >
            <div className="flex items-start justify-between mb-3">
              <div className="p-2 bg-black/20 rounded-lg relative">
                {getIcon(file.type)}
                {file.isEmbedded && (
                    <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full border border-[#0a0a1a]" title="Embedded"></div>
                )}
              </div>
              <div 
                onClick={(e) => { e.stopPropagation(); onToggleSelect(file.id); }}
                className={`w-5 h-5 rounded-full border flex items-center justify-center transition-colors cursor-pointer hover:border-purple-400 ${
                  selectedIds.has(file.id) ? 'bg-purple-500 border-purple-500' : 'border-white/20'
                }`}
              >
                {selectedIds.has(file.id) && <div className="w-2 h-2 bg-white rounded-full" />}
              </div>
            </div>

            <h3 className="text-sm font-medium text-gray-200 truncate mb-1" title={file.name}>
              {file.name}
            </h3>
            
            <div className="flex items-center justify-between text-xs text-gray-500 mt-2">
              <span>{file.size}</span>
              <span>{file.uploadTime.split(' ')[0]}</span>
            </div>

            {/* Hover Actions */}
            <div className="absolute bottom-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-1">
               <button 
                 onClick={(e) => handleDelete(file, e)}
                 className="p-1.5 bg-red-500/20 text-red-400 rounded-lg hover:bg-red-500 hover:text-white shadow-lg"
                 title="Delete file"
               >
                 <Trash2 size={14} />
               </button>
            </div>
          </div>
        ))}
      </div>
      
      {/* Bottom Bar for Vector Embedding */}
      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-20">
          <button
            onClick={() => setShowEmbedConfig(true)}
            disabled={selectedIds.size === 0 || isEmbedding}
            className={`px-6 py-3 rounded-full font-medium shadow-xl backdrop-blur-md border border-white/10 transition-all flex items-center gap-2 ${
                selectedIds.size > 0 
                ? 'bg-gradient-to-r from-purple-600 to-blue-600 text-white hover:scale-105' 
                : 'bg-black/40 text-gray-500 cursor-not-allowed'
            }`}
          >
              {isEmbedding ? (
                  <>
                    <Loader2 className="animate-spin" size={18} />
                    Processing...
                  </>
              ) : (
                  <>
                    <Database size={18} />
                    向量入库 {selectedIds.size > 0 ? `(${selectedIds.size})` : ''}
                  </>
              )}
          </button>
      </div>

      {/* Config Modal */}
      {showEmbedConfig && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={() => setShowEmbedConfig(false)}>
            <div className="bg-[#0a0a1a] border border-white/10 rounded-xl p-6 w-full max-w-md shadow-2xl" onClick={e => e.stopPropagation()}>
                <h3 className="text-lg font-medium text-white mb-4 flex items-center gap-2">
                    <Database className="text-purple-500" />
                    Embedding 配置
                </h3>
                
                <div className="space-y-4">
                    <div>
                        <label className="block text-xs text-gray-400 mb-1">API URL</label>
                        <select 
                            value={embedConfig.api_url} 
                            onChange={e => {
                                const val = e.target.value;
                                setEmbedConfig({...embedConfig, api_url: val});
                            }}
                            className="w-full bg-black/20 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:border-purple-500/50 outline-none"
                        >
                            {API_URL_OPTIONS.map((url: string) => (
                                <option key={url} value={url}>{url}</option>
                            ))}
                        </select>
                    </div>
                    <div>
                        <label className="block text-xs text-gray-400 mb-1">API Key</label>
                        <input 
                            type="password" 
                            value={embedConfig.api_key}
                            onChange={e => setEmbedConfig({...embedConfig, api_key: e.target.value})}
                            placeholder="sk-..."
                            className="w-full bg-black/20 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:border-purple-500/50 outline-none"
                        />
                    </div>
                    <div>
                        <label className="block text-xs text-gray-400 mb-1">Model Name (Embedding)</label>
                        <input 
                            type="text" 
                            value={embedConfig.model_name}
                            onChange={e => setEmbedConfig({...embedConfig, model_name: e.target.value})}
                            className="w-full bg-black/20 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:border-purple-500/50 outline-none"
                        />
                    </div>
                    <div>
                        <label className="block text-xs text-gray-400 mb-1">Image Model</label>
                        <input 
                            type="text" 
                            value={embedConfig.image_model}
                            onChange={e => setEmbedConfig({...embedConfig, image_model: e.target.value})}
                            placeholder="e.g. gemini-2.5-flash"
                            className="w-full bg-black/20 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:border-purple-500/50 outline-none"
                        />
                    </div>
                    <div>
                        <label className="block text-xs text-gray-400 mb-1">Video Model</label>
                        <input 
                            type="text" 
                            value={embedConfig.video_model}
                            onChange={e => setEmbedConfig({...embedConfig, video_model: e.target.value})}
                            placeholder="e.g. gemini-2.5-flash"
                            className="w-full bg-black/20 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:border-purple-500/50 outline-none"
                        />
                    </div>
                </div>

                <div className="flex justify-end gap-3 mt-6">
                    <button 
                        onClick={() => setShowEmbedConfig(false)}
                        className="px-4 py-2 text-sm text-gray-400 hover:text-white transition-colors"
                    >
                        取消
                    </button>
                    <button 
                        onClick={startEmbeddingProcess}
                        className="px-4 py-2 bg-purple-600 hover:bg-purple-500 text-white rounded-lg text-sm font-medium transition-colors"
                    >
                        开始入库
                    </button>
                </div>
            </div>
        </div>
      )}
    </div>
  );
};
