import { useState } from 'react';
import { KnowledgeFile } from './types';
import { FileText, Image, Video, Link as LinkIcon, Trash2, Search, Filter, X, Eye } from 'lucide-react';
import { supabase } from '../../lib/supabase';

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
      // Clear selection is handled by parent or we should assume parent will clear or we keep selectedIds as is but they are gone.
      // Ideally we should call a clearSelection callback but we don't have one in props directly, 
      // but onRefresh will update the list.
    } catch (err) {
      console.error('Bulk delete error:', err);
      alert('Delete failed');
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

  return (
    <div className="h-full flex flex-col relative">
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
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 overflow-y-auto pb-20">
        {files.map(file => (
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
              <div className="p-2 bg-black/20 rounded-lg">
                {getIcon(file.type)}
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
    </div>
  );
};
