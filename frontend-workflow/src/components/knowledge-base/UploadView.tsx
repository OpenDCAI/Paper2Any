import React, { useState, useCallback } from 'react';
import { UploadCloud, Link as LinkIcon, FileText, X, Loader2 } from 'lucide-react';
import { useAuthStore } from '../../stores/authStore';
import { supabase } from '../../lib/supabase';

interface UploadViewProps {
  onSuccess: () => void;
  onUploadFile?: (files: any[], type: any) => void; // Legacy support
  onProcessLinks?: (links: any[]) => void; // Legacy support
  isUploading?: boolean; // Legacy support
}

export const UploadView = ({ onSuccess }: UploadViewProps) => {
  const { user } = useAuthStore();
  const [activeTab, setActiveTab] = useState<'file' | 'link'>('file');
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [urls, setUrls] = useState('');

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFiles(e.dataTransfer.files);
    }
  }, []);

  const handleFiles = async (fileList: FileList) => {
    if (!user) return;
    setUploading(true);

    try {
      const promises = Array.from(fileList).map(async (file) => {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('email', user.email || '');
        formData.append('user_id', user.id);

        // 1. Upload to Backend
        const res = await fetch('/api/kb/upload', {
          method: 'POST',
          headers: {
            'X-API-Key': 'df-internal-2024-workflow-key'
          },
          body: formData,
          credentials: 'include',
        });

        if (!res.ok) throw new Error('Upload failed');
        const data = await res.json();

        // 2. Save to DB
        const { error } = await supabase.from('knowledge_base_files').insert({
          user_id: user.id,
          user_email: user.email,
          file_name: data.filename,
          file_type: data.file_type || file.type,
          file_size: data.file_size,
          storage_path: data.static_url
        });

        if (error) throw error;
      });

      await Promise.all(promises);
      onSuccess();
    } catch (err) {
      console.error(err);
      alert('Upload failed');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto">
      <div className="flex justify-center gap-4 mb-8">
        <button
          onClick={() => setActiveTab('file')}
          className={`px-6 py-2 rounded-full text-sm font-medium transition-all ${
            activeTab === 'file' 
              ? 'bg-purple-600 text-white' 
              : 'bg-white/5 text-gray-400 hover:bg-white/10'
          }`}
        >
          <div className="flex items-center gap-2">
            <UploadCloud size={16} />
            Upload Files
          </div>
        </button>
        <button
          onClick={() => setActiveTab('link')}
          className={`px-6 py-2 rounded-full text-sm font-medium transition-all ${
            activeTab === 'link' 
              ? 'bg-purple-600 text-white' 
              : 'bg-white/5 text-gray-400 hover:bg-white/10'
          }`}
        >
          <div className="flex items-center gap-2">
            <LinkIcon size={16} />
            Import Links
          </div>
        </button>
      </div>

      {activeTab === 'file' ? (
        <div
          className={`border-2 border-dashed rounded-xl p-12 flex flex-col items-center justify-center transition-all ${
            dragActive 
              ? 'border-purple-500 bg-purple-500/10' 
              : 'border-white/10 bg-white/5 hover:border-white/20'
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <div className="w-16 h-16 bg-white/5 rounded-full flex items-center justify-center mb-4">
            {uploading ? (
              <Loader2 className="animate-spin text-purple-400" size={32} />
            ) : (
              <UploadCloud className="text-purple-400" size={32} />
            )}
          </div>
          <p className="text-white font-medium mb-2">
            {uploading ? 'Uploading...' : 'Drag & drop files here'}
          </p>
          <p className="text-gray-500 text-sm mb-6">
            Supported formats: PDF, DOCX, PPTX, PNG, JPG, MP4
          </p>
          
          <input
            type="file"
            multiple
            id="file-upload"
            className="hidden"
            accept=".pdf,.docx,.pptx,.png,.jpg,.jpeg,.mp4"
            onChange={(e) => e.target.files && handleFiles(e.target.files)}
            disabled={uploading}
          />
          <label
            htmlFor="file-upload"
            className={`px-6 py-2.5 bg-white text-black font-medium rounded-lg hover:bg-gray-100 transition-colors cursor-pointer ${uploading ? 'opacity-50 pointer-events-none' : ''}`}
          >
            Browse Files
          </label>
        </div>
      ) : (
        <div className="bg-white/5 border border-white/10 rounded-xl p-6">
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Enter URLs (one per line)
          </label>
          <textarea
            value={urls}
            onChange={(e) => setUrls(e.target.value)}
            className="w-full h-32 bg-black/20 border border-white/10 rounded-lg p-3 text-white text-sm outline-none focus:border-purple-500/50 resize-none mb-4"
            placeholder="https://example.com/article&#10;https://example.com/paper"
          />
          <div className="flex justify-end">
            <button
              onClick={() => {
                /* Link processing logic would go here */
                alert('Link processing not implemented yet');
              }}
              className="px-6 py-2 bg-purple-600 hover:bg-purple-500 text-white rounded-lg text-sm font-medium transition-colors"
            >
              Process Links
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
