import React, { useState, useCallback } from 'react';
import { UploadCloud, Link as LinkIcon, FileText, X, Loader2, Trash2, CheckCircle } from 'lucide-react';
import { useAuthStore } from '../../stores/authStore';
import { supabase } from '../../lib/supabase';

interface UploadViewProps {
  onSuccess: () => void;
  onUploadFile?: (files: any[], type: any) => void; // Legacy support
  onProcessLinks?: (links: any[]) => void; // Legacy support
  isUploading?: boolean; // Legacy support
}

interface FileItem {
  file: File;
  description: string;
}

export const UploadView = ({ onSuccess }: UploadViewProps) => {
  const { user } = useAuthStore();
  const [activeTab, setActiveTab] = useState<'file' | 'link'>('file');
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [urls, setUrls] = useState('');
  
  // New state for file selection
  const [selectedFiles, setSelectedFiles] = useState<FileItem[]>([]);
  const [uploadProgress, setUploadProgress] = useState<{current: number, total: number} | null>(null);

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
      addFiles(e.dataTransfer.files);
    }
  }, []);

  const addFiles = (fileList: FileList) => {
    const newFiles = Array.from(fileList);
    // Filter duplicates based on name and size
    const uniqueFiles = newFiles.filter(newFile => 
      !selectedFiles.some(existing => 
        existing.file.name === newFile.name && existing.file.size === newFile.size
      )
    );
    const newItems = uniqueFiles.map(file => ({ file, description: '' }));
    setSelectedFiles(prev => [...prev, ...newItems]);
  };

  const removeFile = (index: number) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const updateDescription = (index: number, desc: string) => {
    setSelectedFiles(prev => {
        const newFiles = [...prev];
        newFiles[index].description = desc;
        return newFiles;
    });
  };

  const startUpload = async () => {
    if (!user || selectedFiles.length === 0) return;
    setUploading(true);
    setUploadProgress({ current: 0, total: selectedFiles.length });

    try {
      let successCount = 0;
      
      for (let i = 0; i < selectedFiles.length; i++) {
        const item = selectedFiles[i];
        try {
            const formData = new FormData();
            formData.append('file', item.file);
            formData.append('email', user.email || '');
            formData.append('user_id', user.id);

            // 1. Upload to Backend
            const res = await fetch('/api/v1/kb/upload', {
              method: 'POST',
              headers: {
                'X-API-Key': 'df-internal-2024-workflow-key'
              },
              body: formData
            });

            if (!res.ok) {
                const errorData = await res.json();
                console.error(`Failed to upload ${item.file.name}:`, errorData);
                // Continue to next file
                continue;
            }
            
            const data = await res.json();

            // 2. Save to DB
            const { error } = await supabase.from('knowledge_base_files').insert({
              user_id: user.id,
              user_email: user.email,
              file_name: data.filename,
              file_type: data.file_type || item.file.type,
              file_size: data.file_size,
              storage_path: data.static_url,
              is_embedded: false,
              description: item.description
            });

            if (error) throw error;
            successCount++;
            setUploadProgress(prev => prev ? { ...prev, current: prev.current + 1 } : null);

        } catch (err) {
            console.error(`Error uploading file ${item.file.name}:`, err);
        }
      }

      if (successCount > 0) {
        // Clear files only if some succeeded. If partial fail, maybe keep failed ones?
        // For simplicity, we clear all and assume user checks library.
        setSelectedFiles([]);
        onSuccess();
      } else {
        alert("Upload failed for all files. Please check file types and try again.");
      }
      
    } catch (err) {
      console.error(err);
      alert('Upload process encountered an error');
    } finally {
      setUploading(false);
      setUploadProgress(null);
    }
  };

  return (
    <div className="max-w-4xl mx-auto">
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
        <div className="flex flex-col gap-6">
            {/* Drop Zone */}
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
                <UploadCloud className="text-purple-400" size={32} />
            </div>
            <p className="text-white font-medium mb-2">
                Drag & drop files here
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
                onChange={(e) => e.target.files && addFiles(e.target.files)}
                disabled={uploading}
            />
            <label
                htmlFor="file-upload"
                className={`px-6 py-2.5 bg-white text-black font-medium rounded-lg hover:bg-gray-100 transition-colors cursor-pointer ${uploading ? 'opacity-50 pointer-events-none' : ''}`}
            >
                Browse Files
            </label>
            </div>

            {/* File List */}
            {selectedFiles.length > 0 && (
                <div className="bg-white/5 border border-white/10 rounded-xl p-6">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="text-lg font-medium text-white">Selected Files ({selectedFiles.length})</h3>
                        <button 
                            onClick={() => setSelectedFiles([])}
                            className="text-sm text-red-400 hover:text-red-300"
                            disabled={uploading}
                        >
                            Clear All
                        </button>
                    </div>
                    <div className="space-y-4 max-h-[600px] overflow-y-auto mb-6 pr-2">
                        {selectedFiles.map((item, idx) => (
                            <div key={`${item.file.name}-${idx}`} className="flex flex-col gap-3 bg-black/20 p-4 rounded-lg border border-white/5">
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-3 overflow-hidden">
                                        <FileText className="text-purple-400 flex-shrink-0" size={20} />
                                        <div className="flex flex-col min-w-0">
                                            <span className="text-sm text-gray-200 truncate">{item.file.name}</span>
                                            <span className="text-xs text-gray-500">{(item.file.size / 1024 / 1024).toFixed(2)} MB • {item.file.type || 'Unknown Type'}</span>
                                        </div>
                                    </div>
                                    <button 
                                        onClick={() => removeFile(idx)}
                                        className="p-2 text-gray-500 hover:text-red-400 hover:bg-red-400/10 rounded-lg transition-colors"
                                        disabled={uploading}
                                    >
                                        <Trash2 size={16} />
                                    </button>
                                </div>
                                <div className="w-full">
                                    <input
                                        type="text"
                                        value={item.description}
                                        onChange={(e) => updateDescription(idx, e.target.value)}
                                        placeholder="Add a description or caption (optional)..."
                                        className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2 text-sm text-gray-300 placeholder-gray-600 focus:border-purple-500/50 outline-none transition-colors"
                                        disabled={uploading}
                                    />
                                </div>
                            </div>
                        ))}
                    </div>

                    <div className="flex justify-end">
                        <button
                            onClick={startUpload}
                            disabled={uploading}
                            className={`px-8 py-3 bg-purple-600 text-white rounded-xl font-medium transition-all flex items-center gap-2 ${
                                uploading ? 'opacity-70 cursor-not-allowed' : 'hover:bg-purple-500 shadow-lg shadow-purple-900/20'
                            }`}
                        >
                            {uploading ? (
                                <>
                                    <Loader2 className="animate-spin" size={20} />
                                    Uploading ({uploadProgress?.current}/{uploadProgress?.total})...
                                </>
                            ) : (
                                <>
                                    <UploadCloud size={20} />
                                    Start Upload
                                </>
                            )}
                        </button>
                    </div>
                </div>
            )}
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
