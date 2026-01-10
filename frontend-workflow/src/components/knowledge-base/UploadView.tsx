import { useRef, useState, ChangeEvent } from 'react';
import { UploadCloud, FileText, Image as ImageIcon, Video, Link as LinkIcon, Plus, Loader2, Bot, X } from 'lucide-react';
import { MaterialType } from './types';

interface UploadViewProps {
  onUploadFile: (files: { file: File; desc?: string }[], type: MaterialType) => Promise<void>;
  onProcessLinks: (links: string[]) => Promise<void>;
  isUploading: boolean;
}

export const UploadView = ({ onUploadFile, onProcessLinks, isUploading }: UploadViewProps) => {
  const [uploadType, setUploadType] = useState<MaterialType>('image');
  const [linkUrl, setLinkUrl] = useState('');
  const [linkQueue, setLinkQueue] = useState<string[]>([]);
  const [fileQueue, setFileQueue] = useState<{ id: string; file: File; desc: string; preview: string }[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      if (uploadType === 'image') {
        const newFiles = Array.from(e.target.files).map(file => ({
          id: Math.random().toString(36).substr(2, 9),
          file,
          desc: '',
          preview: URL.createObjectURL(file)
        }));
        setFileQueue(prev => [...prev, ...newFiles]);
      } else {
        // Direct upload for other types (or keep same logic if desired, but user only asked for image multi-desc)
        // For now, let's keep direct upload for non-image types or unify it.
        // The user request was specific to "Upload image part can support multi-picture; and can input description one by one"
        // So for doc/video, we can stick to single/direct or also queue. Let's stick to direct for simplicity unless they want description there too.
        // Actually, let's unify it slightly but keep image special for description.
        const files = Array.from(e.target.files).map(f => ({ file: f, desc: '' }));
        onUploadFile(files, uploadType);
      }
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleRemoveFile = (id: string) => {
    setFileQueue(prev => prev.filter(f => f.id !== id));
  };

  const handleUpdateDesc = (id: string, desc: string) => {
    setFileQueue(prev => prev.map(f => f.id === id ? { ...f, desc } : f));
  };

  const handleUploadImages = async () => {
    if (fileQueue.length === 0) return;
    const filesToUpload = fileQueue.map(({ file, desc }) => ({ file, desc }));
    await onUploadFile(filesToUpload, 'image');
    setFileQueue([]);
  };

  const handleAddToLinkQueue = () => {
    if (!linkUrl.trim()) return;
    setLinkQueue(prev => [...prev, linkUrl]);
    setLinkUrl('');
  };

  const handleProcessLinks = async () => {
    if (linkQueue.length === 0) return;
    await onProcessLinks(linkQueue);
    setLinkQueue([]);
  };

  const UploadTabBtn = ({ type, label, icon: Icon }: any) => (
    <button
      onClick={() => setUploadType(type)}
      className={`flex-1 flex flex-col items-center justify-center p-3 rounded-xl transition-all ${
        uploadType === type
          ? 'bg-primary-500 text-white shadow-lg'
          : 'bg-white/5 text-gray-400 hover:bg-white/10 hover:text-gray-200'
      }`}
    >
      <Icon size={20} className="mb-1" />
      <span className="text-xs font-medium">{label}</span>
    </button>
  );

  return (
    <div className="max-w-4xl mx-auto pt-8">
      <div className="text-center mb-10">
        <h3 className="text-2xl font-bold text-white mb-2">添加素材</h3>
        <p className="text-gray-400">支持多种格式文件导入，构建您的专属知识库</p>
      </div>

      <div className="grid grid-cols-4 gap-4 mb-8">
        <UploadTabBtn type="image" label="图片" icon={ImageIcon} />
        <UploadTabBtn type="doc" label="文档" icon={FileText} />
        <UploadTabBtn type="video" label="视频" icon={Video} />
        <UploadTabBtn type="link" label="链接" icon={LinkIcon} />
      </div>

      <div className="bg-white/[0.02] border border-white/10 rounded-3xl p-8 backdrop-blur-sm relative">
        {/* Upload Area - Image Queue Mode or Normal Dropzone */}
        {uploadType === 'image' && fileQueue.length > 0 ? (
          <div className="min-h-[16rem] flex flex-col">
            <div className="flex-1 space-y-3 mb-4 max-h-[400px] overflow-y-auto pr-2">
              {fileQueue.map((item) => (
                <div key={item.id} className="flex gap-4 p-3 rounded-xl bg-white/5 border border-white/10">
                  <div className="w-20 h-20 rounded-lg bg-black/50 overflow-hidden flex-shrink-0">
                    <img src={item.preview} alt="preview" className="w-full h-full object-cover" />
                  </div>
                  <div className="flex-1 min-w-0 flex flex-col justify-center gap-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-200 truncate">{item.file.name}</span>
                      <button onClick={() => handleRemoveFile(item.id)} className="text-gray-500 hover:text-red-400">
                        <X size={16} />
                      </button>
                    </div>
                    <input
                      type="text"
                      placeholder="输入图片描述..."
                      value={item.desc}
                      onChange={(e) => handleUpdateDesc(item.id, e.target.value)}
                      className="w-full bg-black/20 border border-white/10 rounded-lg px-3 py-1.5 text-xs text-gray-300 outline-none focus:border-primary-500 transition-colors"
                    />
                  </div>
                </div>
              ))}
              {/* Add more button */}
              <button 
                onClick={() => fileInputRef.current?.click()}
                className="w-full py-3 border border-dashed border-white/10 rounded-xl text-gray-500 hover:text-gray-300 hover:bg-white/5 transition-all text-sm flex items-center justify-center gap-2"
              >
                <Plus size={16} /> 继续添加图片
              </button>
            </div>
            
            <div className="flex justify-end pt-4 border-t border-white/5">
              <button
                onClick={handleUploadImages}
                disabled={isUploading}
                className="bg-primary-600 hover:bg-primary-700 text-white px-8 py-2.5 rounded-xl text-sm font-medium transition-colors disabled:opacity-50 flex items-center gap-2"
              >
                <UploadCloud size={16} /> 确认上传 ({fileQueue.length})
              </button>
            </div>
          </div>
        ) : (
          <div 
            onClick={() => uploadType !== 'link' && fileInputRef.current?.click()}
            className={`
              border-2 border-dashed rounded-2xl flex flex-col items-center justify-center transition-all overflow-hidden relative
              ${uploadType === 'link' ? 'border-transparent cursor-default min-h-[16rem]' : 'h-64 cursor-pointer border-white/10 hover:border-primary-500/50 hover:bg-white/5'}
            `}
          >
            {uploadType === 'link' ? (
              <div className="w-full h-full flex flex-col">
                <div className="flex gap-2 mb-6">
                  <input
                    type="text"
                    placeholder="https://..."
                    value={linkUrl}
                    onChange={(e) => setLinkUrl(e.target.value)}
                    className="flex-1 bg-black/40 border border-white/10 rounded-xl px-5 py-4 text-sm outline-none focus:border-primary-500 transition-colors"
                  />
                  <button 
                    onClick={handleAddToLinkQueue}
                    disabled={!linkUrl}
                    className="bg-white/10 hover:bg-white/20 text-white px-4 rounded-xl transition-colors disabled:opacity-50"
                  >
                    <Plus size={20} />
                  </button>
                </div>
                
                <div className="flex-1 bg-black/20 rounded-xl p-4 overflow-y-auto mb-4 border border-white/5">
                  <p className="text-xs text-gray-500 mb-2 font-medium">待处理链接队列 ({linkQueue.length})</p>
                  {linkQueue.length === 0 ? (
                    <p className="text-xs text-gray-600 italic">暂无链接，请添加...</p>
                  ) : (
                    <div className="space-y-2">
                      {linkQueue.map((url, idx) => (
                        <div key={idx} className="flex items-center justify-between text-xs bg-white/5 p-2 rounded-lg gap-2">
                          <span className="truncate text-gray-300 flex-1 break-all">{url}</span>
                          <button onClick={() => setLinkQueue(q => q.filter((_, i) => i !== idx))} className="text-gray-500 hover:text-red-400 flex-shrink-0">
                            <X size={14} />
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                <div className="flex items-center justify-between">
                  <p className="text-[10px] text-gray-500 flex items-center gap-1">
                    <Bot size={12} /> 系统将自动爬取并总结链接内容
                  </p>
                  <button
                    onClick={handleProcessLinks}
                    disabled={isUploading || linkQueue.length === 0}
                    className="bg-primary-600 hover:bg-primary-700 text-white px-6 py-2.5 rounded-xl text-sm font-medium transition-colors disabled:opacity-50 flex items-center gap-2"
                  >
                    <UploadCloud size={16} /> 开始处理
                  </button>
                </div>
              </div>
            ) : (
              <>
                <div className="w-16 h-16 rounded-full bg-white/5 flex items-center justify-center mb-4">
                  <UploadCloud className="text-primary-400" size={32} />
                </div>
                <span className="text-base text-gray-200 font-medium mb-1">点击或拖拽文件到此处</span>
                <span className="text-xs text-gray-500">
                  {uploadType === 'image' && '支持多图上传 (JPG, PNG, WEBP)'}
                  {uploadType === 'doc' && '支持 PDF, DOCX, TXT'}
                  {uploadType === 'video' && '支持 MP4, MOV (Max 50MB)'}
                </span>
              </>
            )}
          </div>
        )}

        <input ref={fileInputRef} type="file" multiple={uploadType === 'image'} className="hidden" onChange={handleFileChange} />
        
        {isUploading && (
          <div className="absolute inset-0 bg-black/80 backdrop-blur-sm rounded-3xl flex items-center justify-center flex-col z-20">
            <Loader2 size={40} className="text-primary-500 animate-spin mb-4" />
            <p className="text-white font-medium">正在处理...</p>
          </div>
        )}
      </div>
    </div>
  );
};
