import { useEffect, useMemo, useState } from 'react';
import { KnowledgeBaseEntry, KnowledgeFile, ToolType } from './types';
import { FileText, Image, Video, Link as LinkIcon, Trash2, Search, Filter, X, Database, Loader2, AlertCircle, Folder, Download, PencilLine, Plus, List, LayoutGrid, ChevronLeft, ChevronRight } from 'lucide-react';
import { supabase } from '../../lib/supabase';
import { API_URL_OPTIONS } from '../../config/api';
import { useAuthStore } from '../../stores/authStore';
import { getApiSettings } from '../../services/apiSettingsService';
import { backendFetch } from '../../services/backendClient';
import { downloadSecureAsset } from '../../services/secureAssetService';

interface LibraryViewProps {
  files: KnowledgeFile[];
  knowledgeBases: KnowledgeBaseEntry[];
  kbLoading?: boolean;
  selectedIds: Set<string>;
  onToggleSelect: (id: string) => void;
  onGoToUpload: () => void;
  onRefresh: () => Promise<void>;
  onRefreshKnowledgeBases?: () => Promise<void>;
  onPreview: (file: KnowledgeFile) => void;
  onDelete: (file: KnowledgeFile) => void;
  activeTool: ToolType;
}

// 定义每个工具支持的文件类型
const TOOL_SUPPORTED_TYPES: Record<ToolType, string[]> = {
  chat: ['doc', 'image', 'video', 'link'], // Chat 支持所有类型（通过向量检索）
  search: ['doc', 'image', 'video', 'link'], // 语义检索支持所有类型
  deepresearch: ['doc', 'image', 'video', 'link'], // 深度研究支持多种类型（可选素材）
  report: ['doc'], // 报告生成仅支持文档
  ppt: ['doc', 'image'], // PPT 生成支持 PDF/PPTX + 图片
  podcast: ['doc'], // Podcast 仅支持文档类型（PDF/DOCX/PPTX）
  mindmap: ['doc'], // MindMap 暂定支持文档
  video: ['doc', 'image', 'video'], // Video 暂定支持多种类型
};

// 获取工具的友好提示名称
const TOOL_DISPLAY_NAMES: Record<ToolType, string> = {
  chat: '智能问答',
  search: '语义检索',
  deepresearch: '深度研究',
  report: '报告生成',
  ppt: 'PPT生成',
  podcast: '播客生成',
  mindmap: '思维导图',
  video: '视频生成',
};

const TOOL_HINTS: Record<ToolType, string> = {
  chat: '',
  search: '语义检索无需选择文件，但仅支持已入库的素材。',
  deepresearch: '深度研究可直接输入主题，也可选文件增强上下文。',
  report: '报告生成仅支持文档类型（PDF/DOCX/PPTX）。',
  ppt: 'PPT 生成支持 PDF / PPTX / DOCX 文档和图片素材；文档可多选并合并。',
  podcast: '播客生成仅支持文档类型（PDF/DOCX/PPTX）。',
  mindmap: '思维导图当前仅支持文档类型。',
  video: '视频生成支持文档、图片和视频素材。',
};

export const LibraryView = ({ files, knowledgeBases, kbLoading = false, selectedIds, onToggleSelect, onGoToUpload, onRefresh, onRefreshKnowledgeBases, onPreview, onDelete, activeTool }: LibraryViewProps) => {
  const { user } = useAuthStore();
  const [leftPanelWidth, setLeftPanelWidth] = useState(() => {
    if (typeof window === 'undefined') return 320;
    const saved = window.localStorage.getItem('kb_left_panel_width');
    const parsed = saved ? parseInt(saved, 10) : 320;
    return Number.isNaN(parsed) ? 320 : parsed;
  });
  const [filterType, setFilterType] = useState<'all' | 'embedded'>('all');
  const [activeKbId, setActiveKbId] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [kbPanelCollapsed, setKbPanelCollapsed] = useState(false);
  const [kbModalOpen, setKbModalOpen] = useState(false);
  const [kbModalMode, setKbModalMode] = useState<'create' | 'rename'>('create');
  const [kbEditing, setKbEditing] = useState<KnowledgeBaseEntry | null>(null);
  const [kbNameInput, setKbNameInput] = useState('');
  const [kbDescInput, setKbDescInput] = useState('');
  const [kbSaving, setKbSaving] = useState(false);
  const [kbDeleteTarget, setKbDeleteTarget] = useState<KnowledgeBaseEntry | null>(null);
  const [kbDeleteFiles, setKbDeleteFiles] = useState(false);
  const [kbActionLoading, setKbActionLoading] = useState(false);
  const [isEmbedding, setIsEmbedding] = useState(false);
  const [showManifest, setShowManifest] = useState(false);
  const [manifestLoading, setManifestLoading] = useState(false);
  const [manifestData, setManifestData] = useState<any>(null);
  const [manifestError, setManifestError] = useState('');
  const leftPanelMin = 260;
  const leftPanelMax = 460;

  // 判断文件是否被当前工具支持
  const isFileSupported = (file: KnowledgeFile): boolean => {
    if (activeTool === 'ppt') {
      if (file.type === 'image') return true;
      if (file.type === 'doc') {
        const name = file.name.toLowerCase();
        return name.endsWith('.pdf') || name.endsWith('.pptx') || name.endsWith('.ppt') || name.endsWith('.docx') || name.endsWith('.doc');
      }
      return false;
    }
    const supportedTypes = TOOL_SUPPORTED_TYPES[activeTool];
    return supportedTypes.includes(file.type);
  };

  const formatBytes = (bytes: number) => {
    if (!bytes) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };
  
  // Embedding Config Modal
  const [showEmbedConfig, setShowEmbedConfig] = useState(false);
  const [embedConfig, setEmbedConfig] = useState({
      api_url: 'https://api.apiyi.com/v1/embeddings',
      api_key: '',
      model_name: 'text-embedding-3-small',
      image_model: 'gemini-2.5-flash',
      video_model: 'gemini-2.5-flash'
  });

  useEffect(() => {
    const settings = getApiSettings(user?.id || null);
    if (settings) {
      const baseUrl = settings.apiUrl || '';
      const embedUrl = baseUrl
        ? (baseUrl.includes('/embeddings') ? baseUrl : `${baseUrl.replace(/\/$/, '')}/embeddings`)
        : '';
      setEmbedConfig(prev => ({
        ...prev,
        api_url: embedUrl || prev.api_url,
        api_key: settings.apiKey || prev.api_key
      }));
    }
  }, [user?.id]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    window.localStorage.setItem('kb_left_panel_width', String(leftPanelWidth));
  }, [leftPanelWidth]);

  const kbStats = useMemo(() => {
    const stats: Record<string, { count: number; size: number }> = {};
    files.forEach(file => {
      const key = file.kbId || 'uncategorized';
      if (!stats[key]) {
        stats[key] = { count: 0, size: 0 };
      }
      stats[key].count += 1;
      stats[key].size += file.sizeBytes || 0;
    });
    return stats;
  }, [files]);

  const sortedKnowledgeBases = useMemo(() => {
    return knowledgeBases.slice().sort((a, b) => a.name.localeCompare(b.name));
  }, [knowledgeBases]);

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

  const openCreateKb = () => {
    setKbModalMode('create');
    setKbEditing(null);
    setKbNameInput('');
    setKbDescInput('');
    setKbModalOpen(true);
  };

  const openRenameKb = (kb: KnowledgeBaseEntry) => {
    setKbModalMode('rename');
    setKbEditing(kb);
    setKbNameInput(kb.name || '');
    setKbDescInput(kb.description || '');
    setKbModalOpen(true);
  };

  const saveKnowledgeBase = async () => {
    if (!user?.id || !kbNameInput.trim()) return;
    setKbSaving(true);
    try {
      if (kbModalMode === 'create') {
        const { error } = await supabase
          .from('knowledge_bases')
          .insert({
            user_id: user.id,
            name: kbNameInput.trim(),
            description: kbDescInput.trim() || null
          });
        if (error) throw error;
      } else if (kbEditing?.id) {
        const { error } = await supabase
          .from('knowledge_bases')
          .update({
            name: kbNameInput.trim(),
            description: kbDescInput.trim() || null,
            updated_at: new Date().toISOString()
          })
          .eq('id', kbEditing.id);
        if (error) throw error;
      }

      if (onRefreshKnowledgeBases) {
        await onRefreshKnowledgeBases();
      }
      setKbModalOpen(false);
    } catch (err) {
      console.error('Save knowledge base failed:', err);
      alert('保存知识库失败');
    } finally {
      setKbSaving(false);
    }
  };

  const confirmDeleteKb = (kb: KnowledgeBaseEntry) => {
    setKbDeleteTarget(kb);
    setKbDeleteFiles(false);
  };

  const deleteKnowledgeBase = async () => {
    if (!user?.id || !kbDeleteTarget) return;
    setKbActionLoading(true);
    try {
      const targetId = kbDeleteTarget.id;
      const filesInKb = files.filter(f => f.kbId === targetId);

      if (kbDeleteFiles && filesInKb.length > 0) {
        const storagePaths = filesInKb.map(f => f.url).filter(Boolean);
        if (storagePaths.length > 0) {
          await backendFetch('/api/v1/kb/delete-batch', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ storage_paths: storagePaths })
          });
        }

        const { error: deleteFilesError } = await supabase
          .from('knowledge_base_files')
          .delete()
          .in('id', filesInKb.map(f => f.id));
        if (deleteFilesError) throw deleteFilesError;
      } else if (filesInKb.length > 0) {
        const { error: moveError } = await supabase
          .from('knowledge_base_files')
          .update({ kb_id: null })
          .in('id', filesInKb.map(f => f.id));
        if (moveError) throw moveError;
      }

      const { error } = await supabase
        .from('knowledge_bases')
        .delete()
        .eq('id', targetId);
      if (error) throw error;

      if (onRefreshKnowledgeBases) {
        await onRefreshKnowledgeBases();
      }
      await onRefresh();
      if (activeKbId === targetId) {
        setActiveKbId('all');
      }
    } catch (err) {
      console.error('Delete knowledge base failed:', err);
      alert('删除知识库失败');
    } finally {
      setKbActionLoading(false);
      setKbDeleteTarget(null);
    }
  };

  const moveSelectedToKb = async (kbId: string | null) => {
    if (selectedIds.size === 0) return;
    setKbActionLoading(true);
    try {
      const { error } = await supabase
        .from('knowledge_base_files')
        .update({ kb_id: kbId })
        .in('id', Array.from(selectedIds));
      if (error) throw error;
      await onRefresh();
    } catch (err) {
      console.error('Move files failed:', err);
      alert('移动文件失败');
    } finally {
      setKbActionLoading(false);
    }
  };

  const exportKnowledgeBase = async (kb: KnowledgeBaseEntry) => {
    const filesInKb = files.filter(f => f.kbId === kb.id);
    if (filesInKb.length === 0) {
      alert('该知识库暂无文件可导出');
      return;
    }
    setKbActionLoading(true);
    try {
      const res = await backendFetch('/api/v1/kb/export-zip', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          files: filesInKb.map(f => f.url),
          email: user?.email || null,
          kb_name: kb.name,
          include_root_dir: true
        })
      });
      if (!res.ok) {
        throw new Error(await res.text());
      }
      const data = await res.json();
      if (data?.zip_path) {
        await downloadSecureAsset(data.zip_path, `${kb.name || 'knowledge-base'}.zip`);
      } else {
        alert('导出完成，但未返回下载链接');
      }
    } catch (err) {
      console.error('Export KB failed:', err);
      alert('导出失败');
    } finally {
      setKbActionLoading(false);
    }
  };

  const openManifest = async () => {
    setShowManifest(true);
    if (!user?.email) {
      setManifestError('未检测到用户信息，请先登录。');
      return;
    }
    setManifestLoading(true);
    setManifestError('');
    try {
      const res = await backendFetch(`/api/v1/kb/list?email=${encodeURIComponent(user.email)}`);
      if (!res.ok) {
        throw new Error(await res.text());
      }
      const data = await res.json();
      setManifestData(data);
    } catch (err: any) {
      setManifestError(err?.message || '获取结构化清单失败');
    } finally {
      setManifestLoading(false);
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
        const res = await backendFetch('/api/v1/kb/embedding', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
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
      if (filterType === 'embedded' && !file.isEmbedded) return false;
      if (activeKbId === 'uncategorized' && file.kbId) return false;
      if (activeKbId !== 'all' && activeKbId !== 'uncategorized' && file.kbId !== activeKbId) return false;
      if (searchQuery.trim()) {
        const q = searchQuery.trim().toLowerCase();
        const hay = `${file.name || ''} ${file.desc || ''}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
  });

  return (
    <div className="h-full flex flex-col relative">
      {/* Tool File Type Hint */}
      {TOOL_HINTS[activeTool] && (
        <div className="mb-4 bg-blue-500/10 border border-blue-500/20 rounded-lg p-3 flex items-start gap-3">
          <AlertCircle className="text-blue-400 mt-0.5 flex-shrink-0" size={16} />
          <div className="text-xs text-blue-300">
            <span className="font-medium">{TOOL_DISPLAY_NAMES[activeTool]}</span>：{TOOL_HINTS[activeTool]}
          </div>
        </div>
      )}

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

      <div className="flex flex-col lg:flex-row gap-6 flex-1 min-h-0">
        {!kbPanelCollapsed && (
          <div
            className="relative w-full flex-shrink-0 bg-white/5 border border-white/10 rounded-xl p-4 h-fit lg:h-full overflow-hidden"
            style={{ width: `min(100%, ${leftPanelWidth}px)` }}
          >
            <div
              className="hidden lg:block absolute top-0 right-0 h-full w-1.5 cursor-col-resize bg-transparent hover:bg-purple-500/30 transition-colors"
              onMouseDown={(e) => {
                e.preventDefault();
                const startX = e.clientX;
                const startWidth = leftPanelWidth;
                const onMove = (evt: MouseEvent) => {
                  const next = Math.min(leftPanelMax, Math.max(leftPanelMin, startWidth + (evt.clientX - startX)));
                  setLeftPanelWidth(next);
                };
                const onUp = () => {
                  document.removeEventListener('mousemove', onMove);
                  document.removeEventListener('mouseup', onUp);
                };
                document.addEventListener('mousemove', onMove);
                document.addEventListener('mouseup', onUp);
              }}
            />
            <div className="flex items-center justify-between mb-4">
              <div className="text-sm font-medium text-white flex items-center gap-2">
                <Folder size={16} className="text-purple-400" />
                知识库分类
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setKbPanelCollapsed(true)}
                  className="text-xs px-2 py-1 rounded-md bg-white/10 text-gray-300 hover:bg-white/20 flex items-center gap-1"
                  title="收起分类"
                >
                  <ChevronLeft size={12} /> 收起
                </button>
                <button
                  onClick={openCreateKb}
                  className="text-xs px-2 py-1 rounded-md bg-purple-500/20 text-purple-300 hover:bg-purple-500/30 flex items-center gap-1"
                >
                  <Plus size={12} /> 新建
                </button>
              </div>
            </div>

            <div className="space-y-2 max-h-[70vh] overflow-y-auto pr-1">
              <button
                onClick={() => setActiveKbId('all')}
                className={`w-full text-left px-3 py-2 rounded-lg border transition-all ${
                  activeKbId === 'all'
                    ? 'border-purple-500/50 bg-purple-500/10 text-white'
                    : 'border-white/10 bg-black/20 text-gray-300 hover:bg-white/5'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">全部文件</span>
                  <span className="text-xs text-gray-500">{files.length}</span>
                </div>
              </button>

              <button
                onClick={() => setActiveKbId('uncategorized')}
                className={`w-full text-left px-3 py-2 rounded-lg border transition-all ${
                  activeKbId === 'uncategorized'
                    ? 'border-purple-500/50 bg-purple-500/10 text-white'
                    : 'border-white/10 bg-black/20 text-gray-300 hover:bg-white/5'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">未分类</span>
                  <span className="text-xs text-gray-500">{kbStats.uncategorized?.count || 0}</span>
                </div>
                <div className="text-xs text-gray-500 mt-1">大小：{formatBytes(kbStats.uncategorized?.size || 0)}</div>
              </button>

              {kbLoading && (
                <div className="text-xs text-gray-500 flex items-center gap-2">
                  <Loader2 className="animate-spin" size={12} /> 正在加载知识库...
                </div>
              )}

              {sortedKnowledgeBases.map(kb => {
                const stats = kbStats[kb.id] || { count: 0, size: 0 };
                return (
                  <div
                    key={kb.id}
                    className={`w-full px-3 py-2 rounded-lg border transition-all ${
                      activeKbId === kb.id
                        ? 'border-purple-500/50 bg-purple-500/10 text-white'
                        : 'border-white/10 bg-black/20 text-gray-300 hover:bg-white/5'
                    }`}
                  >
                    <button onClick={() => setActiveKbId(kb.id)} className="w-full text-left">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium truncate">{kb.name}</span>
                        <span className="text-xs text-gray-500">{stats.count}</span>
                      </div>
                      <div className="text-xs text-gray-500 mt-1 flex items-center justify-between">
                        <span>大小：{formatBytes(stats.size)}</span>
                        <span>{new Date(kb.createdAt).toLocaleDateString()}</span>
                      </div>
                    </button>
                    <div className="flex items-center gap-2 mt-2">
                      <button
                        onClick={() => exportKnowledgeBase(kb)}
                        className="text-xs px-2 py-1 rounded-md bg-white/10 hover:bg-white/20 text-gray-200 flex items-center gap-1 disabled:opacity-50 disabled:cursor-not-allowed"
                        title="导出 ZIP"
                        disabled={kbActionLoading}
                      >
                        <Download size={12} /> 导出
                      </button>
                      <button
                        onClick={() => openRenameKb(kb)}
                        className="text-xs px-2 py-1 rounded-md bg-white/10 hover:bg-white/20 text-gray-200 flex items-center gap-1 disabled:opacity-50 disabled:cursor-not-allowed"
                        title="重命名"
                        disabled={kbActionLoading}
                      >
                        <PencilLine size={12} /> 重命名
                      </button>
                      <button
                        onClick={() => confirmDeleteKb(kb)}
                        className="text-xs px-2 py-1 rounded-md bg-red-500/20 hover:bg-red-500/30 text-red-300 flex items-center gap-1 disabled:opacity-50 disabled:cursor-not-allowed"
                        title="删除知识库"
                        disabled={kbActionLoading}
                      >
                        <Trash2 size={12} /> 删除
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Files Panel */}
        <div className="flex-1 min-w-0 flex flex-col">
          {/* Toolbar */}
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-4 flex-1">
              <div className="relative flex-1 max-w-md">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" size={16} />
                <input 
                  type="text" 
                  placeholder="Search files..." 
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-gray-200 outline-none focus:border-purple-500/50"
                />
              </div>
              <button className="p-2 text-gray-400 hover:text-white bg-white/5 rounded-lg border border-white/10">
                <Filter size={18} />
              </button>
              <button
                onClick={() => setViewMode(prev => (prev === 'grid' ? 'list' : 'grid'))}
                className="p-2 text-gray-400 hover:text-white bg-white/5 rounded-lg border border-white/10"
                title={viewMode === 'grid' ? '切换为列表' : '切换为网格'}
              >
                {viewMode === 'grid' ? <List size={18} /> : <LayoutGrid size={18} />}
              </button>
              <button
                onClick={openManifest}
                className="p-2 text-gray-400 hover:text-white bg-white/5 rounded-lg border border-white/10"
                title="结构化清单"
              >
                <Database size={18} />
              </button>
            </div>
            <div className="flex items-center gap-2">
              {selectedIds.size > 0 && (
                <>
                  <div className="relative">
                    <select
                      onChange={(e) => {
                        const val = e.target.value;
                        if (val === '') return;
                        if (val === 'uncategorized') {
                          moveSelectedToKb(null);
                        } else {
                          moveSelectedToKb(val);
                        }
                        e.target.value = '';
                      }}
                      className="px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-xs text-gray-200 outline-none"
                      defaultValue=""
                      disabled={kbActionLoading}
                    >
                      <option value="" disabled>批量移动到...</option>
                      <option value="uncategorized">未分类</option>
                      {sortedKnowledgeBases.map(kb => (
                        <option key={kb.id} value={kb.id}>{kb.name}</option>
                      ))}
                    </select>
                  </div>
                  <button
                    onClick={handleBulkDelete}
                    className="px-3 py-2 bg-red-500/10 hover:bg-red-500/20 text-red-300 border border-red-500/30 rounded-lg text-xs font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                    disabled={kbActionLoading}
                  >
                    批量删除
                  </button>
                </>
              )}
              <button 
                onClick={onGoToUpload}
                className="px-4 py-2 bg-purple-600 hover:bg-purple-500 text-white rounded-lg text-sm font-medium transition-colors"
              >
                + Upload
              </button>
            </div>
          </div>

          <div className="flex items-center justify-between mb-3">
            <div className="text-xs text-gray-500">
              {kbPanelCollapsed ? '分类已隐藏' : '分类已展开'}
            </div>
            {kbPanelCollapsed && (
              <button
                onClick={() => setKbPanelCollapsed(false)}
                className="text-xs px-2 py-1 rounded-md bg-white/10 text-gray-300 hover:bg-white/20 flex items-center gap-1"
              >
                <ChevronRight size={12} /> 展开分类
              </button>
            )}
          </div>

          {viewMode === 'grid' ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 overflow-y-auto pb-20 flex-1">
              {filteredFiles.map(file => {
                const isSupported = isFileSupported(file);
                return (
                <div
                  key={file.id}
                  onClick={() => onPreview(file)}
                  className={`group relative p-4 rounded-xl border transition-all ${
                    !isSupported
                      ? 'opacity-40 cursor-not-allowed bg-white/5 border-white/5'
                      : selectedIds.has(file.id)
                        ? 'bg-purple-500/10 border-purple-500/50 cursor-pointer'
                        : 'bg-white/5 border-white/10 hover:border-white/20 hover:bg-white/10 cursor-pointer'
                  }`}
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="p-2 bg-black/20 rounded-lg relative">
                      {getIcon(file.type)}
                      {file.isEmbedded && (
                          <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full border border-[#0a0a1a]" title="Embedded"></div>
                      )}
                      {!isSupported && (
                          <div className="absolute -top-1 -right-1 w-4 h-4 bg-red-500/80 rounded-full border border-[#0a0a1a] flex items-center justify-center" title="当前工具不支持此文件类型">
                            <X size={10} className="text-white" />
                          </div>
                      )}
                    </div>
                    <div
                      onClick={(e) => {
                        e.stopPropagation();
                        if (isSupported) {
                          onToggleSelect(file.id);
                        }
                      }}
                      className={`w-5 h-5 rounded-full border flex items-center justify-center transition-colors ${
                        !isSupported
                          ? 'cursor-not-allowed border-white/10 bg-white/5'
                          : selectedIds.has(file.id)
                            ? 'bg-purple-500 border-purple-500 cursor-pointer'
                            : 'border-white/20 cursor-pointer hover:border-purple-400'
                      }`}
                    >
                      {selectedIds.has(file.id) && <div className="w-2 h-2 bg-white rounded-full" />}
                    </div>
                  </div>

                  <h3 className="text-sm font-medium text-gray-200 truncate mb-1" title={file.name}>
                    {file.name}
                  </h3>
                  
                  <div className="text-xs text-gray-500 truncate">
                    {file.kbId ? (knowledgeBases.find(kb => kb.id === file.kbId)?.name || '未命名知识库') : '未分类'}
                  </div>

                  <div className="flex items-center justify-between text-xs text-gray-500 mt-2">
                    <span>{file.size}</span>
                    <span>{file.uploadTime.split(' ')[0]}</span>
                  </div>

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
                );
              })}
            </div>
          ) : (
            <div className="space-y-3 overflow-y-auto pb-20 flex-1">
              {filteredFiles.map(file => {
                const isSupported = isFileSupported(file);
                return (
                  <div
                    key={file.id}
                    onClick={() => onPreview(file)}
                    className={`group relative px-4 py-3 rounded-xl border transition-all ${
                      !isSupported
                        ? 'opacity-40 cursor-not-allowed bg-white/5 border-white/5'
                        : selectedIds.has(file.id)
                          ? 'bg-purple-500/10 border-purple-500/50 cursor-pointer'
                          : 'bg-white/5 border-white/10 hover:border-white/20 hover:bg-white/10 cursor-pointer'
                    }`}
                  >
                    <div className="flex items-center gap-4">
                      <div className="p-2 bg-black/20 rounded-lg relative flex-shrink-0">
                        {getIcon(file.type)}
                        {file.isEmbedded && (
                          <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full border border-[#0a0a1a]" title="Embedded"></div>
                        )}
                        {!isSupported && (
                          <div className="absolute -top-1 -right-1 w-4 h-4 bg-red-500/80 rounded-full border border-[#0a0a1a] flex items-center justify-center" title="当前工具不支持此文件类型">
                            <X size={10} className="text-white" />
                          </div>
                        )}
                      </div>

                      <div className="min-w-0 flex-1">
                        <div className="text-sm font-medium text-gray-200 truncate" title={file.name}>
                          {file.name}
                        </div>
                        <div className="text-xs text-gray-500 truncate">
                          {file.kbId ? (knowledgeBases.find(kb => kb.id === file.kbId)?.name || '未命名知识库') : '未分类'}
                        </div>
                      </div>

                      <div className="text-xs text-gray-500 w-24 text-right flex-shrink-0">
                        {file.size}
                      </div>
                      <div className="text-xs text-gray-500 w-24 text-right flex-shrink-0">
                        {file.uploadTime.split(' ')[0]}
                      </div>

                      <div
                        onClick={(e) => {
                          e.stopPropagation();
                          if (isSupported) {
                            onToggleSelect(file.id);
                          }
                        }}
                        className={`w-5 h-5 rounded-full border flex items-center justify-center transition-colors flex-shrink-0 ${
                          !isSupported
                            ? 'cursor-not-allowed border-white/10 bg-white/5'
                            : selectedIds.has(file.id)
                              ? 'bg-purple-500 border-purple-500 cursor-pointer'
                              : 'border-white/20 cursor-pointer hover:border-purple-400'
                        }`}
                      >
                        {selectedIds.has(file.id) && <div className="w-2 h-2 bg-white rounded-full" />}
                      </div>
                      <button
                        onClick={(e) => handleDelete(file, e)}
                        className="p-1.5 bg-red-500/20 text-red-400 rounded-lg hover:bg-red-500 hover:text-white shadow-lg flex-shrink-0"
                        title="Delete file"
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
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

      {/* Knowledge Base Create/Rename Modal */}
      {kbModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={() => setKbModalOpen(false)}>
          <div className="bg-[#0a0a1a] border border-white/10 rounded-xl p-6 w-full max-w-md shadow-2xl" onClick={e => e.stopPropagation()}>
            <h3 className="text-lg font-medium text-white mb-4">
              {kbModalMode === 'create' ? '创建知识库' : '重命名知识库'}
            </h3>
            <div className="space-y-3">
              <div>
                <label className="block text-xs text-gray-400 mb-1">名称</label>
                <input
                  type="text"
                  value={kbNameInput}
                  onChange={(e) => setKbNameInput(e.target.value)}
                  className="w-full bg-black/30 border border-white/10 rounded-lg px-3 py-2 text-sm text-gray-200 outline-none focus:border-purple-500/50"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">描述</label>
                <input
                  type="text"
                  value={kbDescInput}
                  onChange={(e) => setKbDescInput(e.target.value)}
                  className="w-full bg-black/30 border border-white/10 rounded-lg px-3 py-2 text-sm text-gray-200 outline-none focus:border-purple-500/50"
                />
              </div>
            </div>
            <div className="flex justify-end gap-3 mt-6">
              <button
                onClick={() => setKbModalOpen(false)}
                className="px-4 py-2 text-sm text-gray-400 hover:text-white transition-colors"
              >
                取消
              </button>
              <button
                onClick={saveKnowledgeBase}
                disabled={kbSaving || !kbNameInput.trim()}
                className="px-4 py-2 bg-purple-600 hover:bg-purple-500 text-white rounded-lg text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {kbSaving ? '保存中...' : '保存'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Knowledge Base Delete Modal */}
      {kbDeleteTarget && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={() => setKbDeleteTarget(null)}>
          <div className="bg-[#0a0a1a] border border-white/10 rounded-xl p-6 w-full max-w-md shadow-2xl" onClick={e => e.stopPropagation()}>
            <h3 className="text-lg font-medium text-white mb-3">删除知识库</h3>
            <p className="text-sm text-gray-400 mb-4">
              确认删除 <span className="text-white font-medium">{kbDeleteTarget.name}</span> 吗？
            </p>
            <label className="flex items-center gap-2 text-sm text-gray-300 mb-4">
              <input
                type="checkbox"
                checked={kbDeleteFiles}
                onChange={(e) => setKbDeleteFiles(e.target.checked)}
                className="accent-purple-500"
              />
              同时删除该知识库下的文件
            </label>
            <div className="flex justify-end gap-3">
              <button
                onClick={() => setKbDeleteTarget(null)}
                className="px-4 py-2 text-sm text-gray-400 hover:text-white transition-colors"
              >
                取消
              </button>
              <button
                onClick={deleteKnowledgeBase}
                disabled={kbActionLoading}
                className="px-4 py-2 bg-red-500 hover:bg-red-400 text-white rounded-lg text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {kbActionLoading ? '处理中...' : '删除'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Manifest Modal */}
      {showManifest && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={() => setShowManifest(false)}>
          <div className="bg-[#0a0a1a] border border-white/10 rounded-xl p-6 w-full max-w-2xl shadow-2xl" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-medium text-white flex items-center gap-2">
                <Database className="text-purple-500" />
                知识库结构化清单
              </h3>
              <button
                onClick={() => setShowManifest(false)}
                className="p-2 hover:bg-white/10 rounded-lg text-gray-400 hover:text-white transition-colors"
              >
                <X size={18} />
              </button>
            </div>

            {manifestLoading && (
              <div className="flex items-center gap-2 text-sm text-gray-400">
                <Loader2 className="animate-spin" size={16} /> 加载中...
              </div>
            )}

            {manifestError && (
              <div className="text-sm text-red-400 mb-4">{manifestError}</div>
            )}

            {manifestData && (
              <div className="space-y-4 max-h-[60vh] overflow-y-auto pr-2">
                <div className="text-xs text-gray-400">
                  项目：{manifestData.project_name || 'kb_project'} • 文件数：{manifestData.files?.length || 0}
                </div>
                {(manifestData.files || []).map((f: any) => (
                  <div key={f.id} className="bg-white/5 border border-white/10 rounded-lg p-3">
                    <div className="flex items-center justify-between">
                      <div className="text-sm text-white truncate">{f.original_path?.split('/').pop() || f.id}</div>
                      <span className="text-xs text-gray-400">{f.file_type || 'unknown'}</span>
                    </div>
                    <div className="mt-2 text-xs text-gray-500 flex items-center gap-4">
                      <span>状态：{f.status || 'unknown'}</span>
                      <span>文本块：{f.chunks_count ?? 0}</span>
                      <span>多模态描述：{f.media_desc_count ?? 0}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
