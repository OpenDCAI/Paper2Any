import { useState } from 'react';
import { MaterialType, KnowledgeFile, SectionType, ToolType } from './types';
import { Sidebar } from './Sidebar';
import { LibraryView } from './LibraryView';
import { UploadView } from './UploadView';
import { OutputView } from './OutputView';
import { RightPanel } from './RightPanel';

const KnowledgeBase = () => {
  // State
  const [activeSection, setActiveSection] = useState<SectionType>('library');
  const [activeTool, setActiveTool] = useState<ToolType>('chat');
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [isUploading, setIsUploading] = useState(false);

  // Mock Data
  const [files, setFiles] = useState<KnowledgeFile[]>([
    {
      id: '1',
      name: 'DeepSeek-V3 Technical Report.pdf',
      type: 'doc',
      size: '2.4 MB',
      uploadTime: '2025-01-09 10:00',
      url: '#'
    },
    {
      id: '2',
      name: 'System Architecture Diagram.png',
      type: 'image',
      desc: 'High level system design',
      size: '1.2 MB',
      uploadTime: '2025-01-09 11:30',
      url: '#'
    }
  ]);

  const [outputFiles, setOutputFiles] = useState<KnowledgeFile[]>([
    {
      id: 'o1',
      name: 'DeepSeek Report Presentation.pptx',
      type: 'doc',
      size: '1.8 MB',
      uploadTime: '2025-01-09 10:30',
      url: '#',
      desc: 'Generated PPT'
    }
  ]);

  // Handlers
  const handleToggleSelect = (id: string) => {
    const newSet = new Set(selectedIds);
    if (newSet.has(id)) {
      newSet.delete(id);
    } else {
      newSet.add(id);
    }
    setSelectedIds(newSet);
  };

  const handleUploadFile = async (fileItems: { file: File; desc?: string }[], type: MaterialType) => {
    setIsUploading(true);
    await new Promise(resolve => setTimeout(resolve, 1000));

    const newFiles: KnowledgeFile[] = fileItems.map(item => ({
      id: Math.random().toString(36).substr(2, 9),
      name: item.file.name,
      type: type,
      file: item.file,
      size: (item.file.size / 1024 / 1024).toFixed(2) + ' MB',
      desc: item.desc,
      uploadTime: new Date().toLocaleString()
    }));

    setFiles(prev => [...newFiles, ...prev]);
    setIsUploading(false);
    setActiveSection('library');
  };

  const handleProcessLinks = async (links: string[]) => {
    setIsUploading(true);
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    const newFiles: KnowledgeFile[] = links.map(url => ({
      id: Math.random().toString(36).substr(2, 9),
      name: url,
      type: 'link',
      url: url,
      size: 'Scraped',
      uploadTime: new Date().toLocaleString(),
      desc: 'Auto-summarized content'
    }));
    
    setFiles(prev => [...newFiles, ...prev]);
    setIsUploading(false);
    setActiveSection('library');
  };

  const handleGenerateSuccess = (file: KnowledgeFile) => {
    setOutputFiles(prev => [file, ...prev]);
    setActiveSection('output');
  };

  return (
    <div className="w-full h-full flex bg-[#02020a] text-gray-200 overflow-hidden font-sans">
      
      {/* 1. Sidebar */}
      <Sidebar 
        activeSection={activeSection} 
        onSectionChange={setActiveSection}
        filesCount={files.length}
        outputCount={outputFiles.length}
      />

      {/* 2. Main Content */}
      <div className="flex-1 flex flex-col min-w-0 bg-gradient-to-br from-[#050512] to-[#0a0a1a] relative z-10">
        {/* Header */}
        <div className="h-16 border-b border-white/5 flex items-center px-8 justify-between backdrop-blur-sm bg-[#050512]/50 sticky top-0 z-10">
          <h2 className="text-lg font-medium text-white">
            {activeSection === 'library' && '我的知识库'}
            {activeSection === 'upload' && '上传新素材'}
            {activeSection === 'output' && '知识产出成果'}
          </h2>
          <div className="flex items-center gap-2">
            {selectedIds.size > 0 && activeSection === 'library' && (
               <button onClick={() => setSelectedIds(new Set())} className="text-xs px-3 py-1.5 rounded-lg border border-white/10 hover:bg-white/5 transition-colors">
                 取消选择 ({selectedIds.size})
               </button>
            )}
          </div>
        </div>

        {/* Views */}
        <div className="flex-1 overflow-y-auto p-8">
          {activeSection === 'library' && (
            <LibraryView 
              files={files} 
              selectedIds={selectedIds} 
              onToggleSelect={handleToggleSelect} 
              onGoToUpload={() => setActiveSection('upload')}
            />
          )}
          {activeSection === 'upload' && (
            <UploadView 
              onUploadFile={handleUploadFile}
              onProcessLinks={handleProcessLinks}
              isUploading={isUploading}
            />
          )}
          {activeSection === 'output' && (
            <OutputView 
              files={outputFiles} 
              onGoToTool={(tool) => setActiveTool(tool)}
            />
          )}
        </div>
      </div>

      {/* 3. Right Panel */}
      <RightPanel 
        activeTool={activeTool} 
        onToolChange={setActiveTool}
        files={files}
        selectedIds={selectedIds}
        onGenerateSuccess={handleGenerateSuccess}
      />
    </div>
  );
};

export default KnowledgeBase;
