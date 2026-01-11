import { Grid, UploadCloud, Presentation, Bot, Settings2 } from 'lucide-react';
import { SectionType } from './types';

interface SidebarProps {
  activeSection: SectionType;
  onSectionChange: (section: SectionType) => void;
  filesCount: number;
  outputCount: number;
}

export const Sidebar = ({ activeSection, onSectionChange, filesCount, outputCount }: SidebarProps) => {
  const SidebarItem = ({ id, label, icon: Icon, count }: { id: SectionType, label: string, icon: any, count?: number }) => (
    <button
      onClick={() => onSectionChange(id)}
      className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all mb-2 ${
        activeSection === id
          ? 'bg-purple-500/20 text-purple-400 border border-purple-500/30'
          : 'text-gray-400 hover:bg-white/5 hover:text-gray-200 border border-transparent'
      }`}
    >
      <Icon size={18} />
      <span className="text-sm font-medium">{label}</span>
      {count !== undefined && count > 0 && (
        <span className="ml-auto text-xs bg-white/10 px-2 py-0.5 rounded-full">{count}</span>
      )}
    </button>
  );

  return (
    <div className="w-[260px] flex-shrink-0 flex flex-col border-r border-white/5 bg-[#050512]/80 backdrop-blur-xl relative z-20">
      <div className="p-6"></div>

      <nav className="flex-1 px-4 overflow-y-auto">
        <SidebarItem id="library" label="我的知识库" icon={Grid} count={filesCount} />
        <SidebarItem id="upload" label="上传文件" icon={UploadCloud} />
        <div className="h-px bg-white/5 my-4 mx-2" />
        <SidebarItem id="output" label="知识产出" icon={Presentation} count={outputCount} />
      </nav>
    </div>
  );
};
