import { useTranslation } from 'react-i18next';
import {
  X,
  Sparkles,
  Presentation,
  FileText,
  ImagePlus,
  Wand2,
  BookOpen,
  FolderOpen,
  Network
} from 'lucide-react';

interface NavigationItem {
  id: string;
  labelKey: string;
  icon: any;
  gradient: string;
}

interface AppSidebarProps {
  isOpen: boolean;
  onClose: () => void;
  activePage: string;
  onPageChange: (page: string) => void;
}

export const AppSidebar = ({ isOpen, onClose, activePage, onPageChange }: AppSidebarProps) => {
  const { t } = useTranslation('common');

  const navigationItems: NavigationItem[] = [
    {
      id: 'paper2figure',
      labelKey: t('app.nav.paper2figure'),
      icon: Sparkles,
      gradient: 'from-primary-500 to-primary-600'
    },
    {
      id: 'paper2drawio',
      labelKey: 'Paper2Drawio',
      icon: Network,
      gradient: 'from-teal-500 to-cyan-500'
    },
    {
      id: 'paper2ppt',
      labelKey: t('app.nav.paper2ppt'),
      icon: Presentation,
      gradient: 'from-purple-500 to-pink-500'
    },
    {
      id: 'pdf2ppt',
      labelKey: t('app.nav.pdf2ppt'),
      icon: FileText,
      gradient: 'from-orange-500 to-red-500'
    },
    {
      id: 'image2ppt',
      labelKey: 'Image2PPT',
      icon: ImagePlus,
      gradient: 'from-cyan-500 to-blue-500'
    },
    {
      id: 'ppt2polish',
      labelKey: t('app.nav.ppt2polish'),
      icon: Wand2,
      gradient: 'from-cyan-500 to-teal-500'
    },
    {
      id: 'knowledge',
      labelKey: t('app.nav.knowledge'),
      icon: BookOpen,
      gradient: 'from-indigo-500 to-purple-500'
    },
    {
      id: 'files',
      labelKey: t('app.nav.files'),
      icon: FolderOpen,
      gradient: 'from-emerald-500 to-green-500'
    }
  ];

  const handleNavigation = (pageId: string) => {
    onPageChange(pageId);
    onClose();
  };

  return (
    <>
      {/* Backdrop Overlay */}
      <div
        className={`fixed inset-0 bg-black/60 backdrop-blur-sm z-30 transition-opacity duration-300 ${
          isOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'
        }`}
        onClick={onClose}
      />

      {/* Sidebar Panel */}
      <aside className={`fixed top-0 left-0 h-full w-[280px] glass-dark border-r border-white/10 z-40 transition-transform duration-300 ease-in-out ${
        isOpen ? 'translate-x-0' : '-translate-x-full'
      }`}>
        {/* Header */}
        <div className="h-16 flex items-center justify-between px-4 border-b border-white/10">
          <h2 className="text-lg font-bold text-white">{t('app.sidebar.navigation')}</h2>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white transition-colors"
            aria-label="Close sidebar"
          >
            <X size={20} />
          </button>
        </div>

        {/* Navigation Items */}
        <nav className="flex-1 p-4 overflow-y-auto">
          {navigationItems.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                onClick={() => handleNavigation(item.id)}
                className={`w-full flex items-center gap-3 px-4 py-3.5 rounded-xl transition-all duration-200 mb-2 ${
                  activePage === item.id
                    ? `bg-gradient-to-r ${item.gradient} text-white shadow-lg shadow-${item.gradient.split('-')[1]}-500/30 border border-white/20 scale-[1.02]`
                    : 'text-gray-300 bg-white/5 border border-white/10 hover:bg-white/10 hover:border-white/20 hover:text-white hover:shadow-md hover:scale-[1.02]'
                }`}
              >
                <Icon size={22} className={activePage === item.id ? 'drop-shadow-lg' : ''} />
                <span className="text-sm font-medium">{item.labelKey}</span>
              </button>
            );
          })}
        </nav>
      </aside>
    </>
  );
};
