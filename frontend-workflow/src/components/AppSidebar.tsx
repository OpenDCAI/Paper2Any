import { useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  Home,
  X,
  Sparkles,
  Flame,
  Presentation,
  FileText,
  MonitorSmartphone,
  ImagePlus,
  Image,
  Wand2,
  BrainCircuit,
  // BookOpen,
  FolderOpen,
  Network,
  MessageSquare,
  ChevronRight,
  ArrowLeft,
  Video,
  LayoutTemplate,
  Quote
} from 'lucide-react';
import NavTooltip from './NavTooltip';

interface NavigationItem {
  id: string;
  labelKey: string;
  tooltipKey: string;
  icon: any;
  gradient: string;
  hot?: boolean;
}

interface AppSidebarProps {
  isOpen: boolean;
  onClose: () => void;
  activePage: string;
  onPageChange: (page: string) => void;
}

export const AppSidebar = ({ isOpen, onClose, activePage, onPageChange }: AppSidebarProps) => {
  const { t } = useTranslation('common');
  const [menuView, setMenuView] = useState<'main' | 'paper2figure' | 'paper2ppt'>('main');

  useEffect(() => {
    if (!isOpen) setMenuView('main');
  }, [isOpen]);

  const paper2figureChildren = useMemo(() => ([
    {
      id: 'paper2figure-model-drawio',
      labelKey: t('app.navSub.paper2figureModelDrawio'),
      tooltipKey: t('app.navSubTooltip.paper2figureModelDrawio'),
      icon: Wand2,
      gradient: 'from-emerald-500 to-teal-500'
    },
    {
      id: 'paper2figure-tech-exp',
      labelKey: t('app.navSub.paper2figureTechExp'),
      tooltipKey: t('app.navSubTooltip.paper2figureTechExp'),
      icon: Sparkles,
      gradient: 'from-sky-500 to-cyan-500'
    },
    {
      id: 'paper2drawio-ai',
      labelKey: t('app.navSub.paper2drawioAi'),
      tooltipKey: t('app.navSubTooltip.paper2drawioAi'),
      icon: Network,
      gradient: 'from-violet-500 to-fuchsia-500'
    }
  ]), [t]);

  const paper2pptChildren = useMemo(() => ([
    {
      id: 'paper2ppt-image',
      labelKey: t('app.navSub.paper2pptImage'),
      tooltipKey: t('app.navSubTooltip.paper2pptImage'),
      icon: Presentation,
      gradient: 'from-purple-500 to-pink-500'
    },
    {
      id: 'paper2ppt-frontend',
      labelKey: t('app.navSub.paper2pptFrontend'),
      tooltipKey: t('app.navSubTooltip.paper2pptFrontend'),
      icon: MonitorSmartphone,
      gradient: 'from-amber-500 to-orange-500'
    }
  ]), [t]);

  const navigationItems: NavigationItem[] = [
    {
      id: 'home',
      labelKey: t('app.nav.home'),
      tooltipKey: t('app.navTooltip.home'),
      icon: Home,
      gradient: 'from-slate-500 to-cyan-500'
    },
    {
      id: 'paper2figure',
      labelKey: t('app.nav.paper2figure'),
      tooltipKey: t('app.navTooltip.paper2figure'),
      icon: Sparkles,
      gradient: 'from-primary-500 to-primary-600'
    },
    {
      id: 'image-playground',
      labelKey: t('app.nav.imagePlayground'),
      tooltipKey: t('app.navTooltip.imagePlayground'),
      icon: Flame,
      gradient: 'from-orange-500 to-rose-500',
      hot: true,
    },
    {
      id: 'mindmap',
      labelKey: t('app.nav.mindmap'),
      tooltipKey: t('app.navTooltip.mindmap'),
      icon: BrainCircuit,
      gradient: 'from-cyan-500 to-blue-500'
    },
    {
      id: 'image2drawio',
      labelKey: t('app.nav.image2drawio'),
      tooltipKey: t('app.navTooltip.image2drawio'),
      icon: Image,
      gradient: 'from-amber-500 to-lime-500'
    },
    {
      id: 'paper2rebuttal',
      labelKey: t('app.nav.paper2rebuttal'),
      tooltipKey: t('app.navTooltip.paper2rebuttal'),
      icon: MessageSquare,
      gradient: 'from-rose-500 to-pink-500'
    },
    {
      id: 'paper2ppt',
      labelKey: t('app.nav.paper2ppt'),
      tooltipKey: t('app.navTooltip.paper2ppt'),
      icon: Presentation,
      gradient: 'from-purple-500 to-pink-500'
    },
    {
      id: 'paper2video',
      labelKey: t('app.nav.paper2video'),
      tooltipKey: t('app.navTooltip.paper2video'),
      icon: Video,
      gradient: 'from-teal-500 to-cyan-500'
    },
    {
      id: 'paper2poster',
      labelKey: t('app.nav.paper2poster'),
      tooltipKey: t('app.navTooltip.paper2poster'),
      icon: LayoutTemplate,
      gradient: 'from-fuchsia-500 to-rose-500'
    },
    {
      id: 'paper2citation',
      labelKey: t('app.nav.paper2citation'),
      tooltipKey: t('app.navTooltip.paper2citation'),
      icon: Quote,
      gradient: 'from-cyan-500 to-sky-500'
    },
    {
      id: 'ppt2polish',
      labelKey: t('app.nav.ppt2polish'),
      tooltipKey: t('app.navTooltip.ppt2polish'),
      icon: Wand2,
      gradient: 'from-cyan-500 to-teal-500'
    },
    {
      id: 'pdf2ppt',
      labelKey: t('app.nav.pdf2ppt'),
      tooltipKey: t('app.navTooltip.pdf2ppt'),
      icon: FileText,
      gradient: 'from-orange-500 to-red-500'
    },
    {
      id: 'image2ppt',
      labelKey: t('app.nav.image2ppt'),
      tooltipKey: t('app.navTooltip.image2ppt'),
      icon: ImagePlus,
      gradient: 'from-cyan-500 to-blue-500'
    },
    // {
    //   id: 'knowledge',
    //   labelKey: t('app.nav.knowledge'),
    //   tooltipKey: t('app.navTooltip.knowledge'),
    //   icon: BookOpen,
    //   gradient: 'from-indigo-500 to-purple-500'
    // },
    {
      id: 'files',
      labelKey: t('app.nav.files'),
      tooltipKey: t('app.navTooltip.files'),
      icon: FolderOpen,
      gradient: 'from-emerald-500 to-green-500'
    }
  ];

  const handleNavigation = (pageId: string) => {
    onPageChange(pageId);
    onClose();
  };

  const paper2figureActive = paper2figureChildren.some(child => child.id === activePage);
  const paper2pptActive = paper2pptChildren.some(child => child.id === activePage) || activePage === 'paper2ppt';
  const activeSubmenu = menuView === 'paper2figure'
    ? { title: t('app.nav.paper2figure'), items: paper2figureChildren }
    : menuView === 'paper2ppt'
      ? { title: t('app.nav.paper2ppt'), items: paper2pptChildren }
      : null;

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
      <aside className={`fixed top-0 left-0 h-full w-[280px] glass-dark border-r border-white/10 z-40 transition-transform duration-300 ease-in-out overflow-hidden flex flex-col ${
        isOpen ? 'translate-x-0' : '-translate-x-full'
      }`}>
        {/* Header */}
        <div className="h-16 flex items-center justify-between px-4 border-b border-white/10">
          <div className="flex items-center gap-2">
            {menuView !== 'main' && (
              <button
                onClick={() => setMenuView('main')}
                className="p-2 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white transition-colors"
                aria-label="Back"
              >
                <ArrowLeft size={18} />
              </button>
            )}
            <h2 className="text-lg font-bold text-white">
              {activeSubmenu ? activeSubmenu.title : t('app.sidebar.navigation')}
            </h2>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white transition-colors"
            aria-label="Close sidebar"
          >
            <X size={20} />
          </button>
        </div>

        {/* Navigation Items */}
        <nav className="flex-1 overflow-hidden relative">
          <div
            className="absolute inset-0 p-4 overflow-y-auto overflow-x-hidden transition-transform duration-300"
            style={{ transform: menuView === 'main' ? 'translateX(0)' : 'translateX(-100%)' }}
          >
              {navigationItems.map((item) => {
                const Icon = item.icon;
                const isPaper2Figure = item.id === 'paper2figure';
                const isPaper2Ppt = item.id === 'paper2ppt';
                const hasSubmenu = isPaper2Figure || isPaper2Ppt;
                const isActive = isPaper2Figure
                  ? paper2figureActive
                  : isPaper2Ppt
                    ? paper2pptActive
                    : activePage === item.id;

                const button = (
                  <button
                    onClick={() => {
                      if (isPaper2Figure) {
                        setMenuView('paper2figure');
                        return;
                      }
                      if (isPaper2Ppt) {
                        setMenuView('paper2ppt');
                        return;
                      }
                      handleNavigation(item.id);
                    }}
                    className={`w-full flex items-center gap-3 px-4 py-3.5 rounded-xl transition-all duration-200 mb-2 ${
                      isActive
                        ? `bg-gradient-to-r ${item.gradient} text-white shadow-lg shadow-${item.gradient.split('-')[1]}-500/30 border border-white/20 scale-[1.02]`
                        : 'text-gray-300 bg-white/5 border border-white/10 hover:bg-white/10 hover:border-white/20 hover:text-white hover:shadow-md hover:scale-[1.02]'
                    }`}
                  >
                    <Icon size={22} className={isActive ? 'drop-shadow-lg' : ''} />
                    <span className="text-sm font-medium flex-1 text-left">{item.labelKey}</span>
                    {item.hot && (
                      <span className="rounded-full bg-orange-500 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-white">
                        HOT
                      </span>
                    )}
                    {hasSubmenu && (
                      <ChevronRight size={16} className="text-white/60 group-hover:text-white transition-colors" />
                    )}
                  </button>
                );

                return (
                  <div key={item.id} className="relative">
                    {hasSubmenu ? button : (
                      <NavTooltip content={item.tooltipKey}>
                        {button}
                      </NavTooltip>
                    )}
                  </div>
                );
              })}
          </div>

          <div
            className="absolute inset-0 p-4 overflow-y-auto overflow-x-hidden transition-transform duration-300"
            style={{ transform: menuView === 'main' ? 'translateX(100%)' : 'translateX(0)' }}
          >
            {(activeSubmenu?.items || []).map((child) => {
              const ChildIcon = child.icon;
              const isChildActive = activePage === child.id;
              return (
                <NavTooltip key={child.id} content={child.tooltipKey}>
                  <button
                    onClick={() => handleNavigation(child.id)}
                    className={`w-full flex items-center gap-3 px-4 py-3.5 rounded-xl text-left transition-all mb-2 ${
                      isChildActive
                        ? `bg-gradient-to-r ${child.gradient} text-white shadow-lg shadow-${child.gradient.split('-')[1]}-500/20`
                        : 'text-slate-200 bg-white/5 border border-white/10 hover:bg-white/10 hover:border-white/20'
                    }`}
                  >
                    <ChildIcon size={20} />
                    <span className="text-sm font-semibold">{child.labelKey}</span>
                  </button>
                </NavTooltip>
              );
            })}
          </div>
        </nav>
      </aside>
    </>
  );
};
