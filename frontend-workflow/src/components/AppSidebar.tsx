import { useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  X,
  Sparkles,
  Presentation,
  FileText,
  ImagePlus,
  Image,
  Wand2,
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
}

interface AppSidebarProps {
  isOpen: boolean;
  onClose: () => void;
  activePage: string;
  onPageChange: (page: string) => void;
}

export const AppSidebar = ({ isOpen, onClose, activePage, onPageChange }: AppSidebarProps) => {
  const { t } = useTranslation('common');
  const [menuView, setMenuView] = useState<'main' | 'paper2figure'>('main');

  useEffect(() => {
    if (!isOpen) setMenuView('main');
  }, [isOpen]);

  const paper2figureChildren = useMemo(() => ([
    {
      id: 'paper2figure-model-drawio',
      labelKey: t('app.navSub.paper2figureModelDrawio'),
      tooltipKey: t('app.navSubTooltip.paper2figureModelDrawio'),
      icon: Wand2,
      gradient: 'from-primary-600 to-amber-500'
    },
    {
      id: 'paper2figure-tech-exp',
      labelKey: t('app.navSub.paper2figureTechExp'),
      tooltipKey: t('app.navSubTooltip.paper2figureTechExp'),
      icon: Sparkles,
      gradient: 'from-primary-500 to-primary-700'
    },
    {
      id: 'paper2drawio-ai',
      labelKey: t('app.navSub.paper2drawioAi'),
      tooltipKey: t('app.navSubTooltip.paper2drawioAi'),
      icon: Network,
      gradient: 'from-primary-700 to-amber-500'
    }
  ]), [t]);

  const navigationItems: NavigationItem[] = [
    {
      id: 'paper2figure',
      labelKey: t('app.nav.paper2figure'),
      tooltipKey: t('app.navTooltip.paper2figure'),
      icon: Sparkles,
      gradient: 'from-primary-500 to-primary-600'
    },
    {
      id: 'image2drawio',
      labelKey: t('app.nav.image2drawio'),
      tooltipKey: t('app.navTooltip.image2drawio'),
      icon: Image,
      gradient: 'from-amber-500 to-primary-500'
    },
    {
      id: 'paper2rebuttal',
      labelKey: t('app.nav.paper2rebuttal'),
      tooltipKey: t('app.navTooltip.paper2rebuttal'),
      icon: MessageSquare,
      gradient: 'from-primary-600 to-primary-500'
    },
    {
      id: 'paper2ppt',
      labelKey: t('app.nav.paper2ppt'),
      tooltipKey: t('app.navTooltip.paper2ppt'),
      icon: Presentation,
      gradient: 'from-primary-700 to-amber-500'
    },
    {
      id: 'paper2video',
      labelKey: t('app.nav.paper2video'),
      tooltipKey: t('app.navTooltip.paper2video'),
      icon: Video,
      gradient: 'from-primary-600 to-amber-500'
    },
    {
      id: 'paper2poster',
      labelKey: t('app.nav.paper2poster'),
      tooltipKey: t('app.navTooltip.paper2poster'),
      icon: LayoutTemplate,
      gradient: 'from-primary-500 to-amber-500'
    },
    {
      id: 'paper2citation',
      labelKey: t('app.nav.paper2citation'),
      tooltipKey: t('app.navTooltip.paper2citation'),
      icon: Quote,
      gradient: 'from-primary-600 to-primary-500'
    },
    {
      id: 'ppt2polish',
      labelKey: t('app.nav.ppt2polish'),
      tooltipKey: t('app.navTooltip.ppt2polish'),
      icon: Wand2,
      gradient: 'from-primary-600 to-amber-500'
    },
    {
      id: 'pdf2ppt',
      labelKey: t('app.nav.pdf2ppt'),
      tooltipKey: t('app.navTooltip.pdf2ppt'),
      icon: FileText,
      gradient: 'from-amber-500 to-primary-500'
    },
    {
      id: 'image2ppt',
      labelKey: t('app.nav.image2ppt'),
      tooltipKey: t('app.navTooltip.image2ppt'),
      icon: ImagePlus,
      gradient: 'from-primary-500 to-amber-500'
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
      gradient: 'from-primary-600 to-amber-500'
    }
  ];

  const handleNavigation = (pageId: string) => {
    onPageChange(pageId);
    onClose();
  };

  const paper2figureActive = paper2figureChildren.some(child => child.id === activePage);

  return (
    <>
      {/* Backdrop Overlay */}
      <div
        className={`fixed inset-0 bg-[rgba(69,39,48,0.22)] backdrop-blur-sm z-30 transition-opacity duration-300 ${
          isOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'
        }`}
        onClick={onClose}
      />

      {/* Sidebar Panel */}
      <aside className={`portal-sidebar fixed top-0 left-0 h-full w-[280px] z-40 transition-transform duration-300 ease-in-out overflow-hidden flex flex-col ${
        isOpen ? 'translate-x-0' : '-translate-x-full'
      }`}>
        {/* Header */}
        <div className="portal-sidebar-header h-16 flex items-center justify-between px-4">
          <div className="flex items-center gap-2">
            {menuView === 'paper2figure' && (
              <button
                onClick={() => setMenuView('main')}
                className="portal-sidebar-icon-button"
                aria-label="Back"
              >
                <ArrowLeft size={18} />
              </button>
            )}
            <h2 className="text-lg font-bold font-display text-primary-900">
              {menuView === 'paper2figure' ? t('app.nav.paper2figure') : t('app.sidebar.navigation')}
            </h2>
          </div>
          <button
            onClick={onClose}
            className="portal-sidebar-icon-button"
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
                const isActive = isPaper2Figure ? paper2figureActive : activePage === item.id;

                const button = (
                  <button
                    onClick={() => {
                      if (isPaper2Figure) {
                        setMenuView('paper2figure');
                        return;
                      }
                      handleNavigation(item.id);
                    }}
                    className={`portal-sidebar-item ${isActive ? 'portal-sidebar-item-active' : ''}`}
                  >
                    <Icon size={22} className="shrink-0" />
                    <span className="text-sm font-semibold flex-1 text-left leading-6">{item.labelKey}</span>
                    {isPaper2Figure && (
                      <ChevronRight size={16} className={`shrink-0 ${isActive ? 'text-[rgba(255,241,225,0.86)]' : 'text-[rgba(103,95,88,0.88)]'}`} />
                    )}
                  </button>
                );

                return (
                  <div key={item.id} className="relative">
                    {isPaper2Figure ? button : (
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
            {paper2figureChildren.map((child) => {
              const ChildIcon = child.icon;
              const isChildActive = activePage === child.id;
              return (
                <NavTooltip key={child.id} content={child.tooltipKey}>
                  <button
                    onClick={() => handleNavigation(child.id)}
                    className={`portal-sidebar-item ${isChildActive ? 'portal-sidebar-item-active' : ''}`}
                  >
                    <ChildIcon size={20} className="shrink-0" />
                    <span className="text-sm font-semibold leading-6">{child.labelKey}</span>
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
