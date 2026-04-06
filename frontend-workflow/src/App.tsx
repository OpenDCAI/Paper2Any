import { useEffect, useState } from 'react';
import ParticleBackground from './components/ParticleBackground';
import Paper2GraphTechExpPage from './components/Paper2GraphTechExpPage';
import Paper2GraphDrawioPage from './components/Paper2GraphDrawioPage';
import Paper2PptPage from './components/Paper2PptPage';
import Pdf2PptPage from './components/Pdf2PptPage';
import Image2PptPage from './components/Image2PptPage';
import Image2DrawioPage from './components/Image2DrawioPage';
import Ppt2PolishPage from './components/Ppt2PolishPage';
import KnowledgeBasePage from './components/KnowledgeBasePage';
import { FilesPage } from './components/FilesPage';
import Paper2DrawioAiPage from './components/Paper2DrawioAiPage';
import Paper2DrawioPage from './components/paper2drawio';
import MindMapPage from './components/MindMapPage';
import Paper2RebuttalPage from './components/Paper2RebuttalPage';
import Paper2VideoPage from './components/Paper2VideoPage';
import Paper2PosterPage from './components/Paper2PosterPage';
import Paper2CitationPage from './components/Paper2CitationPage';
import { AccountPage } from './components/AccountPage';
import { useTranslation } from 'react-i18next';
import { PointsDisplay } from './components/PointsDisplay';
import { PurchaseEntry } from './components/PurchaseEntry';
import { UserMenu } from './components/UserMenu';
import { LanguageSwitcher } from './components/LanguageSwitcher';
import { Workflow, X, Menu } from 'lucide-react';
import { AppSidebar } from './components/AppSidebar';
import { HomePage } from './components/HomePage';

const pageIds = [
  'home',
  'paper2figure-tech-exp',
  'paper2figure-model-drawio',
  'paper2drawio-ai',
  'mindmap',
  'paper2ppt',
  'paper2ppt-image',
  'paper2ppt-frontend',
  'paper2video',
  'paper2poster',
  'paper2citation',
  'pdf2ppt',
  'image2ppt',
  'image2drawio',
  'ppt2polish',
  'knowledge',
  'files',
  'paper2drawio',
  'paper2rebuttal',
] as const;

type ActivePage = typeof pageIds[number];

const DEFAULT_PAGE: ActivePage = 'home';

const pagePaths: Record<ActivePage, string> = {
  'home': '/',
  'paper2figure-tech-exp': '/paper2figure/tech-exp',
  'paper2figure-model-drawio': '/paper2figure/model-drawio',
  'paper2drawio-ai': '/paper2drawio-ai',
  'mindmap': '/mindmap',
  'paper2ppt': '/paper2ppt',
  'paper2ppt-image': '/paper2ppt/image',
  'paper2ppt-frontend': '/paper2ppt/frontend',
  'paper2video': '/paper2video',
  'paper2poster': '/paper2poster',
  'paper2citation': '/paper2citation',
  'pdf2ppt': '/pdf2ppt',
  'image2ppt': '/image2ppt',
  'image2drawio': '/image2drawio',
  'ppt2polish': '/ppt2polish',
  'knowledge': '/knowledge',
  'files': '/files',
  'paper2drawio': '/paper2drawio',
  'paper2rebuttal': '/paper2rebuttal',
};

function normalizePathname(pathname: string): string {
  if (!pathname || pathname === '/') {
    return '/';
  }

  return pathname.endsWith('/') ? pathname.slice(0, -1) : pathname;
}

const pathToPage = new Map<string, ActivePage>(
  Object.entries(pagePaths).flatMap(([page, path]) => {
    const normalizedPath = normalizePathname(path);
    return [
      [normalizedPath, page as ActivePage],
      [page, page as ActivePage],
    ];
  }),
);

function getPageFromLegacyHash(hash: string): ActivePage | null {
  const normalizedHash = hash.replace(/^#/, '').trim();

  if (!normalizedHash) {
    return null;
  }

  const legacyPath = normalizePathname(
    normalizedHash.startsWith('/') ? normalizedHash : `/${normalizedHash}`,
  );

  return pathToPage.get(legacyPath) ?? null;
}

function getPageFromLocation(pathname: string, hash: string): ActivePage {
  const pageFromHash = getPageFromLegacyHash(hash);
  if (pageFromHash) {
    return pageFromHash;
  }

  const normalizedPath = normalizePathname(pathname);
  if (normalizedPath === '/') {
    return DEFAULT_PAGE;
  }

  return pathToPage.get(normalizedPath) ?? DEFAULT_PAGE;
}

function App() {
  const { t } = useTranslation('common');
  const [activePage, setActivePage] = useState<ActivePage>(() => {
    if (typeof window === 'undefined') {
      return DEFAULT_PAGE;
    }
    return getPageFromLocation(window.location.pathname, window.location.hash);
  });
  const [showFilesModal, setShowFilesModal] = useState(false);
  const [showAccountModal, setShowAccountModal] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  useEffect(() => {
    if (typeof window === 'undefined') {
      return undefined;
    }

    const syncPageFromLocation = () => {
      const nextPage = getPageFromLocation(window.location.pathname, window.location.hash);
      setActivePage(nextPage);

      const nextPath = pagePaths[nextPage];
      const currentPath = normalizePathname(window.location.pathname);
      if (window.location.hash || currentPath !== nextPath) {
        window.history.replaceState(null, '', `${nextPath}${window.location.search}`);
      }
    };

    syncPageFromLocation();
    window.addEventListener('popstate', syncPageFromLocation);

    return () => {
      window.removeEventListener('popstate', syncPageFromLocation);
    };
  }, []);

  const handlePageChange = (page: ActivePage) => {
    if (typeof window === 'undefined') {
      setActivePage(page);
      return;
    }

    const nextPath = pagePaths[page];
    const currentPath = normalizePathname(window.location.pathname);
    if (currentPath === nextPath && !window.location.hash) {
      setActivePage(page);
      return;
    }

    window.history.pushState(null, '', `${nextPath}${window.location.search}`);
    setActivePage(page);
  };

  return (
    <div className="w-screen h-screen bg-[#0a0a1a] overflow-hidden relative">
      {/* 粒子背景 */}
      <ParticleBackground />

      {/* 顶部导航栏 */}
      <header className="absolute top-0 left-0 right-0 h-16 glass-dark border-b border-white/10 z-10">
        <div className="h-full px-6 flex items-center justify-between">
          {/* Logo */}
          <div className="flex items-center gap-3">
            {/* Hamburger Menu Button */}
            <button
              onClick={() => setSidebarOpen(true)}
              className="group flex items-center gap-2 px-3 py-2 rounded-xl glass border border-white/10 text-gray-300 hover:bg-white/10 hover:text-white transition-all duration-200 shadow-[0_10px_30px_rgba(0,0,0,0.2)]"
              aria-label={t('app.sidebar.toggle')}
            >
              <span className="relative">
                <Menu size={20} />
                <span className="absolute -top-1 -right-1 h-2 w-2 rounded-full bg-emerald-400 animate-ping" />
                <span className="absolute -top-1 -right-1 h-2 w-2 rounded-full bg-emerald-400" />
              </span>
              <span className="text-xs font-semibold tracking-wide">菜单 / Menu</span>
            </button>
            <button
              type="button"
              onClick={() => handlePageChange('home')}
              className="flex items-center gap-3 rounded-2xl border border-white/10 bg-white/5 px-3 py-2 text-left transition-all duration-200 hover:border-white/20 hover:bg-white/10"
            >
              <div className="p-2 rounded-lg bg-primary-500/20">
                <Workflow className="text-primary-400" size={24} />
              </div>
              <div>
                <h1 className="text-lg font-bold text-white glow-text">
                  Paper2Any
                </h1>
                <p className="text-xs text-gray-400">{t('app.subtitle')}</p>
              </div>
            </button>
          </div>

          {/* 工具栏 */}
          <div className="flex items-center gap-4">
            {/* 右侧：配额显示 & 用户菜单 */}
            <div className="flex items-center gap-3">
              <LanguageSwitcher />
              <PointsDisplay />
              <PurchaseEntry />
              <UserMenu 
                onShowFiles={() => setShowFilesModal(true)}
                onShowAccount={() => setShowAccountModal(true)}
              />
            </div>
          </div>
        </div>
      </header>

      {/* 主内容区 */}
      <main className="absolute top-16 bottom-8 left-0 right-0 flex">
        <div className="flex-1">
          {activePage === 'home' && <HomePage onNavigate={handlePageChange} />}
          {activePage === 'paper2figure-tech-exp' && <Paper2GraphTechExpPage />}
          {activePage === 'paper2figure-model-drawio' && <Paper2GraphDrawioPage />}
          {activePage === 'paper2drawio-ai' && <Paper2DrawioAiPage />}
          {activePage === 'mindmap' && <MindMapPage />}
          {(activePage === 'paper2ppt' || activePage === 'paper2ppt-image') && (
            <Paper2PptPage initialMode="image" />
          )}
          {activePage === 'paper2ppt-frontend' && <Paper2PptPage initialMode="frontend" />}
          {activePage === 'paper2video' && <Paper2VideoPage />}
          {activePage === 'paper2poster' && <Paper2PosterPage />}
          {activePage === 'paper2citation' && <Paper2CitationPage />}
          {activePage === 'pdf2ppt' && <Pdf2PptPage />}
          {activePage === 'image2ppt' && <Image2PptPage />}
          {activePage === 'image2drawio' && <Image2DrawioPage />}
          {activePage === 'ppt2polish' && <Ppt2PolishPage />}
          {activePage === 'knowledge' && <KnowledgeBasePage />}
          {activePage === 'files' && <FilesPage />}
          {activePage === 'paper2drawio' && <Paper2DrawioPage />}
          {activePage === 'paper2rebuttal' && <Paper2RebuttalPage />}
        </div>
      </main>

      {/* 历史文件模态框 */}
      {showFilesModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="w-full max-w-6xl h-[80vh] m-4 glass-dark rounded-2xl border border-white/10 shadow-2xl flex flex-col">
            <div className="flex items-center justify-between p-4 border-b border-white/10">
              <h2 className="text-xl font-bold text-white">历史文件</h2>
              <button
                onClick={() => setShowFilesModal(false)}
                className="p-2 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white transition-colors"
              >
                <X size={20} />
              </button>
            </div>
            <div className="flex-1 overflow-hidden">
              <FilesPage />
            </div>
          </div>
        </div>
      )}

      {/* 账户设置模态框 */}
      {showAccountModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="w-full max-w-6xl h-[80vh] m-4 glass-dark rounded-2xl border border-white/10 shadow-2xl flex flex-col">
            <div className="flex items-center justify-between p-4 border-b border-white/10">
              <h2 className="text-xl font-bold text-white">账户设置</h2>
              <button
                onClick={() => setShowAccountModal(false)}
                className="p-2 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white transition-colors"
              >
                <X size={20} />
              </button>
            </div>
            <div className="flex-1 overflow-hidden">
              <AccountPage />
            </div>
          </div>
        </div>
      )}

      {/* 底部状态栏 */}
      <footer className="absolute bottom-0 left-0 right-0 h-8 glass-dark border-t border-white/10 z-10">
        <div className="h-full px-4 flex items-center justify-between text-xs text-gray-500">
          <div className="flex items-center gap-4">
            <span>{t('app.footer.version')}</span>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
              <span>{t('app.footer.ready')}</span>
            </div>
          </div>
        </div>
      </footer>

      {/* 侧边栏 */}
      <AppSidebar
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        activePage={activePage}
        onPageChange={(page) => {
          handlePageChange(page as ActivePage);
          setSidebarOpen(false);
        }}
      />
    </div>
  );
}

export default App;
