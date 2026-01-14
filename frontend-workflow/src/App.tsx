import { useState } from 'react';
import ParticleBackground from './components/ParticleBackground';
import Paper2GraphPage from './components/Paper2GraphPage';
import Paper2PptPage from './components/Paper2PptPage';
import Pdf2PptPage from './components/Pdf2PptPage';
import Image2PptPage from './components/Image2PptPage';
import Ppt2PolishPage from './components/Ppt2PolishPage';
// import KnowledgeBasePage from './components/KnowledgeBasePage';
import { FilesPage } from './components/FilesPage';
import { useTranslation } from 'react-i18next';
import { PricingProvider } from './contexts/PricingContext';

import { UserMenu } from './components/UserMenu';
import { LanguageSwitcher } from './components/LanguageSwitcher';
import { Workflow } from 'lucide-react';

function App() {
  const { t } = useTranslation('common');
  const [activePage, setActivePage] = useState<'paper2figure' | 'paper2ppt' | 'pdf2ppt' | 'image2ppt' | 'ppt2polish' | 'knowledge' | 'files'>('paper2figure');

  return (
    <PricingProvider>
      <div className="w-screen h-screen bg-[#0a0a1a] overflow-hidden relative">
      {/* 粒子背景 */}
      <ParticleBackground />

      {/* 顶部导航栏 */}
      <header className="absolute top-0 left-0 right-0 h-16 glass-dark border-b border-white/10 z-10">
        <div className="h-full px-6 flex items-center justify-between">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-primary-500/20">
              <Workflow className="text-primary-400" size={24} />
            </div>
            <div>
              <h1 className="text-lg font-bold text-white glow-text">
                Paper2Any
              </h1>
              <p className="text-xs text-gray-400">{t('app.subtitle')}</p>
            </div>
          </div>

          {/* 工具栏 */}
          <div className="flex items-center gap-4">
            {/* 页面切换 Tab */}
            <div className="flex items-center gap-2">
              <button
                onClick={() => setActivePage('paper2figure')}
                className={`px-3 py-1.5 rounded-full text-sm ${
                  activePage === 'paper2figure'
                    ? 'bg-primary-500 text-white shadow'
                    : 'glass text-gray-300 hover:bg-white/10'
                }`}
              >
                {t('app.nav.paper2figure')}
              </button>
              <button
                onClick={() => setActivePage('paper2ppt')}
                className={`px-3 py-1.5 rounded-full text-sm ${
                  activePage === 'paper2ppt'
                    ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white shadow'
                    : 'glass text-gray-300 hover:bg-white/10'
                }`}
              >
                {t('app.nav.paper2ppt')}
              </button>
              <button
                onClick={() => setActivePage('pdf2ppt')}
                className={`px-3 py-1.5 rounded-full text-sm ${
                  activePage === 'pdf2ppt'
                    ? 'bg-gradient-to-r from-orange-500 to-red-500 text-white shadow'
                    : 'glass text-gray-300 hover:bg-white/10'
                }`}
              >
                {t('app.nav.pdf2ppt')}
              </button>
              <button
                onClick={() => setActivePage('image2ppt')}
                className={`px-3 py-1.5 rounded-full text-sm ${
                  activePage === 'image2ppt'
                    ? 'bg-gradient-to-r from-cyan-500 to-blue-500 text-white shadow'
                    : 'glass text-gray-300 hover:bg-white/10'
                }`}
              >
                Image2PPT
              </button>
              <button
                onClick={() => setActivePage('ppt2polish')}
                className={`px-3 py-1.5 rounded-full text-sm ${
                  activePage === 'ppt2polish'
                    ? 'bg-gradient-to-r from-cyan-500 to-teal-500 text-white shadow'
                    : 'glass text-gray-300 hover:bg-white/10'
                }`}
              >
                {t('app.nav.ppt2polish')}
              </button>
              {/* <button
                onClick={() => setActivePage('knowledge')}
                className={`px-3 py-1.5 rounded-full text-sm ${
                  activePage === 'knowledge'
                    ? 'bg-gradient-to-r from-indigo-500 to-purple-500 text-white shadow'
                    : 'glass text-gray-300 hover:bg-white/10'
                }`}
              >
                {t('app.nav.knowledge')}
              </button> */}
              <button
                onClick={() => setActivePage('files')}
                className={`px-3 py-1.5 rounded-full text-sm ${
                  activePage === 'files'
                    ? 'bg-gradient-to-r from-emerald-500 to-green-500 text-white shadow'
                    : 'glass text-gray-300 hover:bg-white/10'
                }`}
              >
                {t('app.nav.files')}
              </button>
            </div>

            {/* 右侧：用户菜单 */}
            <div className="flex items-center gap-3">
              <LanguageSwitcher />
              <UserMenu />
            </div>
          </div>
        </div>
      </header>

      {/* 主内容区 */}
      <main className="absolute top-16 bottom-0 left-0 right-0 flex">
        <div className="flex-1">
          {activePage === 'paper2figure' && <Paper2GraphPage />}
          {activePage === 'paper2ppt' && <Paper2PptPage />}
          {activePage === 'pdf2ppt' && <Pdf2PptPage />}
          {activePage === 'image2ppt' && <Image2PptPage />}
          {activePage === 'ppt2polish' && <Ppt2PolishPage />}
          {/* {activePage === 'knowledge' && <KnowledgeBasePage />} */}
          {activePage === 'files' && <FilesPage />}
        </div>
      </main>

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
    </div>
    </PricingProvider>
  );
}

export default App;
