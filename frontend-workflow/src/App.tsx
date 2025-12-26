import { useState } from 'react';
import ParticleBackground from './components/ParticleBackground';
import Paper2GraphPage from './components/Paper2GraphPage';
import Paper2PptPage from './components/Paper2PptPage';
import Pdf2PptPage from './components/Pdf2PptPage';
import Ppt2PolishPage from './components/Ppt2PolishPage';
import { FilesPage } from './components/FilesPage';
import { QuotaDisplay } from './components/QuotaDisplay';
import { UserMenu } from './components/UserMenu';
import { Workflow } from 'lucide-react';

function App() {
  const [activePage, setActivePage] = useState<'paper2figure' | 'paper2ppt' | 'pdf2ppt' | 'ppt2polish' | 'files'>('paper2figure');

  return (
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
                DataFlow Agent
              </h1>
              <p className="text-xs text-gray-400">Workflow Editor</p>
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
                Paper2Figure
              </button>
              <button
                onClick={() => setActivePage('paper2ppt')}
                className={`px-3 py-1.5 rounded-full text-sm ${
                  activePage === 'paper2ppt'
                    ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white shadow'
                    : 'glass text-gray-300 hover:bg-white/10'
                }`}
              >
                Paper2PPT
              </button>
              <button
                onClick={() => setActivePage('pdf2ppt')}
                className={`px-3 py-1.5 rounded-full text-sm ${
                  activePage === 'pdf2ppt'
                    ? 'bg-gradient-to-r from-orange-500 to-red-500 text-white shadow'
                    : 'glass text-gray-300 hover:bg-white/10'
                }`}
              >
                Pdf2PPT
              </button>
              <button
                onClick={() => setActivePage('ppt2polish')}
                className={`px-3 py-1.5 rounded-full text-sm ${
                  activePage === 'ppt2polish'
                    ? 'bg-gradient-to-r from-cyan-500 to-teal-500 text-white shadow'
                    : 'glass text-gray-300 hover:bg-white/10'
                }`}
              >
                PptPolish
              </button>
              <button
                onClick={() => setActivePage('files')}
                className={`px-3 py-1.5 rounded-full text-sm ${
                  activePage === 'files'
                    ? 'bg-gradient-to-r from-emerald-500 to-green-500 text-white shadow'
                    : 'glass text-gray-300 hover:bg-white/10'
                }`}
              >
                My Files
              </button>
            </div>

            {/* 右侧：配额显示 & 用户菜单 */}
            <div className="flex items-center gap-3">
              <QuotaDisplay />
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
          {activePage === 'ppt2polish' && <Ppt2PolishPage />}
          {activePage === 'files' && <FilesPage />}
        </div>
      </main>

      {/* 底部状态栏 */}
      <footer className="absolute bottom-0 left-0 right-0 h-8 glass-dark border-t border-white/10 z-10">
        <div className="h-full px-4 flex items-center justify-between text-xs text-gray-500">
          <div className="flex items-center gap-4">
            <span>DataFlow Agent v1.0.0</span>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
              <span>就绪</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
