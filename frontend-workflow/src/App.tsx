import { useState } from 'react';
import ParticleBackground from './components/ParticleBackground';
import Paper2GraphPage from './components/Paper2GraphPage';
import Paper2PptPage from './components/Paper2PptPage';
import { Workflow, Zap } from 'lucide-react';

function App() {
  const [activePage, setActivePage] = useState<'paper2figure' | 'paper2ppt'>('paper2figure');

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
                Paper2Figure 生成科研绘图
              </button>
              <button
                onClick={() => setActivePage('paper2ppt')}
                className={`px-3 py-1.5 rounded-full text-sm ${
                  activePage === 'paper2ppt'
                    ? 'bg-primary-500 text-white shadow'
                    : 'glass text-gray-300 hover:bg-white/10'
                }`}
              >
                Paper2PPT 生成
              </button>
            </div>

            {/* 右侧操作按钮（保留占位，具体生成操作由各页面内部按钮触发） */}
            <div className="flex items-center gap-2">
              <button className="flex items-center gap-2 px-4 py-2 rounded-lg bg-primary-500 hover:bg-primary-600 transition-colors glow">
                <Zap size={18} className="text-white" />
                <span className="text-sm text-white font-medium">
                  {activePage === 'paper2figure' ? 'Paper2Figure' : 'Paper2PPT'}
                </span>
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* 主内容区 */}
      <main className="absolute top-16 bottom-0 left-0 right-0 flex">
        <div className="flex-1">
          {activePage === 'paper2figure' ? <Paper2GraphPage /> : <Paper2PptPage />}
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
