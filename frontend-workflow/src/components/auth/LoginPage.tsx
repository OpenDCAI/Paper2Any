/**
 * Login page component.
 *
 * Email/password authentication form using Supabase Auth.
 */

import { useState, useEffect } from "react";
import { useAuthStore } from "../../stores/authStore";
import { Mail, Lock, AlertCircle, Loader2, ArrowRight, Sparkles, FileText, Presentation, Palette } from "lucide-react";

interface Props {
  onSwitchToRegister: () => void;
  footer?: React.ReactNode;
}

export function LoginPage({ onSwitchToRegister, footer }: Props) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  
  // 动态文字索引
  const [featureIndex, setFeatureIndex] = useState(0);

  const features = [
    {
      icon: Sparkles,
      title: "Paper2Figure",
      desc: "论文一键转科研绘图",
      color: "text-purple-400",
      bg: "bg-purple-500/10",
      border: "border-purple-500/20"
    },
    {
      icon: FileText,
      title: "Paper2PPT",
      desc: "论文内容智能生成 PPT，支持超级长PPT",
      color: "text-blue-400",
      bg: "bg-blue-500/10",
      border: "border-blue-500/20"
    },
    {
      icon: Presentation,
      title: "PDF2PPT",
      desc: "PDF版本PPT转文字图标可编辑",
      color: "text-pink-400",
      bg: "bg-pink-500/10",
      border: "border-pink-500/20"
    },
    {
      icon: Palette,
      title: "PPT Polish",
      desc: "专业级 PPT 智能润色",
      color: "text-emerald-400",
      bg: "bg-emerald-500/10",
      border: "border-emerald-500/20"
    }
  ];

  // 自动轮播功能
  useEffect(() => {
    const interval = setInterval(() => {
      setFeatureIndex((prev) => (prev + 1) % features.length);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const { signInWithEmail, loading, error, clearError } = useAuthStore();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    clearError();
    await signInWithEmail(email, password);
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#050512] p-4 relative overflow-hidden">
      {/* 动态背景装饰 */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-purple-600/20 rounded-full blur-[120px] animate-pulse"></div>
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-blue-600/20 rounded-full blur-[120px] animate-pulse delay-1000"></div>
      </div>

      <div className="w-full max-w-5xl grid grid-cols-1 lg:grid-cols-2 gap-8 items-center relative z-10">
        
        {/* 左侧：功能展示区 */}
        <div className="hidden lg:flex flex-col justify-center space-y-8 pr-8">
          <div>
            <h1 className="text-5xl font-bold text-white mb-4 leading-tight">
              Paper to <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400">Any</span>
            </h1>
            <p className="text-gray-400 text-lg max-w-md">
              一站式 AI 科研与演示文稿助手，让知识流动更自由，让创作更简单。
            </p>
          </div>

          <div className="space-y-4">
            {features.map((feature, idx) => (
              <div 
                key={idx}
                className={`transform transition-all duration-500 border rounded-xl p-4 flex items-center gap-4 ${
                  idx === featureIndex 
                    ? `scale-105 ${feature.bg} ${feature.border} shadow-lg shadow-purple-900/20 translate-x-4` 
                    : 'bg-white/5 border-white/5 opacity-60 hover:opacity-80 hover:translate-x-2'
                }`}
                onClick={() => setFeatureIndex(idx)}
              >
                <div className={`p-3 rounded-lg ${idx === featureIndex ? 'bg-white/10' : 'bg-white/5'}`}>
                  <feature.icon className={feature.color} size={24} />
                </div>
                <div>
                  <h3 className={`font-semibold text-lg ${idx === featureIndex ? 'text-white' : 'text-gray-300'}`}>
                    {feature.title}
                  </h3>
                  <p className="text-sm text-gray-400">{feature.desc}</p>
                </div>
                {idx === featureIndex && (
                  <div className="ml-auto">
                    <ArrowRight className="text-white/50 animate-bounce-x" size={20} />
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* 右侧：登录表单 */}
        <div className="glass-dark p-8 md:p-10 rounded-2xl w-full border border-white/10 shadow-2xl backdrop-blur-xl bg-black/40">
          <div className="lg:hidden mb-8 text-center">
             <h2 className="text-3xl font-bold text-white mb-2">Paper2Any</h2>
             <p className="text-gray-400 text-sm">AI 驱动的科研与演示助手</p>
          </div>

          <h2 className="text-2xl font-bold text-white mb-2">欢迎回来 👋</h2>
          <p className="text-gray-400 mb-8 text-sm">登录您的账户以继续使用</p>

          {error && (
            <div className="mb-6 p-4 bg-red-500/10 border border-red-500/20 rounded-xl flex items-start gap-3 text-red-300 animate-in fade-in slide-in-from-top-2">
              <AlertCircle size={20} className="mt-0.5 shrink-0" />
              <span className="text-sm leading-relaxed">{error}</span>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-5">
            <div className="space-y-1.5">
              <label className="block text-xs font-medium text-gray-400 ml-1">电子邮箱</label>
              <div className="relative group">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Mail className="text-gray-500 group-focus-within:text-purple-400 transition-colors" size={18} />
                </div>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full pl-10 pr-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-600 focus:outline-none focus:ring-2 focus:ring-purple-500/50 focus:border-purple-500/50 transition-all"
                  placeholder="name@example.com"
                  required
                  disabled={loading}
                />
              </div>
            </div>

            <div className="space-y-1.5">
              <div className="flex justify-between items-center ml-1">
                <label className="block text-xs font-medium text-gray-400">密码</label>
                {/* <a href="#" className="text-xs text-purple-400 hover:text-purple-300 transition-colors">忘记密码？</a> */}
              </div>
              <div className="relative group">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Lock className="text-gray-500 group-focus-within:text-purple-400 transition-colors" size={18} />
                </div>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full pl-10 pr-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-600 focus:outline-none focus:ring-2 focus:ring-purple-500/50 focus:border-purple-500/50 transition-all"
                  placeholder="请输入您的密码"
                  required
                  disabled={loading}
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full py-3.5 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white font-bold rounded-xl disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:scale-[1.02] active:scale-[0.98] shadow-lg shadow-purple-900/30 flex items-center justify-center gap-2 mt-4"
            >
              {loading ? (
                <>
                  <Loader2 size={20} className="animate-spin" />
                  <span>正在登录...</span>
                </>
              ) : (
                <>
                  <span>立即登录</span>
                  <ArrowRight size={18} />
                </>
              )}
            </button>
          </form>

          <div className="mt-8 text-center">
            <p className="text-gray-400 text-sm">
              还没有账号？{" "}
              <button
                onClick={onSwitchToRegister}
                className="text-purple-400 hover:text-purple-300 font-medium hover:underline transition-colors"
              >
                免费注册
              </button>
            </p>
          </div>

          {footer}
        </div>
      </div>

      <style>{`
        @keyframes bounce-x {
          0%, 100% { transform: translateX(0); }
          50% { transform: translateX(25%); }
        }
        .animate-bounce-x {
          animation: bounce-x 1s infinite;
        }
      `}</style>
    </div>
  );
}
