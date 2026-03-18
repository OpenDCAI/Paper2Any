/**
 * Registration page component.
 *
 * Email/password signup form with password confirmation.
 * After signup, redirects to OTP verification if email confirmation is required.
 */

import { useState, useEffect } from "react";
import { useAuthStore } from "../../stores/authStore";
import { Mail, Lock, AlertCircle, Loader2, ArrowRight, Sparkles, FileText, Presentation, Palette } from "lucide-react";

interface Props {
  onSwitchToLogin: () => void;
  footer?: React.ReactNode;
}

export function RegisterPage({ onSwitchToLogin, footer }: Props) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [localError, setLocalError] = useState<string | null>(null);

  const [inviteCode, setInviteCode] = useState("");
  const INVITE_CODE_STORAGE_KEY = "paper2any_invite_code";
  
  // 动态文字索引
  const [featureIndex, setFeatureIndex] = useState(0);

  const features = [
    {
      icon: Sparkles,
      title: "Paper2Figure",
      desc: "论文一键转科研绘图",
      color: "text-primary-700",
      bg: "bg-primary-500/10",
      border: "border-primary-500/20"
    },
    {
      icon: FileText,
      title: "Paper2PPT",
      desc: "论文内容智能生成 PPT，支持超级长PPT",
      color: "text-amber-700",
      bg: "bg-amber-500/10",
      border: "border-amber-500/20"
    },
    {
      icon: Presentation,
      title: "PDF2PPT",
      desc: "PDF版本PPT转文字图标可编辑",
      color: "text-primary-600",
      bg: "bg-primary-500/8",
      border: "border-primary-500/16"
    },
    {
      icon: Palette,
      title: "PPT Polish",
      desc: "专业级 PPT 智能润色",
      color: "text-amber-600",
      bg: "bg-amber-500/8",
      border: "border-amber-500/16"
    }
  ];

  // 自动轮播功能
  useEffect(() => {
    const interval = setInterval(() => {
      setFeatureIndex((prev) => (prev + 1) % features.length);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const { signUpWithEmail, loading, error, clearError } = useAuthStore();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    clearError();
    setLocalError(null);

    try {
      if (inviteCode.trim()) {
        localStorage.setItem(INVITE_CODE_STORAGE_KEY, inviteCode.trim());
      }
    } catch {
      // ignore
    }

    // Client-side validation
    if (password !== confirmPassword) {
      setLocalError("两次输入的密码不一致");
      return;
    }

    if (password.length < 6) {
      setLocalError("密码长度至少为 6 位");
      return;
    }

    // signUpWithEmail will set needsOtpVerification if email confirmation is required
    // AuthGate will automatically show the OTP verification page
    await signUpWithEmail(email, password);
  };

  const displayError = localError || error;

  return (
    <div className="portal-shell min-h-screen flex items-center justify-center p-4 relative overflow-hidden">
      {/* 动态背景装饰 */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none">
        <div className="absolute top-[-10%] right-[-10%] w-[40%] h-[40%] bg-primary-500/12 rounded-full blur-[120px] animate-pulse"></div>
        <div className="absolute bottom-[-10%] left-[-10%] w-[40%] h-[40%] bg-amber-500/12 rounded-full blur-[120px] animate-pulse delay-1000"></div>
      </div>

      <div className="w-full max-w-5xl grid grid-cols-1 lg:grid-cols-2 gap-8 items-center relative z-10">
        
        {/* 左侧：功能展示区 (顺序交换，注册页放右边，或者保持左边一致性) 
            保持一致性通常更好，这里依旧放左侧 */}
        <div className="hidden lg:flex flex-col justify-center space-y-8 pr-8">
          <div>
            <h1 className="text-5xl font-bold font-display text-primary-900 mb-4 leading-tight">
              加入 <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary-700 to-amber-600">Paper2Any</span>
            </h1>
            <p className="text-slate-600 text-lg max-w-md">
              立即注册，开启 AI 驱动的科研创作新体验。
            </p>
          </div>

          <div className="space-y-4">
            {features.map((feature, idx) => (
              <div 
                key={idx}
                className={`transform transition-all duration-500 border rounded-xl p-4 flex items-center gap-4 ${
                  idx === featureIndex 
                    ? `scale-105 ${feature.bg} ${feature.border} shadow-lg shadow-primary-900/10 translate-x-4 portal-hero-card` 
                    : 'bg-white/60 border-primary-500/8 opacity-75 hover:opacity-100 hover:translate-x-2'
                }`}
                onClick={() => setFeatureIndex(idx)}
              >
                <div className={`p-3 rounded-2xl ${idx === featureIndex ? 'bg-white/80' : 'bg-white/60'}`}>
                  <feature.icon className={feature.color} size={24} />
                </div>
                <div>
                  <h3 className={`font-semibold text-lg ${idx === featureIndex ? 'text-primary-900' : 'text-slate-700'}`}>
                    {feature.title}
                  </h3>
                  <p className="text-sm text-slate-500">{feature.desc}</p>
                </div>
                {idx === featureIndex && (
                  <div className="ml-auto">
                    <ArrowRight className="text-primary-500 animate-bounce-x" size={20} />
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* 右侧：注册表单 */}
        <div className="portal-panel p-8 md:p-10 rounded-[28px] w-full shadow-shell">
          <div className="lg:hidden mb-8 text-center">
             <h2 className="text-3xl font-bold font-display text-primary-900 mb-2">Paper2Any</h2>
             <p className="text-slate-500 text-sm">创建您的新账号</p>
          </div>

          <h2 className="text-2xl font-bold font-display text-primary-900 mb-2">创建账号 ✨</h2>
          <p className="text-slate-500 mb-8 text-sm">填写以下信息以完成注册</p>

          {displayError && (
            <div className="mb-6 p-4 bg-red-500/10 border border-red-500/20 rounded-xl flex items-start gap-3 text-red-300 animate-in fade-in slide-in-from-top-2">
              <AlertCircle size={20} className="mt-0.5 shrink-0" />
              <span className="text-sm leading-relaxed">{displayError}</span>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-5">
            <div className="space-y-1.5">
              <label className="block text-xs font-medium text-slate-600 ml-1">电子邮箱</label>
              <div className="relative group">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Mail className="text-slate-400 group-focus-within:text-primary-600 transition-colors" size={18} />
                </div>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="portal-input w-full pl-10 pr-4 py-3 rounded-xl transition-all"
                  placeholder="name@example.com"
                  required
                  disabled={loading}
                />
              </div>
            </div>

            <div className="space-y-1.5">
              <label className="block text-xs font-medium text-slate-600 ml-1">邀请码（可选）</label>
              <input
                type="text"
                value={inviteCode}
                onChange={(e) => setInviteCode(e.target.value)}
                className="portal-input w-full px-4 py-3 rounded-xl transition-all"
                placeholder="填写邀请码可获奖励"
                disabled={loading}
              />
            </div>

            <div className="space-y-1.5">
              <label className="block text-xs font-medium text-slate-600 ml-1">密码</label>
              <div className="relative group">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Lock className="text-slate-400 group-focus-within:text-primary-600 transition-colors" size={18} />
                </div>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="portal-input w-full pl-10 pr-4 py-3 rounded-xl transition-all"
                  placeholder="至少 6 位字符"
                  required
                  disabled={loading}
                />
              </div>
            </div>

            <div className="space-y-1.5">
              <label className="block text-xs font-medium text-slate-600 ml-1">确认密码</label>
              <div className="relative group">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Lock className="text-slate-400 group-focus-within:text-primary-600 transition-colors" size={18} />
                </div>
                <input
                  type="password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  className="portal-input w-full pl-10 pr-4 py-3 rounded-xl transition-all"
                  placeholder="再次输入您的密码"
                  required
                  disabled={loading}
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="portal-button-primary w-full py-3.5 font-bold rounded-xl disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:scale-[1.02] active:scale-[0.98] flex items-center justify-center gap-2 mt-4"
            >
              {loading ? (
                <>
                  <Loader2 size={20} className="animate-spin" />
                  <span>正在注册...</span>
                </>
              ) : (
                <>
                  <span>立即注册</span>
                  <ArrowRight size={18} />
                </>
              )}
            </button>
          </form>

          <div className="mt-8 text-center">
            <p className="text-slate-500 text-sm">
              已有账号？{" "}
              <button
                onClick={onSwitchToLogin}
                className="text-primary-700 hover:text-primary-800 font-medium hover:underline transition-colors"
              >
                立即登录
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
