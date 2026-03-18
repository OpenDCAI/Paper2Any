/**
 * Login page component.
 *
 * Tab-based login with email/password and phone OTP options.
 */

import { useState, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { useAuthStore } from "../../stores/authStore";
import { Mail, Lock, AlertCircle, Loader2, ArrowRight, Sparkles, FileText, Presentation, Palette, Phone, CheckCircle2 } from "lucide-react";

type LoginMethod = "email" | "phone";

interface Props {
  onSwitchToRegister: () => void;
  footer?: React.ReactNode;
}

export function LoginPage({ onSwitchToRegister, footer }: Props) {
  const { t } = useTranslation('login');
  
  // Tab state
  const [loginMethod, setLoginMethod] = useState<LoginMethod>("phone");
  
  // Email login
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  // Phone login
  const [phone, setPhone] = useState("");
  const [smsCode, setSmsCode] = useState("");
  const [smsStep, setSmsStep] = useState<"idle" | "sent">("idle");
  const [smsSent, setSmsSent] = useState(false);
  const [sendingSms, setSendingSms] = useState(false);

  
  // 动态文字索引
  const [featureIndex, setFeatureIndex] = useState(0);

  const features = [
    {
      icon: Sparkles,
      title: t('features.paper2figure.title'),
      desc: t('features.paper2figure.desc'),
      color: "text-primary-700",
      bg: "bg-primary-500/10",
      border: "border-primary-500/20"
    },
    {
      icon: FileText,
      title: t('features.paper2ppt.title'),
      desc: t('features.paper2ppt.desc'),
      color: "text-amber-700",
      bg: "bg-amber-500/10",
      border: "border-amber-500/20"
    },
    {
      icon: Presentation,
      title: t('features.pdf2ppt.title'),
      desc: t('features.pdf2ppt.desc'),
      color: "text-primary-600",
      bg: "bg-primary-500/8",
      border: "border-primary-500/16"
    },
    {
      icon: Palette,
      title: t('features.pptPolish.title'),
      desc: t('features.pptPolish.desc'),
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

  const {
    signInWithEmail,
    signInWithPhoneOtp,
    verifyPhoneOtp,
    loading,
    error,
    clearError,
  } = useAuthStore();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    clearError();
    await signInWithEmail(email, password);
  };

  const handleSendSms = async () => {
    clearError();
    setSmsSent(false);
    setSendingSms(true);
    const success = await signInWithPhoneOtp(phone);
    setSendingSms(false);
    if (success) {
      setSmsStep("sent");
      setSmsSent(true);
    }
  };

  const handleVerifySms = async () => {
    clearError();
    await verifyPhoneOtp(phone, smsCode);
  };

  return (
    <div className="portal-shell min-h-screen flex items-center justify-center p-4 relative overflow-hidden">
      {/* 动态背景装饰 */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-primary-500/12 rounded-full blur-[120px] animate-pulse"></div>
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-amber-500/12 rounded-full blur-[120px] animate-pulse delay-1000"></div>
      </div>

      <div className="w-full max-w-5xl grid grid-cols-1 lg:grid-cols-2 gap-8 items-center relative z-10">
        
        {/* 左侧：功能展示区 */}
        <div className="hidden lg:flex flex-col justify-center space-y-8 pr-8">
          <div>
            <h1 className="text-5xl font-bold font-display text-primary-900 mb-4 leading-tight">
              {t('heroTitlePrefix')} <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary-700 to-amber-600">Any</span>
            </h1>
            <p className="text-slate-600 text-lg max-w-md">
              {t('heroDesc')}
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

        {/* 右侧：登录表单 */}
        <div className="portal-panel p-8 md:p-10 rounded-[28px] w-full shadow-shell">
          <div className="lg:hidden mb-8 text-center">
             <h2 className="text-3xl font-bold font-display text-primary-900 mb-2">{t('title')}</h2>
             <p className="text-slate-500 text-sm">{t('subtitle')}</p>
          </div>

          <h2 className="text-2xl font-bold font-display text-primary-900 mb-2">{t('welcome')}</h2>
          <p className="text-slate-500 mb-6 text-sm">{t('loginSubtitle')}</p>

          {error && (
            <div className="mb-6 p-4 bg-red-500/10 border border-red-500/20 rounded-xl flex items-start gap-3 text-red-300 animate-in fade-in slide-in-from-top-2">
              <AlertCircle size={20} className="mt-0.5 shrink-0" />
              <span className="text-sm leading-relaxed">{error}</span>
            </div>
          )}

          {/* Tab 切换 */}
          <div className="flex mb-6 p-1 bg-primary-500/5 rounded-2xl">
            <button
              type="button"
              onClick={() => setLoginMethod("phone")}
              className={`flex-1 py-2.5 rounded-lg text-sm font-medium transition-all flex items-center justify-center gap-2 ${
                loginMethod === "phone"
                  ? "portal-button-primary text-white shadow-lg"
                  : "text-slate-500 hover:text-primary-700"
              }`}
            >
              <Phone size={16} />
              <span>手机号登录</span>
            </button>
            <button
              type="button"
              onClick={() => setLoginMethod("email")}
              className={`flex-1 py-2.5 rounded-lg text-sm font-medium transition-all flex items-center justify-center gap-2 ${
                loginMethod === "email"
                  ? "portal-button-primary text-white shadow-lg"
                  : "text-slate-500 hover:text-primary-700"
              }`}
            >
              <Mail size={16} />
              <span>邮箱登录</span>
            </button>
          </div>

          {/* 手机号登录表单 */}
          {loginMethod === "phone" && (
            <div className="space-y-4">
              {/* 第一行：手机号 + 发送按钮 */}
              <div className="flex gap-3">
                <div className="flex-1 space-y-1.5">
                  <label className="block text-xs font-medium text-slate-600 ml-1">手机号</label>
                  <div className="relative group">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                      <Phone className="text-slate-400 group-focus-within:text-primary-600 transition-colors" size={18} />
                    </div>
                    <input
                      type="tel"
                      value={phone}
                      onChange={(e) => setPhone(e.target.value)}
                      className="portal-input w-full pl-10 pr-4 py-3 rounded-xl transition-all"
                      placeholder="输入手机号"
                      disabled={loading}
                    />
                  </div>
                </div>
                <div className="space-y-1.5">
                  <label className="block text-xs font-medium text-transparent ml-1">-</label>
                  <button
                    type="button"
                    onClick={handleSendSms}
                    disabled={sendingSms || !phone.trim()}
                    className="portal-button-primary px-6 py-3 font-medium rounded-xl disabled:opacity-50 disabled:cursor-not-allowed transition-all whitespace-nowrap flex items-center justify-center gap-2 min-w-[120px]"
                  >
                    {sendingSms ? (
                      <>
                        <Loader2 size={18} className="animate-spin" />
                        <span>发送中</span>
                      </>
                    ) : smsStep === "sent" ? (
                      <span>重新发送</span>
                    ) : (
                      <span>发送验证码</span>
                    )}
                  </button>
                </div>
              </div>

              {/* 发送成功提示 */}
              {smsSent && (
                <div className="flex items-center gap-2 text-sm text-emerald-700 bg-emerald-500/10 border border-emerald-500/20 rounded-xl px-3 py-2">
                  <CheckCircle2 size={16} className="shrink-0" />
                  <span>验证码已发送，请查收短信</span>
                </div>
              )}

              {/* 第二行：验证码输入框（始终显示） */}
              <div className="space-y-1.5">
                <label className="block text-xs font-medium text-slate-600 ml-1">验证码</label>
                <input
                  type="text"
                  inputMode="numeric"
                  value={smsCode}
                  onChange={(e) => setSmsCode(e.target.value.replace(/\D/g, "").slice(0, 6))}
                  className="portal-input w-full px-4 py-3 rounded-xl transition-all tracking-widest text-center text-lg"
                  placeholder="输入 6 位验证码"
                  disabled={sendingSms || loading || smsStep === "idle"}
                  maxLength={6}
                />
              </div>

              {/* 登录按钮 */}
              <button
                type="button"
                onClick={handleVerifySms}
                disabled={loading || smsCode.trim().length < 4 || smsStep === "idle"}
                className="portal-button-primary w-full py-3.5 font-bold rounded-xl disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:scale-[1.02] active:scale-[0.98] flex items-center justify-center gap-2"
              >
                {loading && smsStep === "sent" ? (
                  <>
                    <Loader2 size={20} className="animate-spin" />
                    <span>登录中...</span>
                  </>
                ) : (
                  <>
                    <span>登录</span>
                    <ArrowRight size={18} />
                  </>
                )}
              </button>
            </div>
          )}

          {/* 邮箱登录表单 */}
          {loginMethod === "email" && (
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="space-y-1.5">
                <label className="block text-xs font-medium text-slate-600 ml-1">{t('emailLabel')}</label>
                <div className="relative group">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <Mail className="text-slate-400 group-focus-within:text-primary-600 transition-colors" size={18} />
                  </div>
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="portal-input w-full pl-10 pr-4 py-3 rounded-xl transition-all"
                    placeholder={t('emailPlaceholder')}
                    required
                    disabled={loading}
                  />
                </div>
              </div>

              <div className="space-y-1.5">
                <label className="block text-xs font-medium text-slate-600 ml-1">{t('passwordLabel')}</label>
                <div className="relative group">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <Lock className="text-slate-400 group-focus-within:text-primary-600 transition-colors" size={18} />
                  </div>
                  <input
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="portal-input w-full pl-10 pr-4 py-3 rounded-xl transition-all"
                    placeholder={t('passwordPlaceholder')}
                    required
                    disabled={loading}
                  />
                </div>
              </div>

              <button
                type="submit"
                disabled={loading}
                className="portal-button-primary w-full py-3.5 font-bold rounded-xl disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:scale-[1.02] active:scale-[0.98] flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <Loader2 size={20} className="animate-spin" />
                    <span>{t('loggingIn')}</span>
                  </>
                ) : (
                  <>
                    <span>{t('loginButton')}</span>
                    <ArrowRight size={18} />
                  </>
                )}
              </button>
            </form>
          )}

          <div className="mt-8 text-center">
            <p className="text-slate-500 text-sm">
              {t('noAccount')}{" "}
              <button
                onClick={onSwitchToRegister}
                className="text-primary-700 hover:text-primary-800 font-medium hover:underline transition-colors"
              >
                {t('registerLink')}
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
