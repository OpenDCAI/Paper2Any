/**
 * UserMenu dropdown component.
 *
 * Shows user email with a dropdown menu containing sign out option.
 * For anonymous users, shows "Guest" and option to sign in.
 * Hidden when Supabase is not configured (no auth mode).
 */

import { useState, useRef, useEffect } from "react";
import { useAuthStore } from "../stores/authStore";
import { isSupabaseConfigured } from "../lib/supabase";
import { User, LogOut, ChevronDown, LogIn, Sparkles, Crown } from "lucide-react";

export function UserMenu() {
  const { user, signOut } = useAuthStore();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // Close on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  // Hide when Supabase is not configured or no user
  if (!isSupabaseConfigured() || !user) return null;

  // Check if user is anonymous (no email means anonymous/guest)
  const isAnonymous = user.is_anonymous || !user.email;
  const displayName = isAnonymous ? "匿名用户" : (user.email?.split('@')[0] || "用户");
  const fullEmail = user.email || "";

  const handleSignOut = async () => {
    setOpen(false);
    await signOut();
  };

  return (
    <div ref={ref} className="relative z-50">
      <button
        onClick={() => setOpen(!open)}
        className={`group relative flex items-center gap-2 px-1 pl-1.5 pr-3 py-1 rounded-full border transition-all duration-300 ${
          open 
            ? "bg-white/10 border-white/20 shadow-[0_0_15px_rgba(168,85,247,0.3)]" 
            : "bg-black/20 border-white/10 hover:bg-white/10 hover:border-white/20 hover:shadow-[0_0_10px_rgba(168,85,247,0.15)]"
        }`}
      >
        {/* Avatar / Icon */}
        <div className={`w-8 h-8 rounded-full flex items-center justify-center shadow-inner relative overflow-hidden ${
          isAnonymous 
            ? "bg-gradient-to-br from-yellow-500/20 to-orange-500/20 border border-yellow-500/30" 
            : "bg-gradient-to-br from-purple-500/20 to-pink-500/20 border border-purple-500/30"
        }`}>
          {isAnonymous ? (
             <User size={16} className="text-yellow-200" />
          ) : (
             <Crown size={16} className="text-purple-200" />
          )}
          
          {/* Shine effect */}
          <div className="absolute inset-0 bg-gradient-to-tr from-transparent via-white/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
        </div>

        <div className="flex flex-col items-start mr-1">
           <span className={`text-sm font-medium leading-none ${
             isAnonymous 
               ? "text-yellow-100 group-hover:text-yellow-50" 
               : "bg-gradient-to-r from-purple-200 to-pink-200 bg-clip-text text-transparent group-hover:from-white group-hover:to-white transition-all"
           }`}>
             {displayName}
           </span>
           {!isAnonymous && <span className="text-[10px] text-gray-400 leading-tight scale-90 origin-left">PRO MEMBER</span>}
           {isAnonymous && <span className="text-[10px] text-yellow-500/70 leading-tight scale-90 origin-left">GUEST</span>}
        </div>

        <ChevronDown
          size={14}
          className={`text-gray-400 transition-transform duration-300 ${open ? "rotate-180 text-white" : "group-hover:text-gray-200"}`}
        />
      </button>

      {/* Dropdown Menu */}
      <div 
        className={`absolute right-0 mt-3 w-64 origin-top-right transition-all duration-200 ease-out ${
          open 
            ? "opacity-100 scale-100 translate-y-0" 
            : "opacity-0 scale-95 -translate-y-2 pointer-events-none"
        }`}
      >
        <div className="glass-dark rounded-xl border border-white/10 shadow-[0_10px_40px_-10px_rgba(0,0,0,0.5)] backdrop-blur-xl overflow-hidden relative">
           {/* Decorative background gradients */}
           <div className="absolute top-0 left-0 w-full h-24 bg-gradient-to-b from-purple-500/10 to-transparent pointer-events-none" />
           <div className="absolute -top-10 -right-10 w-32 h-32 bg-purple-500/20 rounded-full blur-3xl pointer-events-none" />

           {/* Header Info */}
           <div className="p-4 border-b border-white/5 relative">
              <p className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-1">
                {isAnonymous ? "当前身份" : "已登录"}
              </p>
              <div className="flex items-center gap-3">
                 <div className={`w-10 h-10 rounded-full flex items-center justify-center text-lg font-bold shadow-lg ${
                    isAnonymous
                      ? "bg-gradient-to-br from-yellow-400 to-orange-600 text-white"
                      : "bg-gradient-to-br from-purple-500 to-pink-600 text-white"
                 }`}>
                    {displayName.charAt(0).toUpperCase()}
                 </div>
                 <div className="overflow-hidden">
                    <p className="text-sm font-bold text-white truncate">{displayName}</p>
                    <p className="text-xs text-gray-400 truncate max-w-[150px]">{isAnonymous ? "访客模式" : fullEmail}</p>
                 </div>
              </div>

              {/* Status Badge */}
              <div className={`mt-3 py-1.5 px-2.5 rounded-lg flex items-center gap-2 text-xs font-medium ${
                 isAnonymous 
                   ? "bg-yellow-500/10 border border-yellow-500/20 text-yellow-300" 
                   : "bg-gradient-to-r from-purple-500/20 to-pink-500/20 border border-purple-500/20 text-purple-200"
              }`}>
                 {isAnonymous ? (
                    <>
                      <Sparkles size={12} />
                      <span>仅限部分功能可用</span>
                    </>
                 ) : (
                    <>
                      <Crown size={12} className="text-yellow-300" />
                      <span>尊贵会员功能已解锁</span>
                    </>
                 )}
              </div>
           </div>

           {/* Actions */}
           <div className="p-2 space-y-1">
              {isAnonymous ? (
                 <button
                   onClick={handleSignOut}
                   className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium text-white transition-all duration-200 group relative overflow-hidden"
                 >
                   <div className="absolute inset-0 bg-gradient-to-r from-purple-600 to-pink-600 opacity-80 group-hover:opacity-100 transition-opacity" />
                   <div className="relative flex items-center gap-3">
                      <LogIn size={16} className="text-white" />
                      登录以获取完整权限
                   </div>
                 </button>
              ) : (
                <div className="px-3 py-2 text-xs text-gray-500 text-center italic">
                   感谢您的使用与支持
                </div>
              )}

              <button
                onClick={handleSignOut}
                className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-gray-300 hover:text-white hover:bg-white/5 transition-all duration-200 group"
              >
                <div className={`p-1.5 rounded-md transition-colors ${
                   isAnonymous ? "bg-red-500/10 text-red-400 group-hover:bg-red-500/20" : "bg-gray-700/50 text-gray-400 group-hover:bg-gray-600"
                }`}>
                   <LogOut size={14} />
                </div>
                {isAnonymous ? "退出访客模式" : "退出登录"}
              </button>
           </div>
        </div>
      </div>
    </div>
  );
}
