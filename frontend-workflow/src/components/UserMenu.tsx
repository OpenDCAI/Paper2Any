/**
 * UserMenu dropdown component.
 *
 * Shows user email with a dropdown menu containing sign out option.
 * For anonymous users, shows "Guest" and option to sign in.
 * Hidden when Supabase is not configured (no auth mode).
 */

import { useState, useRef, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { useAuthStore } from "../stores/authStore";
import { isSupabaseConfigured } from "../lib/supabase";
import { User, LogOut, ChevronDown, LogIn, Sparkles, Crown, FolderOpen, Settings } from "lucide-react";

interface UserMenuProps {
  onShowFiles?: () => void;
  onShowAccount?: () => void;
}

export function UserMenu({ onShowFiles, onShowAccount }: UserMenuProps = {}) {
  const { t } = useTranslation('common');
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

  // Check if user is anonymous (only check is_anonymous flag)
  const isAnonymous = user.is_anonymous === true;
  const displayName = isAnonymous 
    ? t('userMenu.anonymous') 
    : (user.email?.split('@')[0] || user.phone?.slice(-4) || t('userMenu.user'));
  const fullEmail = user.email || user.phone || "";

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
            ? "bg-white/95 border-primary-500/20 shadow-[0_18px_42px_rgba(117,36,57,0.14)]" 
            : "bg-white/72 border-primary-500/10 hover:bg-white hover:border-primary-500/20 hover:shadow-[0_14px_32px_rgba(117,36,57,0.10)]"
        }`}
      >
        {/* Avatar / Icon */}
        <div className={`w-8 h-8 rounded-full flex items-center justify-center shadow-inner relative overflow-hidden ${
          isAnonymous 
            ? "bg-gradient-to-br from-yellow-500/20 to-orange-500/20 border border-yellow-500/30" 
            : "bg-gradient-to-br from-primary-500/18 to-amber-500/20 border border-primary-500/22"
        }`}>
          {isAnonymous ? (
             <User size={16} className="text-amber-700" />
          ) : (
             <Crown size={16} className="text-primary-700" />
          )}
          
          {/* Shine effect */}
          <div className="absolute inset-0 bg-gradient-to-tr from-transparent via-white/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
        </div>

        <div className="flex flex-col items-start mr-1">
           <span className={`text-sm font-medium leading-none ${
             isAnonymous 
               ? "text-amber-700 group-hover:text-amber-800" 
               : "bg-gradient-to-r from-primary-700 to-primary-500 bg-clip-text text-transparent transition-all"
           }`}>
             {displayName}
           </span>
           {!isAnonymous && <span className="text-[10px] text-slate-500 leading-tight scale-90 origin-left">PRO MEMBER</span>}
           {isAnonymous && <span className="text-[10px] text-amber-600/80 leading-tight scale-90 origin-left">GUEST</span>}
        </div>

        <ChevronDown
          size={14}
          className={`text-slate-500 transition-transform duration-300 ${open ? "rotate-180 text-primary-700" : "group-hover:text-primary-700"}`}
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
           <div className="absolute top-0 left-0 w-full h-24 bg-gradient-to-b from-primary-500/12 to-transparent pointer-events-none" />
           <div className="absolute -top-10 -right-10 w-32 h-32 bg-primary-500/20 rounded-full blur-3xl pointer-events-none" />

           {/* Header Info */}
           <div className="p-4 border-b border-white/5 relative">
              <p className="text-xs font-medium text-[#d8c8bf] uppercase tracking-wider mb-1">
                {isAnonymous ? t('userMenu.identity') : t('userMenu.loggedIn')}
              </p>
              <div className="flex items-center gap-3">
                 <div className={`w-10 h-10 rounded-full flex items-center justify-center text-lg font-bold shadow-lg ${
                    isAnonymous
                      ? "bg-gradient-to-br from-yellow-400 to-orange-600 text-white"
                      : "bg-gradient-to-br from-primary-500 to-amber-500 text-white"
                 }`}>
                    {displayName.charAt(0).toUpperCase()}
                 </div>
                 <div className="overflow-hidden">
                    <p className="text-sm font-bold text-white truncate">{displayName}</p>
                    <p className="text-xs text-[#d8c8bf] truncate max-w-[150px]">{isAnonymous ? t('userMenu.guestMode') : fullEmail}</p>
                 </div>
              </div>

              {/* Status Badge */}
              <div className={`mt-3 py-1.5 px-2.5 rounded-lg flex items-center gap-2 text-xs font-medium ${
                 isAnonymous 
                   ? "bg-yellow-500/10 border border-yellow-500/20 text-yellow-200" 
                   : "bg-gradient-to-r from-primary-500/20 to-amber-400/10 border border-primary-500/20 text-[#fff0e6]"
              }`}>
                 {isAnonymous ? (
                    <>
                      <Sparkles size={12} />
                      <span>{t('userMenu.limited')}</span>
                    </>
                 ) : (
                    <>
                      <Crown size={12} className="text-yellow-300" />
                      <span>{t('userMenu.pro')}</span>
                    </>
                 )}
              </div>
           </div>

           {/* Actions */}
           <div className="p-2 space-y-1">
              <button
                onClick={() => {
                  setOpen(false);
                  onShowFiles?.();
                }}
                className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-[#f4e4d9] hover:text-white hover:bg-white/5 transition-all duration-200 group"
              >
                <div className="p-1.5 rounded-md bg-white/5 text-gray-300 group-hover:bg-white/10">
                  <FolderOpen size={14} />
                </div>
                历史文件
              </button>

              <button
                onClick={() => {
                  setOpen(false);
                  onShowAccount?.();
                }}
                className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-[#f4e4d9] hover:text-white hover:bg-white/5 transition-all duration-200 group"
              >
                <div className="p-1.5 rounded-md bg-white/5 text-gray-300 group-hover:bg-white/10">
                  <Settings size={14} />
                </div>
                账户设置
              </button>

              {isAnonymous ? (
                 <button
                   onClick={handleSignOut}
                   className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium text-white transition-all duration-200 group relative overflow-hidden"
                 >
                   <div className="absolute inset-0 bg-gradient-to-r from-primary-600 to-primary-500 opacity-90 group-hover:opacity-100 transition-opacity" />
                   <div className="relative flex items-center gap-3">
                      <LogIn size={16} className="text-white" />
                      {t('userMenu.signIn')}
                   </div>
                 </button>
              ) : (
                <div className="px-3 py-2 text-xs text-[#d8c8bf] text-center italic">
                   {t('userMenu.thanks')}
                </div>
              )}

              <button
                onClick={handleSignOut}
                className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-[#f4e4d9] hover:text-white hover:bg-white/5 transition-all duration-200 group"
              >
                <div className={`p-1.5 rounded-md transition-colors ${
                   isAnonymous ? "bg-red-500/10 text-red-400 group-hover:bg-red-500/20" : "bg-gray-700/50 text-gray-400 group-hover:bg-gray-600"
                }`}>
                   <LogOut size={14} />
                </div>
                {isAnonymous ? t('userMenu.exitGuest') : t('userMenu.signOut')}
              </button>
           </div>
        </div>
      </div>
    </div>
  );
}
