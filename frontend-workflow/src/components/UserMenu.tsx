/**
 * UserMenu dropdown component.
 *
 * Shows user identity with a dropdown menu containing account actions.
 * Hidden when Supabase is not configured (no auth mode).
 */

import { useState, useRef, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { useAuthStore } from "../stores/authStore";
import { isSupabaseConfigured } from "../lib/supabase";
import { DEFAULT_LLM_API_URL, getPurchaseUrl } from "../config/api";
import { useRuntimeBilling } from "../hooks/useRuntimeBilling";
import { LogOut, ChevronDown, Crown, FolderOpen, Settings, ExternalLink, Ticket } from "lucide-react";

interface UserMenuProps {
  onShowFiles?: () => void;
  onShowAccount?: () => void;
}

export function UserMenu({ onShowFiles, onShowAccount }: UserMenuProps = {}) {
  const { t } = useTranslation('common');
  const { user, signOut } = useAuthStore();
  const { runtimeConfig } = useRuntimeBilling();
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

  const displayName = user.email?.split('@')[0] || user.phone?.slice(-4) || t('userMenu.user');
  const fullEmail = user.email || user.phone || "";
  const purchaseUrl =
    runtimeConfig.points_purchase_url?.trim()
    || getPurchaseUrl(runtimeConfig.managed_api_url || DEFAULT_LLM_API_URL);

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
        <div className="w-8 h-8 rounded-full flex items-center justify-center shadow-inner relative overflow-hidden bg-gradient-to-br from-purple-500/20 to-pink-500/20 border border-purple-500/30">
          <Crown size={16} className="text-purple-200" />
          
          {/* Shine effect */}
          <div className="absolute inset-0 bg-gradient-to-tr from-transparent via-white/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
        </div>

        <div className="flex flex-col items-start mr-1">
           <span className="text-sm font-medium leading-none bg-gradient-to-r from-purple-200 to-pink-200 bg-clip-text text-transparent group-hover:from-white group-hover:to-white transition-all">
             {displayName}
           </span>
           <span className="text-[10px] text-gray-400 leading-tight scale-90 origin-left">PRO MEMBER</span>
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
                {t('userMenu.loggedIn')}
              </p>
              <div className="flex items-center gap-3">
                 <div className="w-10 h-10 rounded-full flex items-center justify-center text-lg font-bold shadow-lg bg-gradient-to-br from-purple-500 to-pink-600 text-white">
                    {displayName.charAt(0).toUpperCase()}
                 </div>
                 <div className="overflow-hidden">
                    <p className="text-sm font-bold text-white truncate">{displayName}</p>
                    <p className="text-xs text-gray-400 truncate max-w-[150px]">{fullEmail}</p>
                 </div>
              </div>

              {/* Status Badge */}
              <div className="mt-3 py-1.5 px-2.5 rounded-lg flex items-center gap-2 text-xs font-medium bg-gradient-to-r from-purple-500/20 to-pink-500/20 border border-purple-500/20 text-purple-200">
                 <>
                   <Crown size={12} className="text-yellow-300" />
                   <span>{t('userMenu.pro')}</span>
                 </>
              </div>
           </div>

           {/* Actions */}
           <div className="p-2 space-y-1">
              {runtimeConfig.billing_mode === 'free' && purchaseUrl && (
                <a
                  href={purchaseUrl}
                  target="_blank"
                  rel="noreferrer"
                  onClick={() => setOpen(false)}
                  className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-amber-100 hover:text-white hover:bg-amber-500/10 transition-all duration-200 group"
                >
                  <div className="p-1.5 rounded-md bg-amber-500/10 text-amber-300 group-hover:bg-amber-500/20">
                    <ExternalLink size={14} />
                  </div>
                  {t('userMenu.buyPoints')}
                </a>
              )}

              {runtimeConfig.points_redeem_enabled && (
                <button
                  onClick={() => {
                    setOpen(false);
                    onShowAccount?.();
                  }}
                  className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-gray-300 hover:text-white hover:bg-white/5 transition-all duration-200 group"
                >
                  <div className="p-1.5 rounded-md bg-white/5 text-gray-300 group-hover:bg-white/10">
                    <Ticket size={14} />
                  </div>
                  {t('userMenu.redeemPoints')}
                </button>
              )}

              <button
                onClick={() => {
                  setOpen(false);
                  onShowFiles?.();
                }}
                className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-gray-300 hover:text-white hover:bg-white/5 transition-all duration-200 group"
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
                className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-gray-300 hover:text-white hover:bg-white/5 transition-all duration-200 group"
              >
                <div className="p-1.5 rounded-md bg-white/5 text-gray-300 group-hover:bg-white/10">
                  <Settings size={14} />
                </div>
                账户设置
              </button>

              <div className="px-3 py-2 text-xs text-gray-500 text-center italic">
                 {t('userMenu.thanks')}
              </div>

              <button
                onClick={handleSignOut}
                className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-gray-300 hover:text-white hover:bg-white/5 transition-all duration-200 group"
              >
                <div className="p-1.5 rounded-md transition-colors bg-gray-700/50 text-gray-400 group-hover:bg-gray-600">
                   <LogOut size={14} />
                </div>
                {t('userMenu.signOut')}
              </button>
           </div>
        </div>
      </div>
    </div>
  );
}
