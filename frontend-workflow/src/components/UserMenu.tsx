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
import { User, LogOut, ChevronDown, LogIn } from "lucide-react";

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
  const displayName = isAnonymous ? "Guest" : user.email;

  const handleSignOut = async () => {
    setOpen(false);
    await signOut();
  };

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 transition-colors"
      >
        <User size={18} className={isAnonymous ? "text-yellow-500" : "text-gray-400"} />
        <span className={`text-sm max-w-[120px] truncate ${isAnonymous ? "text-yellow-400" : "text-gray-300"}`}>
          {displayName}
        </span>
        <ChevronDown
          size={14}
          className={`text-gray-500 transition-transform ${open ? "rotate-180" : ""}`}
        />
      </button>

      {open && (
        <div className="absolute right-0 mt-2 w-48 glass-dark rounded-lg border border-white/10 shadow-xl z-50 overflow-hidden">
          <div className="px-3 py-2 border-b border-white/10">
            <p className="text-xs text-gray-500">
              {isAnonymous ? "Browsing as" : "Signed in as"}
            </p>
            <p className={`text-sm truncate ${isAnonymous ? "text-yellow-400" : "text-gray-300"}`}>
              {displayName}
            </p>
            {isAnonymous && (
              <p className="text-xs text-gray-500 mt-1">Limited features available</p>
            )}
          </div>
          <div className="p-1">
            {isAnonymous && (
              <button
                onClick={handleSignOut}
                className="w-full flex items-center gap-2 px-3 py-2 text-sm text-primary-400 hover:bg-primary-500/10 rounded-lg transition-colors"
              >
                <LogIn size={16} />
                Sign in for full access
              </button>
            )}
            <button
              onClick={handleSignOut}
              className="w-full flex items-center gap-2 px-3 py-2 text-sm text-red-400 hover:bg-red-500/10 rounded-lg transition-colors"
            >
              <LogOut size={16} />
              {isAnonymous ? "Exit guest mode" : "Sign out"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
