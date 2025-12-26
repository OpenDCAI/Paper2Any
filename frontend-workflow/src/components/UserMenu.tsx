/**
 * UserMenu dropdown component.
 *
 * Shows user email with a dropdown menu containing sign out option.
 */

import { useState, useRef, useEffect } from "react";
import { useAuthStore } from "../stores/authStore";
import { User, LogOut, ChevronDown } from "lucide-react";

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

  if (!user) return null;

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
        <User size={18} className="text-gray-400" />
        <span className="text-sm text-gray-300 max-w-[120px] truncate">
          {user.email}
        </span>
        <ChevronDown
          size={14}
          className={`text-gray-500 transition-transform ${open ? "rotate-180" : ""}`}
        />
      </button>

      {open && (
        <div className="absolute right-0 mt-2 w-44 glass-dark rounded-lg border border-white/10 shadow-xl z-50 overflow-hidden">
          <div className="px-3 py-2 border-b border-white/10">
            <p className="text-xs text-gray-500">Signed in as</p>
            <p className="text-sm text-gray-300 truncate">{user.email}</p>
          </div>
          <div className="p-1">
            <button
              onClick={handleSignOut}
              className="w-full flex items-center gap-2 px-3 py-2 text-sm text-red-400 hover:bg-red-500/10 rounded-lg transition-colors"
            >
              <LogOut size={16} />
              Sign out
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
