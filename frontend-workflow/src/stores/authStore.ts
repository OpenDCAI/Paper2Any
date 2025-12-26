/**
 * Zustand store for authentication state.
 *
 * Manages user session, login/logout, quota, and session refresh.
 */

import { create } from "zustand";
import { User, Session } from "@supabase/supabase-js";
import { supabase, isSupabaseConfigured } from "../lib/supabase";

interface Quota {
  used: number;
  limit: number;
  remaining: number;
}

interface AuthState {
  user: User | null;
  session: Session | null;
  loading: boolean;
  error: string | null;
  quota: Quota | null;

  // Actions
  setSession: (session: Session | null) => void;
  signInWithEmail: (email: string, password: string) => Promise<void>;
  signUpWithEmail: (email: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
  clearError: () => void;
  refreshQuota: () => Promise<void>;
}

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export const useAuthStore = create<AuthState>((set, get) => ({
  user: null,
  session: null,
  loading: true,
  error: null,
  quota: null,

  setSession: (session) => {
    set({
      session,
      user: session?.user ?? null,
      loading: false,
    });
    // Auto-refresh quota when session changes
    if (session) {
      get().refreshQuota();
    } else {
      set({ quota: null });
    }
  },

  signInWithEmail: async (email, password) => {
    if (!isSupabaseConfigured()) {
      set({ error: "Supabase is not configured", loading: false });
      return;
    }

    set({ loading: true, error: null });

    const { data, error } = await supabase.auth.signInWithPassword({
      email,
      password,
    });

    if (error) {
      set({ error: error.message, loading: false });
      return;
    }

    set({
      session: data.session,
      user: data.user,
      loading: false,
    });

    // Fetch quota after successful login
    get().refreshQuota();
  },

  signUpWithEmail: async (email, password) => {
    if (!isSupabaseConfigured()) {
      set({ error: "Supabase is not configured", loading: false });
      return;
    }

    set({ loading: true, error: null });

    const { data, error } = await supabase.auth.signUp({
      email,
      password,
    });

    if (error) {
      set({ error: error.message, loading: false });
      return;
    }

    // Note: User may need to verify email before session is active
    set({
      session: data.session,
      user: data.user,
      loading: false,
    });
  },

  signOut: async () => {
    if (!isSupabaseConfigured()) {
      set({ user: null, session: null, loading: false, quota: null });
      return;
    }

    set({ loading: true });

    const { error } = await supabase.auth.signOut();

    if (error) {
      set({ error: error.message, loading: false });
      return;
    }

    set({
      user: null,
      session: null,
      quota: null,
      loading: false,
    });
  },

  clearError: () => set({ error: null }),

  refreshQuota: async () => {
    const { session } = get();
    if (!session?.access_token) {
      return;
    }

    try {
      const response = await fetch(`${API_BASE}/api/quota`, {
        headers: {
          Authorization: `Bearer ${session.access_token}`,
        },
      });

      if (response.ok) {
        const quota: Quota = await response.json();
        set({ quota });
      }
    } catch {
      // Silently fail - quota display will just be hidden
    }
  },
}));

/**
 * Get the current access token for API calls.
 * Returns null if not authenticated.
 */
export function getAccessToken(): string | null {
  return useAuthStore.getState().session?.access_token ?? null;
}
