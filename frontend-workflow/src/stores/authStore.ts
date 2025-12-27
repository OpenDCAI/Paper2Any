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
  // For OTP verification flow
  pendingEmail: string | null;
  needsOtpVerification: boolean;

  // Actions
  setSession: (session: Session | null) => void;
  signInWithEmail: (email: string, password: string) => Promise<void>;
  signUpWithEmail: (email: string, password: string) => Promise<{ needsVerification: boolean }>;
  verifyOtp: (email: string, token: string) => Promise<void>;
  resendOtp: (email: string) => Promise<void>;
  signInAnonymously: () => Promise<void>;
  signOut: () => Promise<void>;
  clearError: () => void;
  clearPendingVerification: () => void;
  refreshQuota: () => Promise<void>;
}

// Note: We use relative paths ("/api/...") which go through Vite proxy in dev mode
// This allows the backend URL to be configured at proxy level, not hardcoded here

export const useAuthStore = create<AuthState>((set, get) => ({
  user: null,
  session: null,
  loading: true,
  error: null,
  quota: null,
  pendingEmail: null,
  needsOtpVerification: false,

  setSession: (session) => {
    set({
      session,
      user: session?.user ?? null,
      loading: false,
    });
    // Always refresh quota - backend returns mock data if not authenticated
    get().refreshQuota();
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
      return { needsVerification: false };
    }

    set({ loading: true, error: null });

    const { data, error } = await supabase.auth.signUp({
      email,
      password,
    });

    if (error) {
      set({ error: error.message, loading: false });
      return { needsVerification: false };
    }

    // Check if email confirmation is required
    // If session is null but user exists, email confirmation is pending
    if (data.user && !data.session) {
      set({
        pendingEmail: email,
        needsOtpVerification: true,
        loading: false,
      });
      return { needsVerification: true };
    }

    // No verification needed - user is logged in
    set({
      session: data.session,
      user: data.user,
      loading: false,
    });
    return { needsVerification: false };
  },

  verifyOtp: async (email, token) => {
    if (!isSupabaseConfigured()) {
      set({ error: "Supabase is not configured", loading: false });
      return;
    }

    set({ loading: true, error: null });

    const { data, error } = await supabase.auth.verifyOtp({
      email,
      token,
      type: "email",
    });

    if (error) {
      set({ error: error.message, loading: false });
      return;
    }

    set({
      session: data.session,
      user: data.user,
      pendingEmail: null,
      needsOtpVerification: false,
      loading: false,
    });

    // Fetch quota after successful verification
    get().refreshQuota();
  },

  resendOtp: async (email) => {
    if (!isSupabaseConfigured()) {
      set({ error: "Supabase is not configured" });
      return;
    }

    set({ loading: true, error: null });

    const { error } = await supabase.auth.resend({
      type: "signup",
      email,
    });

    if (error) {
      set({ error: error.message, loading: false });
      return;
    }

    set({ loading: false });
  },

  signInAnonymously: async () => {
    if (!isSupabaseConfigured()) {
      set({ error: "Supabase is not configured", loading: false });
      return;
    }

    set({ loading: true, error: null });

    const { data, error } = await supabase.auth.signInAnonymously();

    if (error) {
      set({ error: error.message, loading: false });
      return;
    }

    set({
      session: data.session,
      user: data.user,
      loading: false,
    });

    // Fetch quota for anonymous user
    get().refreshQuota();
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

  clearPendingVerification: () => set({ pendingEmail: null, needsOtpVerification: false }),

  refreshQuota: async () => {
    const { session } = get();

    try {
      // Build headers - include auth token if available
      const headers: HeadersInit = {};
      if (session?.access_token) {
        headers.Authorization = `Bearer ${session.access_token}`;
      }

      // Always try to fetch quota - backend returns mock data if not authenticated
      const response = await fetch("/api/quota", { headers });

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
