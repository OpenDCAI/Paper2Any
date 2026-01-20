/**
 * Zustand store for authentication state.
 *
 * Manages user session, login/logout, quota, and session refresh.
 */

import { create } from "zustand";
import { User, Session, Provider } from "@supabase/supabase-js";
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
  signInWithPhoneOtp: (phone: string) => Promise<boolean>;
  verifyPhoneOtp: (phone: string, token: string) => Promise<void>;
  signInWithOAuth: (provider: Provider) => Promise<void>;
  linkOAuthIdentity: (provider: Provider) => Promise<void>;
  claimInviteCode: (inviteCode: string) => Promise<void>;
  signInAnonymously: () => Promise<void>;
  signOut: () => Promise<void>;
  clearError: () => void;
  clearPendingVerification: () => void;
  refreshQuota: () => Promise<void>;
}

const INVITE_CODE_STORAGE_KEY = "paper2any_invite_code";

function normalizePhoneE164China(input: string): string {
  const s = input.trim();
  if (s.startsWith("+")) return s;
  if (s.startsWith("86")) return `+${s}`;
  return `+86${s.replace(/\D/g, "")}`;
}

async function tryClaimInviteCode(inviteCode: string): Promise<void> {
  // Database side will enforce idempotency.
  // This RPC name will be added by migration.
  const { data, error } = await supabase.rpc("apply_invite_code", { p_code: inviteCode });
  
  if (error) {
    throw new Error("邀请码兑换失败，请稍后重试");
  }
  
  // Check response from RPC function
  if (data && !data.success) {
    const errorCode = data.error;
    switch (errorCode) {
      case 'not_authenticated':
        throw new Error("请先登录后再填写邀请码");
      case 'already_claimed':
        throw new Error("您已经使用过邀请码了");
      case 'invalid_code':
        throw new Error("邀请码无效，请检查后重试");
      case 'self_invite':
        throw new Error("不能使用自己的邀请码");
      default:
        throw new Error("邀请码兑换失败");
    }
  }
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
      // Translate common error messages to user-friendly Chinese
      let friendlyError = error.message;
      if (error.message.toLowerCase().includes("invalid login credentials")) {
        friendlyError = "邮箱或密码错误，请检查后重试";
      } else if (error.message.toLowerCase().includes("email not confirmed")) {
        friendlyError = "邮箱未验证，请查收验证邮件";
      } else if (error.message.toLowerCase().includes("invalid")) {
        friendlyError = "登录信息无效，请检查后重试";
      }
      set({ error: friendlyError, loading: false });
      return;
    }

    set({
      session: data.session,
      user: data.user,
      loading: false,
    });

    // Try claim invite code after login
    try {
      const stored = localStorage.getItem(INVITE_CODE_STORAGE_KEY);
      if (stored) {
        await tryClaimInviteCode(stored);
        localStorage.removeItem(INVITE_CODE_STORAGE_KEY);
      }
    } catch {
      // ignore
    }

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

    // Try claim invite code after signup
    try {
      const stored = localStorage.getItem(INVITE_CODE_STORAGE_KEY);
      if (stored) {
        await tryClaimInviteCode(stored);
        localStorage.removeItem(INVITE_CODE_STORAGE_KEY);
      }
    } catch {
      // ignore
    }
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

    // Try claim invite code after verification
    try {
      const stored = localStorage.getItem(INVITE_CODE_STORAGE_KEY);
      if (stored) {
        await tryClaimInviteCode(stored);
        localStorage.removeItem(INVITE_CODE_STORAGE_KEY);
      }
    } catch {
      // ignore
    }

    // Fetch quota after successful verification
    get().refreshQuota();
  },

  signInWithPhoneOtp: async (phone) => {
    if (!isSupabaseConfigured()) {
      set({ error: "Supabase is not configured" });
      return false;
    }

    set({ error: null });
    const phoneE164 = normalizePhoneE164China(phone);
    const { error } = await supabase.auth.signInWithOtp({ phone: phoneE164 });

    if (error) {
      // Translate common error messages to user-friendly Chinese
      let friendlyError = error.message;
      if (error.message.toLowerCase().includes("sms") || error.message.toLowerCase().includes("phone")) {
        friendlyError = "发送验证码失败，请检查手机号后重试";
      } else if (error.message.toLowerCase().includes("rate limit")) {
        friendlyError = "发送过于频繁，请稍后再试";
      }
      set({ error: friendlyError });
      return false;
    }

    return true;
  },

  verifyPhoneOtp: async (phone, token) => {
    if (!isSupabaseConfigured()) {
      set({ error: "Supabase is not configured", loading: false });
      return;
    }

    set({ loading: true, error: null });
    const phoneE164 = normalizePhoneE164China(phone);
    const { data, error } = await supabase.auth.verifyOtp({
      phone: phoneE164,
      token,
      type: "sms",
    });

    if (error) {
      // Translate common error messages to user-friendly Chinese
      let friendlyError = error.message;
      if (error.message.toLowerCase().includes("token") && error.message.toLowerCase().includes("expired")) {
        friendlyError = "验证码已过期，请重新获取";
      } else if (error.message.toLowerCase().includes("invalid")) {
        friendlyError = "验证码错误，请检查后重试";
      } else if (error.message.toLowerCase().includes("token")) {
        friendlyError = "验证码无效或已过期，请重新获取";
      }
      set({ error: friendlyError, loading: false });
      return;
    }

    set({
      session: data.session,
      user: data.user,
      loading: false,
    });

    // Try claim invite code after phone login
    try {
      const stored = localStorage.getItem(INVITE_CODE_STORAGE_KEY);
      if (stored) {
        await tryClaimInviteCode(stored);
        localStorage.removeItem(INVITE_CODE_STORAGE_KEY);
      }
    } catch {
      // ignore
    }

    get().refreshQuota();
  },

  signInWithOAuth: async (provider) => {
    if (!isSupabaseConfigured()) {
      set({ error: "Supabase is not configured", loading: false });
      return;
    }

    set({ loading: true, error: null });
    const { error } = await supabase.auth.signInWithOAuth({
      provider,
      options: {
        redirectTo: window.location.origin,
      },
    });

    if (error) {
      set({ error: error.message, loading: false });
      return;
    }

    // Redirect happens; keep loading true to avoid flicker
  },

  linkOAuthIdentity: async (provider) => {
    if (!isSupabaseConfigured()) {
      set({ error: "Supabase is not configured", loading: false });
      return;
    }
    const { user } = get();
    if (!user) {
      set({ error: "Please sign in first", loading: false });
      return;
    }

    set({ loading: true, error: null });
    // supabase-js v2 supports linkIdentity; if not available, this will throw.
    try {
      const authAny = supabase.auth as any;
      const { error } = await authAny.linkIdentity({
        provider,
        options: {
          redirectTo: window.location.origin,
        },
      });
      if (error) {
        set({ error: error.message, loading: false });
        return;
      }
    } catch (e) {
      set({ error: `linkIdentity not supported: ${String(e)}`, loading: false });
      return;
    }

    // Redirect happens
  },

  claimInviteCode: async (inviteCode) => {
    if (!isSupabaseConfigured()) {
      const error = "Supabase is not configured";
      set({ error });
      throw new Error(error);
    }

    set({ error: null });

    try {
      await tryClaimInviteCode(inviteCode.trim());
      localStorage.removeItem(INVITE_CODE_STORAGE_KEY);
      await get().refreshQuota();
    } catch (e) {
      // If DB isn't ready yet, store locally and retry later.
      try {
        localStorage.setItem(INVITE_CODE_STORAGE_KEY, inviteCode.trim());
      } catch {
        // ignore
      }
      const errorMsg = String(e);
      set({ error: errorMsg });
      throw e; // Re-throw so caller can handle it
    }
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
      // Import quota service and check quota directly
      const { checkQuota } = await import('../services/quotaService');
      const userId = session?.user?.id || null;
      const isAnonymous = session?.user?.is_anonymous || false;
      const quotaInfo = await checkQuota(userId, isAnonymous);

      set({
        quota: {
          used: quotaInfo.used,
          limit: quotaInfo.limit,
          remaining: quotaInfo.remaining,
        }
      });
    } catch (err) {
      console.error('[authStore] Failed to refresh quota:', err);
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
