/**
 * AuthGate component that shows login/register when not authenticated.
 *
 * Supports:
 * - Email/password login and registration
 * - OTP verification for email confirmation
 * - Bypass when Supabase is not configured (no auth mode)
 */

import { useState } from "react";
import { useAuthStore } from "../../stores/authStore";
import { isSupabaseConfigured } from "../../lib/supabase";
import { LoginPage } from "./LoginPage";
import { RegisterPage } from "./RegisterPage";
import { VerifyOtpPage } from "./VerifyOtpPage";
import { Loader2 } from "lucide-react";

interface Props {
  children: React.ReactNode;
}

export function AuthGate({ children }: Props) {
  const {
    user,
    loading,
    needsOtpVerification,
    pendingEmail,
    clearPendingVerification,
  } = useAuthStore();
  const [authMode, setAuthMode] = useState<"login" | "register">("login");

  // Skip auth when Supabase is not configured
  if (!isSupabaseConfigured()) {
    return <>{children}</>;
  }

  // Show loading spinner during initial session check
  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#0a0a1a]">
        <div className="flex flex-col items-center gap-3">
          <Loader2 size={32} className="animate-spin text-primary-500" />
          <span className="text-gray-400 text-sm">Loading...</span>
        </div>
      </div>
    );
  }

  // Show OTP verification page if needed
  if (needsOtpVerification && pendingEmail) {
    return (
      <VerifyOtpPage
        email={pendingEmail}
        onBack={() => {
          clearPendingVerification();
          setAuthMode("login");
        }}
      />
    );
  }

  // Show auth pages when not authenticated
  if (!user) {
    if (authMode === "login") {
      return (
        <LoginPage
          onSwitchToRegister={() => setAuthMode("register")}
        />
      );
    }
    return (
      <RegisterPage
        onSwitchToLogin={() => setAuthMode("login")}
      />
    );
  }

  // User is authenticated - render children
  return <>{children}</>;
}
