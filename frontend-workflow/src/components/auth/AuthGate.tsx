/**
 * AuthGate component that shows login/register when not authenticated.
 *
 * Supports:
 * - Email/password login and registration
 * - OTP verification for email confirmation
 * - Anonymous login for limited access
 */

import { useState } from "react";
import { useAuthStore } from "../../stores/authStore";
import { LoginPage } from "./LoginPage";
import { RegisterPage } from "./RegisterPage";
import { VerifyOtpPage } from "./VerifyOtpPage";
import { Loader2, User } from "lucide-react";

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
    signInAnonymously,
  } = useAuthStore();
  const [authMode, setAuthMode] = useState<"login" | "register">("login");
  const [isAnonymousLoading, setIsAnonymousLoading] = useState(false);

  // Show loading spinner during initial session check
  if (loading && !isAnonymousLoading) {
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
    const handleAnonymousLogin = async () => {
      setIsAnonymousLoading(true);
      await signInAnonymously();
      setIsAnonymousLoading(false);
    };

    const authFooter = (
      <div className="mt-4 pt-4 border-t border-white/10">
        <button
          onClick={handleAnonymousLogin}
          disabled={isAnonymousLoading}
          className="w-full py-2 text-gray-400 hover:text-white text-sm flex items-center justify-center gap-2 transition-colors disabled:opacity-50"
        >
          {isAnonymousLoading ? (
            <Loader2 size={16} className="animate-spin" />
          ) : (
            <User size={16} />
          )}
          Continue as guest (limited features)
        </button>
      </div>
    );

    if (authMode === "login") {
      return (
        <LoginPage
          onSwitchToRegister={() => setAuthMode("register")}
          footer={authFooter}
        />
      );
    }
    return (
      <RegisterPage
        onSwitchToLogin={() => setAuthMode("login")}
        footer={authFooter}
      />
    );
  }

  // User is authenticated - render children
  return <>{children}</>;
}
