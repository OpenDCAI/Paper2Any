/**
 * AuthGate component that shows login/register when not authenticated.
 *
 * Wraps the main app content and displays auth pages when user is not logged in.
 */

import { useState } from "react";
import { useAuthStore } from "../../stores/authStore";
import { LoginPage } from "./LoginPage";
import { RegisterPage } from "./RegisterPage";
import { Loader2 } from "lucide-react";

interface Props {
  children: React.ReactNode;
}

export function AuthGate({ children }: Props) {
  const { user, loading } = useAuthStore();
  const [authMode, setAuthMode] = useState<"login" | "register">("login");

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

  // Show auth pages when not authenticated
  if (!user) {
    if (authMode === "login") {
      return <LoginPage onSwitchToRegister={() => setAuthMode("register")} />;
    }
    return <RegisterPage onSwitchToLogin={() => setAuthMode("login")} />;
  }

  // User is authenticated - render children
  return <>{children}</>;
}
