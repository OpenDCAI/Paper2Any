/**
 * AuthProvider component for session lifecycle management.
 *
 * Wraps the app and handles:
 * - Initial session recovery from localStorage
 * - Auth state change subscription
 * - Automatic token refresh
 */

import { useEffect } from "react";
import { supabase, isSupabaseConfigured } from "../lib/supabase";
import { useAuthStore } from "../stores/authStore";

interface AuthProviderProps {
  children: React.ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  const setSession = useAuthStore((state) => state.setSession);

  useEffect(() => {
    if (!isSupabaseConfigured()) {
      // No Supabase config - mark as loaded with no session
      setSession(null);
      return;
    }

    // Get initial session
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session);
    });

    // Subscribe to auth changes
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      setSession(session);
    });

    // Cleanup subscription on unmount
    return () => {
      subscription.unsubscribe();
    };
  }, [setSession]);

  return <>{children}</>;
}
