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

    const syncSession = async (session: any) => {
      if (session?.user?.is_anonymous) {
        await supabase.auth.signOut();
        setSession(null);
        return;
      }
      setSession(session);
    };

    // Get initial session
    supabase.auth.getSession().then(({ data: { session } }) => {
      void syncSession(session);
    });

    // Subscribe to auth changes
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      void syncSession(session);
    });

    // Cleanup subscription on unmount
    return () => {
      subscription.unsubscribe();
    };
  }, [setSession]);

  return <>{children}</>;
}
