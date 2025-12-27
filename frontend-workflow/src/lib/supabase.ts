/**
 * Supabase client singleton for frontend.
 *
 * Reads VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY from environment.
 * Uses the anon key for client-side auth (RLS enforced).
 * Returns null when not configured - use isSupabaseConfigured() to check.
 */

import { createClient, SupabaseClient } from "@supabase/supabase-js";

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

/**
 * Check if Supabase is properly configured.
 */
export function isSupabaseConfigured(): boolean {
  return Boolean(supabaseUrl && supabaseAnonKey);
}

// Only create client when configured
const supabaseClient: SupabaseClient | null = isSupabaseConfigured()
  ? createClient(supabaseUrl!, supabaseAnonKey!, {
      auth: {
        autoRefreshToken: true,
        persistSession: true,
        detectSessionInUrl: true,
      },
    })
  : null;

/**
 * Get Supabase client. Use after checking isSupabaseConfigured().
 * Exported as non-null for convenience - callers should check isSupabaseConfigured() first.
 */
export const supabase = supabaseClient as SupabaseClient;

if (!isSupabaseConfigured()) {
  console.info(
    "[Supabase] Not configured. Auth, quotas, and cloud storage disabled."
  );
}
