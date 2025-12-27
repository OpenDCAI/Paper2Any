/**
 * Supabase client singleton for frontend.
 *
 * Reads VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY from environment.
 * Uses the anon key for client-side auth (RLS enforced).
 */

import { createClient, SupabaseClient } from "@supabase/supabase-js";

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseAnonKey) {
  console.warn(
    "[Supabase] Missing VITE_SUPABASE_URL or VITE_SUPABASE_ANON_KEY. Auth features will be disabled."
  );
}

export const supabase: SupabaseClient = createClient(
  supabaseUrl || "",
  supabaseAnonKey || "",
  {
    auth: {
      autoRefreshToken: true,
      persistSession: true,
      detectSessionInUrl: true,
    },
  }
);

/**
 * Check if Supabase is properly configured.
 */
export function isSupabaseConfigured(): boolean {
  return Boolean(supabaseUrl && supabaseAnonKey);
}
