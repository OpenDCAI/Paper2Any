/**
 * Supabase client singleton for frontend.
 *
 * Reads VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY from environment.
 * Uses the anon key for client-side auth (RLS enforced).
 * Returns null when not configured - use isSupabaseConfigured() to check.
 */

import { createClient, SupabaseClient } from "@supabase/supabase-js";

// TEMPORARY: Hardcoded for debugging env variable issue
const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || 'https://xciveaaildyzbreltihu.supabase.co';
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhjaXZlYWFpbGR5emJyZWx0aWh1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjY3NzA3OTcsImV4cCI6MjA4MjM0Njc5N30.HuxOQfZe4S8aQHiSD7P4A0kHjZ1I2VPAKl1KmduYIMM';

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
