/**
 * Supabase client singleton for frontend.
 *
 * PKU intranet deployment runs in auth-disabled mode, so the frontend never
 * initializes Supabase even if env vars are present.
 */

import type { SupabaseClient } from "@supabase/supabase-js";

const PLATFORM_AUTH_DISABLED = true;

/**
 * Check if Supabase is properly configured.
 */
export function isSupabaseConfigured(): boolean {
  return !PLATFORM_AUTH_DISABLED;
}

const supabaseClient: SupabaseClient | null = null;

/**
 * Get Supabase client. Use after checking isSupabaseConfigured().
 * Exported as non-null for convenience - callers should check isSupabaseConfigured() first.
 */
export const supabase = supabaseClient as unknown as SupabaseClient;

console.info("[Supabase] PKU platform mode enabled. Auth and Supabase features are disabled.");
