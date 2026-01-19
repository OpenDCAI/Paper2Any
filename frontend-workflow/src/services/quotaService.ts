/**
 * Quota service for tracking API usage.
 *
 * Handles quota checking and usage recording for both
 * logged-in users (Supabase) and anonymous users (fingerprint).
 *
 * Quota limits:
 * - Anonymous users: 5 calls/day
 * - Logged-in users: 10 calls/day
 */

import { supabase, isSupabaseConfigured } from '../lib/supabase';
import { getUserIdentifier } from './fingerprintService';

// Quota limits
const ANONYMOUS_DAILY_LIMIT = 15;
const AUTHENTICATED_DAILY_LIMIT = 20;

// Local storage key for anonymous usage tracking (fallback when Supabase not configured)
const LOCAL_USAGE_KEY = 'df_usage';

export interface QuotaInfo {
  used: number;
  limit: number;
  remaining: number;
  isAuthenticated: boolean;
}

/**
 * Get today's date string in YYYY-MM-DD format.
 */
function getTodayDate(): string {
  return new Date().toISOString().split('T')[0];
}

/**
 * Get local usage data (fallback for when Supabase is not configured).
 */
function getLocalUsage(): { date: string; count: number } {
  try {
    const data = localStorage.getItem(LOCAL_USAGE_KEY);
    if (data) {
      return JSON.parse(data);
    }
  } catch {
    // Ignore
  }
  return { date: getTodayDate(), count: 0 };
}

/**
 * Set local usage data.
 */
function setLocalUsage(date: string, count: number): void {
  localStorage.setItem(LOCAL_USAGE_KEY, JSON.stringify({ date, count }));
}

/**
 * Check current quota for a user.
 *
 * @param userId - Supabase user ID if logged in, null for anonymous
 * @param isAnonymous - Whether the user is an anonymous user (optional, defaults to false if userId exists)
 * @returns QuotaInfo with used, limit, and remaining counts
 */
export async function checkQuota(userId: string | null, isAnonymous: boolean = false): Promise<QuotaInfo> {
  // A user is authenticated (non-anonymous) only if they have a userId AND are not anonymous
  const isAuthenticated = Boolean(userId) && !isAnonymous;
  const limit = isAuthenticated ? AUTHENTICATED_DAILY_LIMIT : ANONYMOUS_DAILY_LIMIT;
  const userIdentifier = getUserIdentifier(userId);

  // If Supabase is not configured (self-hosted), no quota limits
  if (!isSupabaseConfigured()) {
    return {
      used: 0,
      limit: Number.MAX_SAFE_INTEGER, // Unlimited
      remaining: Number.MAX_SAFE_INTEGER,
      isAuthenticated: false,
    };
  }

  try {
    // For authenticated users, check points balance instead of daily usage
    if (isAuthenticated && userId) {
      const { data: balanceData, error: balanceError } = await supabase
        .from('points_balance')
        .select('balance')
        .eq('user_id', userId)
        .single();

      if (balanceError && balanceError.code !== 'PGRST116') {
        console.error('[quotaService] Failed to check points balance:', balanceError);
      }

      const balance = balanceData?.balance || 0;

      return {
        used: 0, // Not applicable for points-based system
        limit: balance, // Current balance is the "limit"
        remaining: balance,
        isAuthenticated,
      };
    }

    // For anonymous users, check daily usage count
    const today = getTodayDate();

    const { count, error } = await supabase
      .from('usage_records')
      .select('*', { count: 'exact', head: true })
      .eq('user_id', userIdentifier)
      .gte('created_at', `${today}T00:00:00Z`)
      .lt('created_at', `${today}T23:59:59Z`);

    if (error) {
      console.error('[quotaService] Failed to check quota:', error);
      return {
        used: 0,
        limit,
        remaining: limit,
        isAuthenticated,
      };
    }

    const used = count || 0;
    const remaining = Math.max(0, limit - used);

    return {
      used,
      limit,
      remaining,
      isAuthenticated,
    };
  } catch (err) {
    console.error('[quotaService] Error checking quota:', err);
    return {
      used: 0,
      limit,
      remaining: limit,
      isAuthenticated,
    };
  }
}

/**
 * Record a workflow usage.
 *
 * @param userId - Supabase user ID if logged in, null for anonymous
 * @param workflowType - Type of workflow used (e.g., 'paper2figure')
 * @returns true if recorded successfully
 */
export async function recordUsage(
  userId: string | null,
  workflowType: string
): Promise<boolean> {
  const userIdentifier = getUserIdentifier(userId);

  // If Supabase is not configured, use local storage
  if (!isSupabaseConfigured()) {
    const local = getLocalUsage();
    const today = getTodayDate();

    // Reset if new day
    const newCount = local.date === today ? local.count + 1 : 1;
    setLocalUsage(today, newCount);

    return true;
  }

  try {
    // For authenticated users (non-fingerprint), deduct 1 point
    if (userId && !userIdentifier.startsWith('fp_')) {
      const { data, error: rpcError } = await supabase.rpc('deduct_points', {
        p_user_id: userId,
        p_amount: 1,
        p_reason: `workflow_${workflowType}`
      });

      if (rpcError) {
        console.error('[quotaService] Failed to deduct points:', rpcError);
        return false;
      }

      // If deduction failed (insufficient points), return false
      if (!data) {
        console.warn('[quotaService] Insufficient points');
        return false;
      }
    }

    // Record usage in usage_records table
    const { error } = await supabase.from('usage_records').insert({
      user_id: userIdentifier,
      workflow_type: workflowType,
    });

    if (error) {
      console.error('[quotaService] Failed to record usage:', error);
      return false;
    }

    return true;
  } catch (err) {
    console.error('[quotaService] Error recording usage:', err);
    return false;
  }
}

/**
 * Check if user has remaining quota.
 *
 * @param userId - Supabase user ID if logged in, null for anonymous
 * @returns true if user has remaining quota
 */
export async function hasQuota(userId: string | null): Promise<boolean> {
  const quota = await checkQuota(userId);
  return quota.remaining > 0;
}
