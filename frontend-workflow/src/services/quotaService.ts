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
    // For authenticated users, check and grant daily usage, then return balance
    if (isAuthenticated && userId) {
      // Call RPC to check and grant daily usage (if eligible)
      const { data: newBalance, error: rpcError } = await supabase.rpc(
        'check_and_grant_daily_usage',
        { p_user_id: userId }
      );

      if (rpcError) {
        console.error('[quotaService] Failed to check/grant daily usage:', rpcError);
        // Fallback: query balance directly
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
          used: 0,
          limit: balance,
          remaining: balance,
          isAuthenticated,
        };
      }

      const balance = newBalance || 0;

      return {
        used: 0, // Not applicable for points-based system
        limit: balance, // Current balance is the "limit"
        remaining: balance,
        isAuthenticated,
      };
    }

    // For anonymous users, use local storage (cannot query Supabase with non-UUID user_id)
    const localUsage = getLocalUsage();
    const today = getTodayDate();
    
    // Reset count if it's a new day
    if (localUsage.date !== today) {
      setLocalUsage(today, 0);
      return {
        used: 0,
        limit,
        remaining: limit,
        isAuthenticated,
      };
    }

    return {
      used: localUsage.count,
      limit,
      remaining: Math.max(0, limit - localUsage.count),
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
    // For authenticated users, deduct 1 point
    if (userId) {
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
      
      return true;
    }

    // For anonymous users, use local storage (cannot insert non-UUID into usage_records)
    const local = getLocalUsage();
    const today = getTodayDate();
    const newCount = local.date === today ? local.count + 1 : 1;
    setLocalUsage(today, newCount);

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
