import { backendFetch } from './backendClient';
import { fetchRuntimeConfig, getRuntimeConfigSync } from './runtimeConfigService';

export interface QuotaInfo {
  used: number;
  limit: number;
  remaining: number;
  isAuthenticated: boolean;
  billingMode?: string;
}

export interface RecordUsageOptions {
  amount?: number;
  // Kept only for backward-compatible call sites. Guest mode has been removed.
  isAnonymous?: boolean;
}

interface CachedQuotaEntry {
  value: QuotaInfo;
  expiresAt: number;
  staleUntil: number;
}

const QUOTA_CACHE_TTL_MS = 10_000;
const QUOTA_STALE_TTL_MS = 60_000;
const quotaCache = new Map<string, CachedQuotaEntry>();
const inflightQuotaRequests = new Map<string, Promise<QuotaInfo>>();

function buildUnlimitedQuota(): QuotaInfo {
  return {
    used: 0,
    limit: Number.MAX_SAFE_INTEGER,
    remaining: Number.MAX_SAFE_INTEGER,
    isAuthenticated: false,
    billingMode: getRuntimeConfigSync().billing_mode,
  };
}

function getQuotaCacheKey(userId: string | null): string {
  return userId || '__anonymous__';
}

function readCachedQuota(userId: string | null, allowStale: boolean = false): QuotaInfo | null {
  const key = getQuotaCacheKey(userId);
  const entry = quotaCache.get(key);
  if (!entry) {
    return null;
  }

  const now = Date.now();
  if (entry.staleUntil <= now) {
    quotaCache.delete(key);
    return null;
  }
  if (!allowStale && entry.expiresAt <= now) {
    return null;
  }
  return { ...entry.value };
}

function writeCachedQuota(userId: string | null, quota: QuotaInfo): void {
  const now = Date.now();
  quotaCache.set(getQuotaCacheKey(userId), {
    value: { ...quota },
    expiresAt: now + QUOTA_CACHE_TTL_MS,
    staleUntil: now + QUOTA_STALE_TTL_MS,
  });
}

export function invalidateQuotaCache(userId: string | null): void {
  const key = getQuotaCacheKey(userId);
  quotaCache.delete(key);
  inflightQuotaRequests.delete(key);
}

function normalizeQuotaResponse(data: any): QuotaInfo {
  return {
    used: Number(data.used || 0),
    limit: Number(data.limit || 0),
    remaining: Number(data.remaining || 0),
    isAuthenticated: Boolean(data.is_authenticated),
    billingMode: data.billing_mode || getRuntimeConfigSync().billing_mode,
  };
}

/**
 * Check current quota for the active deployment mode.
 *
 * - With Supabase configured, quota comes from the authenticated account.
 * - Without Supabase, backend returns unlimited local usage.
 */
export async function checkQuota(userId: string | null, _legacyIsAnonymous: boolean = false): Promise<QuotaInfo> {
  const cachedQuota = readCachedQuota(userId);
  if (cachedQuota) {
    return cachedQuota;
  }

  const cacheKey = getQuotaCacheKey(userId);
  const inflight = inflightQuotaRequests.get(cacheKey);
  if (inflight) {
    return inflight;
  }

  const requestPromise = (async () => {
    await fetchRuntimeConfig().catch(() => undefined);
    try {
      const response = await backendFetch('/api/v1/account/quota');
      if (!response.ok) {
        console.warn('[quotaService] Quota request failed:', response.status);
        return readCachedQuota(userId, true) ?? buildUnlimitedQuota();
      }

      const data = await response.json();
      const quota = normalizeQuotaResponse(data);
      writeCachedQuota(userId, quota);
      return quota;
    } catch (err) {
      console.error('[quotaService] Error checking quota:', err);
      return readCachedQuota(userId, true) ?? buildUnlimitedQuota();
    } finally {
      inflightQuotaRequests.delete(cacheKey);
    }
  })();

  inflightQuotaRequests.set(cacheKey, requestPromise);
  return requestPromise;
}

/**
 * Record a workflow usage.
 *
 * @param userId - Supabase user ID if logged in
 * @param workflowType - Type of workflow used (e.g., 'paper2figure')
 * @returns true if recorded successfully
 */
export async function recordUsage(
  userId: string | null,
  workflowType: string,
  options: RecordUsageOptions = {}
): Promise<boolean> {
  await fetchRuntimeConfig().catch(() => undefined);
  if (getRuntimeConfigSync().server_side_billing_enforced) {
    return true;
  }

  const amount = Math.max(1, options.amount ?? 1);

  try {
    const response = await backendFetch('/api/v1/account/quota/consume', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        workflow_type: workflowType,
        amount,
      }),
    });
    if (!response.ok) {
      console.warn('[quotaService] Failed to consume quota:', response.status);
      return false;
    }
    invalidateQuotaCache(userId);
    return true;
  } catch (err) {
    console.error('[quotaService] Error recording usage:', err);
    return false;
  }
}

/**
 * Check if user has remaining quota.
 *
 * @param userId - Supabase user ID if logged in
 * @returns true if user has remaining quota
 */
export async function hasQuota(userId: string | null): Promise<boolean> {
  const quota = await checkQuota(userId);
  return quota.remaining > 0;
}
