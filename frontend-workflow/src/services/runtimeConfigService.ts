import { DEFAULT_LLM_API_URL } from '../config/api';
import { backendFetch } from './backendClient';

export interface RuntimeConfig {
  billing_mode: 'paid' | 'free';
  user_api_config_required: boolean;
  managed_api_enabled: boolean;
  managed_api_url: string;
  server_side_billing_enforced: boolean;
  workflow_costs: Record<string, number>;
  guest_daily_limit: number;
  signup_bonus_points: number;
  daily_grant_points: number;
  daily_grant_balance_cap: number;
  referral_inviter_points: number;
  referral_invitee_points: number;
  points_purchase_url: string;
  points_redeem_enabled: boolean;
}

const STORAGE_KEY = 'paper2any_runtime_config';

const DEFAULT_RUNTIME_CONFIG: RuntimeConfig = {
  billing_mode: 'paid',
  user_api_config_required: true,
  managed_api_enabled: false,
  managed_api_url: DEFAULT_LLM_API_URL,
  server_side_billing_enforced: false,
  workflow_costs: {},
  guest_daily_limit: 0,
  signup_bonus_points: 0,
  daily_grant_points: 5,
  daily_grant_balance_cap: 15,
  referral_inviter_points: 5,
  referral_invitee_points: 0,
  points_purchase_url: '',
  points_redeem_enabled: false,
};

let runtimeConfigCache: RuntimeConfig | null = null;
let runtimeConfigPromise: Promise<RuntimeConfig> | null = null;

function loadStoredRuntimeConfig(): RuntimeConfig | null {
  if (typeof window === 'undefined') {
    return null;
  }
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return null;
    }
    const parsed = JSON.parse(raw);
    return { ...DEFAULT_RUNTIME_CONFIG, ...parsed };
  } catch (err) {
    console.warn('[runtimeConfigService] Failed to load cached runtime config:', err);
    return null;
  }
}

function persistRuntimeConfig(config: RuntimeConfig): void {
  if (typeof window === 'undefined') {
    return;
  }
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
  } catch (err) {
    console.warn('[runtimeConfigService] Failed to persist runtime config:', err);
  }
}

export function getRuntimeConfigSync(): RuntimeConfig {
  if (runtimeConfigCache) {
    return runtimeConfigCache;
  }
  const stored = loadStoredRuntimeConfig();
  if (stored) {
    runtimeConfigCache = stored;
    return runtimeConfigCache;
  }
  runtimeConfigCache = DEFAULT_RUNTIME_CONFIG;
  return runtimeConfigCache;
}

export async function fetchRuntimeConfig(force: boolean = false): Promise<RuntimeConfig> {
  if (!force && runtimeConfigCache && runtimeConfigCache !== DEFAULT_RUNTIME_CONFIG) {
    return runtimeConfigCache;
  }
  if (!force && runtimeConfigPromise) {
    return runtimeConfigPromise;
  }

  runtimeConfigPromise = (async () => {
    try {
      const response = await backendFetch('/api/v1/account/runtime-config');
      if (!response.ok) {
        throw new Error(`Runtime config request failed: ${response.status}`);
      }
      const data = await response.json();
      const config: RuntimeConfig = {
        ...DEFAULT_RUNTIME_CONFIG,
        ...data,
      };
      runtimeConfigCache = config;
      persistRuntimeConfig(config);
      return config;
    } catch (err) {
      console.warn('[runtimeConfigService] Falling back to cached/default runtime config:', err);
      const fallbackConfig = loadStoredRuntimeConfig() || DEFAULT_RUNTIME_CONFIG;
      runtimeConfigCache = fallbackConfig;
      return fallbackConfig;
    } finally {
      runtimeConfigPromise = null;
    }
  })();

  return runtimeConfigPromise as Promise<RuntimeConfig>;
}
