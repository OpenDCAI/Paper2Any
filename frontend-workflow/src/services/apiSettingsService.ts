/**
 * API settings service for managing user's API Base URL and API Key.
 * 
 * Settings are stored in localStorage per user_id to avoid re-entering
 * credentials across different tools.
 */

import { DEFAULT_LLM_API_URL } from '../config/api';
import { getRuntimeConfigSync } from './runtimeConfigService';

export interface ApiSettings {
  apiUrl: string;
  apiKey: string;
}

const STORAGE_KEY_PREFIX = "paper2any_api_settings_";
export const MANAGED_API_KEY_PLACEHOLDER = "__managed_by_backend__";

/**
 * Get API settings for a user from localStorage.
 * 
 * @param userId - User ID (null for anonymous users)
 * @returns ApiSettings or null if not found
 */
export function getApiSettings(userId: string | null): ApiSettings | null {
  const runtimeConfig = getRuntimeConfigSync();
  if (!runtimeConfig.user_api_config_required) {
    return {
      apiUrl: runtimeConfig.managed_api_url || DEFAULT_LLM_API_URL,
      apiKey: MANAGED_API_KEY_PLACEHOLDER,
    };
  }

  if (!userId) return null;

  try {
    const key = `${STORAGE_KEY_PREFIX}${userId}`;
    const saved = localStorage.getItem(key);
    if (!saved) return null;

    const parsed = JSON.parse(saved);
    return {
      apiUrl: parsed.apiUrl || "",
      apiKey: parsed.apiKey || "",
    };
  } catch (err) {
    console.error("[apiSettingsService] Failed to load settings:", err);
    return null;
  }
}

/**
 * Save API settings for a user to localStorage.
 * 
 * @param userId - User ID (null for anonymous users)
 * @param settings - API settings to save
 * @returns true if saved successfully
 */
export function saveApiSettings(userId: string | null, settings: ApiSettings): boolean {
  const runtimeConfig = getRuntimeConfigSync();
  if (!runtimeConfig.user_api_config_required) {
    return true;
  }
  if (!userId) return false;

  try {
    const key = `${STORAGE_KEY_PREFIX}${userId}`;
    localStorage.setItem(key, JSON.stringify(settings));
    return true;
  } catch (err) {
    console.error("[apiSettingsService] Failed to save settings:", err);
    return false;
  }
}

/**
 * Clear API settings for a user from localStorage.
 * 
 * @param userId - User ID (null for anonymous users)
 */
export function clearApiSettings(userId: string | null): void {
  const runtimeConfig = getRuntimeConfigSync();
  if (!runtimeConfig.user_api_config_required) {
    return;
  }
  if (!userId) return;

  try {
    const key = `${STORAGE_KEY_PREFIX}${userId}`;
    localStorage.removeItem(key);
  } catch (err) {
    console.error("[apiSettingsService] Failed to clear settings:", err);
  }
}
