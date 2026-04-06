import { backendFetch } from './backendClient';

type SecureAssetCacheEntry = {
  accessUrl: string;
  expiresAtMs: number;
};

const secureAssetCache = new Map<string, SecureAssetCacheEntry>();
const CACHE_SKEW_MS = 30_000;
const DEFAULT_TTL_MS = 5 * 60 * 1000;

function getCacheKey(pathOrUrl: string): string {
  return (pathOrUrl || '').trim();
}

function getCachedAccessUrl(pathOrUrl: string): string | null {
  const key = getCacheKey(pathOrUrl);
  if (!key) {
    return null;
  }

  const cached = secureAssetCache.get(key);
  if (!cached) {
    return null;
  }

  if (cached.expiresAtMs <= Date.now() + CACHE_SKEW_MS) {
    secureAssetCache.delete(key);
    return null;
  }

  return cached.accessUrl;
}

function storeCachedAccessUrl(pathOrUrl: string, accessUrl: string, expiresAt?: string): void {
  const key = getCacheKey(pathOrUrl);
  if (!key || !accessUrl) {
    return;
  }

  const expiresAtMs = expiresAt ? Date.parse(expiresAt) : Date.now() + DEFAULT_TTL_MS;
  secureAssetCache.set(key, {
    accessUrl,
    expiresAtMs: Number.isNaN(expiresAtMs) ? Date.now() + DEFAULT_TTL_MS : expiresAtMs,
  });
}

export async function getSecureAssetUrl(pathOrUrl: string): Promise<string> {
  const key = getCacheKey(pathOrUrl);
  if (!key) {
    return '';
  }

  const cached = getCachedAccessUrl(key);
  if (cached) {
    return cached;
  }

  const res = await backendFetch('/api/v1/files/access-url', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ path: key }),
  });

  if (!res.ok) {
    throw new Error((await res.text()) || `Failed to get secure asset URL: ${res.status}`);
  }

  const data = await res.json();
  const accessUrl = typeof data?.access_url === 'string' ? data.access_url : '';
  if (!accessUrl) {
    throw new Error('Secure asset URL missing in response');
  }

  storeCachedAccessUrl(key, accessUrl, data?.expires_at);
  return accessUrl;
}

export async function downloadSecureAsset(pathOrUrl: string, filename?: string): Promise<void> {
  const accessUrl = await getSecureAssetUrl(pathOrUrl);
  if (!accessUrl || typeof document === 'undefined') {
    return;
  }

  const link = document.createElement('a');
  link.href = accessUrl;
  link.rel = 'noopener noreferrer';
  if (filename) {
    link.download = filename;
  } else {
    link.download = '';
  }
  document.body.appendChild(link);
  link.click();
  link.remove();
}

export async function openSecureAsset(pathOrUrl: string): Promise<void> {
  const accessUrl = await getSecureAssetUrl(pathOrUrl);
  if (!accessUrl || typeof window === 'undefined') {
    return;
  }
  window.open(accessUrl, '_blank', 'noopener,noreferrer');
}

export function clearSecureAssetCache(pathOrUrl?: string): void {
  const key = getCacheKey(pathOrUrl || '');
  if (!key) {
    secureAssetCache.clear();
    return;
  }
  secureAssetCache.delete(key);
}
