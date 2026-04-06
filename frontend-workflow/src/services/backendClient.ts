import { API_KEY } from '../config/api';
import { isSupabaseConfigured, supabase } from '../lib/supabase';

const BACKEND_ORIGIN = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');

export async function getBackendHeaders(initialHeaders: HeadersInit = {}): Promise<Headers> {
  const headers = new Headers(initialHeaders);
  headers.set('X-API-Key', API_KEY);

  if (isSupabaseConfigured()) {
    try {
      const { data } = await supabase.auth.getSession();
      const token = data.session?.access_token;
      if (token) {
        headers.set('Authorization', `Bearer ${token}`);
      }
    } catch (err) {
      console.warn('[backendClient] Failed to get session for Authorization header:', err);
    }
  }

  return headers;
}

export async function backendFetch(url: string, options: RequestInit = {}): Promise<Response> {
  const headers = await getBackendHeaders(options.headers);
  return fetch(url, {
    ...options,
    headers,
  });
}

function getPublicBackendOrigin(): string {
  if (BACKEND_ORIGIN) {
    return BACKEND_ORIGIN;
  }
  if (typeof window !== 'undefined' && window.location?.origin) {
    return window.location.origin.replace(/\/$/, '');
  }
  return '';
}

function extractOutputsPath(value: string): string | null {
  if (!value) {
    return null;
  }
  const outputsIndex = value.indexOf('/outputs/');
  if (outputsIndex >= 0) {
    return value.slice(outputsIndex);
  }
  if (value.startsWith('/outputs/')) {
    return value;
  }
  return null;
}

export function normalizeBackendAssetUrl(value: string): string {
  if (!value) {
    return value;
  }

  const outputsPath = extractOutputsPath(value);
  if (outputsPath) {
    const origin = getPublicBackendOrigin();
    return origin ? `${origin}${outputsPath}` : outputsPath;
  }

  if (
    typeof window !== 'undefined'
    && window.location.protocol === 'https:'
    && value.startsWith('http://')
  ) {
    try {
      const currentHost = window.location.hostname;
      const parsed = new URL(value);
      if (parsed.hostname === currentHost) {
        parsed.protocol = 'https:';
        return parsed.toString();
      }
    } catch {
      return value;
    }
  }

  return value;
}
