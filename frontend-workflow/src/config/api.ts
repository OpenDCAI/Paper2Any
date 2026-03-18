/**
 * API configuration for backend calls.
 *
 * Contains the API key for authenticating with the backend and
 * default LLM provider settings for the frontend UI.
 */

// API key for backend authentication (must match backend BACKEND_API_KEY)
export const API_KEY = import.meta.env.VITE_API_KEY || '';

// LLM Provider Default Configuration (for frontend UI defaults)
export const DEFAULT_LLM_API_URL = import.meta.env.VITE_DEFAULT_LLM_API_URL || 'https://api.apiyi.com/v1';

// List of available LLM API URLs
export const API_URL_OPTIONS = (import.meta.env.VITE_LLM_API_URLS || 'https://api.apiyi.com/v1,http://b.apiyi.com:16888/v1,https://ai.comfly.chat/v1,http://123.129.219.111:3000/v1').split(',').map((url: string) => url.trim());

/**
 * Get purchase link based on selected LLM API URL.
 */
export function getPurchaseUrl(apiUrl: string): string {
  if (apiUrl.includes('ai.comfly.chat')) {
    return 'https://ai.comfly.chat/register?aff=HsQn96268';
  }
  if (apiUrl.includes('123.129.219.111')) {
    return 'http://123.129.219.111:3000';
  }
  if (apiUrl.includes('apiyi')) {
    return 'https://api.apiyi.com/register/?aff_code=TbrD';
  }
  return 'https://api.apiyi.com/register/?aff_code=TbrD';
}

/**
 * Get headers for API calls including the API key.
 */
export function getApiHeaders(): HeadersInit {
  return {
    'X-API-Key': API_KEY,
  };
}

/**
 * Create a fetch wrapper that includes the API key.
 *
 * @param url - URL to fetch
 * @param options - Fetch options
 * @returns Fetch response
 */
export async function apiFetch(
  url: string,
  options: RequestInit = {}
): Promise<Response> {
  const headers = new Headers(options.headers);
  headers.set('X-API-Key', API_KEY);

  return fetch(url, {
    ...options,
    headers,
  });
}
