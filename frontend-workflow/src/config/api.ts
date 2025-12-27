/**
 * API configuration for backend calls.
 *
 * Contains the API key for authenticating with the backend.
 */

// API key must match backend's API_KEY
export const API_KEY = 'df-internal-2024-workflow-key';

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
