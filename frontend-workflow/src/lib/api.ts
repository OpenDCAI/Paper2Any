/**
 * API client for authenticated requests to the backend.
 *
 * Automatically includes the Supabase access token in Authorization header.
 */

import { getAccessToken } from "../stores/authStore";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

interface RequestOptions extends RequestInit {
  skipAuth?: boolean;
}

/**
 * Make an authenticated fetch request to the API.
 */
export async function apiFetch(
  endpoint: string,
  options: RequestOptions = {}
): Promise<Response> {
  const { skipAuth = false, headers = {}, ...rest } = options;

  const requestHeaders: HeadersInit = {
    "Content-Type": "application/json",
    ...headers,
  };

  // Add auth token if available and not skipped
  if (!skipAuth) {
    const token = getAccessToken();
    if (token) {
      (requestHeaders as Record<string, string>)["Authorization"] =
        `Bearer ${token}`;
    }
  }

  const url = endpoint.startsWith("http")
    ? endpoint
    : `${API_BASE}${endpoint.startsWith("/") ? endpoint : `/${endpoint}`}`;

  return fetch(url, {
    ...rest,
    headers: requestHeaders,
  });
}

/**
 * GET request with auth.
 */
export async function apiGet<T>(
  endpoint: string,
  options?: RequestOptions
): Promise<T> {
  const response = await apiFetch(endpoint, { ...options, method: "GET" });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new ApiError(response.status, error.detail || response.statusText);
  }

  return response.json();
}

/**
 * POST request with auth.
 */
export async function apiPost<T>(
  endpoint: string,
  data?: unknown,
  options?: RequestOptions
): Promise<T> {
  const response = await apiFetch(endpoint, {
    ...options,
    method: "POST",
    body: data ? JSON.stringify(data) : undefined,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new ApiError(response.status, error.detail || response.statusText);
  }

  return response.json();
}

/**
 * DELETE request with auth.
 */
export async function apiDelete<T>(
  endpoint: string,
  options?: RequestOptions
): Promise<T> {
  const response = await apiFetch(endpoint, { ...options, method: "DELETE" });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new ApiError(response.status, error.detail || response.statusText);
  }

  return response.json();
}

/**
 * Custom error class for API errors.
 */
export class ApiError extends Error {
  constructor(
    public status: number,
    message: string
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/**
 * Check if error is an authentication error.
 */
export function isAuthError(error: unknown): boolean {
  return error instanceof ApiError && error.status === 401;
}

// ============ File API Types ============

export interface UserFile {
  id: string;
  file_name: string;
  file_size: number | null;
  workflow_type: string | null;
  created_at: string;
  download_url: string | null;
}

// ============ File API Methods ============

/**
 * Get all files for the current user.
 */
export async function getFiles(): Promise<UserFile[]> {
  return apiGet<UserFile[]>("/api/files");
}

/**
 * Delete a file by ID.
 */
export async function deleteFile(
  fileId: string
): Promise<{ success: boolean; message: string }> {
  return apiDelete<{ success: boolean; message: string }>(
    `/api/files/${fileId}`
  );
}
