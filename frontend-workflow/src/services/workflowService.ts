/**
 * Workflow service - wraps API calls with quota checking and usage recording.
 *
 * Usage:
 *   const result = await callWorkflow('paper2figure', '/api/paper2figure/generate', formData);
 *   if (result.error) {
 *     // Handle error (quota exceeded, API error, etc.)
 *   } else {
 *     // Use result.response
 *   }
 */

import { API_KEY } from '../config/api';
import { checkQuota, recordUsage, QuotaInfo } from './quotaService';
import { saveFileRecord } from './fileService';

export interface WorkflowResult {
  success: boolean;
  response?: Response;
  data?: any;
  error?: string;
  quota?: QuotaInfo;
}

/**
 * Get the current user ID and anonymous status from Supabase session.
 */
async function getCurrentUserInfo(): Promise<{ userId: string | null; isAnonymous: boolean }> {
  try {
    const { supabase, isSupabaseConfigured } = await import('../lib/supabase');
    if (!isSupabaseConfigured()) return { userId: null, isAnonymous: true };

    const { data: { user } } = await supabase.auth.getUser();
    return {
      userId: user?.id || null,
      isAnonymous: user?.is_anonymous || false
    };
  } catch {
    return { userId: null, isAnonymous: true };
  }
}

/**
 * Call a workflow endpoint with quota checking and usage recording.
 *
 * Flow:
 * 1. Check quota
 * 2. Call API with API key
 * 3. Record usage on success
 * 4. Save file record on success
 *
 * @param workflowType - Type of workflow (e.g., 'paper2figure')
 * @param url - API endpoint URL
 * @param body - Request body (FormData or object)
 * @param options - Additional options
 * @returns WorkflowResult with response or error
 */
export async function callWorkflow(
  workflowType: string,
  url: string,
  body: FormData | Record<string, any>,
  options: {
    outputFileName?: string;
    expectBlob?: boolean;
  } = {}
): Promise<WorkflowResult> {
  const { userId, isAnonymous } = await getCurrentUserInfo();

  // 1. Check quota
  const quota = await checkQuota(userId, isAnonymous);
  if (quota.remaining <= 0) {
    return {
      success: false,
      error: quota.isAuthenticated
        ? '今日配额已用完（10次/天），请明天再试'
        : '今日配额已用完（5次/天），登录后可获得更多配额',
      quota,
    };
  }

  // 2. Call API with API key
  try {
    const headers: HeadersInit = {
      'X-API-Key': API_KEY,
    };

    // Don't set Content-Type for FormData - browser will set it with boundary
    const isFormData = body instanceof FormData;
    if (!isFormData) {
      headers['Content-Type'] = 'application/json';
    }

    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: isFormData ? body : JSON.stringify(body),
    });

    if (!response.ok) {
      let errorMsg = `请求失败 (${response.status})`;
      try {
        const errorData = await response.json();
        errorMsg = errorData.detail || errorData.message || errorMsg;
      } catch {
        // Ignore JSON parse error
      }
      return {
        success: false,
        error: errorMsg,
        quota,
      };
    }

    // 3. Record usage on success
    await recordUsage(userId, workflowType);

    // 4. Save file record if filename provided
    if (options.outputFileName) {
      const fileSize = options.expectBlob
        ? (await response.clone().blob()).size
        : undefined;
      await saveFileRecord(options.outputFileName, workflowType, fileSize);
    }

    // 5. Trigger quota refresh in auth store (async, don't wait)
    try {
      const { useAuthStore } = await import('../stores/authStore');
      useAuthStore.getState().refreshQuota();
    } catch {
      // Silently fail if store import fails
    }

    // Return response for further processing
    return {
      success: true,
      response,
      quota: {
        ...quota,
        used: quota.used + 1,
        remaining: quota.remaining - 1,
      },
    };
  } catch (err) {
    return {
      success: false,
      error: err instanceof Error ? err.message : '网络错误，请重试',
      quota,
    };
  }
}

/**
 * Call a workflow and parse JSON response.
 */
export async function callWorkflowJson<T = any>(
  workflowType: string,
  url: string,
  body: FormData | Record<string, any>,
  outputFileName?: string
): Promise<WorkflowResult & { data?: T }> {
  const result = await callWorkflow(workflowType, url, body, {
    outputFileName,
    expectBlob: false,
  });

  if (!result.success || !result.response) {
    return result;
  }

  try {
    const data = await result.response.json();
    return { ...result, data };
  } catch {
    return { ...result, success: false, error: '解析响应失败' };
  }
}

/**
 * Call a workflow and get blob response.
 */
export async function callWorkflowBlob(
  workflowType: string,
  url: string,
  body: FormData | Record<string, any>,
  outputFileName?: string
): Promise<WorkflowResult & { blob?: Blob; filename?: string }> {
  const result = await callWorkflow(workflowType, url, body, {
    outputFileName,
    expectBlob: true,
  });

  if (!result.success || !result.response) {
    return result;
  }

  try {
    const blob = await result.response.blob();

    // Try to get filename from Content-Disposition header
    const disposition = result.response.headers.get('content-disposition') || '';
    const match = disposition.match(/filename="?([^";]+)"?/i);
    const filename = match?.[1] ? decodeURIComponent(match[1]) : outputFileName;

    return { ...result, blob, filename };
  } catch {
    return { ...result, success: false, error: '下载文件失败' };
  }
}
