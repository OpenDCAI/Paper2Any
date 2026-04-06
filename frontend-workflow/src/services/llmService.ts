import { backendFetch } from './backendClient';
import { fetchRuntimeConfig, getRuntimeConfigSync } from './runtimeConfigService';

const DEFAULT_VERIFY_TIMEOUT_MS = 30000;

function getVerifyTimeoutMs(): number {
  const raw = Number(import.meta.env.VITE_LLM_VERIFY_TIMEOUT_MS ?? DEFAULT_VERIFY_TIMEOUT_MS);
  if (!Number.isFinite(raw) || raw <= 0) {
    return DEFAULT_VERIFY_TIMEOUT_MS;
  }
  return raw;
}

/**
 * Verify LLM connection by sending a simple "Hi" message.
 * 
 * @param apiUrl Base URL of the LLM API (e.g., https://api.apiyi.com/v1)
 * @param apiKey API Key
 * @param model Model name (optional, defaults to gpt-4 or user provided)
 * @returns Promise that resolves to true if successful, throws error otherwise
 */
export async function verifyLlmConnection(
  apiUrl: string,
  apiKey: string,
  model: string = 'deepseek-v3.2'
): Promise<boolean> {
  const runtimeConfig = getRuntimeConfigSync();
  if (!runtimeConfig.user_api_config_required) {
    await fetchRuntimeConfig().catch(() => undefined);
    if (!getRuntimeConfigSync().user_api_config_required) {
      return true;
    }
  }

  // Normalize URL
  let baseUrl = apiUrl.trim();
  if (baseUrl.endsWith('/')) {
    baseUrl = baseUrl.slice(0, -1);
  }
  
  // Use the backend verification endpoint to avoid Mixed Content issues
  // The backend will proxy the request to the LLM API (even if it is HTTP)
  const verifyUrl = '/api/v1/system/verify-llm';

  try {
    const controller = new AbortController();
    const timeoutMs = getVerifyTimeoutMs();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    const res = await backendFetch(verifyUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        api_url: baseUrl,
        api_key: apiKey,
        model: model
      }),
      signal: controller.signal
    });

    clearTimeout(timeoutId);

    if (!res.ok) {
      const errText = await res.text().catch(() => '');
      let errMsg = `API Error: ${res.status}`;
      try {
        const errJson = JSON.parse(errText);
        if (errJson.detail) {
           errMsg += ` - ${errJson.detail}`;
        } else if (errJson.error) {
           errMsg += ` - ${errJson.error}`;
        }
      } catch (e) {
        if (errText) {
            errMsg += ` - ${errText.slice(0, 100)}`;
        }
      }
      throw new Error(errMsg);
    }

    const data = await res.json();
    
    if (!data.success) {
      throw new Error(data.error || 'LLM Verification failed');
    }

    return true;
  } catch (err) {
    if (err instanceof Error) {
        if (err.name === 'AbortError') {
            throw new Error(`连接超时，请检查网络、API URL，或把校验超时调大到 ${getVerifyTimeoutMs()}ms 以上`);
        }
        throw err;
    }
    throw new Error('Unknown error during API verification');
  }
}
