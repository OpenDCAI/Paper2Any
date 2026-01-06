import { API_KEY } from '../config/api';

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
  model: string = 'gpt-4o'
): Promise<boolean> {
  // Normalize URL
  let baseUrl = apiUrl.trim();
  if (baseUrl.endsWith('/')) {
    baseUrl = baseUrl.slice(0, -1);
  }
  
  // Use the backend verification endpoint to avoid Mixed Content issues
  // The backend will proxy the request to the LLM API (even if it is HTTP)
  const verifyUrl = '/api/verify-llm';

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10000); // 10s timeout

    const res = await fetch(verifyUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': API_KEY,
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
            throw new Error('连接超时，请检查网络或 API URL');
        }
        throw err;
    }
    throw new Error('Unknown error during API verification');
  }
}
