export function appendManagedApiConfig(
  formData: FormData,
  userApiConfigRequired: boolean,
  apiUrl: string | null | undefined,
  apiKey: string | null | undefined,
  apiUrlKey: string = 'chat_api_url',
  apiKeyKey: string = 'api_key',
): void {
  if (!userApiConfigRequired) {
    return;
  }
  const normalizedApiUrl = (apiUrl || '').trim();
  const normalizedApiKey = (apiKey || '').trim();
  if (normalizedApiUrl) {
    formData.append(apiUrlKey, normalizedApiUrl);
  }
  if (normalizedApiKey) {
    formData.append(apiKeyKey, normalizedApiKey);
  }
}


export function appendManagedModel(
  formData: FormData,
  userApiConfigRequired: boolean,
  key: string,
  value: string | null | undefined,
): void {
  if (!userApiConfigRequired) {
    return;
  }
  const normalizedValue = (value || '').trim();
  if (normalizedValue) {
    formData.append(key, normalizedValue);
  }
}
