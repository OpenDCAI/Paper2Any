import { DEFAULT_LLM_API_URL, getPurchaseUrl } from '../config/api';
import type { RuntimeConfig } from '../services/runtimeConfigService';

type RuntimeConfigLike = Pick<RuntimeConfig, 'billing_mode' | 'managed_api_url' | 'points_purchase_url'>;

export function resolvePointsPurchaseUrl(runtimeConfig?: Partial<RuntimeConfigLike> | null): string {
  const configuredUrl = runtimeConfig?.points_purchase_url?.trim();
  if (configuredUrl) {
    return configuredUrl;
  }

  const managedApiUrl = runtimeConfig?.managed_api_url?.trim() || DEFAULT_LLM_API_URL;
  return getPurchaseUrl(managedApiUrl);
}

export function buildManagedModeDescription(purchaseUrl?: string | null): string {
  if (purchaseUrl) {
    return '当前为平台点数模式，无需手动填写 API URL 或 API Key。功能调用会消耗点数；若点数不足，可前往购买页获取兑换码，再到账户页兑换加点。';
  }
  return '当前为平台点数模式，无需手动填写 API URL 或 API Key。功能调用会消耗点数；若点数不足，请先获取兑换码后到账户页兑换加点。';
}

export function buildInsufficientPointsMessage(
  required: number,
  remaining: number,
  action: string,
  purchaseUrl?: string | null,
): string {
  const suffix = purchaseUrl
    ? '点数不足时可前往购买页获取兑换码，再到账户页兑换加点。'
    : '点数不足时请先获取兑换码，再到账户页兑换加点。';
  return `点数不足：${action}需要 ${required} 点，当前剩余 ${remaining} 点。${suffix}`;
}

export function buildQuotaExhaustedMessage(purchaseUrl?: string | null): string {
  if (purchaseUrl) {
    return '当前点数不足，可前往购买页获取兑换码，再到账户页兑换加点后重试。';
  }
  return '当前点数不足，请先获取兑换码并到账户页兑换加点后重试。';
}

export function isInsufficientPointsError(message?: string | null): boolean {
  if (!message) {
    return false;
  }

  const normalized = message.toLowerCase();
  return [
    '点数不足',
    '积分不足',
    'quota used up',
    'insufficient points',
    '购买兑换码',
    '兑换码充值',
    '兑换加点',
  ].some((keyword) => normalized.includes(keyword.toLowerCase()));
}

export function getManagedValidationText(userApiConfigRequired: boolean): string {
  return userApiConfigRequired ? '正在验证 API Key 有效性...' : '正在验证平台托管模型连接...';
}
