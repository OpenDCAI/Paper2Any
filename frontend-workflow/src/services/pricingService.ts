/**
 * 价格配置服务
 * 从后端 API 获取价格配置
 */

export interface PricingConfig {
  version: string;
  currency: string;
  last_updated?: string;
  pricing: {
    [service: string]: {
      [endpoint: string]: {
        type: 'fixed' | 'per_page' | 'hybrid';
        price?: number;
        price_per_page?: number;
        base_price?: number;
        ai_price_per_page?: number;
        description: string;
        unit?: string;
        note?: string;
        example?: string;
        examples?: {
          [key: string]: string;
        };
      };
    };
  };
}

let cachedPricing: PricingConfig | null = null;

/**
 * 获取价格配置
 */
export async function getPricingConfig(): Promise<PricingConfig> {
  // 如果有缓存，直接返回
  if (cachedPricing) {
    return cachedPricing;
  }

  try {
    const response = await fetch('/api/pricing', {
      credentials: 'include',
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch pricing: ${response.statusText}`);
    }

    const config = await response.json();
    cachedPricing = config;
    return config;
  } catch (error) {
    console.error('[Pricing] Failed to fetch pricing config:', error);
    // 返回默认配置
    return getDefaultPricingConfig();
  }
}

/**
 * 清除缓存（用于强制刷新）
 */
export function clearPricingCache() {
  cachedPricing = null;
}

/**
 * 获取默认价格配置（降级方案）
 */
function getDefaultPricingConfig(): PricingConfig {
  return {
    version: '1.0',
    currency: '积分',
    pricing: {},
  };
}

/**
 * 计算固定价格
 */
export function calculateFixedPrice(
  config: PricingConfig,
  service: string,
  endpoint: string
): number {
  const pricing = config.pricing[service]?.[endpoint];
  if (!pricing || pricing.type !== 'fixed') {
    return 0;
  }
  return pricing.price || 0;
}

/**
 * 计算按页价格
 */
export function calculatePerPagePrice(
  config: PricingConfig,
  service: string,
  endpoint: string,
  pageCount: number
): number {
  const pricing = config.pricing[service]?.[endpoint];
  if (!pricing || pricing.type !== 'per_page') {
    return 0;
  }
  const pricePerPage = pricing.price_per_page || 0;
  return pricePerPage * pageCount;
}

/**
 * 计算混合价格（基础 + AI 增强）
 */
export function calculateHybridPrice(
  config: PricingConfig,
  service: string,
  endpoint: string,
  useAI: boolean,
  pageCount: number = 1
): number {
  const pricing = config.pricing[service]?.[endpoint];
  if (!pricing || pricing.type !== 'hybrid') {
    return 0;
  }

  const basePrice = pricing.base_price || 0;
  if (!useAI) {
    return basePrice;
  }

  const aiPricePerPage = pricing.ai_price_per_page || 0;
  return basePrice + aiPricePerPage * pageCount;
}

/**
 * 格式化价格显示
 */
export function formatPrice(price: number, currency: string = '积分'): string {
  return `${price} ${currency}`;
}

/**
 * 获取价格描述
 */
export function getPriceDescription(
  config: PricingConfig,
  service: string,
  endpoint: string
): string {
  const pricing = config.pricing[service]?.[endpoint];
  if (!pricing) {
    return '';
  }

  const { type, description, note } = pricing;
  let desc = description;

  if (type === 'fixed' && pricing.price !== undefined) {
    desc += ` - ${formatPrice(pricing.price, config.currency)}`;
  } else if (type === 'per_page' && pricing.price_per_page !== undefined) {
    desc += ` - ${formatPrice(pricing.price_per_page, config.currency)}/${pricing.unit || '页'}`;
  } else if (type === 'hybrid') {
    desc += ` - 基础 ${formatPrice(pricing.base_price || 0, config.currency)}`;
    if (pricing.ai_price_per_page) {
      desc += `，AI 增强 ${formatPrice(pricing.ai_price_per_page, config.currency)}/${pricing.unit || '页'}`;
    }
  }

  if (note) {
    desc += ` (${note})`;
  }

  return desc;
}
