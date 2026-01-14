import React from 'react';
import { formatPrice } from '../services/pricingService';
import { usePricing } from '../contexts/PricingContext';

interface PricingDisplayProps {
  service: string;
  endpoint: string;
  pageCount?: number;
  useAI?: boolean;
  className?: string;
  showUnitPriceOnly?: boolean; // 只显示单价，不计算总价
}

/**
 * 价格显示组件
 * 根据服务类型自动计算并显示价格
 */
export const PricingDisplay: React.FC<PricingDisplayProps> = ({
  service,
  endpoint,
  pageCount = 1,
  useAI = false,
  className = '',
  showUnitPriceOnly = false,
}) => {
  const { config, loading } = usePricing();

  // 调试信息
  console.log('[PricingDisplay] Debug:', {
    service,
    endpoint,
    loading,
    hasConfig: !!config,
    pricing: config?.pricing?.[service]?.[endpoint],
    showUnitPriceOnly,
    useAI
  });

  if (loading) {
    return <span className={className}>...</span>;
  }

  if (!config) {
    console.log('[PricingDisplay] No config available');
    return null;
  }

  const pricing = config.pricing[service]?.[endpoint];
  if (!pricing) {
    console.log('[PricingDisplay] No pricing found for', service, endpoint);
    // 如果找不到配置，显示默认提示
    return <span className={className}>价格加载失败，请刷新重试...</span>;
  }

  // 如果只显示单价
  if (showUnitPriceOnly) {
    if (pricing.type === 'per_page') {
      return (
        <span className={className}>
          {formatPrice(pricing.price_per_page || 0, config.currency)}/{pricing.unit || '页'}
        </span>
      );
    }
    if (pricing.type === 'hybrid') {
      if (useAI) {
        return (
          <span className={className}>
            基础总价 {formatPrice(pricing.base_price || 0, config.currency)} + 按页附加 {formatPrice(pricing.ai_price_per_page || 0, config.currency)}/{pricing.unit || '页'}
          </span>
        );
      } else {
        return (
          <span className={className}>
            {formatPrice(pricing.base_price || 0, config.currency)}
          </span>
        );
      }
    }
    if (pricing.type === 'fixed') {
      return (
        <span className={className}>
          {formatPrice(pricing.price || 0, config.currency)}
        </span>
      );
    }
    // 如果没有匹配到任何类型，返回 null
    return null;
  }

  const calculatePrice = () => {
    switch (pricing.type) {
      case 'fixed':
        return pricing.price || 0;
      
      case 'per_page':
        return (pricing.price_per_page || 0) * pageCount;
      
      case 'hybrid':
        const basePrice = pricing.base_price || 0;
        if (!useAI) {
          return basePrice;
        }
        const aiCost = (pricing.ai_price_per_page || 0) * pageCount;
        return basePrice + aiCost;
      
      default:
        return 0;
    }
  };

  const price = calculatePrice();
  const priceText = formatPrice(price, config.currency);

  // 根据价格类型显示不同的信息
  const renderPriceInfo = () => {
    switch (pricing.type) {
      case 'fixed':
        return (
          <span className={className}>
            {priceText}
          </span>
        );
      
      case 'per_page':
        return (
          <span className={className}>
            {priceText}
            {pageCount > 1 && (
              <span className="text-xs opacity-70 ml-1">
                ({pricing.price_per_page} × {pageCount} 页)
              </span>
            )}
          </span>
        );
      
      case 'hybrid':
        if (!useAI) {
          return (
            <span className={className}>
              {priceText}
            </span>
          );
        }
        return (
          <span className={className}>
            {priceText}
            <span className="text-xs opacity-70 ml-1">
              (基础 {pricing.base_price} + AI {pricing.ai_price_per_page} × {pageCount} 页)
            </span>
          </span>
        );
      
      default:
        return null;
    }
  };

  return renderPriceInfo();
};

/**
 * 价格信息卡片组件
 * 显示详细的价格说明
 */
export const PricingInfoCard: React.FC<PricingDisplayProps> = ({
  service,
  endpoint,
  pageCount = 1,
  useAI = false,
  className = '',
}) => {
  const { config, loading } = usePricing();

  if (loading || !config) {
    return null;
  }

  const pricing = config.pricing[service]?.[endpoint];
  if (!pricing) {
    return null;
  }

  return (
    <div className={`glass rounded-lg p-4 ${className}`}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm text-gray-300">{pricing.description}</span>
        <PricingDisplay
          service={service}
          endpoint={endpoint}
          pageCount={pageCount}
          useAI={useAI}
          className="text-lg font-semibold text-purple-400"
        />
      </div>
      
      {pricing.note && (
        <p className="text-xs text-gray-400 mt-1">
          💡 {pricing.note}
        </p>
      )}
      
      {pricing.example && (
        <p className="text-xs text-gray-500 mt-1">
          示例：{pricing.example}
        </p>
      )}
    </div>
  );
};

export default PricingDisplay;
