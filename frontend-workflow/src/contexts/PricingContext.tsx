import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { getPricingConfig, type PricingConfig } from '../services/pricingService';

interface PricingContextType {
  config: PricingConfig | null;
  loading: boolean;
  error: Error | null;
}

const PricingContext = createContext<PricingContextType>({
  config: null,
  loading: true,
  error: null,
});

export const usePricing = () => useContext(PricingContext);

interface PricingProviderProps {
  children: ReactNode;
}

export const PricingProvider: React.FC<PricingProviderProps> = ({ children }) => {
  const [config, setConfig] = useState<PricingConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    loadPricing();
  }, []);

  const loadPricing = async () => {
    try {
      const pricingConfig = await getPricingConfig();
      setConfig(pricingConfig);
      setError(null);
    } catch (err) {
      console.error('Failed to load pricing:', err);
      setError(err instanceof Error ? err : new Error('Failed to load pricing'));
    } finally {
      setLoading(false);
    }
  };

  return (
    <PricingContext.Provider value={{ config, loading, error }}>
      {children}
    </PricingContext.Provider>
  );
};
