import { useEffect, useState } from 'react';

import { fetchRuntimeConfig, getRuntimeConfigSync, RuntimeConfig } from '../services/runtimeConfigService';

export function useRuntimeBilling() {
  const [runtimeConfig, setRuntimeConfig] = useState<RuntimeConfig>(getRuntimeConfigSync());

  useEffect(() => {
    let cancelled = false;

    fetchRuntimeConfig()
      .then((config) => {
        if (!cancelled) {
          setRuntimeConfig(config);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setRuntimeConfig(getRuntimeConfigSync());
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  return {
    runtimeConfig,
    userApiConfigRequired: runtimeConfig.user_api_config_required,
    managedBillingEnabled: !runtimeConfig.user_api_config_required,
  };
}
