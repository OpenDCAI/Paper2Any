import { PropsWithChildren, useEffect, useState } from 'react';
import { fetchRuntimeConfig } from '../services/runtimeConfigService';

export function RuntimeConfigProvider({ children }: PropsWithChildren) {
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetchRuntimeConfig()
      .catch(() => undefined)
      .finally(() => {
        if (!cancelled) {
          setReady(true);
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  if (!ready) {
    return null;
  }

  return <>{children}</>;
}
