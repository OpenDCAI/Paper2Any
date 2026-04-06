import { Coins } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { DEFAULT_LLM_API_URL, getPurchaseUrl } from '../config/api';
import { useRuntimeBilling } from '../hooks/useRuntimeBilling';
import QRCodeTooltip from './QRCodeTooltip';

export function PurchaseEntry() {
  const { t } = useTranslation('common');
  const { runtimeConfig } = useRuntimeBilling();

  if (runtimeConfig.billing_mode !== 'free') {
    return null;
  }

  const purchaseUrl =
    runtimeConfig.points_purchase_url?.trim()
    || getPurchaseUrl(runtimeConfig.managed_api_url || DEFAULT_LLM_API_URL);

  if (!purchaseUrl) {
    return null;
  }

  return (
    <QRCodeTooltip>
      <a
        href={purchaseUrl}
        target="_blank"
        rel="noreferrer"
        className="group inline-flex items-center gap-2 rounded-xl border border-amber-400/20 bg-amber-500/10 px-3 py-1.5 text-sm text-amber-100 transition-all duration-200 hover:border-amber-300/40 hover:bg-amber-500/15 hover:text-white"
        title={t('app.purchaseMore')}
      >
        <Coins size={16} className="text-amber-300 transition-transform duration-200 group-hover:scale-110" />
        <span className="whitespace-nowrap">{t('app.purchaseMore')}</span>
      </a>
    </QRCodeTooltip>
  );
}
