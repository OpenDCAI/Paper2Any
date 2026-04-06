import React from 'react';
import { ExternalLink, ShieldCheck } from 'lucide-react';

import { useRuntimeBilling } from '../hooks/useRuntimeBilling';
import { buildManagedModeDescription, resolvePointsPurchaseUrl } from '../utils/pointsMessaging';

interface ManagedApiNoticeProps {
  className?: string;
  title?: string;
  description?: string;
}

const ManagedApiNotice: React.FC<ManagedApiNoticeProps> = ({
  className = '',
  title = '后端托管模型已开启',
  description,
}) => (
  <ManagedApiNoticeBody className={className} title={title} description={description} />
);

const ManagedApiNoticeBody: React.FC<ManagedApiNoticeProps> = ({
  className = '',
  title = '后端托管模型已开启',
  description,
}) => {
  const { runtimeConfig } = useRuntimeBilling();
  const purchaseUrl = runtimeConfig.billing_mode === 'free'
    ? resolvePointsPurchaseUrl(runtimeConfig)
    : '';
  const resolvedDescription = description || buildManagedModeDescription(purchaseUrl);

  return (
    <div className={`rounded-xl border border-emerald-500/30 bg-emerald-500/10 px-4 py-3 ${className}`.trim()}>
      <div className="flex items-start gap-3">
        <ShieldCheck size={18} className="mt-0.5 text-emerald-300 flex-shrink-0" />
        <div>
          <p className="text-sm font-medium text-emerald-200">{title}</p>
          <p className="mt-1 text-xs leading-relaxed text-emerald-100/80">{resolvedDescription}</p>
          {runtimeConfig.billing_mode === 'free' && purchaseUrl && (
            <a
              href={purchaseUrl}
              target="_blank"
              rel="noreferrer"
              className="mt-2 inline-flex items-center gap-1.5 text-xs font-medium text-emerald-100 transition-colors hover:text-white"
            >
              <span>前往购买页获取兑换码</span>
              <ExternalLink size={12} />
            </a>
          )}
        </div>
      </div>
    </div>
  );
};

export default ManagedApiNotice;
