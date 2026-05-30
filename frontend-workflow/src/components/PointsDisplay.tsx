/**
 * PointsDisplay component showing user's points balance or remaining quota.
 *
 * Displays:
 * - Points balance for authenticated users
 * - "∞" if Supabase is not configured (unlimited local usage)
 */

import { useEffect } from "react";
import { useAuthStore } from "../stores/authStore";
import { Coins, Loader2 } from "lucide-react";

export function PointsDisplay() {
  const { user, quota, refreshQuota } = useAuthStore();
  
  useEffect(() => {
    // Initial fetch
    refreshQuota();

    // Refresh every 60 seconds
    const interval = setInterval(refreshQuota, 60000);

    // Refresh when page becomes visible
    const handleVisibilityChange = () => {
      if (document.visibilityState === "visible") {
        refreshQuota();
      }
    };
    document.addEventListener("visibilitychange", handleVisibilityChange);

    return () => {
      clearInterval(interval);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [refreshQuota]);

  // Show loading state if quota hasn't been fetched yet
  if (!quota) {
    return (
      <div className="flex items-center gap-2 rounded-lg border border-white/10 bg-white/5 px-2 py-1.5 sm:px-3">
        <Loader2 size={16} className="animate-spin text-gray-400" />
        <span className="text-xs text-gray-400 sm:text-sm">...</span>
      </div>
    );
  }

  // Check for "unlimited" quota (returned when Supabase is not configured)
  const isUnlimited = quota.remaining > 1000000;
  const isAuthenticatedUser = Boolean(user);
  const balanceLabel = "点";
  const title = isUnlimited
    ? (quota.billingMode === 'paid' ? '当前为付费模式，平台不扣点' : '当前为无限用量')
    : (isAuthenticatedUser ? '剩余点数' : '当前点数');

  return (
    <div className="flex items-center gap-2 rounded-lg border border-white/10 bg-white/5 px-2 py-1.5 sm:px-3" title={title}>
      <Coins size={16} className="text-yellow-400" />
      <span className="text-xs text-gray-300 sm:text-sm">
        {isUnlimited ? "∞" : `${quota.remaining} ${balanceLabel}`}
      </span>
    </div>
  );
}
