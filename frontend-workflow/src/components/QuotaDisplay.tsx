/**
 * QuotaDisplay component showing remaining API calls.
 *
 * Shows "剩余次数: X/10" with visual states for normal, low, and exhausted.
 */

import { useEffect } from "react";
import { useAuthStore } from "../stores/authStore";
import { Zap, AlertTriangle } from "lucide-react";

export function QuotaDisplay() {
  const { quota, refreshQuota } = useAuthStore();

  useEffect(() => {
    // Always refresh quota on mount - backend returns mock data if not authenticated
    refreshQuota();
    // Refresh every 60 seconds
    const interval = setInterval(refreshQuota, 60000);
    return () => clearInterval(interval);
  }, [refreshQuota]);

  if (!quota) return null;

  const remaining = quota.remaining;
  const isLow = remaining <= 2 && remaining > 0;
  const isExhausted = remaining === 0;

  return (
    <div
      className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border transition-colors ${
        isExhausted
          ? "bg-red-500/20 border-red-500/30"
          : isLow
            ? "bg-yellow-500/20 border-yellow-500/30"
            : "bg-white/5 border-white/10"
      }`}
    >
      {isExhausted || isLow ? (
        <AlertTriangle
          size={16}
          className={isExhausted ? "text-red-400" : "text-yellow-400"}
        />
      ) : (
        <Zap size={16} className="text-primary-400" />
      )}
      <span
        className={`text-sm ${
          isExhausted
            ? "text-red-300"
            : isLow
              ? "text-yellow-300"
              : "text-gray-300"
        }`}
      >
        剩余次数: {remaining}/{quota.limit}
      </span>
    </div>
  );
}
