/**
 * PointsDisplay component showing user's points balance or remaining quota.
 *
 * Displays:
 * - Points balance for authenticated users
 * - Remaining daily quota for anonymous users
 * - "∞" if Supabase is not configured (unlimited local usage)
 */

import { useEffect } from "react";
import { useAuthStore } from "../stores/authStore";
import { Coins, Loader2 } from "lucide-react";

export function PointsDisplay() {
  const { quota, refreshQuota } = useAuthStore();
  
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
      <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg border bg-white/5 border-white/10">
        <Loader2 size={16} className="animate-spin text-gray-400" />
        <span className="text-sm text-gray-400">...</span>
      </div>
    );
  }

  // Check for "unlimited" quota (returned when Supabase is not configured)
  const isUnlimited = quota.remaining > 1000000;

  return (
    <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg border bg-white/5 border-white/10" title={isUnlimited ? "无限次数" : "剩余次数"}>
      <Coins size={16} className="text-yellow-400" />
      <span className="text-sm text-gray-300">
        {isUnlimited ? "∞" : `${quota.remaining} 次`}
      </span>
    </div>
  );
}
