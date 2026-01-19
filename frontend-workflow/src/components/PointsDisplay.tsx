/**
 * PointsDisplay component showing user's points balance.
 *
 * Replaces QuotaDisplay to show points instead of quota.
 * Hidden when Supabase is not configured or user not logged in.
 */

import { useEffect, useState } from "react";
import { useAuthStore } from "../stores/authStore";
import { isSupabaseConfigured, supabase } from "../lib/supabase";
import { Coins, Loader2 } from "lucide-react";

export function PointsDisplay() {
  const { user } = useAuthStore();
  const [points, setPoints] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!isSupabaseConfigured() || !user?.id) {
      setLoading(false);
      return;
    }

    const fetchPoints = async () => {
      try {
        const { data, error } = await supabase
          .from("points_balance")
          .select("balance")
          .eq("user_id", user.id)
          .single();

        if (error) {
          // No points yet, default to 0
          setPoints(0);
        } else {
          setPoints(data.balance);
        }
      } catch (err) {
        console.error("[PointsDisplay] Failed to load points:", err);
        setPoints(0);
      } finally {
        setLoading(false);
      }
    };

    fetchPoints();

    // Refresh every 60 seconds
    const interval = setInterval(fetchPoints, 60000);

    // Refresh when page becomes visible
    const handleVisibilityChange = () => {
      if (document.visibilityState === "visible") {
        fetchPoints();
      }
    };
    document.addEventListener("visibilitychange", handleVisibilityChange);

    return () => {
      clearInterval(interval);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [user?.id]);

  // Hide when Supabase is not configured or user not logged in
  if (!isSupabaseConfigured() || !user) return null;

  if (loading) {
    return (
      <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg border bg-white/5 border-white/10">
        <Loader2 size={16} className="animate-spin text-gray-400" />
        <span className="text-sm text-gray-400">...</span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg border bg-white/5 border-white/10">
      <Coins size={16} className="text-yellow-400" />
      <span className="text-sm text-gray-300">
        {points ?? 0} 积分
      </span>
    </div>
  );
}
