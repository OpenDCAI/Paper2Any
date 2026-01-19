/**
 * Account page showing user invite code, points balance, and API settings.
 */

import { useState, useEffect } from "react";
import { useAuthStore } from "../stores/authStore";
import { supabase } from "../lib/supabase";
import { getApiSettings, saveApiSettings } from "../services/apiSettingsService";
import { Ticket, Coins, Key, AlertCircle, Loader2, Copy, CheckCircle2, Settings } from "lucide-react";

interface ProfileData {
  invite_code: string;
}

interface PointsBalance {
  balance: number;
}

export function AccountPage() {
  const { user, claimInviteCode, error: authError } = useAuthStore();
  const [profile, setProfile] = useState<ProfileData | null>(null);
  const [points, setPoints] = useState<PointsBalance | null>(null);
  const [loadingProfile, setLoadingProfile] = useState(true);
  const [loadingPoints, setLoadingPoints] = useState(true);
  const [inviteCodeInput, setInviteCodeInput] = useState("");
  const [claiming, setClaiming] = useState(false);
  const [claimSuccess, setClaimSuccess] = useState(false);
  const [copied, setCopied] = useState(false);

  // API settings (local storage)
  const [apiUrl, setApiUrl] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [savingSettings, setSavingSettings] = useState(false);
  const [settingsSaved, setSettingsSaved] = useState(false);

  const userId = user?.id;

  // Load profile (invite_code)
  useEffect(() => {
    if (!userId) {
      setLoadingProfile(false);
      return;
    }

    const fetchProfile = async () => {
      try {
        const { data, error } = await supabase
          .from("profiles")
          .select("invite_code")
          .eq("user_id", userId)
          .single();

        if (error) throw error;
        setProfile(data);
      } catch (err) {
        console.error("[AccountPage] Failed to load profile:", err);
      } finally {
        setLoadingProfile(false);
      }
    };

    fetchProfile();
  }, [userId]);

  // Load points balance
  useEffect(() => {
    if (!userId) {
      setLoadingPoints(false);
      return;
    }

    const fetchPoints = async () => {
      try {
        const { data, error } = await supabase
          .from("points_balance")
          .select("balance")
          .eq("user_id", userId)
          .single();

        if (error) {
          // No points yet, default to 0
          setPoints({ balance: 0 });
        } else {
          setPoints(data);
        }
      } catch (err) {
        console.error("[AccountPage] Failed to load points:", err);
        setPoints({ balance: 0 });
      } finally {
        setLoadingPoints(false);
      }
    };

    fetchPoints();
  }, [userId]);

  // Load API settings from localStorage
  useEffect(() => {
    if (!userId) return;

    const settings = getApiSettings(userId);
    if (settings) {
      setApiUrl(settings.apiUrl);
      setApiKey(settings.apiKey);
    }
  }, [userId]);

  const handleClaimInvite = async () => {
    if (!inviteCodeInput.trim()) return;
    setClaiming(true);
    setClaimSuccess(false);

    await claimInviteCode(inviteCodeInput.trim());

    if (!authError) {
      setClaimSuccess(true);
      setInviteCodeInput("");
      // Refresh points
      if (userId) {
        const { data } = await supabase
          .from("points_balance")
          .select("balance")
          .eq("user_id", userId)
          .single();
        if (data) setPoints(data);
      }
    }

    setClaiming(false);
  };

  const handleCopyInviteCode = () => {
    if (!profile?.invite_code) return;
    navigator.clipboard.writeText(profile.invite_code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleSaveSettings = () => {
    if (!userId) return;

    const success = saveApiSettings(userId, { apiUrl, apiKey });
    if (success) {
      setSavingSettings(true);
      setSettingsSaved(true);
      setTimeout(() => {
        setSavingSettings(false);
        setSettingsSaved(false);
      }, 2000);
    }
  };

  if (!user) {
    return (
      <div className="w-full h-full flex items-center justify-center">
        <p className="text-gray-400">请先登录</p>
      </div>
    );
  }

  return (
    <div className="w-full h-full overflow-auto px-6 py-8">
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Header */}
        <div className="glass-dark rounded-xl border border-white/10 p-6">
          <h1 className="text-2xl font-bold text-white mb-2">账户设置</h1>
          <p className="text-gray-400 text-sm">
            管理您的邀请码、积分余额和 API 配置
          </p>
        </div>

        {/* Invite Code Card */}
        <div className="glass-dark rounded-xl border border-white/10 p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 rounded-lg bg-purple-500/20">
              <Ticket size={20} className="text-purple-400" />
            </div>
            <h2 className="text-lg font-semibold text-white">我的邀请码</h2>
          </div>

          {loadingProfile ? (
            <div className="flex items-center gap-2 text-gray-400">
              <Loader2 size={16} className="animate-spin" />
              <span className="text-sm">加载中...</span>
            </div>
          ) : profile?.invite_code ? (
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <code className="flex-1 px-4 py-3 bg-black/30 border border-white/10 rounded-lg text-white font-mono text-lg">
                  {profile.invite_code}
                </code>
                <button
                  onClick={handleCopyInviteCode}
                  className="px-4 py-3 rounded-lg bg-purple-600/80 hover:bg-purple-600 text-white flex items-center gap-2 transition-colors"
                >
                  {copied ? <CheckCircle2 size={16} /> : <Copy size={16} />}
                  {copied ? "已复制" : "复制"}
                </button>
              </div>
              <p className="text-xs text-gray-400">
                分享此邀请码给好友，好友注册后你们都将获得积分奖励
              </p>
            </div>
          ) : (
            <p className="text-gray-400 text-sm">暂无邀请码</p>
          )}
        </div>

        {/* Points Balance Card */}
        <div className="glass-dark rounded-xl border border-white/10 p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 rounded-lg bg-yellow-500/20">
              <Coins size={20} className="text-yellow-400" />
            </div>
            <h2 className="text-lg font-semibold text-white">积分余额</h2>
          </div>

          {loadingPoints ? (
            <div className="flex items-center gap-2 text-gray-400">
              <Loader2 size={16} className="animate-spin" />
              <span className="text-sm">加载中...</span>
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <span className="text-4xl font-bold text-white">
                {points?.balance ?? 0}
              </span>
              <span className="text-gray-400">积分</span>
            </div>
          )}
        </div>

        {/* Claim Invite Code Card */}
        <div className="glass-dark rounded-xl border border-white/10 p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 rounded-lg bg-green-500/20">
              <Ticket size={20} className="text-green-400" />
            </div>
            <h2 className="text-lg font-semibold text-white">填写邀请码</h2>
          </div>

          <div className="space-y-3">
            <input
              type="text"
              value={inviteCodeInput}
              onChange={(e) => setInviteCodeInput(e.target.value.toUpperCase())}
              placeholder="输入邀请码"
              className="w-full px-4 py-3 bg-black/30 border border-white/10 rounded-lg text-white placeholder-gray-600 focus:outline-none focus:ring-2 focus:ring-green-500/50"
              disabled={claiming}
            />

            <button
              onClick={handleClaimInvite}
              disabled={claiming || !inviteCodeInput.trim()}
              className="w-full py-3 rounded-lg bg-green-600/80 hover:bg-green-600 text-white font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
            >
              {claiming ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  <span>提交中...</span>
                </>
              ) : (
                <span>领取奖励</span>
              )}
            </button>

            {claimSuccess && (
              <div className="flex items-start gap-2 text-sm text-green-300 bg-green-500/10 border border-green-500/30 rounded-lg px-3 py-2">
                <CheckCircle2 size={16} className="mt-0.5 shrink-0" />
                <span>邀请码已成功领取，积分已发放！</span>
              </div>
            )}

            {authError && (
              <div className="flex items-start gap-2 text-sm text-red-300 bg-red-500/10 border border-red-500/30 rounded-lg px-3 py-2">
                <AlertCircle size={16} className="mt-0.5 shrink-0" />
                <span>{authError}</span>
              </div>
            )}
          </div>
        </div>

        {/* API Settings Card */}
        <div className="glass-dark rounded-xl border border-white/10 p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 rounded-lg bg-blue-500/20">
              <Settings size={20} className="text-blue-400" />
            </div>
            <h2 className="text-lg font-semibold text-white">API 配置</h2>
          </div>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">
                API Base URL
              </label>
              <input
                type="text"
                value={apiUrl}
                onChange={(e) => setApiUrl(e.target.value)}
                placeholder="https://api.apiyi.com/v1"
                className="w-full px-4 py-3 bg-black/30 border border-white/10 rounded-lg text-white placeholder-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">
                API Key
              </label>
              <input
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="sk-..."
                className="w-full px-4 py-3 bg-black/30 border border-white/10 rounded-lg text-white placeholder-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
              />
            </div>

            <button
              onClick={handleSaveSettings}
              disabled={savingSettings}
              className="w-full py-3 rounded-lg bg-blue-600/80 hover:bg-blue-600 text-white font-medium disabled:opacity-50 transition-colors flex items-center justify-center gap-2"
            >
              {savingSettings ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  <span>保存中...</span>
                </>
              ) : settingsSaved ? (
                <>
                  <CheckCircle2 size={16} />
                  <span>已保存</span>
                </>
              ) : (
                <>
                  <Key size={16} />
                  <span>保存配置</span>
                </>
              )}
            </button>

            <div className="flex items-start gap-2 text-xs text-gray-400 bg-yellow-500/10 border border-yellow-500/20 rounded-lg px-3 py-2">
              <AlertCircle size={14} className="mt-0.5 shrink-0" />
              <p>
                API 配置仅保存在当前设备的浏览器本地存储中（明文），不会上传到服务器。
                请妥善保管您的 API Key，避免在公共设备上保存。
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
