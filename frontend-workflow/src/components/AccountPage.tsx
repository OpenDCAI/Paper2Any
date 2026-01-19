/**
 * Account page showing user invite code, points balance, and API settings.
 */

import { useState, useEffect } from "react";
import { useAuthStore } from "../stores/authStore";
import { supabase } from "../lib/supabase";
import { getApiSettings, saveApiSettings } from "../services/apiSettingsService";
import { Ticket, Coins, Key, AlertCircle, Loader2, Copy, CheckCircle2, Settings, Users, History } from "lucide-react";

interface ProfileData {
  invite_code: string;
}

interface PointsBalance {
  balance: number;
}

interface ReferralRecord {
  id: string;
  referred_user_id: string;
  created_at: string;
}

interface PointsLedgerRecord {
  id: string;
  amount: number;
  description: string;
  created_at: string;
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
  const [referrals, setReferrals] = useState<ReferralRecord[]>([]);
  const [loadingReferrals, setLoadingReferrals] = useState(true);
  const [pointsLedger, setPointsLedger] = useState<PointsLedgerRecord[]>([]);
  const [loadingLedger, setLoadingLedger] = useState(true);

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

  // Load referral history
  useEffect(() => {
    if (!userId) {
      setLoadingReferrals(false);
      return;
    }

    const fetchReferrals = async () => {
      try {
        const { data, error } = await supabase
          .from("referrals")
          .select("id, referred_user_id, created_at")
          .eq("referrer_user_id", userId)
          .order("created_at", { ascending: false })
          .limit(10);

        if (error) throw error;
        setReferrals(data || []);
      } catch (err) {
        console.error("[AccountPage] Failed to load referrals:", err);
      } finally {
        setLoadingReferrals(false);
      }
    };

    fetchReferrals();
  }, [userId]);

  // Load points ledger
  useEffect(() => {
    if (!userId) {
      setLoadingLedger(false);
      return;
    }

    const fetchLedger = async () => {
      try {
        const { data, error } = await supabase
          .from("points_ledger")
          .select("id, amount, description, created_at")
          .eq("user_id", userId)
          .order("created_at", { ascending: false })
          .limit(20);

        if (error) throw error;
        setPointsLedger(data || []);
      } catch (err) {
        console.error("[AccountPage] Failed to load points ledger:", err);
      } finally {
        setLoadingLedger(false);
      }
    };

    fetchLedger();
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
    <div className="w-full h-full overflow-auto px-6 py-8 bg-gradient-to-br from-[#050512] via-[#0a0a1a] to-[#050512]">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-2 bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
            👤 我的账户
          </h1>
          <p className="text-gray-400">
            管理您的个人信息、积分和 API 配置
          </p>
        </div>

        {/* Two Column Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - User Info Cards */}
          <div className="lg:col-span-1 space-y-6">
            {/* User Info Card */}
            <div className="glass-dark rounded-xl border border-white/10 p-6 hover:border-purple-500/30 transition-all">
              <div className="flex flex-col items-center text-center space-y-4">
                <div className="w-20 h-20 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center text-white text-2xl font-bold">
                  {user.email?.charAt(0).toUpperCase() || 'U'}
                </div>
                <div>
                  <p className="text-white font-medium truncate max-w-full">{user.email}</p>
                  <p className="text-xs text-gray-400 mt-1">PRO MEMBER</p>
                </div>
              </div>
            </div>

            {/* Invite Code Card */}
            <div className="glass-dark rounded-xl border border-white/10 p-6 hover:border-purple-500/30 transition-all">
              <div className="flex items-center gap-2 mb-4">
                <Ticket size={18} className="text-purple-400" />
                <h2 className="text-base font-semibold text-white">我的邀请码</h2>
              </div>

              {loadingProfile ? (
                <div className="flex items-center gap-2 text-gray-400">
                  <Loader2 size={16} className="animate-spin" />
                  <span className="text-sm">加载中...</span>
                </div>
              ) : profile?.invite_code ? (
                <div className="space-y-3">
                  <code className="block w-full px-4 py-3 bg-black/30 border border-white/10 rounded-lg text-white font-mono text-center text-lg tracking-wider">
                    {profile.invite_code}
                  </code>
                  <button
                    onClick={handleCopyInviteCode}
                    className="w-full py-2.5 rounded-lg bg-purple-600/80 hover:bg-purple-600 text-white text-sm font-medium flex items-center justify-center gap-2 transition-all transform hover:scale-[1.02]"
                  >
                    {copied ? <CheckCircle2 size={16} /> : <Copy size={16} />}
                    {copied ? "已复制" : "复制邀请码"}
                  </button>
                  <p className="text-xs text-gray-400 text-center">
                    分享给好友获得奖励
                  </p>
                </div>
              ) : (
                <p className="text-gray-400 text-sm text-center">暂无邀请码</p>
              )}
            </div>

            {/* Claim Invite Code Card */}
            <div className="glass-dark rounded-xl border border-white/10 p-6 hover:border-green-500/30 transition-all">
              <div className="flex items-center gap-2 mb-4">
                <Ticket size={18} className="text-green-400" />
                <h2 className="text-base font-semibold text-white">填写邀请码</h2>
              </div>

              <div className="space-y-3">
                <input
                  type="text"
                  value={inviteCodeInput}
                  onChange={(e) => setInviteCodeInput(e.target.value.toUpperCase())}
                  placeholder="输入邀请码"
                  className="w-full px-4 py-2.5 bg-black/30 border border-white/10 rounded-lg text-white placeholder-gray-600 focus:outline-none focus:ring-2 focus:ring-green-500/50 text-center font-mono tracking-wider"
                  disabled={claiming}
                />

                <button
                  onClick={handleClaimInvite}
                  disabled={claiming || !inviteCodeInput.trim()}
                  className="w-full py-2.5 rounded-lg bg-green-600/80 hover:bg-green-600 text-white text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:scale-[1.02] flex items-center justify-center gap-2"
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
                  <div className="flex items-start gap-2 text-xs text-green-300 bg-green-500/10 border border-green-500/30 rounded-lg px-3 py-2">
                    <CheckCircle2 size={14} className="mt-0.5 shrink-0" />
                    <span>邀请码已成功领取！</span>
                  </div>
                )}

                {authError && (
                  <div className="flex items-start gap-2 text-xs text-red-300 bg-red-500/10 border border-red-500/30 rounded-lg px-3 py-2">
                    <AlertCircle size={14} className="mt-0.5 shrink-0" />
                    <span>{authError}</span>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Right Column - Functional Areas */}
          <div className="lg:col-span-2 space-y-6">
            {/* Points Balance Card */}
            <div className="glass-dark rounded-xl border border-white/10 p-8 hover:border-yellow-500/30 transition-all">
              <div className="flex items-center gap-3 mb-6">
                <div className="p-2 rounded-lg bg-yellow-500/20">
                  <Coins size={24} className="text-yellow-400" />
                </div>
                <h2 className="text-xl font-semibold text-white">积分余额</h2>
              </div>

              {loadingPoints ? (
                <div className="flex items-center gap-2 text-gray-400">
                  <Loader2 size={20} className="animate-spin" />
                  <span>加载中...</span>
                </div>
              ) : (
                <div className="flex items-baseline gap-3">
                  <span className="text-6xl font-bold bg-gradient-to-r from-yellow-400 via-orange-400 to-yellow-500 bg-clip-text text-transparent">
                    {points?.balance ?? 0}
                  </span>
                  <span className="text-2xl text-gray-400">积分</span>
                </div>
              )}
            </div>

            {/* API Settings Card */}
            <div className="glass-dark rounded-xl border border-white/10 p-8 hover:border-blue-500/30 transition-all">
              <div className="flex items-center gap-3 mb-6">
                <div className="p-2 rounded-lg bg-blue-500/20">
                  <Settings size={24} className="text-blue-400" />
                </div>
                <h2 className="text-xl font-semibold text-white">API 配置</h2>
              </div>

              <div className="space-y-5">
                <div>
                  <label className="block text-sm font-medium text-gray-400 mb-2">
                    API Base URL
                  </label>
                  <input
                    type="text"
                    value={apiUrl}
                    onChange={(e) => setApiUrl(e.target.value)}
                    placeholder="https://api.apiyi.com/v1"
                    className="w-full px-4 py-3 bg-black/30 border border-white/10 rounded-lg text-white placeholder-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500/50 transition-all"
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
                    className="w-full px-4 py-3 bg-black/30 border border-white/10 rounded-lg text-white placeholder-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500/50 transition-all"
                  />
                </div>

                <button
                  onClick={handleSaveSettings}
                  disabled={savingSettings}
                  className="w-full py-3 rounded-lg bg-blue-600/80 hover:bg-blue-600 text-white font-medium disabled:opacity-50 transition-all transform hover:scale-[1.01] flex items-center justify-center gap-2"
                >
                  {savingSettings ? (
                    <>
                      <Loader2 size={18} className="animate-spin" />
                      <span>保存中...</span>
                    </>
                  ) : settingsSaved ? (
                    <>
                      <CheckCircle2 size={18} />
                      <span>已保存</span>
                    </>
                  ) : (
                    <>
                      <Key size={18} />
                      <span>保存配置</span>
                    </>
                  )}
                </button>

                <div className="flex items-start gap-2 text-xs text-gray-400 bg-yellow-500/10 border border-yellow-500/20 rounded-lg px-4 py-3">
                  <AlertCircle size={16} className="mt-0.5 shrink-0" />
                  <p>
                    API 配置仅保存在当前设备的浏览器本地存储中（明文），不会上传到服务器。
                    请妥善保管您的 API Key，避免在公共设备上保存。
                  </p>
                </div>
              </div>
            </div>

            {/* Invite History Card */}
            <div className="glass-dark rounded-xl border border-white/10 p-6 hover:border-purple-500/30 transition-all">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 rounded-lg bg-purple-500/20">
                  <Users size={20} className="text-purple-400" />
                </div>
                <h2 className="text-lg font-semibold text-white">邀请历史</h2>
              </div>

              {loadingReferrals ? (
                <div className="flex items-center gap-2 text-gray-400">
                  <Loader2 size={16} className="animate-spin" />
                  <span className="text-sm">加载中...</span>
                </div>
              ) : referrals.length > 0 ? (
                <div className="space-y-2">
                  {referrals.map((ref) => (
                    <div key={ref.id} className="flex items-center justify-between px-4 py-3 bg-black/20 border border-white/5 rounded-lg hover:bg-white/5 transition-all">
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center text-white text-xs font-bold">
                          {ref.referred_user_id.slice(0, 2).toUpperCase()}
                        </div>
                        <div>
                          <p className="text-sm text-white font-mono">{ref.referred_user_id.slice(0, 8)}...</p>
                          <p className="text-xs text-gray-400">{new Date(ref.created_at).toLocaleDateString('zh-CN')}</p>
                        </div>
                      </div>
                      <CheckCircle2 size={16} className="text-green-400" />
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-gray-400 text-sm text-center py-4">暂无邀请记录</p>
              )}
            </div>

            {/* Points Ledger Card */}
            <div className="glass-dark rounded-xl border border-white/10 p-6 hover:border-yellow-500/30 transition-all">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 rounded-lg bg-yellow-500/20">
                  <History size={20} className="text-yellow-400" />
                </div>
                <h2 className="text-lg font-semibold text-white">积分记录</h2>
              </div>

              {loadingLedger ? (
                <div className="flex items-center gap-2 text-gray-400">
                  <Loader2 size={16} className="animate-spin" />
                  <span className="text-sm">加载中...</span>
                </div>
              ) : pointsLedger.length > 0 ? (
                <div className="space-y-2 max-h-96 overflow-y-auto">
                  {pointsLedger.map((record) => (
                    <div key={record.id} className="flex items-center justify-between px-4 py-3 bg-black/20 border border-white/5 rounded-lg hover:bg-white/5 transition-all">
                      <div className="flex-1">
                        <p className="text-sm text-white">{record.description}</p>
                        <p className="text-xs text-gray-400">{new Date(record.created_at).toLocaleString('zh-CN')}</p>
                      </div>
                      <span className={`text-base font-bold ${
                        record.amount > 0 ? 'text-green-400' : 'text-red-400'
                      }`}>
                        {record.amount > 0 ? '+' : ''}{record.amount}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-gray-400 text-sm text-center py-4">暂无积分记录</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
