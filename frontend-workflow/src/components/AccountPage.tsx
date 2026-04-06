import { useEffect, useMemo, useState } from "react";
import {
  CheckCircle2,
  Coins,
  Copy,
  ExternalLink,
  History,
  Key,
  Loader2,
  Settings,
  Ticket,
  Users,
} from "lucide-react";

import { API_URL_OPTIONS, DEFAULT_LLM_API_URL, getPurchaseUrl } from "../config/api";
import { backendFetch } from "../services/backendClient";
import { getApiSettings, saveApiSettings } from "../services/apiSettingsService";
import { fetchRuntimeConfig, getRuntimeConfigSync, RuntimeConfig } from "../services/runtimeConfigService";
import { useAuthStore } from "../stores/authStore";

interface ProfileData {
  user_id?: string;
  invite_code?: string;
  created_at?: string;
}

interface PointsData {
  balance: number;
}

interface ReferralRecord {
  id: string;
  invitee_user_id: string;
  invite_code: string;
  created_at: string;
}

interface LedgerRecord {
  id: string;
  points: number;
  reason: string;
  created_at: string;
}

interface AccountProfileResponse {
  billing_mode: "paid" | "free";
  profile: ProfileData;
  points: PointsData;
  referrals: ReferralRecord[];
  points_ledger: LedgerRecord[];
}

function formatLedgerReason(reason: string): string {
  if (reason === "signup_bonus") {
    return "注册赠送";
  }
  if (reason === "daily_grant") {
    return "每日补点";
  }
  if (reason === "referral_inviter") {
    return "邀请奖励";
  }
  if (reason === "referral_invitee") {
    return "被邀请奖励";
  }
  if (reason.startsWith("redeem_code_")) {
    const points = reason.split("_").pop() || "";
    return `兑换码充值 (${points} 点)`;
  }
  if (reason.startsWith("workflow_")) {
    return `工作流消耗: ${reason.replace(/^workflow_/, "")}`;
  }
  return reason;
}

export function AccountPage() {
  const { user, claimInviteCode, error: authError, refreshQuota } = useAuthStore();
  const [runtimeConfig, setRuntimeConfig] = useState<RuntimeConfig>(getRuntimeConfigSync());
  const [profileData, setProfileData] = useState<AccountProfileResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [inviteCodeInput, setInviteCodeInput] = useState("");
  const [claiming, setClaiming] = useState(false);
  const [claimSuccess, setClaimSuccess] = useState(false);
  const [redeemCodeInput, setRedeemCodeInput] = useState("");
  const [redeeming, setRedeeming] = useState(false);
  const [redeemSuccessMessage, setRedeemSuccessMessage] = useState("");
  const [redeemErrorMessage, setRedeemErrorMessage] = useState("");
  const [copied, setCopied] = useState(false);

  const [apiUrl, setApiUrl] = useState(DEFAULT_LLM_API_URL);
  const [apiKey, setApiKey] = useState("");
  const [savingSettings, setSavingSettings] = useState(false);
  const [settingsSaved, setSettingsSaved] = useState(false);

  const showApiSettings = runtimeConfig.user_api_config_required;
  const pointsBalance = profileData?.points?.balance ?? 0;
  const displayModeText = runtimeConfig.billing_mode === "free" ? "免费模式" : "付费模式";
  const purchaseUrl = runtimeConfig.points_purchase_url?.trim()
    || getPurchaseUrl(runtimeConfig.managed_api_url || DEFAULT_LLM_API_URL);

  const inviteRewardText = useMemo(() => {
    if (runtimeConfig.referral_invitee_points > 0) {
      return `邀请人 +${runtimeConfig.referral_inviter_points}，被邀请人 +${runtimeConfig.referral_invitee_points}`;
    }
    return `邀请人 +${runtimeConfig.referral_inviter_points}`;
  }, [runtimeConfig.referral_inviter_points, runtimeConfig.referral_invitee_points]);

  const fetchAccountProfileData = async (): Promise<AccountProfileResponse | null> => {
    if (!user) {
      return null;
    }

    const response = await backendFetch("/api/v1/account/profile");
    if (!response.ok) {
      const errorPayload = await response.json().catch(() => null);
      throw new Error(errorPayload?.detail || `账户信息获取失败 (${response.status})`);
    }
    return (await response.json()) as AccountProfileResponse;
  };

  useEffect(() => {
    fetchRuntimeConfig()
      .then(setRuntimeConfig)
      .catch(() => setRuntimeConfig(getRuntimeConfigSync()));
  }, []);

  useEffect(() => {
    const settings = getApiSettings(user?.id || null);
    if (settings) {
      setApiUrl(settings.apiUrl || DEFAULT_LLM_API_URL);
      setApiKey(settings.apiKey || "");
    }
  }, [user?.id, runtimeConfig.user_api_config_required]);

  useEffect(() => {
    let cancelled = false;

    const loadAccountProfile = async () => {
      if (!user) {
        setLoading(false);
        return;
      }

      try {
        setLoading(true);
        const data = await fetchAccountProfileData();
        if (!cancelled) {
          setProfileData(data);
        }
      } catch (err) {
        console.error("[AccountPage] Failed to load account profile:", err);
        if (!cancelled) {
          setProfileData(null);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    loadAccountProfile();

    return () => {
      cancelled = true;
    };
  }, [user?.id]);

  const handleClaimInvite = async () => {
    if (!inviteCodeInput.trim()) {
      return;
    }

    setClaiming(true);
    setClaimSuccess(false);
    try {
      await claimInviteCode(inviteCodeInput.trim());
      setClaimSuccess(true);
      setInviteCodeInput("");
      await refreshQuota();

      const data = await fetchAccountProfileData().catch(() => null);
      if (data) {
        setProfileData(data);
      }
    } catch (err) {
      console.error("[AccountPage] Failed to claim invite code:", err);
    } finally {
      setClaiming(false);
    }
  };

  const handleRedeemPoints = async () => {
    const normalizedCode = redeemCodeInput.trim();
    if (!normalizedCode) {
      return;
    }

    setRedeeming(true);
    setRedeemSuccessMessage("");
    setRedeemErrorMessage("");

    try {
      const response = await backendFetch("/api/v1/account/points/redeem", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ redeem_code: normalizedCode }),
      });
      const payload = await response.json().catch(() => null);
      if (!response.ok) {
        throw new Error(payload?.detail || `兑换失败 (${response.status})`);
      }

      setRedeemCodeInput("");
      setRedeemSuccessMessage(`兑换成功，已到账 ${payload?.points_added ?? 0} 点`);
      await refreshQuota();

      const data = await fetchAccountProfileData().catch(() => null);
      if (data) {
        setProfileData(data);
      }
    } catch (err) {
      console.error("[AccountPage] Failed to redeem points code:", err);
      setRedeemErrorMessage(err instanceof Error ? err.message : "兑换失败，请稍后重试");
    } finally {
      setRedeeming(false);
    }
  };

  const handleCopyInviteCode = () => {
    const inviteCode = profileData?.profile?.invite_code;
    if (!inviteCode) {
      return;
    }
    navigator.clipboard.writeText(inviteCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleSaveSettings = () => {
    if (!user?.id || !showApiSettings) {
      return;
    }

    setSavingSettings(true);
    const ok = saveApiSettings(user.id, { apiUrl, apiKey });
    setSettingsSaved(ok);
    setTimeout(() => {
      setSavingSettings(false);
      setSettingsSaved(false);
    }, 1200);
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
      <div className="max-w-6xl mx-auto space-y-6">
        <div className="flex flex-col gap-2">
          <h1 className="text-3xl font-bold text-white">账户设置</h1>
          <p className="text-sm text-gray-400">
            当前为 <span className="text-white">{displayModeText}</span>
            {runtimeConfig.billing_mode === "free"
              ? "，业务模型与扣点策略均由后端统一托管。"
              : "，用户自行填写 API 配置，平台默认不扣点。"}
          </p>
        </div>

        {loading ? (
          <div className="rounded-2xl border border-white/10 bg-white/5 p-6 flex items-center gap-3 text-gray-300">
            <Loader2 size={18} className="animate-spin" />
            <span>正在加载账户信息...</span>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="rounded-2xl border border-white/10 bg-white/5 p-6 space-y-4">
                <div className="flex items-center gap-2 text-white">
                  <Settings size={18} className="text-cyan-400" />
                  <span className="font-medium">模式状态</span>
                </div>
                <div className="text-2xl font-semibold text-white">{displayModeText}</div>
                <p className="text-sm text-gray-400">
                  免费模式下右上角点数来自后端配置；付费模式下使用用户自带 API，不额外消耗平台点数。
                </p>
              </div>

              <div className="rounded-2xl border border-white/10 bg-white/5 p-6 space-y-4">
                <div className="flex items-center gap-2 text-white">
                  <Coins size={18} className="text-yellow-400" />
                  <span className="font-medium">点数 / 配额</span>
                </div>
                <div className="text-2xl font-semibold text-white">
                  {runtimeConfig.billing_mode === "free" ? `${pointsBalance}` : "∞"}
                </div>
                <p className="text-sm text-gray-400">
                  {runtimeConfig.billing_mode === "free"
                    ? `每日最多补 ${runtimeConfig.daily_grant_points} 点，余额上限 ${runtimeConfig.daily_grant_balance_cap} 点。`
                    : "当前模式不扣平台点数，主要依赖用户自备 API。"}
                </p>
              </div>

              <div className="rounded-2xl border border-white/10 bg-white/5 p-6 space-y-4">
                <div className="flex items-center gap-2 text-white">
                  <Ticket size={18} className="text-purple-400" />
                  <span className="font-medium">我的邀请码</span>
                </div>
                <code className="block w-full rounded-lg border border-white/10 bg-black/30 px-4 py-3 text-center text-lg text-white">
                  {profileData?.profile?.invite_code || "暂无"}
                </code>
                <button
                  onClick={handleCopyInviteCode}
                  disabled={!profileData?.profile?.invite_code}
                  className="w-full py-2.5 rounded-lg bg-purple-600/80 hover:bg-purple-600 disabled:opacity-50 text-white text-sm font-medium flex items-center justify-center gap-2 transition-all"
                >
                  {copied ? <CheckCircle2 size={16} /> : <Copy size={16} />}
                  {copied ? "已复制" : "复制邀请码"}
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {(runtimeConfig.billing_mode === "free" || runtimeConfig.points_redeem_enabled) && (
                <div className="rounded-2xl border border-white/10 bg-white/5 p-6 space-y-4">
                  <div className="flex items-center gap-2 text-white">
                    <Coins size={18} className="text-amber-300" />
                    <span className="font-medium">点数兑换</span>
                  </div>
                  <p className="text-sm text-gray-400">
                    点数不足时，可先前往购买页获取兑换码，再回到这里兑换加点。
                  </p>
                  <div className="space-y-3">
                    <input
                      type="text"
                      value={redeemCodeInput}
                      onChange={(e) => setRedeemCodeInput(e.target.value)}
                      placeholder="输入点数兑换码"
                      className="w-full px-4 py-3 bg-black/30 border border-white/10 rounded-lg text-white placeholder-gray-600 focus:outline-none focus:ring-2 focus:ring-amber-500/40"
                    />
                    <div className="flex flex-col sm:flex-row gap-3">
                      <button
                        onClick={handleRedeemPoints}
                        disabled={redeeming || !redeemCodeInput.trim()}
                        className="flex-1 px-5 py-3 rounded-lg bg-amber-600/80 hover:bg-amber-600 disabled:opacity-50 text-white text-sm font-medium flex items-center justify-center gap-2"
                      >
                        {redeeming ? <Loader2 size={16} className="animate-spin" /> : <Ticket size={16} />}
                        立即兑换
                      </button>
                      {runtimeConfig.billing_mode === "free" && purchaseUrl && (
                        <a
                          href={purchaseUrl}
                          target="_blank"
                          rel="noreferrer"
                          className="flex-1 px-5 py-3 rounded-lg border border-amber-400/20 bg-amber-500/10 hover:bg-amber-500/15 text-amber-100 text-sm font-medium flex items-center justify-center gap-2 transition-colors"
                        >
                          <ExternalLink size={16} />
                          前往发卡平台
                        </a>
                      )}
                    </div>
                  </div>
                  {(redeemSuccessMessage || redeemErrorMessage) && (
                    <div className={`rounded-lg px-4 py-3 text-sm ${redeemSuccessMessage ? "bg-green-500/10 border border-green-500/20 text-green-300" : "bg-red-500/10 border border-red-500/20 text-red-300"}`}>
                      {redeemSuccessMessage || redeemErrorMessage}
                    </div>
                  )}
                </div>
              )}

              <div className="rounded-2xl border border-white/10 bg-white/5 p-6 space-y-4">
                <div className="flex items-center gap-2 text-white">
                  <Users size={18} className="text-green-400" />
                  <span className="font-medium">填写邀请码</span>
                </div>
                <p className="text-sm text-gray-400">
                  当前邀请策略：{inviteRewardText}
                </p>
                <div className="flex gap-3">
                  <input
                    type="text"
                    value={inviteCodeInput}
                    onChange={(e) => setInviteCodeInput(e.target.value)}
                    placeholder="输入邀请码"
                    className="flex-1 px-4 py-3 bg-black/30 border border-white/10 rounded-lg text-white placeholder-gray-600 focus:outline-none focus:ring-2 focus:ring-green-500/50"
                  />
                  <button
                    onClick={handleClaimInvite}
                    disabled={claiming || !inviteCodeInput.trim()}
                    className="px-5 py-3 rounded-lg bg-green-600/80 hover:bg-green-600 disabled:opacity-50 text-white text-sm font-medium flex items-center gap-2"
                  >
                    {claiming ? <Loader2 size={16} className="animate-spin" /> : <Ticket size={16} />}
                    兑换
                  </button>
                </div>
                {(claimSuccess || authError) && (
                  <div className={`rounded-lg px-4 py-3 text-sm ${claimSuccess ? "bg-green-500/10 border border-green-500/20 text-green-300" : "bg-red-500/10 border border-red-500/20 text-red-300"}`}>
                    {claimSuccess ? "邀请码兑换成功" : authError}
                  </div>
                )}
              </div>

              <div className="rounded-2xl border border-white/10 bg-white/5 p-6 space-y-4">
                <div className="flex items-center gap-2 text-white">
                  <Key size={18} className="text-blue-400" />
                  <span className="font-medium">API 配置</span>
                </div>
                {showApiSettings ? (
                  <>
                    <p className="text-sm text-gray-400">
                      付费模式下，业务调用优先使用当前浏览器保存的用户 API 配置。
                    </p>
                    <div className="space-y-4">
                      <div>
                        <label className="block text-sm text-gray-400 mb-2">API URL</label>
                        <select
                          value={apiUrl}
                          onChange={(e) => setApiUrl(e.target.value)}
                          className="w-full px-4 py-3 bg-black/30 border border-white/10 rounded-lg text-white focus:outline-none"
                        >
                          {[apiUrl, ...API_URL_OPTIONS]
                            .filter((value, index, array) => array.indexOf(value) === index)
                            .map((url) => (
                              <option key={url} value={url}>
                                {url}
                              </option>
                            ))}
                        </select>
                      </div>
                      <div>
                        <label className="block text-sm text-gray-400 mb-2">API Key</label>
                        <input
                          type="password"
                          value={apiKey}
                          onChange={(e) => setApiKey(e.target.value)}
                          placeholder="sk-..."
                          className="w-full px-4 py-3 bg-black/30 border border-white/10 rounded-lg text-white placeholder-gray-600 focus:outline-none"
                        />
                      </div>
                      <button
                        onClick={handleSaveSettings}
                        disabled={savingSettings}
                        className="w-full py-3 rounded-lg bg-blue-600/80 hover:bg-blue-600 text-white font-medium flex items-center justify-center gap-2 disabled:opacity-50"
                      >
                        {savingSettings ? <Loader2 size={18} className="animate-spin" /> : <Key size={18} />}
                        {settingsSaved ? "已保存" : "保存配置"}
                      </button>
                    </div>
                  </>
                ) : (
                  <div className="rounded-xl border border-cyan-500/20 bg-cyan-500/10 px-4 py-4 text-sm text-cyan-100">
                    后端托管模型已开启。功能调用会消耗点数，无需手动填写 API URL 或 API Key；若点数不足，可前往购买页获取兑换码，再到账户页兑换加点。
                  </div>
                )}
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="rounded-2xl border border-white/10 bg-white/5 p-6">
                <div className="flex items-center gap-2 text-white mb-4">
                  <Users size={18} className="text-emerald-400" />
                  <span className="font-medium">邀请记录</span>
                </div>
                {profileData?.referrals?.length ? (
                  <div className="space-y-3">
                    {profileData.referrals.map((ref) => (
                      <div key={ref.id} className="rounded-lg border border-white/10 bg-black/20 px-4 py-3">
                        <div className="text-sm text-white">{ref.invitee_user_id}</div>
                        <div className="text-xs text-gray-400 mt-1">{new Date(ref.created_at).toLocaleString()}</div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-gray-400">暂无邀请记录</p>
                )}
              </div>

              <div className="rounded-2xl border border-white/10 bg-white/5 p-6">
                <div className="flex items-center gap-2 text-white mb-4">
                  <History size={18} className="text-orange-400" />
                  <span className="font-medium">点数流水</span>
                </div>
                {profileData?.points_ledger?.length ? (
                  <div className="space-y-3 max-h-[420px] overflow-auto pr-1">
                    {profileData.points_ledger.map((record) => (
                      <div key={record.id} className="rounded-lg border border-white/10 bg-black/20 px-4 py-3">
                        <div className="flex items-center justify-between gap-3">
                          <span className="text-sm text-white">{formatLedgerReason(record.reason)}</span>
                          <span className={`text-sm font-medium ${record.points >= 0 ? "text-green-300" : "text-red-300"}`}>
                            {record.points >= 0 ? `+${record.points}` : record.points}
                          </span>
                        </div>
                        <div className="text-xs text-gray-400 mt-1">{new Date(record.created_at).toLocaleString()}</div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-gray-400">暂无点数流水</p>
                )}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
