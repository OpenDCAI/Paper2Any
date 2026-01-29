import { useEffect, useState } from 'react';
import { Settings, Key, CheckCircle2, Loader2, AlertCircle } from 'lucide-react';
import { API_URL_OPTIONS, DEFAULT_LLM_API_URL } from '../../config/api';
import { getApiSettings, saveApiSettings } from '../../services/apiSettingsService';
import { useAuthStore } from '../../stores/authStore';

export const SettingsView = () => {
  const { user } = useAuthStore();
  const [apiUrl, setApiUrl] = useState(DEFAULT_LLM_API_URL);
  const [apiKey, setApiKey] = useState('');
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    const settings = getApiSettings(user?.id || null);
    if (settings) {
      setApiUrl(settings.apiUrl || DEFAULT_LLM_API_URL);
      setApiKey(settings.apiKey || '');
    }
  }, [user?.id]);

  const handleSave = () => {
    if (!user?.id) return;
    setSaving(true);
    const ok = saveApiSettings(user.id, { apiUrl, apiKey });
    setSaved(ok);
    setTimeout(() => {
      setSaving(false);
      setSaved(false);
    }, 1200);
  };

  if (!user) {
    return (
      <div className="h-full flex items-center justify-center text-sm text-gray-400">
        请先登录后再配置 API 设置。
      </div>
    );
  }

  return (
    <div className="max-w-3xl">
      <div className="mb-6 bg-gradient-to-br from-blue-900/20 to-cyan-900/20 border border-blue-500/20 rounded-xl p-4 flex items-start gap-3">
        <Settings className="text-blue-400 mt-1 flex-shrink-0" size={18} />
        <div>
          <h4 className="text-sm font-medium text-blue-300 mb-1">API 设置</h4>
          <p className="text-xs text-blue-200/70">
            统一配置 API URL 与 API Key，知识库与其他应用将优先使用该配置，避免重复填写。
          </p>
        </div>
      </div>

      <div className="bg-white/5 border border-white/10 rounded-xl p-6 space-y-5">
        <div>
          <label className="block text-sm font-medium text-gray-400 mb-2">API URL</label>
          <select
            value={apiUrl}
            onChange={(e) => setApiUrl(e.target.value)}
            className="w-full px-4 py-3 bg-black/30 border border-white/10 rounded-lg text-white placeholder-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500/50 transition-all"
          >
            {[apiUrl, ...API_URL_OPTIONS].filter((v, i, a) => a.indexOf(v) === i).map((url: string) => (
              <option key={url} value={url}>{url}</option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-400 mb-2">API Key</label>
          <input
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="sk-..."
            className="w-full px-4 py-3 bg-black/30 border border-white/10 rounded-lg text-white placeholder-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500/50 transition-all"
          />
        </div>

        <button
          onClick={handleSave}
          disabled={saving}
          className="w-full py-3 rounded-lg bg-blue-600/80 hover:bg-blue-600 text-white font-medium disabled:opacity-50 transition-all transform hover:scale-[1.01] flex items-center justify-center gap-2"
        >
          {saving ? (
            <>
              <Loader2 size={18} className="animate-spin" />
              <span>保存中...</span>
            </>
          ) : saved ? (
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
            请避免在公共设备上保存敏感信息。
          </p>
        </div>
      </div>
    </div>
  );
};
