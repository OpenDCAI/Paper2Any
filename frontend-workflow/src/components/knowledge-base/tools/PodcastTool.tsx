import { useEffect, useState } from 'react';
import { Headphones, Loader2, CheckCircle2, X } from 'lucide-react';
import { API_URL_OPTIONS } from '../../../config/api';
import { KnowledgeFile } from '../types';
import { getApiSettings } from '../../../services/apiSettingsService';
import { backendFetch } from '../../../services/backendClient';
import { useAuthStore } from '../../../stores/authStore';
import { useRuntimeBilling } from '../../../hooks/useRuntimeBilling';

const COSYVOICE_TTS_MODELS = ['cosyvoice-v3-flash', 'cosyvoice-v3-plus', 'cosyvoice-v2'];
const COSYVOICE_VOICE_LABEL = '默认';

const OPENAI_TTS_VOICES = [
  'alloy',
  'echo',
  'fable',
  'onyx',
  'nova',
  'shimmer',
  'coral',
  'verse',
  'ballad',
  'ash',
  'sage',
  'marin',
  'cedar',
  'amuch',
  'aster',
  'brook',
  'clover',
  'dan',
  'elan',
  'marilyn',
  'meadow',
  'jazz',
  'rio',
  'breeze',
  'cove',
  'ember',
  'fathom',
  'glimmer',
  'harp',
  'juniper',
  'maple',
  'orbit',
  'vale',
  'megan-wetherall',
  'jade-hardy',
  'megan-wetherall-2025-03-07',
  'jade-hardy-2025-03-07'
];

const isOpenAITtsModel = (model: string) => {
  const lowered = model.toLowerCase();
  return lowered.includes('gpt-4o-mini-tts') || lowered.startsWith('tts-1');
};

const isCosyVoiceTtsModel = (model: string) => {
  return COSYVOICE_TTS_MODELS.includes(model);
};

const normalizeVoice = (voice: string, options: string[]) => {
  if (!voice) return options[0] ?? '';
  const exact = options.find(v => v === voice);
  if (exact) return exact;
  const lower = voice.toLowerCase();
  const matchInsensitive = options.find(v => v.toLowerCase() === lower);
  return matchInsensitive || options[0] || '';
};

interface PodcastToolProps {
  files: KnowledgeFile[];
  selectedIds: Set<string>;
  onGenerateSuccess: (file: KnowledgeFile) => void;
}

export const PodcastTool = ({ files = [], selectedIds, onGenerateSuccess }: PodcastToolProps) => {
  const { user } = useAuthStore();
  const { userApiConfigRequired } = useRuntimeBilling();
  const [podcastGenerating, setPodcastGenerating] = useState(false);
  const [podcastParams, setPodcastParams] = useState({
    api_key: '',
    api_url: 'https://api.apiyi.com/v1',
    model: 'gpt-5.1',
    tts_model: 'cosyvoice-v3-flash',
    voice_name: '',
    voice_name_b: '',
    podcast_mode: 'monologue',
    podcast_length: 'standard',
    language: 'zh'
  });
  const isOpenAITts = isOpenAITtsModel(podcastParams.tts_model);
  const isCosyVoiceTts = isCosyVoiceTtsModel(podcastParams.tts_model);
  const voiceOptions = isOpenAITts ? OPENAI_TTS_VOICES : (isCosyVoiceTts ? [''] : []);

  useEffect(() => {
    const settings = getApiSettings(user?.id || null);
    if (settings) {
      setPodcastParams(prev => ({
        ...prev,
        api_key: settings.apiKey || prev.api_key,
        api_url: settings.apiUrl || prev.api_url
      }));
    }
  }, [user?.id]);

  useEffect(() => {
    const nextOptions = isOpenAITts ? OPENAI_TTS_VOICES : (isCosyVoiceTts ? [''] : []);
    const nextVoice = normalizeVoice(podcastParams.voice_name, nextOptions);
    const nextVoiceB = normalizeVoice(podcastParams.voice_name_b, nextOptions);
    if (nextVoice !== podcastParams.voice_name || nextVoiceB !== podcastParams.voice_name_b) {
      setPodcastParams(prev => ({
        ...prev,
        voice_name: nextVoice,
        voice_name_b: nextVoiceB
      }));
    }
  }, [isOpenAITts, isCosyVoiceTts, podcastParams.voice_name, podcastParams.voice_name_b, podcastParams.tts_model]);

  const handleGeneratePodcast = async () => {
    if (!user?.id || !user?.email) {
      alert('请先登录后再生成播客。');
      return;
    }

    if (selectedIds.size === 0) {
      alert('请至少选择一个文件进行播客生成。');
      return;
    }

    if (userApiConfigRequired && !podcastParams.api_key) {
      alert('请输入 API Key');
      return;
    }

    // Get selected file paths
    const selectedFiles = (files || []).filter(f => selectedIds.has(f.id));
    const filePaths = selectedFiles.map(f => f.url).filter(url => url);

    if (filePaths.length === 0) {
      alert('无法获取文件路径，请重新上传文件。');
      return;
    }

    setPodcastGenerating(true);
    try {
      const res = await backendFetch('/api/v1/kb/generate-podcast', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          file_paths: filePaths,
          user_id: user.id,
          email: user.email,
          voice_name: podcastParams.voice_name,
          voice_name_b: podcastParams.voice_name_b,
          podcast_mode: podcastParams.podcast_mode,
          podcast_length: podcastParams.podcast_length,
          language: podcastParams.language,
          ...(userApiConfigRequired
            ? {
                api_url: podcastParams.api_url,
                api_key: podcastParams.api_key,
                model: podcastParams.model,
                tts_model: podcastParams.tts_model,
              }
            : {})
        })
      });

      if (!res.ok) {
        const errorText = await res.text();
        throw new Error('生成失败: ' + errorText);
      }

      const data = await res.json();

      if (data.success && data.audio_path) {

        onGenerateSuccess({
          id: data.output_file_id || 'o' + Date.now(),
          name: `podcast_${Date.now()}.wav`,
          type: 'audio',
          size: '未知',
          uploadTime: new Date().toLocaleString(),
          url: data.audio_path,
          desc: `Knowledge Podcast from ${selectedFiles.length} file(s)`
        });
      } else {
        throw new Error(data?.detail || '生成失败');
      }

    } catch (e: any) {
      alert('Error: ' + e.message);
    } finally {
      setPodcastGenerating(false);
    }
  };

  const selectedFileNames = (files || [])
    .filter(f => selectedIds.has(f.id))
    .map(f => f.name)
    .join(', ');

  return (
    <div className="flex-1 overflow-y-auto p-6 bg-[#0a0a1a] h-full">
      <div className="mb-6 bg-gradient-to-br from-green-900/20 to-blue-900/20 border border-green-500/20 rounded-xl p-4 flex items-start gap-3">
        <Headphones className="text-green-400 mt-1 flex-shrink-0" size={18} />
        <div>
          <h4 className="text-sm font-medium text-green-300 mb-1">知识播客生成</h4>
          <p className="text-xs text-green-200/70">
            支持选择多个文档（PDF/DOCX/PPTX）。系统将合并内容生成播客脚本并转换为语音。
          </p>
        </div>
      </div>

      <div className="space-y-6">
        {/* Context Info */}
        <div>
          <label className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2 block">
            当前选中素材 ({selectedIds.size} 个文件)
          </label>
          <div className="bg-white/5 border border-white/10 rounded-lg p-3 text-sm text-gray-300 flex items-center justify-between">
            <span className="truncate">{selectedIds.size > 0 ? selectedFileNames : '未选择'}</span>
            {selectedIds.size > 0 ? <CheckCircle2 size={16} className="text-green-500" /> : <X size={16} className="text-red-500" />}
          </div>
        </div>

        {/* Configuration */}
        <div className="space-y-4">
          {userApiConfigRequired ? (
            <>
              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-300">API Key</label>
                <input
                  type="password"
                  value={podcastParams.api_key}
                  onChange={e => setPodcastParams({...podcastParams, api_key: e.target.value})}
                  placeholder="sk-..."
                  className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-gray-200 outline-none focus:border-green-500 font-mono"
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-300">API URL</label>
                <select
                  value={podcastParams.api_url}
                  onChange={e => setPodcastParams({...podcastParams, api_url: e.target.value})}
                  className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-gray-200 outline-none focus:border-green-500"
                >
                  {API_URL_OPTIONS.map((url: string) => (
                    <option key={url} value={url}>{url}</option>
                  ))}
                </select>
              </div>
            </>
          ) : (
            <div className="rounded-lg border border-green-500/20 bg-green-500/5 px-3 py-2 text-xs text-green-200/80">
              Free 模式下由后端统一选择播客脚本模型、TTS 模型和接口配置。
            </div>
          )}

          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-300">LLM Model</label>
            <div className="grid grid-cols-2 gap-2">
              <select
                value={podcastParams.model}
                onChange={e => setPodcastParams({...podcastParams, model: e.target.value})}
                disabled={!userApiConfigRequired}
                className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-gray-200 outline-none focus:border-green-500 disabled:opacity-50"
              >
                <option value="gpt-5.1">gpt-5.1</option>
                <option value="gpt-5.2">gpt-5.2</option>
                <option value="gemini-3-pro-preview">gemini-3-pro-preview</option>
              </select>
              <input
                type="text"
                value={podcastParams.model}
                onChange={e => setPodcastParams({...podcastParams, model: e.target.value})}
                placeholder="自定义模型"
                disabled={!userApiConfigRequired}
                className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-gray-200 outline-none focus:border-green-500 disabled:opacity-50"
              />
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-300">TTS Model</label>
            <select
              value={podcastParams.tts_model}
              onChange={e => {
                const nextModel = e.target.value;
                const nextIsOpenAI = isOpenAITtsModel(nextModel);
                const nextIsCosyVoice = isCosyVoiceTtsModel(nextModel);
                const nextOptions = nextIsOpenAI ? OPENAI_TTS_VOICES : (nextIsCosyVoice ? [''] : []);
                setPodcastParams(prev => ({
                  ...prev,
                  tts_model: nextModel,
                  voice_name: normalizeVoice(prev.voice_name, nextOptions),
                  voice_name_b: normalizeVoice(prev.voice_name_b, nextOptions)
                }));
              }}
              disabled={!userApiConfigRequired}
              className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-gray-200 outline-none focus:border-green-500 disabled:opacity-50"
            >
              {COSYVOICE_TTS_MODELS.map(m => (
                <option key={m} value={m}>{m}</option>
              ))}
              <option value="gpt-4o-mini-tts">gpt-4o-mini-tts</option>
            </select>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-300">声音选择</label>
            <select
              value={podcastParams.voice_name}
              onChange={e => setPodcastParams({...podcastParams, voice_name: e.target.value})}
              className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-gray-200 outline-none focus:border-green-500"
            >
              {isCosyVoiceTts ? (
                <option value="">{COSYVOICE_VOICE_LABEL}</option>
              ) : (
                voiceOptions.map(voice => (
                  <option key={voice} value={voice}>{voice}</option>
                ))
              )}
            </select>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-300">播客模式</label>
            <div className="grid grid-cols-2 gap-3">
              <button
                onClick={() => setPodcastParams({...podcastParams, podcast_mode: 'monologue'})}
                className={`py-2.5 rounded-lg border text-sm transition-all ${
                  podcastParams.podcast_mode === 'monologue'
                    ? 'bg-green-500/20 border-green-500 text-green-300'
                    : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'
                }`}
              >
                单口
              </button>
              <button
                onClick={() => setPodcastParams({...podcastParams, podcast_mode: 'dialog'})}
                className={`py-2.5 rounded-lg border text-sm transition-all ${
                  podcastParams.podcast_mode === 'dialog'
                    ? 'bg-green-500/20 border-green-500 text-green-300'
                    : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'
                }`}
              >
                对话
              </button>
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-300">播客长度</label>
            <select
              value={podcastParams.podcast_length}
              onChange={e => setPodcastParams({...podcastParams, podcast_length: e.target.value})}
              className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-gray-200 outline-none focus:border-green-500"
            >
              <option value="brief">精炼（约2-4分钟）</option>
              <option value="standard">标准（约5-10分钟）</option>
              <option value="long">详细（约10-15分钟）</option>
            </select>
          </div>

          {podcastParams.podcast_mode === 'dialog' && (
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-300">第二位声音</label>
              <select
                value={podcastParams.voice_name_b}
                onChange={e => setPodcastParams({...podcastParams, voice_name_b: e.target.value})}
                className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-gray-200 outline-none focus:border-green-500"
              >
                {isCosyVoiceTts ? (
                  <option value="">{COSYVOICE_VOICE_LABEL}</option>
                ) : (
                  voiceOptions.map(voice => (
                    <option key={voice} value={voice}>{voice}</option>
                  ))
                )}
              </select>
            </div>
          )}

          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-300">目标语言</label>
            <div className="grid grid-cols-2 gap-3">
              <button
                onClick={() => setPodcastParams({...podcastParams, language: 'zh'})}
                className={`py-2.5 rounded-lg border text-sm transition-all ${
                  podcastParams.language === 'zh'
                    ? 'bg-green-500/20 border-green-500 text-green-300'
                    : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'
                }`}
              >
                中文
              </button>
              <button
                onClick={() => setPodcastParams({...podcastParams, language: 'en'})}
                className={`py-2.5 rounded-lg border text-sm transition-all ${
                  podcastParams.language === 'en'
                    ? 'bg-green-500/20 border-green-500 text-green-300'
                    : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'
                }`}
              >
                English
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-8 pb-8">
        <button
          onClick={handleGeneratePodcast}
          disabled={podcastGenerating || selectedIds.size === 0}
          className="w-full bg-gradient-to-r from-green-600 to-blue-600 hover:from-green-500 hover:to-blue-500 text-white py-3.5 rounded-xl font-medium flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-green-500/20 transition-all transform active:scale-95"
        >
          {podcastGenerating ? <Loader2 size={18} className="animate-spin" /> : <Headphones size={18} />}
          {podcastGenerating ? '正在生成播客...' : '开始生成播客'}
        </button>
      </div>
    </div>
  );
};
