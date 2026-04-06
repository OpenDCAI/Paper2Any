import React, { ChangeEvent, useRef, useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { API_URL_OPTIONS } from '../../config/api';
import ManagedApiNotice from '../ManagedApiNotice';
import {
  COSYVOICE_V3_FLASH_VOICES,
  COSYVOICE_VOICE_LIST_URL,
  TTS_MODEL,
} from './constants';
import {
  UploadCloud, Settings2, Loader2, AlertCircle, ArrowRight,
  Key, Globe, Cpu, Mic, Image, X, Play, Square, ExternalLink
} from 'lucide-react';

/** 系统预置数字人：id 对应 public/paper2video/avatar/{id}.png */
export const SYSTEM_AVATARS = [
  { id: 'avatar1', name: 'Avatar 1', url: '/paper2video/avatar/avatar1.png' },
  { id: 'avatar2', name: 'Avatar 2', url: '/paper2video/avatar/avatar2.png' },
];

export type UseAvatar = 'none' | 'yes';

interface UploadStepProps {
  selectedFile: File | null;
  isDragOver: boolean;
  setIsDragOver: (v: boolean) => void;
  useAvatar: UseAvatar;
  setUseAvatar: (v: UseAvatar) => void;
  avatarFile: File | null;
  avatarPreview: string | null;
  avatarPreset: string | null;
  isUploading: boolean;
  progress: number;
  progressStatus: string;
  error: string | null;
  showApiConfig: boolean;
  apiKey: string;
  setApiKey: (v: string) => void;
  scriptApiUrl: string;
  setScriptApiUrl: (v: string) => void;
  scriptModel: string;
  setScriptModel: (v: string) => void;
  ttsModel: string;
  setTtsModel: (v: string) => void;
  ttsVoiceName: string;
  setTtsVoiceName: (v: string) => void;
  language: 'zh' | 'en';
  setLanguage: (v: 'zh' | 'en') => void;
  handleFileChange: (e: ChangeEvent<HTMLInputElement>) => void;
  handleDrop: (e: React.DragEvent<HTMLDivElement>) => void;
  handleAvatarChange: (e: ChangeEvent<HTMLInputElement>) => void;
  handleSelectAvatarPreset: (presetId: string) => void;
  handleRemoveAvatar: () => void;
  handleStartParse: () => void;
}

const UploadStep: React.FC<UploadStepProps> = ({
  selectedFile,
  isDragOver,
  setIsDragOver,
  useAvatar,
  setUseAvatar,
  avatarFile,
  avatarPreview,
  avatarPreset,
  isUploading,
  progress,
  progressStatus,
  error,
  showApiConfig,
  apiKey,
  setApiKey,
  scriptApiUrl,
  setScriptApiUrl,
  scriptModel,
  setScriptModel,
  ttsModel,
  setTtsModel,
  ttsVoiceName,
  setTtsVoiceName,
  language,
  setLanguage,
  handleFileChange,
  handleDrop,
  handleAvatarChange,
  handleSelectAvatarPreset,
  handleRemoveAvatar,
  handleStartParse,
}) => {
  const { t } = useTranslation(['paper2video', 'common']);
  const voiceAudioRef = useRef<HTMLAudioElement | null>(null);
  const [voicePlayingId, setVoicePlayingId] = useState<string | null>(null);

  useEffect(() => {
    return () => {
      if (voiceAudioRef.current) {
        voiceAudioRef.current.pause();
        voiceAudioRef.current = null;
      }
      setVoicePlayingId(null);
    };
  }, []);

  const handleTtsVoicePreview = (e: React.MouseEvent, voiceId: string) => {
    e.stopPropagation();
    const id = `tts-${voiceId}`;
    if (voicePlayingId === id) {
      voiceAudioRef.current?.pause();
      voiceAudioRef.current = null;
      setVoicePlayingId(null);
      return;
    }
    if (voiceAudioRef.current) voiceAudioRef.current.pause();
    const url = `/paper2video/cosyvoice/v3-flash/${voiceId}.wav`;
    const audio = new Audio(url);
    voiceAudioRef.current = audio;
    audio.play().catch(() => {});
    audio.onended = () => {
      setVoicePlayingId(null);
      voiceAudioRef.current = null;
    };
    setVoicePlayingId(id);
  };

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-10 text-center">
        <p className="text-xs uppercase tracking-[0.2em] text-teal-300 mb-3 font-semibold">{t('upload.subtitle')}</p>
        <h1 className="text-4xl md:text-5xl font-bold mb-4">
          <span className="bg-gradient-to-r from-teal-400 via-cyan-400 to-blue-400 bg-clip-text text-transparent">
            {t('upload.title')}
          </span>
        </h1>
        <p className="text-base text-gray-300 max-w-2xl mx-auto leading-relaxed">
          {t('upload.desc')}
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="glass rounded-xl border border-white/10 p-6 relative overflow-hidden">
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-2/3 h-1 bg-gradient-to-r from-transparent via-teal-500 to-transparent opacity-50 blur-sm" />

          <h3 className="text-white font-medium text-sm mb-3 flex items-center gap-2">
            <span className="w-1 h-4 rounded-full bg-teal-500" />
            {t('upload.instruction.pdf')}
          </h3>
          <div
            className={`border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center text-center gap-4 transition-all min-h-[180px] ${
              isDragOver ? 'border-teal-500 bg-teal-500/10' : 'border-white/20 hover:border-teal-400'
            }`}
            onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
            onDragLeave={(e) => { e.preventDefault(); setIsDragOver(false); }}
            onDrop={handleDrop}
          >
            <div className="w-14 h-14 rounded-full bg-gradient-to-br from-teal-500/20 to-cyan-500/20 flex items-center justify-center">
              <UploadCloud size={28} className="text-teal-400" />
            </div>
            <div>
              <p className="text-white font-medium mb-1">{t('upload.dropzone.dragText')}</p>
              <p className="text-sm text-gray-400">{t('upload.dropzone.supportText')}</p>
            </div>
            <label className="px-5 py-2 rounded-full bg-gradient-to-r from-teal-600 to-cyan-600 text-white text-sm font-medium cursor-pointer hover:from-teal-700 hover:to-cyan-700 transition-all">
              {t('upload.dropzone.button')}
              <input type="file" accept=".pdf,.pptx" className="hidden" onChange={handleFileChange} />
            </label>
            {selectedFile && (
              <div className="px-4 py-2 bg-teal-500/20 border border-teal-500/40 rounded-lg">
                <p className="text-sm text-teal-300">✓ {selectedFile.name}</p>
              </div>
            )}
          </div>

          <h3 className="text-white font-medium text-sm mt-6 mb-2 flex items-center gap-2">
            <span className="w-1 h-4 rounded-full bg-teal-500" />
            {t('upload.avatarLabel')} <span className="text-gray-500 text-xs font-normal">({t('upload.optional')})</span>
          </h3>
          <div className="flex rounded-xl bg-white/5 border border-white/10 p-1 gap-0">
            <button
              type="button"
              onClick={() => setUseAvatar('none')}
              className={`flex-1 py-2.5 px-4 rounded-lg text-sm font-medium transition-all ${
                useAvatar === 'none'
                  ? 'bg-teal-500/20 text-teal-300 border border-teal-500/50 shadow-sm'
                  : 'text-gray-400 hover:text-gray-300 hover:bg-white/5'
              }`}
            >
              {t('upload.avatarUseNone')}
            </button>
            <button
              type="button"
              onClick={() => setUseAvatar('yes')}
              className={`flex-1 py-2.5 px-4 rounded-lg text-sm font-medium transition-all ${
                useAvatar === 'yes'
                  ? 'bg-teal-500/20 text-teal-300 border border-teal-500/50 shadow-sm'
                  : 'text-gray-400 hover:text-gray-300 hover:bg-white/5'
              }`}
            >
              {t('upload.avatarUseYes')}
            </button>
          </div>
          {useAvatar === 'yes' && (
            <>
              <p className="text-xs text-amber-400/90 mb-2">{t('upload.avatarTipTime')}</p>
              <p className="text-xs text-gray-400 mb-2">{t('upload.avatarChoiceHint')}</p>
              <p className="text-xs text-teal-300/90 mb-3 mt-3">{t('upload.talkingModelFixed')}</p>
              <p className="text-xs text-gray-500 mb-1">{t('upload.avatarSystemLabel')}</p>
              <div className="flex flex-wrap gap-3 mb-3">
                {SYSTEM_AVATARS.map((a) => (
                  <button
                    key={a.id}
                    type="button"
                    onClick={() => handleSelectAvatarPreset(avatarPreset === a.id ? '' : a.id)}
                    className={`rounded-xl border-2 overflow-hidden transition-all w-20 h-20 flex-shrink-0 ${
                      avatarPreset === a.id && !avatarFile
                        ? 'border-teal-500 ring-2 ring-teal-400/50 shadow-lg shadow-teal-500/20 scale-[1.02]'
                        : 'border-white/20 hover:border-teal-400/60 hover:shadow-md'
                    }`}
                  >
                    <img src={a.url} alt={a.name} className="w-full h-full object-cover" />
                  </button>
                ))}
              </div>
              <p className="text-xs text-gray-500 mb-1">{t('upload.avatarUploadOwn')}</p>
              {avatarPreview || (avatarPreset && !avatarFile) ? (
                <div className="relative">
                  <img
                    src={avatarPreview ?? SYSTEM_AVATARS.find((a) => a.id === avatarPreset)?.url ?? ''}
                    alt="Avatar"
                    className="w-full max-h-32 object-contain rounded-lg border border-white/20 bg-black/40"
                  />
                  <button
                    type="button"
                    onClick={handleRemoveAvatar}
                    className="absolute top-2 right-2 p-1.5 rounded-full bg-black/60 text-white hover:bg-red-500 transition-colors"
                  >
                    <X size={14} />
                  </button>
                  <p className="text-[11px] text-teal-300 mt-1">
                    {avatarFile ? `✓ ${avatarFile.name}` : avatarPreset ? `✓ ${t('upload.avatarSystemLabel')}` : ''}
                  </p>
                </div>
              ) : (
                <label className="border-2 border-dashed border-white/20 rounded-lg p-4 flex flex-col items-center justify-center text-center gap-2 cursor-pointer hover:border-teal-400 transition-all min-h-[80px]">
                  <Image size={20} className="text-gray-400" />
                  <span className="text-xs text-gray-400">{t('upload.avatarUpload')}</span>
                  <input type="file" accept="image/jpeg,image/png,image/jpg" className="hidden" onChange={handleAvatarChange} />
                </label>
              )}
            </>
          )}
        </div>

        <div className="glass rounded-xl border border-white/10 p-6 space-y-4">
          <h3 className="text-white font-semibold flex items-center gap-2">
            <Settings2 size={18} className="text-teal-400" /> {t('upload.config.title')}
          </h3>

          {showApiConfig ? (
            <>
              <div>
                <label className="block text-xs text-gray-400 mb-1 flex items-center gap-1">
                  <Key size={12} /> {t('upload.config.apiKey')}
                </label>
                <input
                  type="password"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder={t('upload.config.apiKeyPlaceholder')}
                  className="w-full rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-teal-500"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs text-gray-400 mb-1 flex items-center gap-1">
                    <Globe size={12} /> {t('upload.config.scriptApiUrl')}
                  </label>
                  <select
                    value={scriptApiUrl}
                    onChange={(e) => setScriptApiUrl(e.target.value)}
                    className="w-full rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-teal-500"
                  >
                    {API_URL_OPTIONS.map((url: string) => (
                      <option key={url} value={url}>{url}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-xs text-gray-400 mb-1 flex items-center gap-1">
                    <Cpu size={12} /> {t('upload.config.scriptModel')}
                  </label>
                  <select
                    value={scriptModel}
                    onChange={(e) => setScriptModel(e.target.value)}
                    className="w-full rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-teal-500"
                  >
                    <option value="gpt-4o">gpt-4o</option>
                    <option value="gpt-4o-mini">gpt-4o-mini</option>
                    <option value="gemini-2.5-flash">gemini-2.5-flash</option>
                    <option value="gemini-2.5-pro">gemini-2.5-pro</option>
                  </select>
                </div>
              </div>
            </>
          ) : (
            <>
              <ManagedApiNotice />
              <div>
                <label className="block text-xs text-gray-400 mb-1 flex items-center gap-1">
                  <Cpu size={12} /> {t('upload.config.scriptModel')}
                </label>
                <select
                  value={scriptModel}
                  onChange={(e) => setScriptModel(e.target.value)}
                  className="w-full rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-teal-500"
                >
                  <option value="gpt-4o">gpt-4o</option>
                  <option value="gpt-4o-mini">gpt-4o-mini</option>
                  <option value="gemini-2.5-flash">gemini-2.5-flash</option>
                  <option value="gemini-2.5-pro">gemini-2.5-pro</option>
                </select>
              </div>
            </>
          )}

          <div>
            <label className="block text-xs text-gray-400 mb-1 flex items-center gap-1">
              <Mic size={12} /> {t('upload.voiceLabel')} <span className="text-gray-500">({t('upload.optional')})</span>
            </label>
            <div className="space-y-3">
              <div className="rounded-lg border border-teal-500/30 bg-teal-500/10 px-3 py-2 text-sm text-teal-200">
                {t('upload.voiceApiOnly')}
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1 flex items-center gap-1">
                  <Mic size={12} /> {t('upload.config.ttsModelLabel')}
                </label>
                <div className="rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm text-teal-300">
                  {TTS_MODEL}
                </div>
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1 flex items-center gap-1">
                  <Mic size={12} /> {t('upload.config.ttsVoiceLabel')}
                </label>
                <p className="text-xs text-gray-500 mb-2">{t('upload.config.ttsVoicePresetHint')}</p>
                <div className="flex flex-wrap gap-2 mb-2">
                  {COSYVOICE_V3_FLASH_VOICES.map((v) => (
                    <div
                      key={v.id}
                      className={`flex items-center gap-0 rounded-xl overflow-hidden border-2 transition-all ${
                        ttsVoiceName === v.id
                          ? 'border-teal-500 bg-teal-500/15 shadow-md shadow-teal-500/10'
                          : 'border-white/20 hover:border-teal-400/50 bg-white/5'
                      }`}
                    >
                      <button
                        type="button"
                        onClick={() => setTtsVoiceName(v.id)}
                        className="pl-3 pr-2 py-2.5 text-sm font-medium text-left text-gray-200 flex-1 min-w-0"
                      >
                        {v.name}
                      </button>
                      <button
                        type="button"
                        onClick={(e) => handleTtsVoicePreview(e, v.id)}
                        className="pr-2.5 py-2.5 text-teal-400 hover:text-teal-300 hover:bg-white/10 flex-shrink-0 transition-colors"
                        title={t('upload.voicePreview')}
                      >
                        {voicePlayingId === `tts-${v.id}` ? (
                          <Square size={16} fill="currentColor" />
                        ) : (
                          <Play size={16} fill="currentColor" />
                        )}
                      </button>
                    </div>
                  ))}
                </div>
                <p className="text-xs text-gray-500 mb-1 flex items-center gap-1 flex-wrap">
                  <span>{t('upload.config.ttsVoiceCustomHint')}</span>
                  <a
                    href={COSYVOICE_VOICE_LIST_URL}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-teal-400 hover:text-teal-300 inline-flex items-center gap-0.5"
                  >
                    {t('upload.config.ttsVoiceListLink')}
                    <ExternalLink size={12} />
                  </a>
                </p>
                <input
                  type="text"
                  value={ttsVoiceName}
                  onChange={(e) => setTtsVoiceName(e.target.value)}
                  placeholder={t('upload.config.ttsVoicePlaceholder')}
                  spellCheck={false}
                  className="w-full rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm text-gray-100 placeholder-gray-500 outline-none focus:ring-2 focus:ring-teal-500"
                />
              </div>
            </div>
          </div>

          <div>
            <label className="block text-xs text-gray-400 mb-1">{t('upload.config.language')}</label>
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value as 'zh' | 'en')}
              className="w-full rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-teal-500"
            >
              <option value="zh">中文</option>
              <option value="en">English</option>
            </select>
          </div>

          <button
            onClick={handleStartParse}
            disabled={!selectedFile || isUploading}
            className="w-full py-3 rounded-lg bg-gradient-to-r from-teal-600 to-cyan-600 hover:from-teal-700 hover:to-cyan-700 disabled:from-gray-600 disabled:to-gray-700 text-white font-semibold flex items-center justify-center gap-2 transition-all"
          >
            {isUploading ? (
              <>
                <Loader2 size={18} className="animate-spin" /> {t('upload.config.parsing')}
              </>
            ) : (
              <>
                <ArrowRight size={18} /> {t('upload.config.startParse')}
              </>
            )}
          </button>

          {isUploading && (
            <div className="mt-4 animate-in fade-in slide-in-from-top-2">
              <div className="flex justify-between text-xs text-gray-400 mb-1">
                <span>{progressStatus}</span>
                <span>{Math.round(progress)}%</span>
              </div>
              <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-teal-500 to-cyan-500 transition-all duration-300 ease-out"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )}
        </div>
      </div>

      {error && (
        <div className="mt-4 flex items-center gap-2 text-sm text-red-300 bg-red-500/10 border border-red-500/40 rounded-lg px-4 py-3">
          <AlertCircle size={16} /> {error}
        </div>
      )}
    </div>
  );
};

export default UploadStep;
