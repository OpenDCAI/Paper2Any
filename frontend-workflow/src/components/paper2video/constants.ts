export const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB
export const STORAGE_KEY = 'paper2video-storage';
export const VIDEO_GENERATION_COST = 10;

/** 仅使用 cosyvoice-v3-flash */
export const TTS_MODEL = 'cosyvoice-v3-flash';
export const TTS_MODEL_DEFAULT = TTS_MODEL;

/** CosyVoice v3-flash 预置音色（试听文件在 public/paper2video/cosyvoice/v3-flash/{id}.wav，仅保留目录中存在的） */
export const COSYVOICE_V3_FLASH_VOICES = [
  { id: 'longanyang', name: '龙安洋' },
  { id: 'longanhuan', name: '龙安欢' },
  { id: 'longanwen_v3', name: '龙安温' },
  { id: 'longanzhi_v3', name: '龙安智' },
] as const;

/** 阿里云 CosyVoice 音色列表文档（用户可查 voice 参数并自定义填入） */
export const COSYVOICE_VOICE_LIST_URL = 'https://help.aliyun.com/zh/model-studio/cosyvoice-voice-list?spm=a2c4g.11186623.help-menu-2400256.d_2_6_0_9.68755c77EWIRG9#b259a0ab83et1';

/** 数字人视频统一走云端 LivePortrait */
export const TALKING_MODEL_LIVEPORTRAIT = 'liveportrait';
export const TALKING_MODEL_DEFAULT = TALKING_MODEL_LIVEPORTRAIT;
