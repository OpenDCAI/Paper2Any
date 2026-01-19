import { GenerationStage } from './types';

export const BACKEND_API = '/api/v1/paper2figure/generate';
export const JSON_API = '/api/v1/paper2figure/generate-json';
export const HISTORY_API = '/api/v1/paper2figure/history';

export const IMAGE_EXTENSIONS = ['png', 'jpg', 'jpeg', 'bmp', 'gif', 'webp', 'tiff'];

export const GENERATION_STAGES: GenerationStage[] = [
  { id: 1, message: '正在分析论文内容...', duration: 30 },
  { id: 2, message: '正在生成科研绘图...', duration: 30 },
  { id: 3, message: '正在转为可编辑绘图...', duration: 30 },
  { id: 4, message: '正在合成 PPT...', duration: 30 },
];

export const MAX_FILE_SIZE = 20 * 1024 * 1024; // 20MB
export const STORAGE_KEY = 'paper2figure_config_v1';
