const env = import.meta.env as Record<string, string | undefined>;

const readEnvList = (key: string, fallback: string[]) => {
  const value = env[key];
  if (typeof value === 'string') {
    const items = value
      .split(',')
      .map((item) => item.trim())
      .filter((item) => item.length > 0);
    if (items.length > 0) return items;
  }
  return fallback;
};

const firstOr = (list: string[], fallback: string) => list[0] || fallback;

export const PAPER2FIGURE_MODEL_ARCH_MODELS = readEnvList('VITE_PAPER2FIGURE_MODEL_MODEL_ARCH', [
  'gemini-3-pro-image-preview',
  'gemini-2.5-flash-image-preview',
]);
export const PAPER2FIGURE_EXP_DATA_MODELS = readEnvList('VITE_PAPER2FIGURE_MODEL_EXP_DATA', [
  'gemini-3-pro-image-preview',
  'gemini-2.5-flash-image-preview',
]);
export const PAPER2FIGURE_TECH_ROUTE_MODELS = readEnvList('VITE_PAPER2FIGURE_MODEL_TECH_ROUTE', [
  'gpt-5.2-medium',
  'gemini-3-pro-preview',
  'claude-sonnet-4-5-20250929',
  'claude-haiku-4-5-20251001',
]);

export const DEFAULT_PAPER2FIGURE_MODELS = {
  model_arch: firstOr(PAPER2FIGURE_MODEL_ARCH_MODELS, 'gemini-3-pro-image-preview'),
  exp_data: firstOr(PAPER2FIGURE_EXP_DATA_MODELS, 'gemini-3-pro-image-preview'),
  tech_route: firstOr(PAPER2FIGURE_TECH_ROUTE_MODELS, 'gpt-5.2-medium'),
} as const;

export const PAPER2PPT_MODELS = readEnvList('VITE_PAPER2PPT_MODEL', [
  'gpt-5.1',
  'gpt-5.2',
  'gemini-3-pro-preview',
]);
export const PAPER2PPT_GEN_FIG_MODELS = readEnvList('VITE_PAPER2PPT_GEN_FIG_MODEL', [
  'gemini-3-pro-image-preview',
  'gemini-2.5-flash-image',
]);
export const DEFAULT_PAPER2PPT_MODEL = firstOr(PAPER2PPT_MODELS, 'gpt-5.1');
export const DEFAULT_PAPER2PPT_GEN_FIG_MODEL = firstOr(PAPER2PPT_GEN_FIG_MODELS, 'gemini-3-pro-image-preview');

export const PDF2PPT_GEN_FIG_MODELS = readEnvList('VITE_PDF2PPT_GEN_FIG_MODEL', [
  'gemini-3-pro-image-preview',
]);
export const DEFAULT_PDF2PPT_GEN_FIG_MODEL = firstOr(PDF2PPT_GEN_FIG_MODELS, 'gemini-3-pro-image-preview');

export const IMAGE2PPT_GEN_FIG_MODELS = readEnvList('VITE_IMAGE2PPT_GEN_FIG_MODEL', [
  'gemini-3-pro-image-preview',
]);
export const DEFAULT_IMAGE2PPT_GEN_FIG_MODEL = firstOr(IMAGE2PPT_GEN_FIG_MODELS, 'gemini-3-pro-image-preview');

export const PPT2POLISH_MODELS = readEnvList('VITE_PPT2POLISH_MODEL', [
  'gpt-5.1',
  'gpt-5.2',
  'gemini-3-pro-preview',
]);
export const PPT2POLISH_GEN_FIG_MODELS = readEnvList('VITE_PPT2POLISH_GEN_FIG_MODEL', [
  'gemini-3-pro-image-preview',
]);
export const DEFAULT_PPT2POLISH_MODEL = firstOr(PPT2POLISH_MODELS, 'gpt-5.1');
export const DEFAULT_PPT2POLISH_GEN_FIG_MODEL = firstOr(PPT2POLISH_GEN_FIG_MODELS, 'gemini-3-pro-image-preview');

export const PAPER2DRAWIO_MODELS = readEnvList('VITE_PAPER2DRAWIO_MODEL', [
  'claude-sonnet-4-5-20250929',
]);
export const PAPER2DRAWIO_IMAGE_MODELS = readEnvList('VITE_PAPER2DRAWIO_IMAGE_MODEL', [
  'gemini-3-pro-image-preview',
]);
export const DEFAULT_PAPER2DRAWIO_MODEL = firstOr(PAPER2DRAWIO_MODELS, 'claude-sonnet-4-5-20250929');
export const DEFAULT_PAPER2DRAWIO_IMAGE_MODEL = firstOr(PAPER2DRAWIO_IMAGE_MODELS, 'gemini-3-pro-image-preview');

export const IMAGE2DRAWIO_GEN_FIG_MODELS = readEnvList('VITE_IMAGE2DRAWIO_GEN_FIG_MODEL', [
  'gemini-3-pro-image-preview',
]);
export const IMAGE2DRAWIO_VLM_MODELS = readEnvList('VITE_IMAGE2DRAWIO_VLM_MODEL', [
  'qwen-vl-ocr-2025-11-20',
]);
export const DEFAULT_IMAGE2DRAWIO_GEN_FIG_MODEL = firstOr(IMAGE2DRAWIO_GEN_FIG_MODELS, 'gemini-3-pro-image-preview');
export const DEFAULT_IMAGE2DRAWIO_VLM_MODEL = firstOr(IMAGE2DRAWIO_VLM_MODELS, 'qwen-vl-ocr-2025-11-20');

export const PAPER2REBUTTAL_MODELS = readEnvList('VITE_PAPER2REBUTTAL_MODEL', [
  'deepseek-v3.1',
  'kimi-k2.5',
]);
export const DEFAULT_PAPER2REBUTTAL_MODEL = firstOr(PAPER2REBUTTAL_MODELS, 'deepseek-v3.1');

export const withModelOptions = (base: string[], current?: string) => {
  const seen = new Set<string>();
  const ordered: string[] = [];
  const push = (value?: string) => {
    if (!value) return;
    if (seen.has(value)) return;
    seen.add(value);
    ordered.push(value);
  };
  push(current);
  base.forEach((value) => push(value));
  return ordered;
};
