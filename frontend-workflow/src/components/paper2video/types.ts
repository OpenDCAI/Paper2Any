export type Step = 'upload' | 'script' | 'complete';

export interface ScriptPage {
  pageNum: number;
  imageUrl: string;
  scriptText: string;
}
