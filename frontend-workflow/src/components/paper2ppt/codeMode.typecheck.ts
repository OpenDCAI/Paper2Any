import type { PptGenerationMode } from './types';

const codeMode: PptGenerationMode = 'code';

const modeLabels: Record<PptGenerationMode, string> = {
  image: 'Image deck',
  frontend: 'Frontend editable deck',
  code: 'Code editable deck',
};

void codeMode;
void modeLabels;
