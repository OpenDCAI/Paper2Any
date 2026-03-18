import React from 'react';
import { useTranslation } from 'react-i18next';
import { ArrowLeft, ArrowRight, AlertCircle, Loader2 } from 'lucide-react';
import { ScriptPage, Step } from './types';
import { VIDEO_GENERATION_COST } from './constants';

interface ScriptStepProps {
  scriptPages: ScriptPage[];
  setScriptPages: React.Dispatch<React.SetStateAction<ScriptPage[]>>;
  handleConfirmScript: () => void;
  setCurrentStep: (step: Step) => void;
  error: string | null;
  isGenerating: boolean;
}

const ScriptStep: React.FC<ScriptStepProps> = ({
  scriptPages,
  setScriptPages,
  handleConfirmScript,
  setCurrentStep,
  error,
  isGenerating,
}) => {
  const { t } = useTranslation(['paper2video', 'common']);
  const disabledClass = 'disabled:opacity-50 disabled:cursor-not-allowed';

  const handleScriptChange = (pageNum: number, value: string) => {
    setScriptPages((prev) =>
      prev.map((p) => (p.pageNum === pageNum ? { ...p, scriptText: value } : p))
    );
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="text-center mb-8">
        <h2 className="text-2xl font-bold text-white mb-2">{t('script.title')}</h2>
        <p className="text-gray-400">{t('script.desc')}</p>
      </div>

      <div className="space-y-8">
        {scriptPages.map((page) => (
          <div
            key={page.pageNum}
            className="glass rounded-xl border border-white/10 p-4 space-y-3"
          >
            <div className="flex items-center gap-2">
              <span className="w-8 h-8 rounded-full bg-teal-500/20 text-teal-300 text-sm font-medium flex items-center justify-center">
                {page.pageNum + 1}
              </span>
              <span className="text-sm text-gray-400">{t('script.pageWithNum', { num: page.pageNum + 1 })}</span>
            </div>
            <div className="rounded-lg border border-white/10 overflow-hidden bg-white/5">
              {page.imageUrl ? (
                <img
                  src={page.imageUrl}
                  alt={`Page ${page.pageNum + 1}`}
                  className="w-full max-h-[400px] object-contain"
                />
              ) : (
                <div className="w-full h-48 flex items-center justify-center text-gray-500 text-sm">
                  {t('script.loadingImage')}
                </div>
              )}
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">{t('script.scriptLabel')}</label>
              <textarea
                value={page.scriptText}
                onChange={(e) => handleScriptChange(page.pageNum, e.target.value)}
                disabled={isGenerating}
                rows={4}
                className={`w-full rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-teal-500 resize-none ${disabledClass}`}
                placeholder={t('script.scriptPlaceholder')}
              />
            </div>
          </div>
        ))}
      </div>

      <div className="mt-6 text-sm text-amber-300 bg-amber-500/10 border border-amber-500/30 rounded-lg px-4 py-3">
        {t('script.costHint', { count: VIDEO_GENERATION_COST })}
      </div>

      <div className="flex justify-between mt-8">
        <button
          onClick={() => setCurrentStep('upload')}
          disabled={isGenerating}
          className={`px-6 py-2.5 rounded-lg border border-white/20 text-gray-300 hover:bg-white/10 flex items-center gap-2 ${disabledClass}`}
        >
          <ArrowLeft size={18} /> {t('script.back')}
        </button>
        <button
          onClick={handleConfirmScript}
          disabled={isGenerating}
          className={`px-6 py-2.5 rounded-lg bg-gradient-to-r from-teal-600 to-cyan-600 text-white font-semibold flex items-center gap-2 transition-all ${disabledClass}`}
        >
          {isGenerating ? (
            <>
              <Loader2 size={18} className="animate-spin" /> {t('script.generating')}
            </>
          ) : (
            <>
              {t('script.confirmAndGenerate')} <ArrowRight size={18} />
            </>
          )}
        </button>
      </div>

      {error && (
        <div className="mt-4 flex items-center gap-2 text-sm text-red-300 bg-red-500/10 border border-red-500/40 rounded-lg px-4 py-3">
          <AlertCircle size={16} /> {error}
        </div>
      )}
    </div>
  );
};

export default ScriptStep;
