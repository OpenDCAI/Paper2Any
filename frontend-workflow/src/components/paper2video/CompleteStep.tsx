import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { CheckCircle2, Download, RotateCcw, Loader2, AlertCircle } from 'lucide-react';

interface CompleteStepProps {
  videoUrl: string | null;
  isGenerating: boolean;
  handleDownload: () => void;
  handleReset: () => void;
  error: string | null;
  progress?: number;
  progressStatus?: string;
}

const CompleteStep: React.FC<CompleteStepProps> = ({
  videoUrl,
  isGenerating,
  handleDownload,
  handleReset,
  error,
  progress: progressProp,
  progressStatus: progressStatusProp,
}) => {
  const { t } = useTranslation(['paper2video', 'common']);
  const [simulatedProgress, setSimulatedProgress] = useState(0);
  const [simulatedStatus, setSimulatedStatus] = useState('');

  const progress = progressProp ?? simulatedProgress;
  const progressStatus = progressStatusProp ?? (simulatedStatus || t('complete.generating'));

  useEffect(() => {
    if (!isGenerating) {
      setSimulatedProgress(0);
      setSimulatedStatus('');
      return;
    }
    const messages = [
      t('complete.progressPreparing'),
      t('complete.progressTts'),
      t('complete.progressVideo'),
    ];
    setSimulatedProgress(0);
    setSimulatedStatus(messages[0]);
    const tickMs = 2500;
    const stepAvg = 90 / (480 / (tickMs / 1000));
    const interval = setInterval(() => {
      setSimulatedProgress((prev) => {
        if (prev >= 90) return 90;
        const idx = Math.min(Math.floor(prev / 30), messages.length - 1);
        setSimulatedStatus(messages[idx]);
        return prev + (Math.random() * stepAvg * 0.6 + stepAvg * 0.7);
      });
    }, tickMs);
    return () => clearInterval(interval);
  }, [isGenerating, t]);

  return (
    <div className="max-w-3xl mx-auto text-center">
      <div className="mb-8">
        <div className="w-20 h-20 rounded-full bg-gradient-to-br from-teal-500 to-cyan-500 flex items-center justify-center mx-auto mb-4">
          <CheckCircle2 size={40} className="text-white" />
        </div>
        <h2 className="text-2xl font-bold text-white mb-2">{t('complete.title')}</h2>
        <p className="text-gray-400">{t('complete.desc')}</p>
      </div>

      <div className="glass rounded-xl border border-white/10 p-6 mb-6">
        {!videoUrl ? (
          <div className="py-12">
            {isGenerating ? (
              <div className="animate-in fade-in">
                <Loader2 size={48} className="animate-spin text-teal-400 mx-auto mb-4" />
                <p className="text-gray-400 mb-4">{t('complete.generating')}</p>
                <div className="max-w-md mx-auto text-left">
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
              </div>
            ) : (
              <p className="text-gray-500">{t('complete.noVideo')}</p>
            )}
          </div>
        ) : (
          <div className="space-y-4">
            <div className="rounded-lg border border-white/20 overflow-hidden bg-black/40">
              <video
                src={videoUrl}
                controls
                className="w-full max-h-[400px]"
                preload="metadata"
              >
                {t('complete.videoNotSupported')}
              </video>
            </div>
            <button
              onClick={handleDownload}
              className="px-6 py-3 rounded-lg bg-gradient-to-r from-teal-600 to-cyan-600 text-white font-semibold flex items-center justify-center gap-2 mx-auto transition-all hover:from-teal-700 hover:to-cyan-700"
            >
              <Download size={18} /> {t('complete.download')}
            </button>
          </div>
        )}
      </div>

      <div>
        <button
          onClick={handleReset}
          className="text-sm text-gray-400 hover:text-white transition-colors flex items-center justify-center gap-1 mx-auto"
        >
          <RotateCcw size={14} /> {t('complete.newTask')}
        </button>
      </div>

      {error && (
        <div className="mt-4 flex items-center gap-2 text-sm text-red-300 bg-red-500/10 border border-red-500/40 rounded-lg px-4 py-3 justify-center">
          <AlertCircle size={16} /> {error}
        </div>
      )}
    </div>
  );
};

export default CompleteStep;
