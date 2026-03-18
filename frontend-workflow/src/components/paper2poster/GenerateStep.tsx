import React from 'react';
import { useTranslation } from 'react-i18next';
import { Loader2, CheckCircle, AlertCircle } from 'lucide-react';
import { GenerateResult } from './types';

interface GenerateStepProps {
  result: GenerateResult;
  error: string | null;
}

const GenerateStep: React.FC<GenerateStepProps> = ({ result, error }) => {
  const { t } = useTranslation(['paper2poster', 'common']);

  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-10 text-center">
        <h1 className="text-4xl md:text-5xl font-bold mb-4">
          <span className="bg-gradient-to-r from-green-400 via-emerald-400 to-teal-400 bg-clip-text text-transparent">
            {t('generate.title', '正在生成海报')}
          </span>
        </h1>
        <p className="text-base text-gray-300 max-w-2xl mx-auto leading-relaxed">
          {t('generate.desc', '请稍候，AI正在为您生成精美的学术海报...')}
        </p>
      </div>

      {/* 生成状态 */}
      <div className="glass rounded-2xl p-8 border border-white/10">
        {result.status === 'processing' && (
          <div className="text-center">
            <Loader2 className="w-16 h-16 text-green-400 animate-spin mx-auto mb-4" />
            <p className="text-white font-medium text-lg mb-2">
              {t('generate.processing', '生成中...')}
            </p>
            <p className="text-gray-400 text-sm">
              {t('generate.processingDesc', '这可能需要几分钟时间，请耐心等待')}
            </p>
            {result.progress !== undefined && (
              <div className="mt-6">
                <div className="w-full bg-white/10 rounded-full h-2 overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-green-500 to-emerald-500 transition-all duration-300"
                    style={{ width: `${result.progress}%` }}
                  />
                </div>
                <p className="text-sm text-gray-400 mt-2">{result.progress.toFixed(0)}%</p>
              </div>
            )}
          </div>
        )}

        {result.status === 'done' && (
          <div className="text-center">
            <CheckCircle className="w-16 h-16 text-green-400 mx-auto mb-4" />
            <p className="text-white font-medium text-lg mb-2">
              {t('generate.done', '生成完成！')}
            </p>
            <p className="text-gray-400 text-sm">
              {t('generate.doneDesc', '海报已成功生成')}
            </p>
          </div>
        )}

        {error && (
          <div className="flex items-start gap-3 p-4 bg-red-500/10 border border-red-500/50 rounded-xl">
            <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <p className="text-red-300 text-sm">{error}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default GenerateStep;
