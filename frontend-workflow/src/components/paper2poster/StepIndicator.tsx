import React from 'react';
import { useTranslation } from 'react-i18next';
import { Check, ArrowRight } from 'lucide-react';
import { Step } from './types';

interface StepIndicatorProps {
  currentStep: Step;
}

const StepIndicator: React.FC<StepIndicatorProps> = ({ currentStep }) => {
  const { t } = useTranslation(['paper2poster', 'common']);

  const steps = [
    { key: 'upload', label: t('steps.upload', '上传配置'), num: 1 },
    { key: 'generate', label: t('steps.generate', '生成海报'), num: 2 },
    { key: 'complete', label: t('steps.complete', '完成'), num: 3 },
  ];

  const currentIndex = steps.findIndex(s => s.key === currentStep);

  return (
    <div className="flex items-center justify-center gap-2 mb-8">
      {steps.map((step, index) => (
        <div key={step.key} className="flex items-center">
          <div className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-all ${
            index === currentIndex
              ? 'bg-gradient-to-r from-green-500 to-emerald-500 text-white shadow-lg'
              : index < currentIndex
                ? 'bg-green-500/20 text-green-300 border border-green-500/40'
                : 'bg-white/5 text-gray-500 border border-white/10'
          }`}>
            <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs ${
              index < currentIndex ? 'bg-green-400 text-white' : ''
            }`}>
              {index < currentIndex ? <Check size={14} /> : step.num}
            </span>
            <span className="hidden sm:inline">{step.label}</span>
          </div>
          {index < steps.length - 1 && (
            <ArrowRight size={16} className={`mx-2 ${index < currentIndex ? 'text-green-400' : 'text-gray-600'}`} />
          )}
        </div>
      ))}
    </div>
  );
};

export default StepIndicator;
