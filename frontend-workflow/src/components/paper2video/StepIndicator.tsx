import React from 'react';
import { useTranslation } from 'react-i18next';
import { Check, ArrowRight } from 'lucide-react';
import { Step } from './types';

interface StepIndicatorProps {
  currentStep: Step;
}

const StepIndicator: React.FC<StepIndicatorProps> = ({ currentStep }) => {
  const { t } = useTranslation(['paper2video', 'common']);

  const steps = [
    { key: 'upload', label: t('steps.upload'), num: 1 },
    { key: 'script', label: t('steps.script'), num: 2 },
    { key: 'complete', label: t('steps.complete'), num: 3 },
  ];

  const currentIndex = steps.findIndex((s) => s.key === currentStep);

  return (
    <div className="flex items-center justify-center gap-2 mb-8">
      {steps.map((step, index) => (
        <div key={step.key} className="flex items-center">
          <div
            className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-all ${
              index === currentIndex
                ? 'bg-gradient-to-r from-teal-500 to-cyan-500 text-white shadow-lg'
                : index < currentIndex
                  ? 'bg-teal-500/20 text-teal-300 border border-teal-500/40'
                  : 'bg-white/5 text-gray-500 border border-white/10'
            }`}
          >
            <span
              className={`w-6 h-6 rounded-full flex items-center justify-center text-xs ${
                index < currentIndex ? 'bg-teal-400 text-white' : ''
              }`}
            >
              {index < currentIndex ? <Check size={14} /> : step.num}
            </span>
            <span className="hidden sm:inline">{step.label}</span>
          </div>
          {index < steps.length - 1 && (
            <ArrowRight
              size={16}
              className={`mx-2 ${index < currentIndex ? 'text-teal-400' : 'text-gray-600'}`}
            />
          )}
        </div>
      ))}
    </div>
  );
};

export default StepIndicator;
