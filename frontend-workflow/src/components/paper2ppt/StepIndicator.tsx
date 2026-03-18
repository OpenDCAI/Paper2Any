import React from 'react';
import { useTranslation } from 'react-i18next';
import { Check, ArrowRight } from 'lucide-react';
import { Step } from './types';

interface StepIndicatorProps {
  currentStep: Step;
}

const StepIndicator: React.FC<StepIndicatorProps> = ({ currentStep }) => {
  const { t } = useTranslation(['paper2ppt', 'common']);
  
  const steps = [
    { key: 'upload', label: t('steps.upload'), num: 1 },
    { key: 'outline', label: t('steps.outline'), num: 2 },
    { key: 'generate', label: t('steps.generate'), num: 3 },
    { key: 'complete', label: t('steps.complete'), num: 4 },
  ];
  
  const currentIndex = steps.findIndex(s => s.key === currentStep);
  
  return (
    <div className="mb-8 flex flex-wrap items-center justify-center gap-2">
      {steps.map((step, index) => (
        <div key={step.key} className="flex items-center">
          <div className={`flex items-center gap-2 rounded-full border px-4 py-2 text-sm font-medium transition-all ${
            index === currentIndex 
              ? 'paper2ppt-tab-active border-white/20' 
              : index < currentIndex 
                ? 'paper2ppt-chip-active border-[rgba(140,29,64,0.18)]'
                : 'paper2ppt-chip'
          }`}>
            <span className={`flex h-6 w-6 items-center justify-center rounded-full text-xs ${
              index === currentIndex
                ? 'bg-white/18 text-white'
                : index < currentIndex
                  ? 'bg-[#8c1d40] text-white'
                  : 'bg-[rgba(140,29,64,0.08)] text-[#6c1634]'
            }`}>
              {index < currentIndex ? <Check size={14} /> : step.num}
            </span>
            <span className="hidden sm:inline">{step.label}</span>
          </div>
          {index < steps.length - 1 && (
            <ArrowRight size={16} className={`mx-2 ${index < currentIndex ? 'text-[#8c1d40]' : 'text-[rgba(103,95,88,0.72)]'}`} />
          )}
        </div>
      ))}
    </div>
  );
};

export default StepIndicator;
