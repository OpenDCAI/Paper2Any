import React from 'react';
import { useTranslation } from 'react-i18next';

const Header = () => {
  const { t } = useTranslation('paper2graph');

  return (
    <div className="mb-8 text-center">
      <p className="text-xs uppercase tracking-[0.2em] text-primary-300 mb-2">
        {t('hero.badge')}
      </p>
      <h1 className="text-3xl font-semibold text-white mb-2">
        {t('hero.title')}
      </h1>
      <p className="text-sm text-gray-400 max-w-2xl mx-auto">
        {t('hero.subtitle')}
      </p>
    </div>
  );
};

export default Header;
