import React from 'react';
import { useTranslation } from 'react-i18next';

interface HeaderProps {
  badge?: string;
  title?: string;
  subtitle?: string;
  align?: 'center' | 'left';
}

const Header: React.FC<HeaderProps> = ({ badge, title, subtitle, align = 'center' }) => {
  const { t } = useTranslation('paper2graph');

  const resolvedBadge = badge ?? t('hero.badge');
  const resolvedTitle = title ?? t('hero.title');
  const resolvedSubtitle = subtitle ?? t('hero.subtitle');
  const alignClass = align === 'left' ? 'text-left' : 'text-center';

  return (
    <div className={`mb-8 ${alignClass}`}>
      <p className="mb-2 text-xs uppercase tracking-[0.2em] text-primary-700">
        {resolvedBadge}
      </p>
      <h1 className="mb-2 text-3xl font-semibold text-[var(--text-primary)]">
        {resolvedTitle}
      </h1>
      <p className="mx-auto max-w-2xl text-sm text-[var(--text-secondary)]">
        {resolvedSubtitle}
      </p>
    </div>
  );
};

export default Header;
