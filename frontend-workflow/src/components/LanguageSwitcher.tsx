import { useTranslation } from 'react-i18next';

export function LanguageSwitcher() {
  const { i18n } = useTranslation();

  const changeLanguage = (lng: string) => {
    i18n.changeLanguage(lng);
  };

  const languages = [
    { code: 'zh', label: '中' },
    { code: 'en', label: 'EN' }
  ];

  // 简单判断当前语言前缀
  const currentCode = i18n.language && i18n.language.startsWith('zh') ? 'zh' : 'en';

  return (
    <div className="inline-flex items-center rounded-full portal-pill p-1">
      {languages.map((lang) => {
        const isActive = currentCode === lang.code;
        return (
          <button
            key={lang.code}
            onClick={() => changeLanguage(lang.code)}
            className={`
              px-3 py-1 text-xs font-medium rounded-full transition-all
              ${isActive 
                ? 'bg-primary-600 text-white shadow-sm' 
                : 'text-primary-700 hover:text-primary-900 hover:bg-primary-500/5'
              }
            `}
          >
            {lang.label}
          </button>
        );
      })}
    </div>
  );
}
