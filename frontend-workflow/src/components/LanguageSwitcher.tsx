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
    <div className="inline-flex items-center rounded-full bg-white/10 p-1 border border-white/10">
      {languages.map((lang) => {
        const isActive = currentCode === lang.code;
        return (
          <button
            key={lang.code}
            onClick={() => changeLanguage(lang.code)}
            className={`
              px-3 py-1 text-xs font-medium rounded-full transition-all
              ${isActive 
                ? 'bg-white text-gray-900 shadow-sm' 
                : 'text-gray-400 hover:text-white hover:bg-white/5'
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
