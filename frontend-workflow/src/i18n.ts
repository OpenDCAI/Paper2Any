import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';

// 导入翻译文件
import enLogin from './locales/en/login.json';
import zhLogin from './locales/zh/login.json';
import enPaper2ppt from './locales/en/paper2ppt.json';
import zhPaper2ppt from './locales/zh/paper2ppt.json';
import enPdf2ppt from './locales/en/pdf2ppt.json';
import zhPdf2ppt from './locales/zh/pdf2ppt.json';
import enImage2ppt from './locales/en/image2ppt.json';
import zhImage2ppt from './locales/zh/image2ppt.json';
import enImage2drawio from './locales/en/image2drawio.json';
import zhImage2drawio from './locales/zh/image2drawio.json';
import enPptPolish from './locales/en/pptPolish.json';
import zhPptPolish from './locales/zh/pptPolish.json';
import enCommon from './locales/en/common.json';
import zhCommon from './locales/zh/common.json';
import enPaper2graph from './locales/en/paper2graph.json';
import zhPaper2graph from './locales/zh/paper2graph.json';
import enPaper2drawio from './locales/en/paper2drawio.json';
import zhPaper2drawio from './locales/zh/paper2drawio.json';
import enPaper2rebuttal from './locales/en/paper2rebuttal.json';
import zhPaper2rebuttal from './locales/zh/paper2rebuttal.json';
import enPaper2video from './locales/en/paper2video.json';
import zhPaper2video from './locales/zh/paper2video.json';
import enPaper2citation from './locales/en/paper2citation.json';
import zhPaper2citation from './locales/zh/paper2citation.json';

i18n
  // 检测用户语言
  // 文档: https://github.com/i18next/i18next-browser-languageDetector
  .use(LanguageDetector)
  // 将 i18n 实例传递给 react-i18next
  .use(initReactI18next)
  // 初始化 i18next
  // 所有选项: https://www.i18next.com/overview/configuration-options
  .init({
    debug: true,
    fallbackLng: 'en',
    interpolation: {
      escapeValue: false, // React 默认防止 XSS
    },
    resources: {
      en: {
        common: enCommon,
        login: enLogin,
        paper2ppt: enPaper2ppt,
        pdf2ppt: enPdf2ppt,
        image2ppt: enImage2ppt,
        image2drawio: enImage2drawio,
        pptPolish: enPptPolish,
        paper2graph: enPaper2graph,
        paper2drawio: enPaper2drawio,
        paper2rebuttal: enPaper2rebuttal,
        paper2video: enPaper2video,
        paper2citation: enPaper2citation
      },
      zh: {
        common: zhCommon,
        login: zhLogin,
        paper2ppt: zhPaper2ppt,
        pdf2ppt: zhPdf2ppt,
        image2ppt: zhImage2ppt,
        image2drawio: zhImage2drawio,
        pptPolish: zhPptPolish,
        paper2graph: zhPaper2graph,
        paper2drawio: zhPaper2drawio,
        paper2rebuttal: zhPaper2rebuttal,
        paper2video: zhPaper2video,
        paper2citation: zhPaper2citation
      }
    },
    // 默认命名空间
    defaultNS: 'common',
    // 命名空间
    ns: ['common', 'login', 'paper2ppt', 'pdf2ppt', 'image2ppt', 'image2drawio', 'pptPolish', 'paper2graph', 'paper2drawio', 'paper2rebuttal', 'paper2video', 'paper2citation']
  });

export default i18n;
