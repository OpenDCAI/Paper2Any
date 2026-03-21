export type HomeNavigablePage =
  | 'paper2figure-tech-exp'
  | 'paper2figure-model-drawio'
  | 'paper2drawio-ai'
  | 'paper2ppt'
  | 'paper2video'
  | 'paper2poster'
  | 'paper2citation'
  | 'pdf2ppt'
  | 'image2ppt'
  | 'image2drawio'
  | 'ppt2polish'
  | 'knowledge'
  | 'files'
  | 'paper2drawio'
  | 'paper2rebuttal';

export type HomeIconKey =
  | 'sparkles'
  | 'presentation'
  | 'video'
  | 'gitBranch'
  | 'network'
  | 'layoutTemplate'
  | 'fileStack'
  | 'fileImage'
  | 'fileSearch'
  | 'messageSquare'
  | 'folderKanban';

export interface HomeFeatureCard {
  page: HomeNavigablePage;
  titleKey: string;
  descriptionKey: string;
  icon: HomeIconKey;
  accent: string;
}

export interface HomeFeatureSection {
  titleKey: string;
  descriptionKey: string;
  cards: HomeFeatureCard[];
}

export const featuredHomeCards: HomeFeatureCard[] = [
  {
    page: 'paper2figure-model-drawio',
    titleKey: 'app.navSub.paper2figureModelDrawio',
    descriptionKey: 'app.navSubTooltip.paper2figureModelDrawio',
    icon: 'gitBranch',
    accent: 'from-primary-700 via-primary-600 to-amber-500',
  },
  {
    page: 'paper2ppt',
    titleKey: 'app.nav.paper2ppt',
    descriptionKey: 'app.navTooltip.paper2ppt',
    icon: 'presentation',
    accent: 'from-primary-600 via-primary-500 to-amber-500',
  },
  {
    page: 'paper2video',
    titleKey: 'app.nav.paper2video',
    descriptionKey: 'app.navTooltip.paper2video',
    icon: 'video',
    accent: 'from-amber-500 via-primary-500 to-primary-700',
  },
];

export const homeFeatureSections: HomeFeatureSection[] = [
  {
    titleKey: 'app.home.sections.creation.title',
    descriptionKey: 'app.home.sections.creation.description',
    cards: [
      {
        page: 'paper2figure-tech-exp',
        titleKey: 'app.navSub.paper2figureTechExp',
        descriptionKey: 'app.navSubTooltip.paper2figureTechExp',
        icon: 'sparkles',
        accent: 'from-primary-700 via-primary-500 to-amber-500',
      },
      {
        page: 'paper2drawio-ai',
        titleKey: 'app.navSub.paper2drawioAi',
        descriptionKey: 'app.navSubTooltip.paper2drawioAi',
        icon: 'network',
        accent: 'from-primary-700 via-primary-600 to-amber-500',
      },
      {
        page: 'paper2poster',
        titleKey: 'app.nav.paper2poster',
        descriptionKey: 'app.navTooltip.paper2poster',
        icon: 'layoutTemplate',
        accent: 'from-amber-500 via-primary-500 to-primary-700',
      },
    ],
  },
  {
    titleKey: 'app.home.sections.delivery.title',
    descriptionKey: 'app.home.sections.delivery.description',
    cards: [
      {
        page: 'paper2citation',
        titleKey: 'app.nav.paper2citation',
        descriptionKey: 'app.navTooltip.paper2citation',
        icon: 'fileSearch',
        accent: 'from-primary-700 via-primary-500 to-amber-500',
      },
      {
        page: 'paper2rebuttal',
        titleKey: 'app.nav.paper2rebuttal',
        descriptionKey: 'app.navTooltip.paper2rebuttal',
        icon: 'messageSquare',
        accent: 'from-primary-600 via-primary-500 to-amber-500',
      },
      {
        page: 'files',
        titleKey: 'app.nav.files',
        descriptionKey: 'filesPage.empty.desc',
        icon: 'folderKanban',
        accent: 'from-amber-500 via-primary-500 to-primary-700',
      },
    ],
  },
  {
    titleKey: 'app.home.sections.conversion.title',
    descriptionKey: 'app.home.sections.conversion.description',
    cards: [
      {
        page: 'pdf2ppt',
        titleKey: 'app.nav.pdf2ppt',
        descriptionKey: 'app.navTooltip.pdf2ppt',
        icon: 'fileStack',
        accent: 'from-primary-700 via-primary-600 to-amber-500',
      },
      {
        page: 'image2ppt',
        titleKey: 'app.nav.image2ppt',
        descriptionKey: 'app.navTooltip.image2ppt',
        icon: 'fileImage',
        accent: 'from-primary-600 via-primary-500 to-amber-500',
      },
      {
        page: 'image2drawio',
        titleKey: 'app.nav.image2drawio',
        descriptionKey: 'app.navTooltip.image2drawio',
        icon: 'network',
        accent: 'from-amber-500 via-primary-500 to-primary-700',
      },
      {
        page: 'ppt2polish',
        titleKey: 'app.nav.ppt2polish',
        descriptionKey: 'app.navTooltip.ppt2polish',
        icon: 'sparkles',
        accent: 'from-primary-700 via-primary-500 to-amber-500',
      },
    ],
  },
];
