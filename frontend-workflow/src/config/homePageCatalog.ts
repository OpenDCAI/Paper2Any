export type HomeNavigablePage =
  | 'paper2figure-tech-exp'
  | 'paper2figure-model-drawio'
  | 'paper2drawio-ai'
  | 'mindmap'
  | 'paper2ppt'
  | 'paper2ppt-image'
  | 'paper2ppt-frontend'
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

export type HomePreviewKind = 'image' | 'gif' | 'video';

export type HomeIconKey =
  | 'sparkles'
  | 'presentation'
  | 'video'
  | 'gitBranch'
  | 'brainCircuit'
  | 'network'
  | 'layoutTemplate'
  | 'fileStack'
  | 'fileImage'
  | 'fileSearch'
  | 'messageSquare'
  | 'bookOpen'
  | 'folderKanban';

export interface HomePreviewAsset {
  kind: HomePreviewKind;
  src: string;
  poster?: string;
}

export interface HomeFeatureCard {
  page: HomeNavigablePage;
  titleKey: string;
  descriptionKey: string;
  badgeKey: string;
  icon: HomeIconKey;
  accent: string;
  preview?: HomePreviewAsset;
}

export interface HomeFeatureSection {
  titleKey: string;
  descriptionKey: string;
  cards: HomeFeatureCard[];
}

export const featuredHomeCards: HomeFeatureCard[] = [
  {
    page: 'paper2figure-model-drawio',
    titleKey: 'app.home.cards.paper2figureModel.title',
    descriptionKey: 'app.home.cards.paper2figureModel.description',
    badgeKey: 'app.home.cards.paper2figureModel.badge',
    icon: 'gitBranch',
    accent: 'from-sky-500/80 via-cyan-400/70 to-teal-300/70',
    preview: {
      kind: 'image',
      src: '/home-previews/paper2figure-model.png',
    },
  },
  {
    page: 'paper2ppt-image',
    titleKey: 'app.home.cards.paper2pptImage.title',
    descriptionKey: 'app.home.cards.paper2pptImage.description',
    badgeKey: 'app.home.cards.paper2pptImage.badge',
    icon: 'presentation',
    accent: 'from-fuchsia-500/80 via-pink-400/70 to-rose-300/70',
    preview: {
      kind: 'image',
      src: '/home-previews/paper2ppt.png',
    },
  },
  {
    page: 'paper2ppt-frontend',
    titleKey: 'app.home.cards.paper2pptFrontend.title',
    descriptionKey: 'app.home.cards.paper2pptFrontend.description',
    badgeKey: 'app.home.cards.paper2pptFrontend.badge',
    icon: 'presentation',
    accent: 'from-amber-500/80 via-orange-400/70 to-yellow-300/70',
    preview: {
      kind: 'image',
      src: '/home-previews/paper2ppt-frontend.png',
    },
  },
  {
    page: 'paper2video',
    titleKey: 'app.home.cards.paper2video.title',
    descriptionKey: 'app.home.cards.paper2video.description',
    badgeKey: 'app.home.cards.paper2video.badge',
    icon: 'video',
    accent: 'from-emerald-500/80 via-teal-400/70 to-cyan-300/70',
    preview: {
      kind: 'video',
      src: '/home-previews/paper2video.mp4',
      poster: '/home-previews/paper2video-poster.png',
    },
  },
];

export const homeFeatureSections: HomeFeatureSection[] = [
  {
    titleKey: 'app.home.sections.creation.title',
    descriptionKey: 'app.home.sections.creation.description',
    cards: [
      {
        page: 'paper2figure-tech-exp',
        titleKey: 'app.home.cards.paper2figureTechExp.title',
        descriptionKey: 'app.home.cards.paper2figureTechExp.description',
        badgeKey: 'app.home.cards.paper2figureTechExp.badge',
        icon: 'sparkles',
        accent: 'from-emerald-500/75 via-teal-400/65 to-cyan-300/60',
        preview: {
          kind: 'image',
          src: '/home-previews/paper2figure-tech-route-alt.png',
        },
      },
      {
        page: 'paper2drawio-ai',
        titleKey: 'app.home.cards.paper2drawioAi.title',
        descriptionKey: 'app.home.cards.paper2drawioAi.description',
        badgeKey: 'app.home.cards.paper2drawioAi.badge',
        icon: 'network',
        accent: 'from-violet-500/75 via-fuchsia-400/65 to-pink-300/60',
        preview: {
          kind: 'gif',
          src: '/home-previews/paper2drawio-ai.gif',
        },
      },
      {
        page: 'mindmap',
        titleKey: 'app.home.cards.mindmap.title',
        descriptionKey: 'app.home.cards.mindmap.description',
        badgeKey: 'app.home.cards.mindmap.badge',
        icon: 'brainCircuit',
        accent: 'from-cyan-500/75 via-sky-400/65 to-indigo-300/60',
        preview: {
          kind: 'image',
          src: '/home-previews/mindmap-home.png',
        },
      },
      {
        page: 'paper2poster',
        titleKey: 'app.home.cards.paper2poster.title',
        descriptionKey: 'app.home.cards.paper2poster.description',
        badgeKey: 'app.home.cards.paper2poster.badge',
        icon: 'layoutTemplate',
        accent: 'from-orange-500/75 via-amber-400/65 to-yellow-300/60',
        preview: {
          kind: 'image',
          src: '/home-previews/paper2poster-cover.png',
        },
      },
      {
        page: 'paper2ppt-image',
        titleKey: 'app.home.cards.paper2pptImage.title',
        descriptionKey: 'app.home.cards.paper2pptImage.description',
        badgeKey: 'app.home.cards.paper2pptImage.badge',
        icon: 'presentation',
        accent: 'from-fuchsia-500/75 via-pink-400/65 to-rose-300/60',
        preview: {
          kind: 'image',
          src: '/home-previews/paper2ppt.png',
        },
      },
      {
        page: 'paper2ppt-frontend',
        titleKey: 'app.home.cards.paper2pptFrontend.title',
        descriptionKey: 'app.home.cards.paper2pptFrontend.description',
        badgeKey: 'app.home.cards.paper2pptFrontend.badge',
        icon: 'presentation',
        accent: 'from-amber-500/75 via-orange-400/65 to-yellow-300/60',
        preview: {
          kind: 'image',
          src: '/home-previews/paper2ppt-frontend.png',
        },
      },
    ],
  },
  {
    titleKey: 'app.home.sections.conversion.title',
    descriptionKey: 'app.home.sections.conversion.description',
    cards: [
      {
        page: 'pdf2ppt',
        titleKey: 'app.home.cards.pdf2ppt.title',
        descriptionKey: 'app.home.cards.pdf2ppt.description',
        badgeKey: 'app.home.cards.pdf2ppt.badge',
        icon: 'fileStack',
        accent: 'from-amber-500/75 via-orange-400/65 to-rose-300/60',
        preview: {
          kind: 'image',
          src: '/home-previews/pdf2ppt.png',
        },
      },
      {
        page: 'image2ppt',
        titleKey: 'app.home.cards.image2ppt.title',
        descriptionKey: 'app.home.cards.image2ppt.description',
        badgeKey: 'app.home.cards.image2ppt.badge',
        icon: 'fileImage',
        accent: 'from-cyan-500/75 via-sky-400/65 to-blue-300/60',
        preview: {
          kind: 'image',
          src: '/home-previews/image2ppt-cover.png',
        },
      },
      {
        page: 'image2drawio',
        titleKey: 'app.home.cards.image2drawio.title',
        descriptionKey: 'app.home.cards.image2drawio.description',
        badgeKey: 'app.home.cards.image2drawio.badge',
        icon: 'network',
        accent: 'from-lime-500/75 via-emerald-400/65 to-teal-300/60',
        preview: {
          kind: 'image',
          src: '/home-previews/image2drawio.png',
        },
      },
      {
        page: 'ppt2polish',
        titleKey: 'app.home.cards.ppt2polish.title',
        descriptionKey: 'app.home.cards.ppt2polish.description',
        badgeKey: 'app.home.cards.ppt2polish.badge',
        icon: 'sparkles',
        accent: 'from-blue-500/75 via-indigo-400/65 to-violet-300/60',
        preview: {
          kind: 'image',
          src: '/home-previews/ppt2polish.png',
        },
      },
    ],
  },
  {
    titleKey: 'app.home.sections.research.title',
    descriptionKey: 'app.home.sections.research.description',
    cards: [
      {
        page: 'paper2citation',
        titleKey: 'app.home.cards.paper2citation.title',
        descriptionKey: 'app.home.cards.paper2citation.description',
        badgeKey: 'app.home.cards.paper2citation.badge',
        icon: 'fileSearch',
        accent: 'from-cyan-500/75 via-sky-400/65 to-indigo-300/60',
        preview: {
          kind: 'image',
          src: '/home-previews/paper2citation-cover.png',
        },
      },
      {
        page: 'paper2rebuttal',
        titleKey: 'app.home.cards.paper2rebuttal.title',
        descriptionKey: 'app.home.cards.paper2rebuttal.description',
        badgeKey: 'app.home.cards.paper2rebuttal.badge',
        icon: 'messageSquare',
        accent: 'from-rose-500/75 via-pink-400/65 to-fuchsia-300/60',
        preview: {
          kind: 'image',
          src: '/home-previews/paper2rebuttal-cover.png',
        },
      },
      {
        page: 'files',
        titleKey: 'app.home.cards.files.title',
        descriptionKey: 'app.home.cards.files.description',
        badgeKey: 'app.home.cards.files.badge',
        icon: 'folderKanban',
        accent: 'from-emerald-500/75 via-green-400/65 to-lime-300/60',
      },
    ],
  },
];
