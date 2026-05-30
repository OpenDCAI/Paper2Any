import {
  ArrowRight,
  BookOpen,
  BrainCircuit,
  Check,
  FileImage,
  FileSearch,
  FileStack,
  Flame,
  FolderKanban,
  GitBranch,
  LayoutTemplate,
  MessageSquare,
  Network,
  Presentation,
  Sparkles,
  Video,
  Wand2,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { featuredHomeCards, homeFeatureSections, HomeFeatureCard, HomeNavigablePage } from '../config/homePageCatalog';

type ActivePage =
  | 'home'
  | HomeNavigablePage;

interface HomePageProps {
  onNavigate: (page: ActivePage) => void;
}

const iconMap = {
  sparkles: Sparkles,
  presentation: Presentation,
  video: Video,
  gitBranch: GitBranch,
  brainCircuit: BrainCircuit,
  network: Network,
  layoutTemplate: LayoutTemplate,
  fileStack: FileStack,
  fileImage: FileImage,
  fileSearch: FileSearch,
  messageSquare: MessageSquare,
  bookOpen: BookOpen,
  folderKanban: FolderKanban,
  flame: Flame,
} as const;

const stats = [
  { value: '14+', labelKey: 'app.home.stats.workflows' },
  { value: '4', labelKey: 'app.home.stats.stages' },
  { value: '16', labelKey: 'app.home.stats.batchImages' },
  { value: '1', labelKey: 'app.home.stats.console' },
] as const;

const modulePages: HomeNavigablePage[] = [
  'paper2figure-tech-exp',
  'paper2ppt-image',
  'paper2ppt-frontend',
  'paper2drawio-ai',
  'pdf2ppt',
  'paper2rebuttal',
];

const highlightPages: HomeNavigablePage[] = [
  'paper2ppt-frontend',
  'paper2figure-tech-exp',
  'paper2poster',
];

const displayFont = {
  fontFamily: '"Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif',
};

function findCard(page: HomeNavigablePage): HomeFeatureCard | undefined {
  return [...featuredHomeCards, ...homeFeatureSections.flatMap((section) => section.cards)].find((card) => card.page === page);
}

function HomePreview({ card, className = '' }: { card: HomeFeatureCard; className?: string }) {
  if (card.preview?.kind === 'video') {
    return (
      <video
        src={card.preview.src}
        poster={card.preview.poster}
        className={`h-full w-full object-cover ${className}`}
        autoPlay
        muted
        loop
        playsInline
        preload="metadata"
      />
    );
  }

  if (card.preview) {
    return (
      <img
        src={card.preview.src}
        alt=""
        className={`h-full w-full object-cover ${className}`}
      />
    );
  }

  const Icon = iconMap[card.icon];
  return (
    <div className={`flex h-full w-full items-center justify-center bg-[#11161f] ${className}`}>
      <div className="flex size-16 items-center justify-center rounded-lg border border-white/10 bg-white/5 text-[#d6b15f]">
        <Icon size={28} />
      </div>
    </div>
  );
}

function HighlightCard({
  card,
  onNavigate,
  compact = false,
}: {
  card: HomeFeatureCard;
  onNavigate: (page: ActivePage) => void;
  compact?: boolean;
}) {
  const { t } = useTranslation('common');
  const Icon = iconMap[card.icon];

  return (
    <button
      type="button"
      onClick={() => onNavigate(card.page)}
      className={`group overflow-hidden border border-white/10 bg-[#0f141b] text-left transition duration-200 hover:border-[#d6b15f]/40 hover:bg-[#141a22] ${
        compact ? 'rounded-lg' : 'rounded-xl'
      }`}
    >
      <div className={`relative overflow-hidden ${compact ? 'h-40' : 'h-[26rem]'}`}>
        <HomePreview card={card} className="transition duration-300 group-hover:scale-[1.02]" />
        <div className="absolute inset-0 bg-[linear-gradient(to_top,rgba(5,7,10,0.88),rgba(5,7,10,0.1),rgba(5,7,10,0))]" />
        <div className="absolute left-4 top-4 inline-flex items-center gap-2 rounded-md border border-white/10 bg-black/35 px-3 py-2 text-xs font-medium text-[#d8d0c2] backdrop-blur-sm">
          <Icon size={14} className="text-[#d6b15f]" />
          <span>{t(card.badgeKey)}</span>
        </div>
        <div className="absolute inset-x-0 bottom-0 p-4">
          <div className={`${compact ? 'text-lg' : 'text-[1.65rem]'} font-semibold leading-tight text-white text-balance`} style={displayFont}>
            {t(card.titleKey)}
          </div>
          <p className={`mt-2 max-w-xl text-pretty ${compact ? 'text-sm leading-6' : 'text-base leading-7'} text-[#b5bdc9]`}>
            {t(card.descriptionKey)}
          </p>
          <div className="mt-4 inline-flex items-center gap-2 text-sm font-medium text-[#d6b15f]">
            <span>{t('app.home.cardAction')}</span>
            <ArrowRight size={15} className="transition duration-200 group-hover:translate-x-1" />
          </div>
        </div>
      </div>
    </button>
  );
}

function ModuleButton({ page, onNavigate }: { page: HomeNavigablePage; onNavigate: (page: ActivePage) => void }) {
  const { t } = useTranslation('common');
  const card = findCard(page);
  if (!card) return null;

  const Icon = iconMap[card.icon];

  return (
    <button
      type="button"
      onClick={() => onNavigate(page)}
      className="group flex items-start gap-4 border-b border-white/10 py-4 text-left transition duration-200 last:border-b-0 hover:border-[#d6b15f]/25"
    >
      <div className="flex size-11 shrink-0 items-center justify-center rounded-lg border border-white/10 bg-white/5 text-[#d6b15f] transition duration-200 group-hover:border-[#d6b15f]/40 group-hover:bg-[#151b22]">
        <Icon size={20} />
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex items-center justify-between gap-3">
          <h3 className="text-base font-semibold text-white">{t(card.titleKey)}</h3>
          <ArrowRight size={15} className="shrink-0 text-[#6d7683] transition duration-200 group-hover:translate-x-1 group-hover:text-[#d6b15f]" />
        </div>
        <p className="mt-2 text-sm leading-6 text-[#98a0ac] text-pretty">{t(card.descriptionKey)}</p>
      </div>
    </button>
  );
}

function WorkflowCard({ card, onNavigate }: { card: HomeFeatureCard; onNavigate: (page: ActivePage) => void }) {
  const { t } = useTranslation('common');
  const Icon = iconMap[card.icon];

  return (
    <button
      type="button"
      onClick={() => onNavigate(card.page)}
      className="group overflow-hidden rounded-lg border border-white/10 bg-[#0f141b] text-left transition duration-200 hover:border-[#d6b15f]/40 hover:bg-[#141a22]"
    >
      <div className="relative h-44 overflow-hidden border-b border-white/10">
        <HomePreview card={card} className="transition duration-300 group-hover:scale-[1.03]" />
        <div className="absolute inset-0 bg-[linear-gradient(to_top,rgba(6,8,11,0.75),rgba(6,8,11,0.05),rgba(6,8,11,0))]" />
        <div className="absolute left-3 top-3 inline-flex items-center gap-2 rounded-md bg-black/35 px-2.5 py-1.5 text-xs text-[#d8d0c2] backdrop-blur-sm">
          <Icon size={13} className="text-[#d6b15f]" />
          <span>{t(card.badgeKey)}</span>
        </div>
      </div>
      <div className="p-4">
        <h3 className="text-lg font-semibold leading-6 text-white text-balance">{t(card.titleKey)}</h3>
        <p className="mt-2 text-sm leading-6 text-[#9aa3af] text-pretty">{t(card.descriptionKey)}</p>
      </div>
    </button>
  );
}

export function HomePage({ onNavigate }: HomePageProps) {
  const { t } = useTranslation('common');
  const leadCard = findCard(highlightPages[0]);
  const supportCards = highlightPages.slice(1).map(findCard).filter(Boolean) as HomeFeatureCard[];

  return (
    <div className="h-full overflow-y-auto bg-[#05070a] text-[#e7ebf1]">
      <div className="min-h-full border-t border-white/5 bg-[linear-gradient(to_bottom,rgba(255,255,255,0.015),rgba(255,255,255,0))]">
        <section className="border-b border-white/8">
          <div className="mx-auto max-w-7xl px-5 pb-14 pt-10 md:px-8 lg:px-10">
            <div className="grid gap-8 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] xl:items-start">
              <div>
                <div className="inline-flex items-center gap-2 rounded-md border border-[#d6b15f]/25 bg-[#18140c] px-3 py-2 text-xs font-medium text-[#d6b15f]">
                  <Sparkles size={14} />
                  <span>{t('app.home.kicker')}</span>
                </div>

                <h2
                  className="mt-6 max-w-3xl text-5xl leading-[0.96] text-white text-balance md:text-7xl"
                  style={displayFont}
                >
                  {t('app.home.title')}
                </h2>

                <p className="mt-5 max-w-3xl text-base leading-7 text-[#a8b1bc] text-pretty md:text-lg">
                  {t('app.home.description')}
                </p>

                <div className="mt-8 flex flex-col gap-3 sm:flex-row">
                  <button
                    type="button"
                    onClick={() => onNavigate('paper2figure-tech-exp')}
                    className="inline-flex items-center justify-center gap-2 rounded-lg bg-[#d6b15f] px-6 py-3.5 text-sm font-semibold text-[#14110a] transition duration-200 hover:bg-[#e1bf73]"
                  >
                    <span>{t('app.home.primaryCta')}</span>
                    <ArrowRight size={16} />
                  </button>
                  <button
                    type="button"
                    onClick={() => onNavigate('paper2ppt-frontend')}
                    className="inline-flex items-center justify-center gap-2 rounded-lg border border-white/10 bg-white/5 px-6 py-3.5 text-sm font-semibold text-white transition duration-200 hover:border-[#d6b15f]/40 hover:bg-white/10"
                  >
                    <Presentation size={16} className="text-[#d6b15f]" />
                    <span>{t('app.home.frontendCta')}</span>
                  </button>
                </div>

                <div className="mt-10 grid gap-px overflow-hidden rounded-lg border border-white/10 bg-white/10 sm:grid-cols-2 xl:grid-cols-4">
                  {stats.map((stat) => (
                    <div key={stat.labelKey} className="bg-[#0b0f14] px-4 py-4">
                      <div className="text-3xl font-semibold text-white tabular-nums">{stat.value}</div>
                      <div className="mt-1 text-sm text-[#8f98a5]">{t(stat.labelKey)}</div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="grid gap-4 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)]">
                {leadCard ? (
                  <HighlightCard card={leadCard} onNavigate={onNavigate} />
                ) : null}
                <div className="grid gap-4">
                  {supportCards.map((card) => (
                    <HighlightCard key={card.page} card={card} onNavigate={onNavigate} compact />
                  ))}
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="border-b border-white/8">
          <div className="mx-auto grid max-w-7xl gap-10 px-5 py-14 md:px-8 lg:grid-cols-[minmax(0,0.75fr)_minmax(0,1.25fr)] lg:px-10">
            <div>
              <p className="text-sm font-medium text-[#d6b15f]">{t('app.home.workflowTitle')}</p>
              <h3 className="mt-3 max-w-xl text-4xl leading-tight text-white text-balance md:text-5xl" style={displayFont}>
                {t('app.home.workflowTitle')}
              </h3>
              <p className="mt-4 max-w-xl text-sm leading-7 text-[#9aa3af] text-pretty md:text-base">
                {t('app.home.outputDescription')}
              </p>
              <button
                type="button"
                onClick={() => onNavigate('paper2ppt-image')}
                className="mt-8 inline-flex items-center gap-2 rounded-lg border border-white/10 bg-white/5 px-5 py-3 text-sm font-semibold text-white transition duration-200 hover:border-[#d6b15f]/40 hover:bg-white/10"
              >
                <span>{t('app.home.secondaryCta')}</span>
                <ArrowRight size={15} />
              </button>
            </div>

            <div className="grid gap-px overflow-hidden rounded-lg border border-white/10 bg-white/10 lg:grid-cols-3">
              {[
                ['app.home.pipelineStep1Title', 'app.home.pipelineStep1Detail'],
                ['app.home.pipelineStep2Title', 'app.home.pipelineStep2Detail'],
                ['app.home.pipelineStep3Title', 'app.home.pipelineStep3Detail'],
              ].map(([titleKey, detailKey], index) => (
                <div key={titleKey} className="bg-[#0b0f14] p-5">
                  <div className="flex size-9 items-center justify-center rounded-lg border border-[#d6b15f]/25 bg-[#18140c] text-sm font-semibold text-[#d6b15f]">
                    {index + 1}
                  </div>
                  <div className="mt-4 text-lg font-semibold text-white">{t(titleKey)}</div>
                  <div className="mt-2 text-sm leading-6 text-[#98a0ac]">{t(detailKey)}</div>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="border-b border-white/8">
          <div className="mx-auto max-w-7xl px-5 py-14 md:px-8 lg:px-10">
            <div className="grid gap-10 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
              <div>
                <div className="text-sm font-medium text-[#d6b15f]">{t('app.home.modulesTitle')}</div>
                <h3 className="mt-3 text-4xl leading-tight text-white text-balance md:text-5xl" style={displayFont}>
                  {t('app.home.modulesTitle')}
                </h3>
                <p className="mt-4 max-w-xl text-sm leading-7 text-[#9aa3af] text-pretty md:text-base">
                  {t('app.home.modulesDescription')}
                </p>
              </div>
              <div className="rounded-lg border border-white/10 bg-[#0b0f14] px-5 py-3">
                {modulePages.map((page) => (
                  <ModuleButton key={page} page={page} onNavigate={onNavigate} />
                ))}
              </div>
            </div>
          </div>
        </section>

        {homeFeatureSections.map((section) => (
          <section key={section.titleKey} className="border-b border-white/8 last:border-b-0">
            <div className="mx-auto max-w-7xl px-5 py-14 md:px-8 lg:px-10">
              <div className="max-w-3xl">
                <div className="text-sm font-medium text-[#d6b15f]">{t(section.titleKey)}</div>
                <h3 className="mt-3 text-3xl leading-tight text-white text-balance md:text-4xl" style={displayFont}>
                  {t(section.titleKey)}
                </h3>
                <p className="mt-4 text-sm leading-7 text-[#9aa3af] text-pretty md:text-base">
                  {t(section.descriptionKey)}
                </p>
              </div>
              <div className="mt-8 grid gap-5 md:grid-cols-2 xl:grid-cols-3">
                {section.cards.map((card) => (
                  <WorkflowCard key={card.page} card={card} onNavigate={onNavigate} />
                ))}
              </div>
            </div>
          </section>
        ))}

        <section className="mx-auto max-w-7xl px-5 py-14 md:px-8 lg:px-10">
          <div className="grid gap-8 rounded-lg border border-white/10 bg-[#0b0f14] px-6 py-8 lg:grid-cols-[minmax(0,1fr)_auto] lg:items-center">
            <div>
              <div className="text-sm font-medium text-[#d6b15f]">{t('app.home.ctaTitle')}</div>
              <h3 className="mt-3 text-3xl leading-tight text-white text-balance md:text-4xl" style={displayFont}>
                {t('app.home.ctaTitle')}
              </h3>
              <p className="mt-4 max-w-2xl text-sm leading-7 text-[#9aa3af] text-pretty md:text-base">
                {t('app.home.ctaDescription')}
              </p>
            </div>
            <div className="flex flex-col gap-3 sm:flex-row lg:flex-col">
              <button
                type="button"
                onClick={() => onNavigate('paper2ppt-image')}
                className="rounded-lg bg-[#d6b15f] px-6 py-3.5 text-sm font-semibold text-[#14110a] transition duration-200 hover:bg-[#e1bf73]"
              >
                {t('app.home.secondaryCta')}
              </button>
              <button
                type="button"
                onClick={() => onNavigate('paper2rebuttal')}
                className="rounded-lg border border-white/10 bg-white/5 px-6 py-3.5 text-sm font-semibold text-white transition duration-200 hover:border-[#d6b15f]/40 hover:bg-white/10"
              >
                {t('app.home.rebuttalCta')}
              </button>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
