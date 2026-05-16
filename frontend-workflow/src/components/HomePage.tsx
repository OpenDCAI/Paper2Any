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

const moduleCards = [
  { page: 'paper2figure-model-drawio', icon: GitBranch, accent: 'text-violet-300', bg: 'bg-violet-400/10', border: 'border-violet-300/20' },
  { page: 'paper2ppt-image', icon: Presentation, accent: 'text-pink-300', bg: 'bg-pink-400/10', border: 'border-pink-300/20' },
  { page: 'pdf2ppt', icon: FileStack, accent: 'text-cyan-300', bg: 'bg-cyan-400/10', border: 'border-cyan-300/20' },
  { page: 'ppt2polish', icon: Wand2, accent: 'text-fuchsia-200', bg: 'bg-fuchsia-300/10', border: 'border-fuchsia-200/20' },
] as const;

const stats = [
  { value: '14+', labelKey: 'app.home.stats.workflows' },
  { value: '4', labelKey: 'app.home.stats.stages' },
  { value: '16', labelKey: 'app.home.stats.batchImages' },
  { value: '1', labelKey: 'app.home.stats.console' },
] as const;

function HomePreview({ card }: { card: HomeFeatureCard }) {
  if (card.preview?.kind === 'video') {
    return (
      <video
        src={card.preview.src}
        poster={card.preview.poster}
        className="h-full w-full object-cover opacity-85 transition duration-700 group-hover:scale-[1.04]"
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
        className="h-full w-full object-cover opacity-85 transition duration-700 group-hover:scale-[1.04]"
      />
    );
  }

  const Icon = iconMap[card.icon];
  return (
    <div className={`flex h-full w-full items-center justify-center bg-gradient-to-br ${card.accent}`}>
      <div className="rounded-2xl border border-white/20 bg-black/20 p-4 text-white backdrop-blur-xl">
        <Icon size={34} />
      </div>
    </div>
  );
}

function FeatureCard({ card, onNavigate }: { card: HomeFeatureCard; onNavigate: (page: ActivePage) => void }) {
  const { t } = useTranslation('common');
  const Icon = iconMap[card.icon];

  return (
    <button
      type="button"
      onClick={() => onNavigate(card.page)}
      className="group flex h-full flex-col rounded-[28px] border border-white/10 bg-white/[0.035] p-4 text-left shadow-[0_24px_90px_rgba(0,0,0,0.28)] backdrop-blur-2xl transition duration-300 hover:-translate-y-1 hover:border-violet-300/35 hover:bg-white/[0.055] hover:shadow-[0_28px_110px_rgba(160,120,255,0.14)]"
    >
      <div className="relative h-40 overflow-hidden rounded-[22px] border border-white/10 bg-slate-950/70">
        <HomePreview card={card} />
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-slate-950/5 to-slate-950/75" />
        <div className={`absolute inset-x-0 bottom-0 h-20 bg-gradient-to-t ${card.accent} opacity-55 blur-2xl`} />
        <div className="absolute left-3 top-3 inline-flex items-center gap-2 rounded-full border border-white/12 bg-black/35 px-3 py-1.5 text-[10px] font-semibold uppercase text-white/75 backdrop-blur-xl">
          <Icon size={13} />
          <span>{t(card.badgeKey)}</span>
        </div>
      </div>

      <div className="flex flex-1 flex-col px-1 pb-1 pt-4">
        <h3 className="text-lg font-semibold leading-6 text-white">{t(card.titleKey)}</h3>
        <p className="mt-2 flex-1 text-sm leading-6 text-slate-300/75">{t(card.descriptionKey)}</p>
        <div className="mt-5 inline-flex items-center gap-2 text-sm font-semibold text-violet-200 transition group-hover:text-white">
          <span>{t('app.home.cardAction')}</span>
          <ArrowRight size={16} className="transition group-hover:translate-x-1" />
        </div>
      </div>
    </button>
  );
}

function ModuleCard({ page, icon: Icon, accent, bg, border, onNavigate }: {
  page: HomeNavigablePage;
  icon: typeof Sparkles;
  accent: string;
  bg: string;
  border: string;
  onNavigate: (page: ActivePage) => void;
}) {
  const { t } = useTranslation('common');
  const card = [...featuredHomeCards, ...homeFeatureSections.flatMap((section) => section.cards)].find((item) => item.page === page);
  if (!card) return null;

  return (
    <button
      type="button"
      onClick={() => onNavigate(page)}
      className="group flex min-h-[268px] flex-col rounded-2xl border border-white/10 bg-white/[0.035] p-6 text-left backdrop-blur-2xl transition duration-300 hover:-translate-y-1 hover:border-violet-300/35 hover:shadow-[0_0_32px_rgba(160,120,255,0.14)]"
    >
      <div className={`mb-6 flex h-12 w-12 items-center justify-center rounded-xl border ${border} ${bg} ${accent} transition group-hover:scale-110`}>
        <Icon size={24} />
      </div>
      <h3 className="text-xl font-semibold text-white">{t(card.titleKey)}</h3>
      <p className="mt-3 flex-1 text-sm leading-6 text-slate-300/72">{t(card.descriptionKey)}</p>
      <div className={`mt-6 inline-flex items-center gap-2 text-sm font-semibold ${accent}`}>
        <span>{t('app.home.cardAction')}</span>
        <ArrowRight size={16} className="transition group-hover:translate-x-1" />
      </div>
    </button>
  );
}

function DashboardPreview({ onNavigate }: { onNavigate: (page: ActivePage) => void }) {
  const { t } = useTranslation('common');
  const previewCards = featuredHomeCards.filter((card) => card.page !== 'image-playground');

  return (
    <div className="relative mx-auto mt-14 max-w-5xl">
      <div className="absolute -inset-1 rounded-[30px] bg-gradient-to-r from-violet-500/24 via-pink-500/18 to-cyan-400/20 blur-xl" />
      <div className="relative overflow-hidden rounded-[30px] border border-white/10 bg-slate-950/70 shadow-2xl shadow-violet-950/35 backdrop-blur-2xl">
        <div className="flex items-center justify-between border-b border-white/10 px-5 py-4">
          <div className="flex items-center gap-2">
            <span className="h-3 w-3 rounded-full bg-rose-400" />
            <span className="h-3 w-3 rounded-full bg-amber-300" />
            <span className="h-3 w-3 rounded-full bg-emerald-400" />
          </div>
          <div className="hidden rounded-full border border-white/10 bg-white/5 px-4 py-1.5 text-xs font-medium text-slate-300 sm:block">
            {t('app.home.dashboardTitle')}
          </div>
        </div>

        <div className="grid gap-0 lg:grid-cols-[0.9fr_1.45fr]">
          <div className="border-b border-white/10 p-5 lg:border-b-0 lg:border-r">
            <div className="rounded-2xl border border-white/10 bg-white/[0.035] p-4">
              <div className="text-xs font-semibold uppercase text-violet-200/80">{t('app.home.pipelineLabel')}</div>
              <div className="mt-4 space-y-3">
                {[
                  ['app.home.pipelineStep1Title', 'app.home.pipelineStep1Detail'],
                  ['app.home.pipelineStep2Title', 'app.home.pipelineStep2Detail'],
                  ['app.home.pipelineStep3Title', 'app.home.pipelineStep3Detail'],
                ].map(([title, detail], index) => (
                  <div key={title} className="flex items-start gap-3">
                    <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-violet-300/25 bg-violet-400/10 text-xs font-bold text-violet-100">
                      {index + 1}
                    </div>
                    <div>
                      <div className="text-sm font-semibold text-white">{t(title)}</div>
                      <div className="mt-1 text-xs text-slate-400">{t(detail)}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <button
              type="button"
              onClick={() => onNavigate('image-playground')}
              className="mt-4 flex w-full items-center justify-between rounded-2xl border border-orange-300/20 bg-orange-400/10 px-4 py-3 text-left transition hover:bg-orange-400/15"
            >
              <div>
                <div className="text-sm font-semibold text-white">{t('app.home.imagePlaygroundTitle')}</div>
                <div className="mt-1 text-xs text-orange-100/70">{t('app.home.imagePlaygroundDescription')}</div>
              </div>
              <Flame size={18} className="text-orange-200" />
            </button>
          </div>

          <div className="grid gap-4 p-5 sm:grid-cols-2">
            {previewCards.map((card) => (
              <button
                type="button"
                key={card.page}
                onClick={() => onNavigate(card.page)}
                className="group min-h-[220px] overflow-hidden rounded-2xl border border-white/10 bg-white/[0.035] text-left transition hover:border-violet-300/30"
              >
                <div className="relative h-32 overflow-hidden">
                  <HomePreview card={card} />
                  <div className="absolute inset-0 bg-gradient-to-t from-slate-950/88 to-transparent" />
                </div>
                <div className="p-4">
                  <div className="text-sm font-semibold leading-5 text-white">{t(card.titleKey)}</div>
                  <div className="mt-2 flex items-center gap-2 text-xs font-medium text-violet-200">
                    <span>{t(card.badgeKey)}</span>
                    <ArrowRight size={13} className="transition group-hover:translate-x-1" />
                  </div>
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export function HomePage({ onNavigate }: HomePageProps) {
  const { t } = useTranslation('common');

  return (
    <div className="h-full overflow-y-auto overflow-x-hidden bg-[#0b1326] text-[#dae2fd]">
      <div className="relative min-h-full">
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_0%_0%,rgba(109,59,215,0.18),transparent_36%),radial-gradient(circle_at_100%_0%,rgba(255,176,205,0.12),transparent_32%),radial-gradient(circle_at_50%_100%,rgba(76,215,246,0.12),transparent_38%)]" />

        <section className="relative px-5 pb-20 pt-10 md:px-8 lg:px-10">
          <div className="mx-auto max-w-7xl text-center">
            <div className="inline-flex items-center gap-2 rounded-full border border-violet-400/20 bg-violet-400/10 px-4 py-2 text-xs font-semibold uppercase text-violet-200">
              <Sparkles size={15} />
              <span>{t('app.home.kicker')}</span>
            </div>
            <h2 className="mx-auto mt-8 max-w-5xl text-5xl font-extrabold leading-[0.98] text-white md:text-7xl lg:text-8xl">
              <span className="bg-gradient-to-r from-violet-300 via-fuchsia-300 to-pink-300 bg-clip-text text-transparent">
                Paper2Any
              </span>
            </h2>
            <p className="mx-auto mt-6 max-w-3xl text-base leading-7 text-slate-300 md:text-lg">
              {t('app.home.description')}
            </p>

            <div className="mt-9 flex flex-col items-center justify-center gap-3 sm:flex-row">
              <button
                type="button"
                onClick={() => onNavigate('paper2figure-tech-exp')}
                className="inline-flex items-center gap-2 rounded-xl bg-[#a078ff] px-7 py-4 text-sm font-bold text-[#23005c] shadow-xl shadow-violet-500/20 transition hover:brightness-110 active:scale-[0.98]"
              >
                <span>{t('app.home.primaryCta')}</span>
                <ArrowRight size={18} />
              </button>
              <button
                type="button"
                onClick={() => onNavigate('paper2ppt-image')}
                className="inline-flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-7 py-4 text-sm font-semibold text-white backdrop-blur-xl transition hover:bg-white/10 active:scale-[0.98]"
              >
                <Presentation size={17} />
                <span>{t('app.home.secondaryCta')}</span>
              </button>
            </div>

            <DashboardPreview onNavigate={onNavigate} />
          </div>
        </section>

        <section className="relative mx-auto max-w-7xl px-5 py-20 md:px-8 lg:px-10">
          <div className="mb-12 text-center">
            <h3 className="text-3xl font-bold text-white md:text-5xl">{t('app.home.modulesTitle')}</h3>
            <p className="mx-auto mt-4 max-w-2xl text-sm leading-6 text-slate-400 md:text-base">{t('app.home.modulesDescription')}</p>
          </div>
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
            {moduleCards.map((card) => (
              <ModuleCard key={card.page} {...card} onNavigate={onNavigate} />
            ))}
          </div>
        </section>

        <section className="relative border-y border-white/5 bg-[#131b2e]/45 py-14">
          <div className="mx-auto grid max-w-7xl grid-cols-2 gap-8 px-5 text-center md:px-8 lg:grid-cols-4 lg:px-10">
            {stats.map((stat) => (
              <div key={stat.labelKey}>
                <div className="text-4xl font-extrabold text-white md:text-5xl">{stat.value}</div>
                <div className="mt-3 text-xs font-semibold uppercase text-slate-500">{t(stat.labelKey)}</div>
              </div>
            ))}
          </div>
        </section>

        <section className="relative mx-auto max-w-7xl px-5 py-20 md:px-8 lg:px-10">
          <div className="overflow-hidden rounded-[32px] border border-white/10 bg-white/[0.035] p-6 shadow-[0_30px_100px_rgba(0,0,0,0.3)] backdrop-blur-2xl md:p-10 lg:p-12">
            <div className="grid gap-10 lg:grid-cols-[0.92fr_1.08fr] lg:items-center">
              <div>
                <h3 className="text-3xl font-bold leading-tight text-white md:text-5xl">
                  {t('app.home.workflowTitle')}
                </h3>
                <div className="mt-8 space-y-5">
                  {[
                    ['app.home.workflowPoint1Title', 'app.home.workflowPoint1Description'],
                    ['app.home.workflowPoint2Title', 'app.home.workflowPoint2Description'],
                    ['app.home.workflowPoint3Title', 'app.home.workflowPoint3Description'],
                  ].map(([titleKey, descriptionKey]) => (
                    <div key={titleKey} className="flex items-start gap-4">
                      <div className="mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-violet-400/16 text-violet-200">
                        <Check size={16} />
                      </div>
                      <div>
                        <h4 className="text-sm font-semibold text-white">{t(titleKey)}</h4>
                        <p className="mt-1 text-sm leading-6 text-slate-400">{t(descriptionKey)}</p>
                      </div>
                    </div>
                  ))}
                </div>
                <button
                  type="button"
                  onClick={() => onNavigate('paper2ppt-frontend')}
                  className="mt-9 rounded-xl bg-white px-7 py-4 text-sm font-bold text-slate-950 transition hover:bg-violet-100 active:scale-[0.98]"
                >
                  {t('app.home.frontendCta')}
                </button>
              </div>

              <div className="relative min-h-[430px] overflow-hidden rounded-3xl border border-white/10 bg-gradient-to-br from-violet-500/18 via-slate-950/50 to-pink-500/14">
                <img
                  src="/home-previews/paper2ppt-frontend.png"
                  alt=""
                  className="absolute inset-0 h-full w-full object-cover opacity-45 mix-blend-screen"
                />
                <div className="absolute inset-0 bg-gradient-to-br from-slate-950/12 via-slate-950/36 to-slate-950/82" />
                <div className="absolute left-6 top-6 rounded-2xl border border-white/10 bg-black/30 px-4 py-3 backdrop-blur-xl">
                  <div className="flex items-center gap-2 text-sm font-semibold text-white">
                    <span className="h-2.5 w-2.5 rounded-full bg-emerald-400 animate-pulse" />
                    <span>{t('app.home.processingLabel')}</span>
                  </div>
                  <div className="mt-3 h-2 w-52 overflow-hidden rounded-full bg-white/10">
                    <div className="h-full w-2/3 rounded-full bg-gradient-to-r from-violet-400 to-pink-400" />
                  </div>
                </div>
                <div className="absolute bottom-6 right-6 max-w-xs rounded-2xl border border-violet-300/20 bg-black/35 p-5 backdrop-blur-xl">
                  <div className="flex items-center gap-2 text-xs font-semibold uppercase text-violet-100">
                    <Sparkles size={14} />
                    <span>{t('app.home.outputLabel')}</span>
                  </div>
                  <p className="mt-3 text-sm leading-6 text-slate-300">
                    {t('app.home.outputDescription')}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="relative mx-auto max-w-7xl space-y-12 px-5 pb-20 md:px-8 lg:px-10">
          {homeFeatureSections.map((section) => (
            <div key={section.titleKey}>
              <div className="mb-5 flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
                <div>
                  <h3 className="text-2xl font-bold text-white md:text-3xl">{t(section.titleKey)}</h3>
                  <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-400 md:text-base">{t(section.descriptionKey)}</p>
                </div>
              </div>
              <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
                {section.cards.map((card) => (
                  <FeatureCard key={card.page} card={card} onNavigate={onNavigate} />
                ))}
              </div>
            </div>
          ))}
        </section>

        <section className="relative px-5 pb-20 text-center md:px-8 lg:px-10">
          <div className="mx-auto max-w-3xl">
            <h3 className="text-3xl font-bold text-white md:text-5xl">{t('app.home.ctaTitle')}</h3>
            <p className="mx-auto mt-4 max-w-2xl text-base leading-7 text-slate-400">{t('app.home.ctaDescription')}</p>
            <div className="mt-9 flex flex-col justify-center gap-3 sm:flex-row">
              <button
                type="button"
                onClick={() => onNavigate('paper2ppt-image')}
                className="rounded-xl bg-gradient-to-r from-violet-500 to-pink-500 px-8 py-4 text-base font-bold text-white shadow-xl shadow-violet-500/25 transition hover:brightness-110 active:scale-[0.98]"
              >
                {t('app.home.secondaryCta')}
              </button>
              <button
                type="button"
                onClick={() => onNavigate('paper2rebuttal')}
                className="rounded-xl border border-white/10 bg-white/[0.035] px-8 py-4 text-base font-semibold text-white backdrop-blur-xl transition hover:bg-white/10 active:scale-[0.98]"
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
