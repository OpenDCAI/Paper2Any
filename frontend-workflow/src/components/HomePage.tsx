import {
  ArrowRight,
  FileImage,
  FileSearch,
  FileStack,
  FolderKanban,
  GitBranch,
  LayoutTemplate,
  MessageSquare,
  Network,
  Presentation,
  Sparkles,
  Video,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';
import {
  featuredHomeCards,
  homeFeatureSections,
  HomeFeatureCard,
  HomeNavigablePage,
} from '../config/homePageCatalog';

type ActivePage = 'home' | HomeNavigablePage;

interface HomePageProps {
  onNavigate: (page: ActivePage) => void;
}

const iconMap = {
  sparkles: Sparkles,
  presentation: Presentation,
  video: Video,
  gitBranch: GitBranch,
  network: Network,
  layoutTemplate: LayoutTemplate,
  fileStack: FileStack,
  fileImage: FileImage,
  fileSearch: FileSearch,
  messageSquare: MessageSquare,
  folderKanban: FolderKanban,
} as const;

function FeatureCard({
  card,
  onNavigate,
}: {
  card: HomeFeatureCard;
  onNavigate: (page: ActivePage) => void;
}) {
  const { t } = useTranslation('common');
  const Icon = iconMap[card.icon];

  return (
    <button
      type="button"
      onClick={() => onNavigate(card.page)}
      className="group relative overflow-hidden rounded-[28px] border border-primary-500/10 bg-white/80 p-5 text-left shadow-[0_18px_46px_rgba(87,48,46,0.08)] transition-all duration-300 hover:-translate-y-1 hover:border-primary-500/20 hover:bg-white"
    >
      <div className={`absolute inset-x-0 top-0 h-1.5 bg-gradient-to-r ${card.accent}`} />
      <div className="relative flex h-full flex-col gap-5">
        <div className="flex items-start justify-between gap-4">
          <div className={`inline-flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br ${card.accent} text-white shadow-[0_16px_34px_rgba(117,36,57,0.18)]`}>
            <Icon size={22} />
          </div>
          <div className="inline-flex h-10 w-10 items-center justify-center rounded-2xl border border-primary-500/10 bg-[rgba(255,250,245,0.9)] text-primary-700 transition-all duration-300 group-hover:translate-x-1 group-hover:border-primary-500/20">
            <ArrowRight size={18} />
          </div>
        </div>

        <div className="space-y-3">
          <h3 className="text-lg font-bold text-primary-900">{t(card.titleKey)}</h3>
          <p className="text-sm leading-6 text-slate-600">{t(card.descriptionKey)}</p>
        </div>
      </div>
    </button>
  );
}

export function HomePage({ onNavigate }: HomePageProps) {
  const { t } = useTranslation('common');

  return (
    <div className="h-full overflow-y-auto overflow-x-hidden">
      <div className="mx-auto flex min-h-full max-w-7xl flex-col gap-8 px-5 pb-14 pt-6 md:px-8 lg:px-10">
        <section className="portal-hero-card relative overflow-hidden rounded-[34px] p-6 md:p-8 lg:p-10">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(140,29,64,0.14),transparent_30%),radial-gradient(circle_at_top_right,rgba(197,155,91,0.16),transparent_26%),linear-gradient(135deg,rgba(255,250,245,0.92),rgba(255,255,255,0.78))]" />
          <div className="relative grid gap-6 lg:grid-cols-[minmax(0,1.3fr)_minmax(300px,0.9fr)]">
            <div className="space-y-6">
              <span className="inline-flex items-center gap-2 rounded-full border border-primary-500/10 bg-[rgba(140,29,64,0.06)] px-4 py-2 text-xs font-semibold uppercase tracking-[0.24em] text-primary-700">
                <Sparkles size={14} />
                <span>{t('app.home.kicker')}</span>
              </span>
              <div className="space-y-4">
                <h2 className="max-w-4xl font-display text-[2.6rem] font-bold leading-[1.02] tracking-[-0.04em] text-primary-900 md:text-[3.4rem] lg:text-[4rem]">
                  {t('app.home.title')}
                </h2>
                <p className="max-w-3xl text-base leading-7 text-slate-600 md:text-lg">
                  {t('app.home.description')}
                </p>
              </div>
              <div className="flex flex-wrap gap-3">
                <button
                  type="button"
                  onClick={() => onNavigate('paper2figure-tech-exp')}
                  className="rounded-full px-5 py-3 text-sm font-semibold portal-button-primary"
                >
                  {t('app.home.primaryCta')}
                </button>
                <button
                  type="button"
                  onClick={() => onNavigate('paper2ppt')}
                  className="rounded-full px-5 py-3 text-sm font-semibold portal-button-secondary"
                >
                  {t('app.home.secondaryCta')}
                </button>
              </div>
            </div>

            <aside className="portal-card-soft grid gap-4 rounded-[30px] p-5 md:p-6">
              <div className="grid gap-3 md:grid-cols-3 lg:grid-cols-1">
                <div className="rounded-[22px] border border-primary-500/10 bg-[rgba(255,251,246,0.92)] p-4">
                  <div className="text-2xl font-bold text-primary-900">12+</div>
                  <p className="mt-2 text-sm leading-6 text-slate-600">{t('app.home.metrics.features')}</p>
                </div>
                <div className="rounded-[22px] border border-primary-500/10 bg-[rgba(255,251,246,0.92)] p-4">
                  <div className="text-2xl font-bold text-primary-900">Path</div>
                  <p className="mt-2 text-sm leading-6 text-slate-600">{t('app.home.metrics.routing')}</p>
                </div>
                <div className="rounded-[22px] border border-primary-500/10 bg-[rgba(255,251,246,0.92)] p-4">
                  <div className="text-2xl font-bold text-primary-900">PKU</div>
                  <p className="mt-2 text-sm leading-6 text-slate-600">{t('app.home.metrics.intranet')}</p>
                </div>
              </div>
              <div className="rounded-[24px] border border-primary-500/10 bg-gradient-to-br from-[rgba(140,29,64,0.08)] via-[rgba(255,252,247,0.88)] to-[rgba(197,155,91,0.12)] p-5">
                <div className="flex items-center gap-2 text-sm font-semibold text-primary-700">
                  <GitBranch size={16} />
                  <span>{t('app.home.panelTitle')}</span>
                </div>
                <p className="mt-3 text-sm leading-7 text-slate-600">
                  {t('app.home.panelDescription')}
                </p>
              </div>
            </aside>
          </div>
        </section>

        <section className="space-y-4">
          <div className="flex flex-col gap-2 px-1 md:flex-row md:items-end md:justify-between">
            <div>
              <h3 className="font-display text-2xl font-bold text-primary-900 md:text-3xl">
                {t('app.home.featuredTitle')}
              </h3>
              <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600 md:text-base">
                {t('app.home.featuredDescription')}
              </p>
            </div>
          </div>
          <div className="grid gap-5 lg:grid-cols-3">
            {featuredHomeCards.map((card) => (
              <FeatureCard key={card.page} card={card} onNavigate={onNavigate} />
            ))}
          </div>
        </section>

        {homeFeatureSections.map((section) => (
          <section key={section.titleKey} className="space-y-4">
            <div className="flex flex-col gap-2 px-1 md:flex-row md:items-end md:justify-between">
              <div>
                <h3 className="font-display text-2xl font-bold text-primary-900 md:text-3xl">
                  {t(section.titleKey)}
                </h3>
                <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600 md:text-base">
                  {t(section.descriptionKey)}
                </p>
              </div>
            </div>
            <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
              {section.cards.map((card) => (
                <FeatureCard key={card.page} card={card} onNavigate={onNavigate} />
              ))}
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}
