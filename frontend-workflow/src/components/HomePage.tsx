import { ArrowRight, BookOpen, BrainCircuit, FileImage, FileSearch, FileStack, FileText, FolderKanban, GitBranch, LayoutTemplate, MessageSquare, Network, Presentation, Sparkles, Video } from 'lucide-react';
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
} as const;

const heroSignals = [
  { label: 'AGENT', top: '18%', left: '10%', color: 'bg-cyan-400', delay: '0s' },
  { label: 'PAPER', top: '34%', left: '32%', color: 'bg-fuchsia-400', delay: '0.7s' },
  { label: 'AI', top: '20%', left: '66%', color: 'bg-emerald-400', delay: '1.2s' },
  { label: 'FLOW', top: '56%', left: '16%', color: 'bg-sky-400', delay: '1.8s' },
  { label: 'MULTI', top: '52%', left: '74%', color: 'bg-violet-400', delay: '0.4s' },
] as const;

function HeroSignalField() {
  return (
    <div className="pointer-events-none relative h-full overflow-hidden rounded-[28px] border border-white/6 bg-[linear-gradient(135deg,rgba(255,255,255,0.05),rgba(255,255,255,0.01))]">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_14%_18%,rgba(80,170,255,0.3),transparent_18%),radial-gradient(circle_at_74%_22%,rgba(82,224,195,0.24),transparent_20%),radial-gradient(circle_at_52%_68%,rgba(225,96,255,0.2),transparent_24%),linear-gradient(180deg,rgba(7,10,22,0.05),rgba(7,10,22,0.45))]" />
      <div className="hero-scan-beam absolute left-[-20%] top-[18%] h-24 w-[58%] bg-[linear-gradient(90deg,rgba(88,196,255,0),rgba(88,196,255,0.18),rgba(88,196,255,0.42),rgba(88,196,255,0.08),rgba(88,196,255,0))] blur-xl" />
      <div className="absolute inset-x-6 top-12 h-px bg-gradient-to-r from-transparent via-cyan-300/35 to-transparent" />
      <div className="absolute inset-y-8 left-[42%] w-px bg-gradient-to-b from-transparent via-white/16 to-transparent" />
      <div className="hero-signal-flow absolute left-[12%] top-[28%] h-px w-[30%] rotate-[8deg] bg-[linear-gradient(90deg,rgba(34,211,238,0),rgba(34,211,238,0.1),rgba(34,211,238,0.85),rgba(34,211,238,0.1),rgba(34,211,238,0))]" />
      <div className="hero-signal-flow absolute left-[34%] top-[46%] h-px w-[34%] -rotate-[11deg] bg-[linear-gradient(90deg,rgba(232,121,249,0),rgba(232,121,249,0.1),rgba(232,121,249,0.8),rgba(232,121,249,0.1),rgba(232,121,249,0))]" />
      <div className="hero-signal-flow absolute left-[58%] top-[24%] h-[36%] w-px rotate-[12deg] bg-[linear-gradient(180deg,rgba(16,185,129,0),rgba(16,185,129,0.12),rgba(16,185,129,0.8),rgba(16,185,129,0.12),rgba(16,185,129,0))]" />
      <div className="hero-orbit-ring absolute left-[39%] top-[18%] h-36 w-36 rounded-full border border-cyan-300/18">
        <div className="absolute left-1/2 top-0 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-cyan-300 shadow-[0_0_20px_rgba(76,170,255,0.7)]" />
      </div>
      <div className="absolute left-[40%] top-[22%] h-28 w-28 rounded-full border border-cyan-300/20 bg-cyan-300/8 blur-[2px]" />
      <div className="hero-drift-slow absolute left-[44%] top-[30%] h-12 w-12 rounded-full border border-white/20 bg-white/8 shadow-[0_0_35px_rgba(76,170,255,0.35)] backdrop-blur-2xl" />
      <div className="hero-drift-fast absolute left-[45.2%] top-[32.5%] h-6 w-6 rounded-full bg-cyan-300/80 shadow-[0_0_28px_rgba(76,170,255,0.55)]" />
      <div className="hero-drift-slow absolute left-[23%] top-[18%] h-2.5 w-2.5 rounded-full bg-cyan-200/90 shadow-[0_0_18px_rgba(125,211,252,0.75)]" style={{ animationDelay: '0.8s' }} />
      <div className="hero-drift-fast absolute left-[71%] top-[38%] h-2 w-2 rounded-full bg-emerald-300/90 shadow-[0_0_18px_rgba(110,231,183,0.75)]" style={{ animationDelay: '1.5s' }} />
      <div className="hero-drift-slow absolute left-[62%] top-[62%] h-2.5 w-2.5 rounded-full bg-fuchsia-300/90 shadow-[0_0_18px_rgba(240,171,252,0.7)]" style={{ animationDelay: '2.1s' }} />

      {heroSignals.map((signal) => (
        <div
          key={signal.label}
          className="hero-drift-slow absolute"
          style={{ top: signal.top, left: signal.left, animationDelay: signal.delay }}
        >
          <div className="relative">
            <div className={`absolute inset-0 rounded-full ${signal.color} opacity-30 blur-xl animate-pulse`} />
            <div className={`relative h-3 w-3 rounded-full ${signal.color} shadow-[0_0_18px_rgba(255,255,255,0.25)]`} />
            <div className="absolute left-5 top-1/2 -translate-y-1/2 rounded-full border border-white/12 bg-black/35 px-2.5 py-1 text-[10px] font-semibold tracking-[0.22em] text-white/70 backdrop-blur-xl">
              {signal.label}
            </div>
          </div>
        </div>
      ))}

      <div className="absolute left-6 top-6 rounded-[20px] border border-white/10 bg-black/25 px-4 py-3 backdrop-blur-xl">
        <div className="text-[10px] font-semibold uppercase tracking-[0.26em] text-white/55">Signal Mesh</div>
        <div className="mt-2 flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-sm text-white/75">Agent + Paper + AI</span>
        </div>
      </div>

      <div className="absolute bottom-6 right-6 max-w-[18rem] rounded-[24px] border border-white/10 bg-black/25 p-4 backdrop-blur-xl">
        <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.24em] text-cyan-200/80">
          <Sparkles size={12} />
          <span>Agentic Pipeline</span>
        </div>
        <p className="mt-3 text-sm leading-6 text-white/70">
          Paper understanding, workflow planning, and AI generation are stitched into one visual entry point.
        </p>
      </div>
    </div>
  );
}

function FeaturePreview({ card, compact = false }: { card: HomeFeatureCard; compact?: boolean }) {
  const Icon = iconMap[card.icon];
  const previewHeight = compact ? 'h-32' : 'h-44';
  const previewRadius = compact ? 'rounded-[20px]' : 'rounded-[24px]';
  const iconRadius = compact ? 'rounded-[18px]' : 'rounded-2xl';
  const iconSize = compact ? 24 : 28;

  if (card.preview?.kind === 'video') {
    return (
      <div className={`relative ${previewHeight} overflow-hidden ${previewRadius} border border-white/10 bg-black/30 shadow-[0_24px_60px_rgba(0,0,0,0.35)]`}>
        <video
          src={card.preview.src}
          poster={card.preview.poster}
          className="h-full w-full object-cover transition-transform duration-700 group-hover:scale-[1.06]"
          autoPlay
          muted
          loop
          playsInline
          preload="metadata"
        />
        <div className="absolute inset-0 bg-gradient-to-br from-black/15 via-black/25 to-black/70" />
        <div className={`absolute inset-x-0 bottom-0 h-24 bg-gradient-to-t ${card.accent} opacity-70 blur-2xl`} />
      </div>
    );
  }

  if (card.preview) {
    return (
      <div className={`relative ${previewHeight} overflow-hidden ${previewRadius} border border-white/10 bg-black/30 shadow-[0_24px_60px_rgba(0,0,0,0.35)]`}>
        <img
          src={card.preview.src}
          alt=""
          className="h-full w-full object-cover transition-transform duration-700 group-hover:scale-[1.06]"
        />
        <div className="absolute inset-0 bg-gradient-to-br from-black/15 via-black/25 to-black/70" />
        <div className={`absolute inset-x-0 bottom-0 h-24 bg-gradient-to-t ${card.accent} opacity-70 blur-2xl`} />
      </div>
    );
  }

  return (
    <div className={`relative flex ${previewHeight} items-end overflow-hidden ${previewRadius} border border-white/10 bg-gradient-to-br ${card.accent} ${compact ? 'p-4' : 'p-5'} shadow-[0_24px_60px_rgba(0,0,0,0.35)]`}>
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(255,255,255,0.22),transparent_42%),radial-gradient(circle_at_bottom_right,rgba(0,0,0,0.28),transparent_58%)]" />
      <div className="absolute -right-8 -top-8 h-28 w-28 rounded-full bg-white/10 blur-2xl" />
      <div className="relative flex items-center gap-3 text-white">
        <div className={`${iconRadius} border border-white/20 bg-black/20 p-3 backdrop-blur-xl`}>
          <Icon size={iconSize} />
        </div>
      </div>
    </div>
  );
}

function FeatureCardBlock({
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
      className={`group relative overflow-hidden border border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.14),rgba(255,255,255,0.05))] text-left shadow-[0_30px_80px_rgba(0,0,0,0.35)] backdrop-blur-2xl transition-all duration-300 hover:-translate-y-1 hover:border-white/20 hover:bg-[linear-gradient(180deg,rgba(255,255,255,0.18),rgba(255,255,255,0.08))] ${
        compact ? 'rounded-[24px] p-3.5' : 'rounded-[30px] p-4'
      }`}
    >
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(255,255,255,0.18),transparent_28%),radial-gradient(circle_at_bottom_left,rgba(255,255,255,0.08),transparent_32%)] opacity-70" />
      <div className={`relative ${compact ? 'space-y-3' : 'space-y-4'}`}>
        <FeaturePreview card={card} compact={compact} />
        <div className="flex items-start justify-between gap-3">
          <div className={compact ? 'space-y-1.5' : 'space-y-2'}>
            <div className={`inline-flex items-center gap-2 rounded-full border border-white/12 bg-white/6 font-semibold uppercase tracking-[0.22em] text-white/75 ${
              compact ? 'px-2.5 py-1 text-[10px]' : 'px-3 py-1 text-[11px]'
            }`}>
              <Icon size={14} />
              <span>{t(card.badgeKey)}</span>
            </div>
            <h3 className={`font-semibold text-white ${compact ? 'text-base leading-6' : 'text-lg md:text-xl'}`}>{t(card.titleKey)}</h3>
            <p className={`max-w-xl text-white/65 ${compact ? 'text-[13px] leading-5' : 'text-sm leading-6'}`}>{t(card.descriptionKey)}</p>
          </div>
          <div className={`mt-1 rounded-2xl border border-white/12 bg-black/25 text-white/75 transition-all duration-300 group-hover:translate-x-0.5 group-hover:text-white ${
            compact ? 'p-2.5' : 'p-3'
          }`}>
            <ArrowRight size={compact ? 16 : 18} />
          </div>
        </div>
      </div>
    </button>
  );
}

export function HomePage({ onNavigate }: HomePageProps) {
  const { t } = useTranslation('common');

  return (
    <div className="h-full overflow-y-auto overflow-x-hidden">
      <div className="relative min-h-full px-5 pb-14 pt-6 md:px-8 lg:px-10">
        <div className="pointer-events-none absolute inset-x-0 top-0 h-72 bg-[radial-gradient(circle_at_top_left,rgba(73,120,255,0.22),transparent_40%),radial-gradient(circle_at_top_right,rgba(0,214,201,0.18),transparent_38%),radial-gradient(circle_at_50%_30%,rgba(255,255,255,0.08),transparent_30%)]" />
        <section className="relative mx-auto max-w-7xl rounded-[36px] border border-white/10 bg-[linear-gradient(135deg,rgba(10,14,26,0.88),rgba(18,24,38,0.62))] p-6 shadow-[0_40px_120px_rgba(0,0,0,0.45)] backdrop-blur-2xl md:p-8 lg:p-9">
          <div className="absolute inset-0 rounded-[36px] bg-[radial-gradient(circle_at_top_left,rgba(255,255,255,0.12),transparent_34%),radial-gradient(circle_at_bottom_right,rgba(93,178,255,0.12),transparent_30%)]" />
          <div className="relative grid gap-6 lg:grid-cols-[1.28fr_0.92fr]">
            <div className="relative overflow-hidden rounded-[30px] border border-white/6 bg-[linear-gradient(180deg,rgba(255,255,255,0.04),rgba(255,255,255,0.01))] p-6 md:p-7">
              <div className="relative h-[220px] md:h-[270px]">
                <HeroSignalField />
              </div>
              <div className="relative mt-6 space-y-5">
                <div className="inline-flex items-center gap-2 rounded-full border border-white/12 bg-white/6 px-4 py-2 text-xs font-semibold uppercase tracking-[0.24em] text-white/70">
                  <Sparkles size={14} />
                  <span>{t('app.home.kicker')}</span>
                </div>
                <div className="space-y-4">
                  <h2 className="max-w-4xl text-[2.4rem] font-semibold leading-[1.02] tracking-tight text-white md:text-[3.3rem] md:leading-[1.04] lg:text-[4rem] lg:leading-[1.04]">
                    {t('app.home.title')}
                  </h2>
                  <p className="max-w-3xl text-base leading-7 text-white/65 md:text-lg">
                    {t('app.home.description')}
                  </p>
                </div>
                <div className="flex flex-wrap gap-3">
                  <button
                    type="button"
                    onClick={() => onNavigate('paper2figure-tech-exp')}
                    className="inline-flex items-center gap-2 rounded-full border border-cyan-300/30 bg-cyan-300/10 px-5 py-3 text-sm font-semibold text-white transition-all duration-300 hover:bg-cyan-300/20"
                  >
                    <Sparkles size={16} />
                    <span>{t('app.home.primaryCta')}</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => onNavigate('paper2ppt-image')}
                    className="inline-flex items-center gap-2 rounded-full border border-white/12 bg-white/6 px-5 py-3 text-sm font-semibold text-white/85 transition-all duration-300 hover:border-white/20 hover:bg-white/10"
                  >
                    <Presentation size={16} />
                    <span>{t('app.home.secondaryCta')}</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => onNavigate('paper2ppt-frontend')}
                    className="inline-flex items-center gap-2 rounded-full border border-amber-300/20 bg-amber-300/10 px-5 py-3 text-sm font-semibold text-white/85 transition-all duration-300 hover:border-amber-300/35 hover:bg-amber-300/15"
                  >
                    <FileText size={16} />
                    <span>{t('app.home.frontendCta')}</span>
                  </button>
                </div>
              </div>
            </div>

            <div className="grid content-start gap-4 sm:grid-cols-2 lg:grid-cols-2">
              {featuredHomeCards.map((card) => (
                <FeatureCardBlock key={card.page} card={card} onNavigate={onNavigate} compact />
              ))}
            </div>
          </div>
        </section>

        <div className="mx-auto mt-8 max-w-7xl space-y-10">
          {homeFeatureSections.map((section) => (
            <section key={section.titleKey} className="space-y-4">
              <div className="flex flex-col gap-2 px-1 md:flex-row md:items-end md:justify-between">
                <div>
                  <h3 className="text-2xl font-semibold text-white">{t(section.titleKey)}</h3>
                  <p className="mt-2 max-w-2xl text-sm leading-6 text-white/55 md:text-base">
                    {t(section.descriptionKey)}
                  </p>
                </div>
              </div>
              <div className="grid gap-5 xl:grid-cols-3 md:grid-cols-2">
                {section.cards.map((card) => (
                  <FeatureCardBlock key={card.page} card={card} onNavigate={onNavigate} />
                ))}
              </div>
            </section>
          ))}
        </div>
      </div>
    </div>
  );
}
