import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  AlertCircle,
  BadgeInfo,
  BookOpen,
  Building2,
  ExternalLink,
  GraduationCap,
  Loader2,
  Search,
  Trophy,
  Users,
} from 'lucide-react';

import { apiFetch } from '../../config/api';
import { checkQuota, recordUsage } from '../../services/quotaService';
import { useAuthStore } from '../../stores/authStore';

import type {
  CitationAuthorCandidate,
  CitationAuthorDetail,
  CitationAuthorItem,
  CitationContextItem,
  CitationMatchedHonoree,
  CitationInstitutionStat,
  CitationMode,
  CitationPaperContextDetail,
  CitationPaperDetail,
  CitationStatItem,
  CitationWorkItem,
} from './types';

type JsonRecord = Record<string, any>;
type TranslationFn = (key: string, options?: Record<string, unknown>) => string;

const MAX_LIST_ITEMS = 30;
const AUTHOR_CANDIDATES_PAGE_SIZE = 8;
const MAX_VISIBLE_AFFILIATIONS = 6;
const AUTHOR_PUBLICATIONS_PAGE_SIZE = 20;

function asArray<T>(value: T[] | undefined | null): T[] {
  return Array.isArray(value) ? value : [];
}

function normalizeString(value: unknown): string {
  return typeof value === 'string' ? value.trim() : '';
}

function normalizeStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((item) => normalizeString(item))
    .filter(Boolean);
}

function normalizeStatItems(
  value: unknown,
  labelMap: Record<string, string> = {},
): CitationStatItem[] {
  if (Array.isArray(value)) {
    return asArray<JsonRecord>(value as JsonRecord[])
      .map((item) => ({
        label: normalizeString(
          item.label
          || item.name
          || labelMap[normalizeString(item.key || item.stat_key || item.honor_label || item.venue)]
          || item.key
          || item.honor_label
          || item.venue,
        ),
        value: Number(item.value || item.count || item.total || 0),
      }))
      .filter((item) => item.label);
  }

  if (value && typeof value === 'object') {
    return Object.entries(value as JsonRecord)
      .filter(([, rawValue]) => typeof rawValue === 'number')
      .map(([key, rawValue]) => ({
        label: labelMap[key] || key.replace(/_/g, ' '),
        value: Number(rawValue || 0),
      }));
  }

  return [];
}

function buildAuthorProfileStats(
  profile: CitationAuthorDetail['profile'],
  t: TranslationFn,
): CitationStatItem[] {
  return [
    { label: t('paper2citation:stats.totalWorks'), value: Number(profile.worksCount || 0) },
    { label: t('paper2citation:stats.totalCitations'), value: Number(profile.citedByCount || 0) },
    ...(profile.hIndex != null ? [{ label: t('paper2citation:stats.hIndex'), value: Number(profile.hIndex || 0) }] : []),
  ].filter((item) => item.value > 0);
}

function getAuthorStatLabelMap(t: TranslationFn): Record<string, string> {
  return {
    loaded_publications_count: t('paper2citation:stats.loadedPublications'),
    linked_publications_count: t('paper2citation:stats.linkedPublications'),
    unlinked_publications_count: t('paper2citation:stats.unlinkedPublications'),
    citing_works_count: t('paper2citation:stats.loadedCitingWorks'),
    citing_authors_count: t('paper2citation:stats.loadedCitingAuthors'),
    citing_institutions_count: t('paper2citation:stats.loadedCitingInstitutions'),
  };
}

function getPaperStatLabelMap(t: TranslationFn): Record<string, string> {
  return {
    citing_works_count: t('paper2citation:stats.loadedCitingWorks'),
    citing_authors_count: t('paper2citation:stats.loadedCitingAuthors'),
    citing_institutions_count: t('paper2citation:stats.loadedCitingInstitutions'),
    paper_cited_by_count: t('paper2citation:stats.paperTotalCitations'),
  };
}

function normalizeInstitutionStats(value: unknown): CitationInstitutionStat[] {
  return asArray<JsonRecord>(value as JsonRecord[])
    .map((item) => ({
      institution: normalizeString(item.institution || item.name || item.display_name),
      count: Number(item.count || item.value || item.citing_works_count || 0),
      country: normalizeString(item.country || item.country_code || item.country_name),
    }))
    .filter((item) => item.institution);
}

function normalizeAuthorItems(value: unknown): CitationAuthorItem[] {
  return asArray<JsonRecord>(value as JsonRecord[])
    .map((item) => ({
      name: normalizeString(item.name || item.display_name),
      openalexAuthorId: normalizeString(item.openalex_author_id || item.openalexAuthorId || item.id),
      affiliations: normalizeStringArray(item.affiliations || item.institutions),
      citedCount: Number(item.cited_count || item.citation_count || item.citing_works_count || item.count || 0) || undefined,
      honors: normalizeStringArray(item.honors),
    }))
    .filter((item) => item.name);
}

function normalizeMatchedHonorees(value: unknown): CitationMatchedHonoree[] {
  return asArray<JsonRecord>(value as JsonRecord[])
    .map((item) => ({
      name: normalizeString(item.display_name || item.name),
      canonicalName: normalizeString(item.canonical_name || item.canonicalName) || undefined,
      honorLabel: normalizeString(item.honor_label || item.honorLabel || item.label),
      openalexAuthorId: normalizeString(item.openalex_author_id || item.openalexAuthorId || item.id) || undefined,
      affiliations: normalizeStringArray(item.affiliations || item.institutions),
      citingWorksCount: Number(item.citing_works_count || item.cited_count || item.count || 0) || undefined,
    }))
    .filter((item) => item.name && item.honorLabel);
}

function decorateCitingAuthors(
  authors: CitationAuthorItem[],
  matchedHonorees: CitationMatchedHonoree[],
): CitationAuthorItem[] {
  if (!authors.length || !matchedHonorees.length) return authors;

  const honorMap = new Map<string, Set<string>>();
  matchedHonorees.forEach((item) => {
    [normalizeString(item.openalexAuthorId), normalizeString(item.name)]
      .filter(Boolean)
      .forEach((key) => {
        if (!honorMap.has(key)) honorMap.set(key, new Set<string>());
        honorMap.get(key)?.add(item.honorLabel);
      });
  });

  return [...authors]
    .map((author) => {
      const mergedHonors = new Set<string>(author.honors || []);
      [normalizeString(author.openalexAuthorId), normalizeString(author.name)]
        .filter(Boolean)
        .forEach((key) => {
          honorMap.get(key)?.forEach((honor) => mergedHonors.add(honor));
        });
      return {
        ...author,
        honors: Array.from(mergedHonors),
      };
    })
    .sort((left, right) => {
      const honorGap = (right.honors?.length || 0) - (left.honors?.length || 0);
      if (honorGap !== 0) return honorGap;
      return (right.citedCount || 0) - (left.citedCount || 0);
    });
}

function normalizeWorkItems(value: unknown): CitationWorkItem[] {
  return asArray<JsonRecord>(value as JsonRecord[])
    .map((item) => ({
      id: normalizeString(item.id || item.openalex_work_id || item.openalexWorkId),
      title: normalizeString(item.title),
      year: item.year == null ? null : Number(item.year),
      venue: normalizeString(item.venue || item.primary_location?.source?.display_name),
      doi: normalizeString(item.doi),
      citedByCount: Number(item.cited_by_count || item.citedByCount || item.citations || 0) || undefined,
      authors: normalizeStringArray(item.authors),
      institutions: normalizeStringArray(item.institutions),
      landingPageUrl: normalizeString(item.landing_page_url || item.landingPageUrl) || undefined,
    }))
    .filter((item) => item.title);
}

function buildAuthorCitationBuildInfo(
  value: unknown,
  t: TranslationFn,
): string[] {
  if (!value || typeof value !== 'object') return [];
  const stats = value as JsonRecord;
  const lines: string[] = [];
  const seedPublicationsCount = Number(stats.seed_publications_count || 0);
  const seedCitingWorksFetchLimit = Number(stats.seed_citing_works_fetch_limit || 0);
  const builtFromPublicationPage = Number(stats.built_from_publication_page || 0);
  const builtFromPublicationPageSize = Number(stats.built_from_publication_page_size || 0);

  if (seedPublicationsCount > 0) {
    lines.push(t('paper2citation:buildInfo.seedPublications', { count: formatNumber(seedPublicationsCount) }));
  }
  if (seedCitingWorksFetchLimit > 0) {
    lines.push(t('paper2citation:buildInfo.citingWorksPerSeed', { count: formatNumber(seedCitingWorksFetchLimit) }));
  }
  if (builtFromPublicationPage > 0) {
    lines.push(
      t('paper2citation:buildInfo.builtFromPage', {
        page: formatNumber(builtFromPublicationPage),
        pageSize: formatNumber(builtFromPublicationPageSize || AUTHOR_PUBLICATIONS_PAGE_SIZE),
      }),
    );
  }
  return lines;
}

function normalizeAuthorCandidates(payload: JsonRecord): CitationAuthorCandidate[] {
  const raw = asArray<JsonRecord>(payload.candidates || payload.results || payload.authors);
  return raw.map((item) => ({
    openalexAuthorId: normalizeString(item.openalex_author_id || item.openalexAuthorId || item.id),
    dblpId: normalizeString(item.dblp_id || item.dblpId) || undefined,
    source: normalizeString(item.source) || undefined,
    displayName: normalizeString(item.display_name || item.displayName || item.name),
    affiliations: normalizeStringArray(item.affiliations || item.institutions),
    worksCount: Number(item.works_count || item.worksCount || 0),
    citedByCount: Number(item.cited_by_count || item.citedByCount || 0),
    hIndex: item.h_index == null ? null : Number(item.h_index),
    summary: normalizeString(item.summary || item.hint) || undefined,
  })).filter((item) => item.displayName && (item.openalexAuthorId || item.dblpId));
}

function normalizeAuthorDetail(payload: JsonRecord, t: TranslationFn): CitationAuthorDetail {
  const profile = payload.author_profile || payload.profile || payload.author || {};
  const matchedHonorees = normalizeMatchedHonorees(payload.matched_honorees || payload.matchedHonorees);
  const normalizedProfile = {
    openalexAuthorId: normalizeString(profile.openalex_author_id || profile.openalexAuthorId || profile.id),
    dblpId: normalizeString(profile.dblp_id || profile.dblpId) || undefined,
    displayName: normalizeString(profile.display_name || profile.displayName || profile.name),
    affiliations: normalizeStringArray(profile.affiliations || profile.institutions),
    titles: normalizeStringArray(profile.titles),
    honors: normalizeStringArray(profile.honors),
    homepage: normalizeString(profile.homepage || profile.url) || undefined,
    worksCount: Number(profile.works_count || profile.worksCount || 0),
    citedByCount: Number(profile.cited_by_count || profile.citedByCount || 0),
    hIndex: profile.h_index == null ? (profile.summary_stats?.h_index == null ? null : Number(profile.summary_stats.h_index)) : Number(profile.h_index),
    summary: normalizeString(profile.summary || payload.summary) || undefined,
  };
  return {
    profile: normalizedProfile,
    bestEffortNotice: normalizeString(payload.best_effort_notice || payload.bestEffortNotice) || undefined,
    profileStats: buildAuthorProfileStats(normalizedProfile, t),
    publicationStats: normalizeStatItems(payload.publication_stats || payload.publicationStats, getAuthorStatLabelMap(t)),
    citationStats: normalizeStatItems(payload.citation_stats || payload.citationStats, getAuthorStatLabelMap(t)),
    citationBuildInfo: buildAuthorCitationBuildInfo(payload.citation_stats || payload.citationStats, t),
    publicationPagination: {
      page: Number(payload.publication_pagination?.page || payload.publicationPagination?.page || 1),
      pageSize: Number(payload.publication_pagination?.page_size || payload.publicationPagination?.pageSize || AUTHOR_PUBLICATIONS_PAGE_SIZE),
      totalItems: Number(payload.publication_pagination?.total_items || payload.publicationPagination?.totalItems || normalizedProfile.worksCount || 0),
      totalPages: Number(payload.publication_pagination?.total_pages || payload.publicationPagination?.totalPages || 1),
    },
    honorsStats: normalizeStatItems(payload.honors_stats || payload.honorsStats),
    publications: normalizeWorkItems(payload.publications || payload.works),
    citingInstitutions: normalizeInstitutionStats(payload.citing_institutions || payload.citingInstitutions),
    citingAuthors: decorateCitingAuthors(
      normalizeAuthorItems(payload.citing_authors || payload.citingAuthors),
      matchedHonorees,
    ),
    matchedHonorees,
  };
}

function normalizeAuthorPublicationPage(
  payload: JsonRecord,
  t: TranslationFn,
  totalItemsFallback = 0,
): Pick<CitationAuthorDetail, 'publicationStats' | 'publicationPagination' | 'publications'> {
  return {
    publicationStats: normalizeStatItems(payload.publication_stats || payload.publicationStats, getAuthorStatLabelMap(t)),
    publicationPagination: {
      page: Number(payload.publication_pagination?.page || payload.publicationPagination?.page || 1),
      pageSize: Number(payload.publication_pagination?.page_size || payload.publicationPagination?.pageSize || AUTHOR_PUBLICATIONS_PAGE_SIZE),
      totalItems: Number(payload.publication_pagination?.total_items || payload.publicationPagination?.totalItems || totalItemsFallback || 0),
      totalPages: Number(payload.publication_pagination?.total_pages || payload.publicationPagination?.totalPages || 1),
    },
    publications: normalizeWorkItems(payload.publications || payload.works),
  };
}

function normalizePaperDetail(payload: JsonRecord, t: TranslationFn): CitationPaperDetail {
  const paper = payload.paper_detail || payload.paper || payload.work || {};
  const matchedHonorees = normalizeMatchedHonorees(payload.matched_honorees || payload.matchedHonorees);
  return {
    paper: {
      id: normalizeString(paper.id || paper.openalex_work_id || paper.openalexWorkId) || undefined,
      title: normalizeString(paper.title),
      year: paper.year == null ? null : Number(paper.year),
      venue: normalizeString(paper.venue || paper.primary_location?.source?.display_name),
      doi: normalizeString(paper.doi) || undefined,
      url: normalizeString(paper.url || paper.landing_page_url) || undefined,
      authors: normalizeStringArray(paper.authors),
      institutions: normalizeStringArray(paper.institutions),
      citedByCount: Number(paper.cited_by_count || paper.citedByCount || 0) || undefined,
    },
    bestEffortNotice: normalizeString(payload.best_effort_notice || payload.bestEffortNotice) || undefined,
    citationStats: normalizeStatItems(payload.citation_stats || payload.citationStats, getPaperStatLabelMap(t)),
    honorsStats: normalizeStatItems(payload.honors_stats || payload.honorsStats),
    citingInstitutions: normalizeInstitutionStats(payload.citing_institutions || payload.citingInstitutions),
    citingAuthors: decorateCitingAuthors(
      normalizeAuthorItems(payload.citing_authors || payload.citingAuthors),
      matchedHonorees,
    ),
    citingWorks: normalizeWorkItems(payload.citing_works || payload.citingWorks || payload.citations),
    matchedHonorees,
  };
}

function normalizePaperContextDetail(payload: JsonRecord): CitationPaperContextDetail {
  const targetReferenceMatch = payload.target_reference_match || payload.targetReferenceMatch || {};
  const citingPaper = payload.citing_paper || payload.citingPaper || {};
  const contexts = asArray<JsonRecord>(payload.contexts || payload.citation_contexts)
    .map((item) => ({
      section: normalizeString(item.section) || undefined,
      sentence: normalizeString(item.sentence),
      paragraph: normalizeString(item.paragraph),
      marker: normalizeString(item.marker) || undefined,
      confidence: item.confidence == null ? undefined : Number(item.confidence),
    }))
    .filter((item) => item.sentence || item.paragraph);
  return {
    sourceUrl: normalizeString(payload.source_url || payload.sourceUrl) || undefined,
    bestEffortNotice: normalizeString(payload.best_effort_notice || payload.bestEffortNotice) || undefined,
    summary: normalizeString(payload.summary) || undefined,
    citationIntents: normalizeStringArray(payload.citation_intents || payload.citationIntents),
    targetReferenceMatch: {
      matched: Boolean(targetReferenceMatch.matched),
      matchedBy: normalizeString(targetReferenceMatch.matched_by || targetReferenceMatch.matchedBy) || undefined,
      marker: normalizeString(targetReferenceMatch.marker) || undefined,
      referenceText: normalizeString(targetReferenceMatch.reference_text || targetReferenceMatch.referenceText) || undefined,
      confidence: targetReferenceMatch.confidence == null ? undefined : Number(targetReferenceMatch.confidence),
    },
    citingPaper: {
      id: normalizeString(citingPaper.id || citingPaper.openalex_work_id || citingPaper.openalexWorkId) || undefined,
      title: normalizeString(citingPaper.title) || undefined,
      venue: normalizeString(citingPaper.venue) || undefined,
      year: citingPaper.year == null ? null : Number(citingPaper.year),
      doi: normalizeString(citingPaper.doi) || undefined,
    },
    contexts,
  };
}

function formatNumber(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return '-';
  return new Intl.NumberFormat().format(value);
}

function buildMetaLine(values: Array<string | number | null | undefined>): string {
  return values
    .map((item) => (typeof item === 'number' ? String(item) : normalizeString(item)))
    .filter(Boolean)
    .join(' · ');
}

async function safeReadJsonResponse(response: Response): Promise<JsonRecord> {
  const raw = await response.text();
  if (!raw.trim()) {
    throw new Error('Backend returned an empty response. Please retry in a moment.');
  }
  try {
    return JSON.parse(raw) as JsonRecord;
  } catch {
    throw new Error('Backend returned a non-JSON response. Please retry in a moment.');
  }
}

const SectionCard = ({
  title,
  subtitle,
  icon,
  children,
}: {
  title: string;
  subtitle?: string;
  icon: React.ReactNode;
  children: React.ReactNode;
}) => (
  <section className="rounded-3xl border border-white/10 bg-slate-950/55 p-5 shadow-[0_24px_60px_rgba(2,6,23,0.45)] backdrop-blur">
    <div className="mb-4 flex items-start justify-between gap-3">
      <div>
        <div className="flex items-center gap-2 text-white">
          <div className="rounded-xl border border-white/10 bg-white/5 p-2 text-cyan-300">
            {icon}
          </div>
          <h3 className="text-lg font-semibold">{title}</h3>
        </div>
        {subtitle ? <p className="mt-1 text-sm text-slate-400">{subtitle}</p> : null}
      </div>
    </div>
    {children}
  </section>
);

const StatGrid = ({ items, emptyText }: { items: CitationStatItem[]; emptyText: string }) => {
  if (!items.length) {
    return <p className="text-sm text-slate-500">{emptyText}</p>;
  }

  return (
    <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
      {items.map((item) => (
        <div key={item.label} className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
          <div className="text-xs uppercase tracking-[0.22em] text-slate-500">{item.label}</div>
          <div className="mt-2 text-2xl font-semibold text-white">{formatNumber(item.value)}</div>
        </div>
      ))}
    </div>
  );
};

const TableLikeList = ({
  items,
  emptyText,
  renderItem,
}: {
  items: any[];
  emptyText: string;
  renderItem: (item: any, index: number) => React.ReactNode;
}) => {
  if (!items.length) {
    return <p className="text-sm text-slate-500">{emptyText}</p>;
  }

  return (
    <div className="space-y-3">
      {items.slice(0, MAX_LIST_ITEMS).map((item, index) => renderItem(item, index))}
    </div>
  );
};

const LoadingPanel = ({
  title,
  subtitle,
}: {
  title: string;
  subtitle: string;
}) => (
  <section className="rounded-3xl border border-cyan-300/20 bg-[linear-gradient(135deg,rgba(8,47,73,0.55),rgba(15,23,42,0.72))] p-6 shadow-[0_24px_60px_rgba(8,145,178,0.18)]">
    <div className="flex items-start gap-4">
      <div className="rounded-2xl border border-cyan-300/20 bg-cyan-400/10 p-3 text-cyan-200">
        <Loader2 size={20} className="animate-spin" />
      </div>
      <div className="min-w-0 flex-1">
        <div className="text-base font-semibold text-white">{title}</div>
        <p className="mt-1 text-sm text-slate-300">{subtitle}</p>
        <div className="mt-5 space-y-3">
          <div className="h-3 w-5/6 animate-pulse rounded-full bg-white/10" />
          <div className="h-3 w-4/6 animate-pulse rounded-full bg-white/10" />
          <div className="h-20 w-full animate-pulse rounded-2xl bg-white/[0.05]" />
          <div className="grid gap-3 md:grid-cols-3">
            <div className="h-20 animate-pulse rounded-2xl bg-white/[0.05]" />
            <div className="h-20 animate-pulse rounded-2xl bg-white/[0.05]" />
            <div className="h-20 animate-pulse rounded-2xl bg-white/[0.05]" />
          </div>
        </div>
      </div>
    </div>
  </section>
);

const NoticeBanner = ({ children }: { children: React.ReactNode }) => (
  <div className="rounded-2xl border border-amber-300/20 bg-amber-500/10 px-4 py-3 text-sm text-amber-50">
    {children}
  </div>
);

const NotableCitersCard = ({
  title,
  subtitle,
  items,
}: {
  title: string;
  subtitle: string;
  items: CitationMatchedHonoree[];
}) => {
  if (!items.length) return null;

  return (
    <SectionCard title={title} subtitle={subtitle} icon={<Trophy size={18} />}>
      <div className="grid gap-3 md:grid-cols-2">
        {items.slice(0, 8).map((item) => (
          <div key={`${item.openalexAuthorId || item.name}-${item.honorLabel}`} className="rounded-2xl border border-amber-300/25 bg-amber-500/10 p-4">
            <div className="flex items-start justify-between gap-3">
              <div>
                <div className="text-sm font-semibold text-white">{item.name}</div>
                <div className="mt-1 text-xs text-amber-100/80">
                  {item.canonicalName && item.canonicalName !== item.name ? item.canonicalName : item.honorLabel}
                </div>
              </div>
              <span className="rounded-full border border-amber-300/30 bg-amber-400/10 px-2 py-1 text-[11px] text-amber-100">
                {item.honorLabel}
              </span>
            </div>
            {item.affiliations.length ? (
              <div className="mt-3 text-xs text-slate-300">{item.affiliations.join(', ')}</div>
            ) : null}
            {item.citingWorksCount ? (
              <div className="mt-3 text-xs font-medium text-cyan-200">{formatNumber(item.citingWorksCount)} citing works</div>
            ) : null}
          </div>
        ))}
      </div>
    </SectionCard>
  );
};

function buildCandidateMeta(candidate: CitationAuthorCandidate, t: (key: string) => string): string {
  const values: string[] = [];
  if (candidate.affiliations[0]) values.push(candidate.affiliations[0]);
  if (candidate.worksCount > 0) values.push(`${formatNumber(candidate.worksCount)} works`);
  if (candidate.citedByCount > 0) values.push(`${formatNumber(candidate.citedByCount)} citations`);
  if (candidate.hIndex != null && candidate.hIndex > 0) values.push(`h-index ${candidate.hIndex}`);
  if (!values.length && candidate.source === 'dblp') values.push(t('paper2citation:states.dblpCandidate'));
  return buildMetaLine(values);
}

const Paper2CitationPage = () => {
  const { t } = useTranslation(['paper2citation', 'common']);
  const { user, refreshQuota } = useAuthStore();

  const [mode, setMode] = useState<CitationMode>('author');
  const [authorQuery, setAuthorQuery] = useState('');
  const [paperQuery, setPaperQuery] = useState('');
  const [authorCandidates, setAuthorCandidates] = useState<CitationAuthorCandidate[]>([]);
  const [candidatePage, setCandidatePage] = useState(1);
  const [selectedAuthorId, setSelectedAuthorId] = useState<string | null>(null);
  const [selectedAuthorCandidate, setSelectedAuthorCandidate] = useState<CitationAuthorCandidate | null>(null);
  const [authorDetail, setAuthorDetail] = useState<CitationAuthorDetail | null>(null);
  const [paperDetail, setPaperDetail] = useState<CitationPaperDetail | null>(null);
  const [authorPublicationPage, setAuthorPublicationPage] = useState(1);
  const [error, setError] = useState<string | null>(null);
  const [isSearchingAuthors, setIsSearchingAuthors] = useState(false);
  const [isLoadingAuthorDetail, setIsLoadingAuthorDetail] = useState(false);
  const [isPagingAuthorPublications, setIsPagingAuthorPublications] = useState(false);
  const [isLoadingPaperDetail, setIsLoadingPaperDetail] = useState(false);
  const [paperContexts, setPaperContexts] = useState<Record<string, CitationPaperContextDetail>>({});
  const [loadingPaperContextKey, setLoadingPaperContextKey] = useState<string | null>(null);
  const [loadingAuthorName, setLoadingAuthorName] = useState('');
  const [loadingPaperValue, setLoadingPaperValue] = useState('');
  const detailAnchorRef = useRef<HTMLDivElement | null>(null);
  const isLoadingDetail = isLoadingAuthorDetail || isLoadingPaperDetail;
  const isBusy = isSearchingAuthors || isLoadingDetail;
  const showPinnedLoadingNotice = isLoadingDetail && !authorDetail && !paperDetail;

  const quotaUserId = user?.id || null;
  const isAnonymous = user?.is_anonymous || false;

  const authorSummary = useMemo(() => {
    if (!authorDetail) return null;
    return buildMetaLine([
      authorDetail.profile.affiliations[0],
      authorDetail.profile.worksCount ? `${formatNumber(authorDetail.profile.worksCount)} works` : '',
      authorDetail.profile.citedByCount ? `${formatNumber(authorDetail.profile.citedByCount)} citations` : '',
      authorDetail.profile.hIndex != null ? `h-index ${authorDetail.profile.hIndex}` : '',
    ]);
  }, [authorDetail]);

  const pagedAuthorCandidates = useMemo(() => {
    const start = (candidatePage - 1) * AUTHOR_CANDIDATES_PAGE_SIZE;
    return authorCandidates.slice(start, start + AUTHOR_CANDIDATES_PAGE_SIZE);
  }, [authorCandidates, candidatePage]);

  const candidatePageCount = Math.max(1, Math.ceil(authorCandidates.length / AUTHOR_CANDIDATES_PAGE_SIZE));

  const paperSummary = useMemo(() => {
    if (!paperDetail) return null;
    return buildMetaLine([
      paperDetail.paper.year,
      paperDetail.paper.venue,
      paperDetail.paper.citedByCount ? `${formatNumber(paperDetail.paper.citedByCount)} citations` : '',
    ]);
  }, [paperDetail]);

  const visibleAuthorAffiliations = useMemo(
    () => authorDetail?.profile.affiliations.slice(0, MAX_VISIBLE_AFFILIATIONS) || [],
    [authorDetail],
  );

  const hiddenAuthorAffiliationCount = Math.max(
    0,
    (authorDetail?.profile.affiliations.length || 0) - visibleAuthorAffiliations.length,
  );

  const resetResults = (nextMode: CitationMode) => {
    setMode(nextMode);
    setError(null);
    setAuthorCandidates([]);
    setCandidatePage(1);
    setSelectedAuthorId(null);
    setSelectedAuthorCandidate(null);
    setAuthorDetail(null);
    setPaperDetail(null);
    setPaperContexts({});
    setLoadingPaperContextKey(null);
    setAuthorPublicationPage(1);
    setLoadingAuthorName('');
    setLoadingPaperValue('');
    setIsPagingAuthorPublications(false);
  };

  useEffect(() => {
    if (isLoadingDetail || authorDetail || paperDetail) {
      requestAnimationFrame(() => {
        detailAnchorRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
    }
  }, [isLoadingDetail, authorDetail, paperDetail]);

  const runQuotaGuard = async () => {
    const quota = await checkQuota(quotaUserId, isAnonymous);
    if (quota.remaining <= 0) {
      throw new Error(
        quota.isAuthenticated ? t('paper2citation:errors.quotaUserExhausted') : t('paper2citation:errors.quotaGuestExhausted')
      );
    }
  };

  const onSearchAuthor = async () => {
    const query = authorQuery.trim();
    if (!query) {
      setError(t('paper2citation:errors.authorQueryRequired'));
      return;
    }

    setError(null);
    setSelectedAuthorId(null);
    setSelectedAuthorCandidate(null);
    setAuthorDetail(null);
    setPaperDetail(null);
    setPaperContexts({});
    setLoadingPaperContextKey(null);
    setAuthorPublicationPage(1);

    setIsSearchingAuthors(true);
    try {
      const response = await apiFetch('/api/v1/paper2citation/authors/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          author_name: query,
          max_author_candidates: 24,
        }),
      });

      const payload = await safeReadJsonResponse(response);
      if (!response.ok) {
        throw new Error(payload?.detail || payload?.error || t('paper2citation:errors.searchFailed'));
      }

      setAuthorCandidates(normalizeAuthorCandidates(payload));
      setCandidatePage(1);
    } catch (err) {
      setAuthorCandidates([]);
      setCandidatePage(1);
      setError(err instanceof Error ? err.message : t('paper2citation:errors.searchFailed'));
    } finally {
      setIsSearchingAuthors(false);
    }
  };

  const getCandidateKey = (candidate: CitationAuthorCandidate) => candidate.dblpId || candidate.openalexAuthorId || candidate.displayName;

  const onLoadAuthorDetail = async (candidate: CitationAuthorCandidate, page = 1) => {
    setError(null);
    const candidateKey = getCandidateKey(candidate);
    const currentCandidateKey = selectedAuthorCandidate ? getCandidateKey(selectedAuthorCandidate) : null;
    const isPaginationOnly = Boolean(
      authorDetail
      && currentCandidateKey
      && currentCandidateKey === candidateKey
      && page !== authorPublicationPage
    );
    setSelectedAuthorId(candidateKey);
    setSelectedAuthorCandidate(candidate);
    setLoadingAuthorName(isPaginationOnly ? '' : (candidate.displayName || candidateKey));
    if (!isPaginationOnly) {
      setAuthorDetail(null);
      setPaperDetail(null);
      setPaperContexts({});
      setLoadingPaperContextKey(null);
    }
    setIsPagingAuthorPublications(isPaginationOnly);
    setIsLoadingAuthorDetail(true);
    try {
      if (!isPaginationOnly) {
        await runQuotaGuard();
      }

      const response = await apiFetch(isPaginationOnly ? '/api/v1/paper2citation/author/publications' : '/api/v1/paper2citation/author/detail', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          openalex_author_id: candidate.openalexAuthorId || '',
          dblp_id: candidate.dblpId || '',
          display_name: candidate.displayName,
          affiliation_hint: candidate.affiliations[0] || '',
          candidate_source: candidate.source || '',
          publication_page: page,
          publication_page_size: AUTHOR_PUBLICATIONS_PAGE_SIZE,
        }),
      });

      const payload = await safeReadJsonResponse(response);
      if (!response.ok) {
        throw new Error(payload?.detail || payload?.error || t('paper2citation:errors.detailFailed'));
      }

      setSelectedAuthorId(candidateKey);
      if (isPaginationOnly) {
        const nextPage = normalizeAuthorPublicationPage(
          payload,
          t,
          authorDetail?.publicationPagination?.totalItems || authorDetail?.profile.worksCount || 0,
        );
        setAuthorDetail((current) => {
          if (!current) return current;
          return {
            ...current,
            publicationStats: nextPage.publicationStats,
            publicationPagination: nextPage.publicationPagination,
            publications: nextPage.publications,
          };
        });
      } else {
        setAuthorDetail(normalizeAuthorDetail(payload, t));
      }
      setAuthorPublicationPage(page);
      setPaperDetail(null);
      if (!isPaginationOnly) {
        const usageRecorded = await recordUsage(quotaUserId, 'paper2citation', { isAnonymous });
        if (usageRecorded) refreshQuota();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : t('paper2citation:errors.detailFailed'));
    } finally {
      setLoadingAuthorName('');
      setIsPagingAuthorPublications(false);
      setIsLoadingAuthorDetail(false);
    }
  };

  const onLoadPaperDetail = async () => {
    const query = paperQuery.trim();
    if (!query) {
      setError(t('paper2citation:errors.paperQueryRequired'));
      return;
    }

    setError(null);
    setPaperDetail(null);
    setAuthorDetail(null);
    setPaperContexts({});
    setLoadingPaperContextKey(null);
    setLoadingPaperValue(query);
    setLoadingAuthorName('');

    setIsLoadingPaperDetail(true);
    try {
      await runQuotaGuard();

      const response = await apiFetch('/api/v1/paper2citation/paper/detail', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ doi_or_url: query }),
      });

      const payload = await safeReadJsonResponse(response);
      if (!response.ok) {
        throw new Error(payload?.detail || payload?.error || t('paper2citation:errors.detailFailed'));
      }

      setPaperDetail(normalizePaperDetail(payload, t));
      setPaperContexts({});
      setLoadingPaperContextKey(null);
      const usageRecorded = await recordUsage(quotaUserId, 'paper2citation', { isAnonymous });
      if (usageRecorded) refreshQuota();
    } catch (err) {
      setError(err instanceof Error ? err.message : t('paper2citation:errors.detailFailed'));
    } finally {
      setLoadingPaperValue('');
      setIsLoadingPaperDetail(false);
    }
  };

  const getWorkKey = (work: CitationWorkItem) => work.id || work.doi || work.title;

  const onLoadPaperContext = async (work: CitationWorkItem) => {
    if (!paperDetail) return;
    const workKey = getWorkKey(work);
    setError(null);
    setLoadingPaperContextKey(workKey);
    try {
      const response = await apiFetch('/api/v1/paper2citation/paper/context', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          target_doi_or_url: paperDetail.paper.id || paperDetail.paper.doi || paperQuery.trim(),
          citing_work_openalex_id: work.id || '',
          citing_work_doi_or_url: work.doi || work.landingPageUrl || '',
          citing_work_title: work.title,
        }),
      });

      const payload = await safeReadJsonResponse(response);
      if (!response.ok) {
        throw new Error(payload?.detail || payload?.error || t('paper2citation:errors.contextFailed'));
      }

      setPaperContexts((current) => ({
        ...current,
        [workKey]: normalizePaperContextDetail(payload),
      }));
    } catch (err) {
      setError(err instanceof Error ? err.message : t('paper2citation:errors.contextFailed'));
    } finally {
      setLoadingPaperContextKey((current) => (current === workKey ? null : current));
    }
  };

  return (
    <div className="h-full overflow-y-auto px-4 py-6 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-7xl space-y-6">
        <section className="relative overflow-hidden rounded-[32px] border border-cyan-400/20 bg-[radial-gradient(circle_at_top_left,_rgba(34,211,238,0.16),_transparent_32%),linear-gradient(135deg,rgba(15,23,42,0.94),rgba(2,6,23,0.92))] p-6 shadow-[0_28px_90px_rgba(8,145,178,0.25)]">
          <div className="absolute inset-y-0 right-0 hidden w-80 bg-[radial-gradient(circle,_rgba(34,211,238,0.18),_transparent_60%)] lg:block" />
          <div className="relative flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-3xl">
              <div className="inline-flex items-center gap-2 rounded-full border border-cyan-300/20 bg-cyan-400/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.26em] text-cyan-200">
                <BadgeInfo size={14} />
                {t('paper2citation:hero.badge')}
              </div>
              <h1 className="mt-4 text-3xl font-bold tracking-tight text-white sm:text-4xl">
                {t('paper2citation:hero.title')}
              </h1>
              <p className="mt-3 max-w-2xl text-sm leading-7 text-slate-300 sm:text-base">
                {t('paper2citation:hero.description')}
              </p>
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              {(['author', 'paper'] as CitationMode[]).map((item) => (
                <button
                  key={item}
                  onClick={() => resetResults(item)}
                  className={`rounded-2xl border px-4 py-3 text-left transition ${
                    mode === item
                      ? 'border-cyan-300/40 bg-cyan-300/12 text-white shadow-[0_16px_30px_rgba(34,211,238,0.18)]'
                      : 'border-white/10 bg-white/[0.04] text-slate-300 hover:border-white/20 hover:bg-white/[0.08]'
                  }`}
                >
                  <div className="text-xs uppercase tracking-[0.2em] text-slate-400">{t(`paper2citation:modes.${item}.label`)}</div>
                  <div className="mt-1 text-sm font-semibold">{t(`paper2citation:modes.${item}.title`)}</div>
                  <div className="mt-1 text-xs text-slate-400">{t(`paper2citation:modes.${item}.description`)}</div>
                </button>
              ))}
            </div>
          </div>
        </section>

        <section className="rounded-3xl border border-white/10 bg-slate-950/60 p-5 shadow-[0_24px_60px_rgba(2,6,23,0.42)]">
          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_auto] lg:items-end">
            {mode === 'author' ? (
              <div>
                <label className="mb-2 block text-sm font-medium text-slate-200">
                  {t('paper2citation:search.authorLabel')}
                </label>
                <input
                  value={authorQuery}
                  onChange={(event) => {
                    const nextValue = event.target.value;
                    setAuthorQuery(nextValue);
                    setError(null);
                    setAuthorCandidates([]);
                    setCandidatePage(1);
                    setSelectedAuthorId(null);
                    setSelectedAuthorCandidate(null);
                    setAuthorDetail(null);
                    setPaperDetail(null);
                    setLoadingAuthorName('');
                  }}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter') void onSearchAuthor();
                  }}
                  placeholder={t('paper2citation:search.authorPlaceholder')}
                  className="w-full rounded-2xl border border-white/10 bg-white/[0.04] px-4 py-3 text-white outline-none transition placeholder:text-slate-500 focus:border-cyan-300/40 focus:bg-white/[0.06]"
                />
              </div>
            ) : (
              <div>
                <label className="mb-2 block text-sm font-medium text-slate-200">
                  {t('paper2citation:search.paperLabel')}
                </label>
                <input
                  value={paperQuery}
                  onChange={(event) => setPaperQuery(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter') void onLoadPaperDetail();
                  }}
                  placeholder={t('paper2citation:search.paperPlaceholder')}
                  className="w-full rounded-2xl border border-white/10 bg-white/[0.04] px-4 py-3 text-white outline-none transition placeholder:text-slate-500 focus:border-cyan-300/40 focus:bg-white/[0.06]"
                />
              </div>
            )}
            <button
              onClick={() => (mode === 'author' ? void onSearchAuthor() : void onLoadPaperDetail())}
              disabled={isBusy}
              className="inline-flex items-center justify-center gap-2 rounded-2xl bg-cyan-400 px-5 py-3 font-semibold text-slate-950 transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:bg-cyan-900/40 disabled:text-slate-300"
            >
              {isBusy ? <Loader2 size={18} className="animate-spin" /> : <Search size={18} />}
              {mode === 'author' ? t('paper2citation:search.authorButton') : t('paper2citation:search.paperButton')}
            </button>
          </div>

          <div className="mt-3 text-xs text-slate-500">
            {mode === 'author' ? t('paper2citation:search.authorHint') : t('paper2citation:search.paperHint')}
          </div>

          {error ? (
            <div className="mt-4 flex items-start gap-3 rounded-2xl border border-rose-400/20 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
              <AlertCircle size={18} className="mt-0.5 shrink-0" />
              <span>{error}</span>
            </div>
          ) : null}
        </section>

        {showPinnedLoadingNotice ? (
          <div ref={detailAnchorRef}>
            <NoticeBanner>{t('paper2citation:states.loadingPinned')}</NoticeBanner>
          </div>
        ) : (
          <div ref={detailAnchorRef} />
        )}

        {mode === 'author' ? (
          <>
            {isSearchingAuthors ? (
              <LoadingPanel
                title={t('paper2citation:states.searchingAuthorsTitle', { query: authorQuery.trim() || t('paper2citation:states.loadingFallbackAuthor') })}
                subtitle={t('paper2citation:states.searchingAuthorsSubtitle')}
              />
            ) : null}

            <SectionCard
              title={t('paper2citation:sections.candidates.title')}
              subtitle={t('paper2citation:sections.candidates.subtitle')}
              icon={<Users size={18} />}
            >
              <TableLikeList
                items={pagedAuthorCandidates}
                emptyText={isSearchingAuthors ? t('paper2citation:states.searching') : t('paper2citation:states.noCandidates')}
                renderItem={(candidate: CitationAuthorCandidate) => {
                  const candidateKey = getCandidateKey(candidate);
                  const selected = candidateKey === selectedAuthorId;
                  return (
                    <button
                      key={candidateKey}
                      onClick={() => void onLoadAuthorDetail(candidate)}
                      className={`w-full rounded-2xl border p-4 text-left transition ${
                        selected
                          ? 'border-cyan-300/40 bg-cyan-400/10 shadow-[0_18px_36px_rgba(34,211,238,0.12)]'
                          : 'border-white/10 bg-white/[0.03] hover:border-white/20 hover:bg-white/[0.06]'
                      }`}
                    >
                      <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                        <div>
                          <div className="text-base font-semibold text-white">{candidate.displayName}</div>
                          <div className="mt-1 text-sm text-slate-400">
                            {buildCandidateMeta(candidate, (key) => t(key)) || t('paper2citation:states.noAffiliation')}
                          </div>
                          {candidate.summary ? (
                            <p className="mt-2 text-sm text-slate-300">{candidate.summary}</p>
                          ) : null}
                        </div>
                        <div className="text-xs text-slate-500">
                          {isLoadingDetail && selected && !isPagingAuthorPublications ? (
                            <span className="inline-flex items-center gap-1 rounded-full border border-cyan-300/30 bg-cyan-400/10 px-2 py-1 text-cyan-200">
                              <Loader2 size={12} className="animate-spin" />
                              {t('paper2citation:states.loadingDetail')}
                            </span>
                          ) : null}
                          {isLoadingDetail && selected && !isPagingAuthorPublications ? ' · ' : null}
                          {candidate.dblpId ? `DBLP: ${candidate.dblpId}` : candidate.openalexAuthorId}
                        </div>
                      </div>
                    </button>
                  );
                }}
              />

              {authorCandidates.length > AUTHOR_CANDIDATES_PAGE_SIZE ? (
                <div className="mt-4 flex items-center justify-between gap-3 border-t border-white/10 pt-4 text-sm text-slate-400">
                  <span>
                    {t('paper2citation:states.candidatePage', {
                      page: candidatePage,
                      total: candidatePageCount,
                      count: authorCandidates.length,
                    })}
                  </span>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setCandidatePage((current) => Math.max(1, current - 1))}
                      disabled={candidatePage <= 1}
                      className="rounded-xl border border-white/10 px-3 py-2 text-sm text-slate-200 transition hover:border-white/20 hover:bg-white/[0.06] disabled:cursor-not-allowed disabled:opacity-40"
                    >
                      {t('paper2citation:actions.prevPage')}
                    </button>
                    <button
                      onClick={() => setCandidatePage((current) => Math.min(candidatePageCount, current + 1))}
                      disabled={candidatePage >= candidatePageCount}
                      className="rounded-xl border border-white/10 px-3 py-2 text-sm text-slate-200 transition hover:border-white/20 hover:bg-white/[0.06] disabled:cursor-not-allowed disabled:opacity-40"
                    >
                      {t('paper2citation:actions.nextPage')}
                    </button>
                  </div>
                </div>
              ) : null}
            </SectionCard>

            {isLoadingDetail && !authorDetail ? (
              <LoadingPanel
                title={t('paper2citation:states.loadingAuthorTitle', { name: loadingAuthorName || t('paper2citation:states.loadingFallbackAuthor') })}
                subtitle={t('paper2citation:states.loadingAuthorSubtitle')}
              />
            ) : null}

            {authorDetail ? (
              <div className="grid gap-6 xl:grid-cols-[minmax(0,1.2fr)_minmax(0,0.8fr)]">
                <div className="space-y-6">
                  <SectionCard
                    title={authorDetail.profile.displayName}
                    subtitle={authorSummary || t('paper2citation:states.noSummary')}
                    icon={<GraduationCap size={18} />}
                  >
                    <div className="space-y-4 text-sm text-slate-300">
                      {selectedAuthorCandidate?.affiliations[0] || selectedAuthorCandidate?.dblpId ? (
                        <div className="rounded-2xl border border-cyan-300/20 bg-cyan-400/10 px-4 py-3">
                          <div className="text-xs uppercase tracking-[0.18em] text-cyan-200">
                            {t('paper2citation:sections.selectedCandidate.title')}
                          </div>
                          <div className="mt-2 text-sm text-white">
                            {buildMetaLine([
                              selectedAuthorCandidate?.affiliations[0],
                              selectedAuthorCandidate?.dblpId ? `DBLP: ${selectedAuthorCandidate.dblpId}` : '',
                            ])}
                          </div>
                        </div>
                      ) : null}
                      {authorDetail.profile.summary ? <p>{authorDetail.profile.summary}</p> : null}
                      {authorDetail.profile.titles?.length ? (
                        <div className="flex flex-wrap gap-2">
                          {authorDetail.profile.titles.map((title) => (
                            <span key={title} className="rounded-full border border-cyan-300/25 bg-cyan-400/10 px-3 py-1 text-xs text-cyan-100">
                              {title}
                            </span>
                          ))}
                        </div>
                      ) : null}
                      {authorDetail.profile.honors?.length ? (
                        <div className="flex flex-wrap gap-2">
                          {authorDetail.profile.honors.map((honor) => (
                            <span key={honor} className="rounded-full border border-amber-300/25 bg-amber-500/10 px-3 py-1 text-xs text-amber-100">
                              {honor}
                            </span>
                          ))}
                        </div>
                      ) : null}
                      <div className="flex flex-wrap gap-2">
                        {visibleAuthorAffiliations.length ? visibleAuthorAffiliations.map((affiliation) => (
                          <span key={affiliation} className="rounded-full border border-white/10 bg-white/[0.05] px-3 py-1 text-xs text-slate-200">
                            {affiliation}
                          </span>
                        )) : <span className="text-slate-500">{t('paper2citation:states.noAffiliation')}</span>}
                        {hiddenAuthorAffiliationCount > 0 ? (
                          <span className="rounded-full border border-dashed border-white/10 bg-white/[0.03] px-3 py-1 text-xs text-slate-400">
                            +{hiddenAuthorAffiliationCount} more
                          </span>
                        ) : null}
                      </div>
                      {authorDetail.profile.homepage ? (
                        <a
                          href={authorDetail.profile.homepage}
                          target="_blank"
                          rel="noreferrer"
                          className="inline-flex items-center gap-2 text-cyan-300 hover:text-cyan-200"
                        >
                          <ExternalLink size={15} />
                          {t('paper2citation:actions.openHomepage')}
                        </a>
                      ) : null}
                    </div>
                  </SectionCard>

                  <NotableCitersCard
                    title={t('paper2citation:sections.notableCiters.title')}
                    subtitle={t('paper2citation:sections.notableCiters.subtitle')}
                    items={authorDetail.matchedHonorees}
                  />

                  <SectionCard
                    title={t('paper2citation:sections.publications.title')}
                    subtitle={t('paper2citation:sections.publications.subtitle')}
                    icon={<BookOpen size={18} />}
                  >
                    <TableLikeList
                      items={authorDetail.publications}
                      emptyText={t('paper2citation:states.noPublications')}
                      renderItem={(work: CitationWorkItem, index) => (
                        <div key={`${work.id || work.title}-${index}`} className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
                          <div className="flex items-start justify-between gap-3">
                            <div>
                              <div className="text-sm font-semibold text-white">{work.title}</div>
                              <div className="mt-1 text-xs text-slate-400">{buildMetaLine([work.year, work.venue])}</div>
                              {work.authors.length ? <div className="mt-2 text-xs text-slate-500">{work.authors.join(', ')}</div> : null}
                            </div>
                            <div className="text-right text-xs text-slate-400">
                              {work.citedByCount != null ? `${formatNumber(work.citedByCount)} cites` : ''}
                            </div>
                          </div>
                          {work.doi ? (
                            <a
                              href={work.doi.startsWith('http') ? work.doi : `https://doi.org/${work.doi.replace(/^https?:\/\/doi\.org\//, '')}`}
                              target="_blank"
                              rel="noreferrer"
                              className="mt-3 inline-flex items-center gap-2 text-xs text-cyan-300 hover:text-cyan-200"
                            >
                              <ExternalLink size={14} />
                              {work.doi}
                            </a>
                          ) : null}
                        </div>
                      )}
                    />
                    {(authorDetail.publicationPagination?.totalPages || 1) > 1 ? (
                      <div className="mt-4 flex items-center justify-between gap-3 border-t border-white/10 pt-4 text-sm text-slate-400">
                        <span>
                          {t('paper2citation:states.publicationPage', {
                            page: authorDetail.publicationPagination?.page || authorPublicationPage,
                            total: authorDetail.publicationPagination?.totalPages || 1,
                            count: authorDetail.publicationPagination?.totalItems || authorDetail.profile.worksCount,
                          })}
                          {isPagingAuthorPublications ? (
                            <span className="ml-3 inline-flex items-center gap-2 text-cyan-300">
                              <Loader2 size={12} className="animate-spin" />
                              {t('paper2citation:states.loadingDetail')}
                            </span>
                          ) : null}
                        </span>
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => selectedAuthorCandidate && void onLoadAuthorDetail(selectedAuthorCandidate, Math.max(1, authorPublicationPage - 1))}
                            disabled={isLoadingAuthorDetail || authorPublicationPage <= 1}
                            className="rounded-xl border border-white/10 px-3 py-2 text-sm text-slate-200 transition hover:border-white/20 hover:bg-white/[0.06] disabled:cursor-not-allowed disabled:opacity-40"
                          >
                            {t('paper2citation:actions.prevPage')}
                          </button>
                          <button
                            onClick={() => selectedAuthorCandidate && void onLoadAuthorDetail(selectedAuthorCandidate, Math.min(authorDetail.publicationPagination?.totalPages || 1, authorPublicationPage + 1))}
                            disabled={isLoadingAuthorDetail || authorPublicationPage >= (authorDetail.publicationPagination?.totalPages || 1)}
                            className="rounded-xl border border-white/10 px-3 py-2 text-sm text-slate-200 transition hover:border-white/20 hover:bg-white/[0.06] disabled:cursor-not-allowed disabled:opacity-40"
                          >
                            {t('paper2citation:actions.nextPage')}
                          </button>
                        </div>
                      </div>
                    ) : null}
                  </SectionCard>
                </div>

                <div className="space-y-6">
                  <SectionCard
                    title={t('paper2citation:sections.authorProfileStats.title')}
                    subtitle={t('paper2citation:sections.authorProfileStats.subtitle')}
                    icon={<GraduationCap size={18} />}
                  >
                    <StatGrid items={authorDetail.profileStats} emptyText={t('paper2citation:states.noStats')} />
                  </SectionCard>

                  <SectionCard
                    title={t('paper2citation:sections.publicationStats.title')}
                    subtitle={t('paper2citation:sections.publicationStats.subtitle')}
                    icon={<BookOpen size={18} />}
                  >
                    <StatGrid items={authorDetail.publicationStats} emptyText={t('paper2citation:states.noStats')} />
                  </SectionCard>

                  <SectionCard
                    title={t('paper2citation:sections.authorCitationStats.title')}
                    subtitle={t('paper2citation:sections.authorCitationStats.subtitle')}
                    icon={<Users size={18} />}
                  >
                    <StatGrid items={authorDetail.citationStats} emptyText={t('paper2citation:states.noStats')} />
                    {authorDetail.citationBuildInfo.length ? (
                      <div className="mt-4 rounded-2xl border border-slate-200 bg-slate-50 p-4">
                        <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">
                          {t('paper2citation:sections.citationBuildInfo.title')}
                        </div>
                        <div className="mt-2 space-y-2 text-sm text-slate-600">
                          {authorDetail.citationBuildInfo.map((line) => (
                            <div key={line}>{line}</div>
                          ))}
                        </div>
                      </div>
                    ) : null}
                  </SectionCard>

                  <SectionCard
                    title={t('paper2citation:sections.honors.title')}
                    subtitle={t('paper2citation:sections.honors.subtitle')}
                    icon={<Trophy size={18} />}
                  >
                    <StatGrid items={authorDetail.honorsStats} emptyText={t('paper2citation:states.noHonors')} />
                    {authorDetail.bestEffortNotice ? (
                      <div className="mt-4">
                        <NoticeBanner>{authorDetail.bestEffortNotice}</NoticeBanner>
                      </div>
                    ) : null}
                  </SectionCard>

                  <SectionCard
                    title={t('paper2citation:sections.institutions.title')}
                    subtitle={t('paper2citation:sections.institutions.subtitle')}
                    icon={<Building2 size={18} />}
                  >
                    <TableLikeList
                      items={authorDetail.citingInstitutions}
                      emptyText={t('paper2citation:states.noInstitutions')}
                      renderItem={(institution: CitationInstitutionStat, index) => (
                        <div key={`${institution.institution}-${index}`} className="flex items-center justify-between rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-3">
                          <div>
                            <div className="text-sm font-medium text-white">{institution.institution}</div>
                            {institution.country ? <div className="text-xs text-slate-500">{institution.country}</div> : null}
                          </div>
                          <div className="text-sm font-semibold text-cyan-300">{formatNumber(institution.count)}</div>
                        </div>
                      )}
                    />
                  </SectionCard>

                  <SectionCard
                    title={t('paper2citation:sections.authors.title')}
                    subtitle={t('paper2citation:sections.authors.subtitle')}
                    icon={<Users size={18} />}
                  >
                    <TableLikeList
                      items={authorDetail.citingAuthors}
                      emptyText={t('paper2citation:states.noAuthors')}
                      renderItem={(item: CitationAuthorItem, index) => (
                        <div key={`${item.name}-${index}`} className="rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-3">
                          <div className="flex items-start justify-between gap-3">
                            <div>
                              <div className="text-sm font-medium text-white">{item.name}</div>
                              <div className="text-xs text-slate-500">{item.affiliations.join(', ') || t('paper2citation:states.noAffiliation')}</div>
                            </div>
                            <div className="text-right text-xs text-slate-400">
                              {item.citedCount != null ? `${formatNumber(item.citedCount)} cites` : ''}
                            </div>
                          </div>
                          {item.honors?.length ? (
                            <div className="mt-2 flex flex-wrap gap-2">
                              {item.honors.map((honor) => (
                                <span key={honor} className="rounded-full border border-amber-300/25 bg-amber-500/10 px-2 py-1 text-[11px] text-amber-100">
                                  {honor}
                                </span>
                              ))}
                            </div>
                          ) : null}
                        </div>
                      )}
                    />
                  </SectionCard>
                </div>
              </div>
            ) : null}
          </>
        ) : (
          isLoadingDetail && !paperDetail ? (
            <LoadingPanel
              title={t('paper2citation:states.loadingPaperTitle', { query: loadingPaperValue || t('paper2citation:states.loadingFallbackPaper') })}
              subtitle={t('paper2citation:states.loadingPaperSubtitle')}
            />
          ) : paperDetail ? (
            <div className="grid gap-6 xl:grid-cols-[minmax(0,1.15fr)_minmax(0,0.85fr)]">
              <div className="space-y-6">
                <SectionCard
                  title={paperDetail.paper.title}
                  subtitle={paperSummary || t('paper2citation:states.noSummary')}
                  icon={<BookOpen size={18} />}
                >
                  <div className="space-y-4 text-sm text-slate-300">
                    <div>{paperDetail.paper.authors.join(', ') || t('paper2citation:states.noAuthors')}</div>
                    <div className="flex flex-wrap gap-2">
                      {paperDetail.paper.institutions.length ? paperDetail.paper.institutions.map((institution) => (
                        <span key={institution} className="rounded-full border border-white/10 bg-white/[0.05] px-3 py-1 text-xs text-slate-200">
                          {institution}
                        </span>
                      )) : <span className="text-slate-500">{t('paper2citation:states.noInstitutions')}</span>}
                    </div>
                    <div className="flex flex-wrap gap-4 text-xs text-slate-400">
                      {paperDetail.paper.doi ? <span>{paperDetail.paper.doi}</span> : null}
                      {paperDetail.paper.citedByCount != null ? <span>{formatNumber(paperDetail.paper.citedByCount)} cites</span> : null}
                    </div>
                    {(paperDetail.paper.url || paperDetail.paper.doi) ? (
                      <a
                        href={paperDetail.paper.url || `https://doi.org/${paperDetail.paper.doi}`}
                        target="_blank"
                        rel="noreferrer"
                        className="inline-flex items-center gap-2 text-cyan-300 hover:text-cyan-200"
                      >
                        <ExternalLink size={15} />
                        {t('paper2citation:actions.openPaper')}
                      </a>
                    ) : null}
                  </div>
                </SectionCard>

                <NotableCitersCard
                  title={t('paper2citation:sections.notableCiters.title')}
                  subtitle={t('paper2citation:sections.notableCiters.subtitle')}
                  items={paperDetail.matchedHonorees}
                />

                <SectionCard
                  title={t('paper2citation:sections.citingWorks.title')}
                  subtitle={t('paper2citation:sections.citingWorks.subtitle')}
                  icon={<BookOpen size={18} />}
                >
                  <TableLikeList
                    items={paperDetail.citingWorks}
                    emptyText={t('paper2citation:states.noCitingWorks')}
                    renderItem={(work: CitationWorkItem, index) => {
                      const workKey = getWorkKey(work);
                      const contextDetail = paperContexts[workKey];
                      const isLoadingContext = loadingPaperContextKey === workKey;
                      return (
                        <div key={`${work.id || work.title}-${index}`} className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
                          <div className="flex items-start justify-between gap-4">
                            <div>
                              <div className="text-sm font-semibold text-white">{work.title}</div>
                              <div className="mt-1 text-xs text-slate-400">{buildMetaLine([work.year, work.venue])}</div>
                              {work.authors.length ? <div className="mt-2 text-xs text-slate-500">{work.authors.join(', ')}</div> : null}
                              {work.institutions?.length ? <div className="mt-1 text-xs text-slate-500">{work.institutions.join(', ')}</div> : null}
                            </div>
                            <button
                              onClick={() => void onLoadPaperContext(work)}
                              disabled={isLoadingContext}
                              className="inline-flex shrink-0 items-center gap-2 rounded-xl border border-cyan-300/30 bg-cyan-400/10 px-3 py-2 text-xs font-medium text-cyan-100 transition hover:border-cyan-200/40 hover:bg-cyan-400/15 disabled:cursor-not-allowed disabled:opacity-60"
                            >
                              {isLoadingContext ? <Loader2 size={14} className="animate-spin" /> : <BookOpen size={14} />}
                              {isLoadingContext ? t('paper2citation:actions.extractingContext') : t('paper2citation:actions.extractContext')}
                            </button>
                          </div>

                          {contextDetail ? (
                            <div className="mt-4 space-y-3 rounded-2xl border border-cyan-300/15 bg-slate-950/40 p-4">
                              {contextDetail.summary ? (
                                <div>
                                  <div className="text-xs uppercase tracking-[0.18em] text-cyan-200">
                                    {t('paper2citation:sections.citationContext.summaryTitle')}
                                  </div>
                                  <p className="mt-2 text-sm leading-6 text-slate-200">{contextDetail.summary}</p>
                                </div>
                              ) : null}

                              {contextDetail.citationIntents.length ? (
                                <div className="flex flex-wrap gap-2">
                                  {contextDetail.citationIntents.map((intent) => (
                                    <span key={intent} className="rounded-full border border-cyan-300/20 bg-cyan-400/10 px-2 py-1 text-[11px] text-cyan-100">
                                      {intent.replace(/_/g, ' ')}
                                    </span>
                                  ))}
                                </div>
                              ) : null}

                              {contextDetail.targetReferenceMatch?.matched ? (
                                <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3 text-xs text-slate-300">
                                  <div className="font-medium text-white">{t('paper2citation:sections.citationContext.referenceTitle')}</div>
                                  <div className="mt-1 text-slate-400">
                                    {buildMetaLine([
                                      contextDetail.targetReferenceMatch.marker,
                                      contextDetail.targetReferenceMatch.matchedBy,
                                      contextDetail.targetReferenceMatch.confidence != null
                                        ? `${Math.round(contextDetail.targetReferenceMatch.confidence * 100)}%`
                                        : '',
                                    ])}
                                  </div>
                                  {contextDetail.targetReferenceMatch.referenceText ? (
                                    <div className="mt-2 line-clamp-4">{contextDetail.targetReferenceMatch.referenceText}</div>
                                  ) : null}
                                </div>
                              ) : null}

                              {contextDetail.contexts.length ? (
                                <div className="space-y-3">
                                  {contextDetail.contexts.map((context: CitationContextItem, contextIndex) => (
                                    <div key={`${workKey}-context-${contextIndex}`} className="rounded-xl border border-white/10 bg-white/[0.03] p-3">
                                      <div className="flex flex-wrap items-center gap-2 text-[11px] uppercase tracking-[0.16em] text-slate-400">
                                        {context.section ? <span>{context.section}</span> : null}
                                        {context.marker ? <span>{context.marker}</span> : null}
                                        {context.confidence != null ? <span>{Math.round(context.confidence * 100)}%</span> : null}
                                      </div>
                                      <div className="mt-2 text-sm font-medium text-white">{context.sentence}</div>
                                      {context.paragraph && context.paragraph !== context.sentence ? (
                                        <p className="mt-2 whitespace-pre-wrap text-xs leading-6 text-slate-400">{context.paragraph}</p>
                                      ) : null}
                                    </div>
                                  ))}
                                </div>
                              ) : (
                                <div className="rounded-xl border border-dashed border-white/10 bg-white/[0.02] px-3 py-3 text-xs text-slate-400">
                                  {t('paper2citation:states.noCitationContexts')}
                                </div>
                              )}

                              <div className="flex flex-wrap items-center gap-3 text-xs text-slate-400">
                                {contextDetail.sourceUrl ? (
                                  <a
                                    href={contextDetail.sourceUrl}
                                    target="_blank"
                                    rel="noreferrer"
                                    className="inline-flex items-center gap-2 text-cyan-300 hover:text-cyan-200"
                                  >
                                    <ExternalLink size={13} />
                                    {t('paper2citation:actions.openContextSource')}
                                  </a>
                                ) : null}
                                {contextDetail.bestEffortNotice ? <span>{contextDetail.bestEffortNotice}</span> : null}
                              </div>
                            </div>
                          ) : null}
                        </div>
                      );
                    }}
                  />
                </SectionCard>
              </div>

              <div className="space-y-6">
                <SectionCard
                  title={t('paper2citation:sections.citationStats.title')}
                  subtitle={t('paper2citation:sections.citationStats.subtitle')}
                  icon={<BookOpen size={18} />}
                >
                  <StatGrid items={paperDetail.citationStats} emptyText={t('paper2citation:states.noStats')} />
                </SectionCard>

                <SectionCard
                  title={t('paper2citation:sections.honors.title')}
                  subtitle={t('paper2citation:sections.honors.subtitle')}
                  icon={<Trophy size={18} />}
                >
                  <StatGrid items={paperDetail.honorsStats} emptyText={t('paper2citation:states.noHonors')} />
                  {paperDetail.bestEffortNotice ? (
                    <div className="mt-4">
                      <NoticeBanner>{paperDetail.bestEffortNotice}</NoticeBanner>
                    </div>
                  ) : null}
                </SectionCard>

                <SectionCard
                  title={t('paper2citation:sections.institutions.title')}
                  subtitle={t('paper2citation:sections.institutions.subtitle')}
                  icon={<Building2 size={18} />}
                >
                  <TableLikeList
                    items={paperDetail.citingInstitutions}
                    emptyText={t('paper2citation:states.noInstitutions')}
                    renderItem={(institution: CitationInstitutionStat, index) => (
                      <div key={`${institution.institution}-${index}`} className="flex items-center justify-between rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-3">
                        <div>
                          <div className="text-sm font-medium text-white">{institution.institution}</div>
                          {institution.country ? <div className="text-xs text-slate-500">{institution.country}</div> : null}
                        </div>
                        <div className="text-sm font-semibold text-cyan-300">{formatNumber(institution.count)}</div>
                      </div>
                    )}
                  />
                </SectionCard>

                <SectionCard
                  title={t('paper2citation:sections.authors.title')}
                  subtitle={t('paper2citation:sections.authors.subtitle')}
                  icon={<Users size={18} />}
                >
                  <TableLikeList
                    items={paperDetail.citingAuthors}
                    emptyText={t('paper2citation:states.noAuthors')}
                    renderItem={(item: CitationAuthorItem, index) => (
                      <div key={`${item.name}-${index}`} className="rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-3">
                        <div className="flex items-start justify-between gap-3">
                          <div>
                            <div className="text-sm font-medium text-white">{item.name}</div>
                            <div className="text-xs text-slate-500">{item.affiliations.join(', ') || t('paper2citation:states.noAffiliation')}</div>
                          </div>
                          <div className="text-right text-xs text-slate-400">
                            {item.citedCount != null ? `${formatNumber(item.citedCount)} cites` : ''}
                          </div>
                        </div>
                        {item.honors?.length ? (
                          <div className="mt-2 flex flex-wrap gap-2">
                            {item.honors.map((honor) => (
                              <span key={honor} className="rounded-full border border-amber-300/25 bg-amber-500/10 px-2 py-1 text-[11px] text-amber-100">
                                {honor}
                              </span>
                            ))}
                          </div>
                        ) : null}
                      </div>
                    )}
                  />
                </SectionCard>
              </div>
            </div>
          ) : (
            <section className="rounded-3xl border border-dashed border-white/10 bg-slate-950/40 p-8 text-center text-sm text-slate-500">
              {isLoadingDetail ? t('paper2citation:states.loadingDetail') : t('paper2citation:states.emptyPaper')}
            </section>
          )
        )}
      </div>
    </div>
  );
};

export default Paper2CitationPage;
