export type CitationMode = 'author' | 'paper';

export interface CitationAuthorCandidate {
  openalexAuthorId: string;
  dblpId?: string;
  source?: string;
  displayName: string;
  affiliations: string[];
  worksCount: number;
  citedByCount: number;
  hIndex?: number | null;
  summary?: string;
}

export interface CitationStatItem {
  label: string;
  value: number;
}

export interface CitationInstitutionStat {
  institution: string;
  count: number;
  country?: string;
}

export interface CitationAuthorItem {
  name: string;
  openalexAuthorId?: string;
  affiliations: string[];
  citedCount?: number;
  honors?: string[];
}

export interface CitationMatchedHonoree {
  name: string;
  canonicalName?: string;
  honorLabel: string;
  openalexAuthorId?: string;
  affiliations: string[];
  citingWorksCount?: number;
}

export interface CitationWorkItem {
  id?: string;
  title: string;
  year?: number | null;
  venue?: string;
  doi?: string;
  citedByCount?: number;
  authors: string[];
  institutions?: string[];
  landingPageUrl?: string;
}

export interface CitationPagination {
  page: number;
  pageSize: number;
  totalItems: number;
  totalPages: number;
}

export interface CitationAuthorDetail {
  profile: {
    openalexAuthorId: string;
    dblpId?: string;
    displayName: string;
    affiliations: string[];
    titles?: string[];
    honors?: string[];
    homepage?: string;
    worksCount: number;
    citedByCount: number;
    hIndex?: number | null;
    summary?: string;
  };
  bestEffortNotice?: string;
  profileStats: CitationStatItem[];
  publicationStats: CitationStatItem[];
  citationStats: CitationStatItem[];
  citationBuildInfo: string[];
  publicationPagination?: CitationPagination;
  honorsStats: CitationStatItem[];
  publications: CitationWorkItem[];
  citingInstitutions: CitationInstitutionStat[];
  citingAuthors: CitationAuthorItem[];
  matchedHonorees: CitationMatchedHonoree[];
}

export interface CitationPaperDetail {
  paper: {
    id?: string;
    title: string;
    year?: number | null;
    venue?: string;
    doi?: string;
    url?: string;
    authors: string[];
    institutions: string[];
    citedByCount?: number;
  };
  bestEffortNotice?: string;
  citationStats: CitationStatItem[];
  honorsStats: CitationStatItem[];
  citingInstitutions: CitationInstitutionStat[];
  citingAuthors: CitationAuthorItem[];
  citingWorks: CitationWorkItem[];
  matchedHonorees: CitationMatchedHonoree[];
}

export interface CitationContextItem {
  section?: string;
  sentence: string;
  paragraph: string;
  marker?: string;
  confidence?: number;
}

export interface CitationPaperContextDetail {
  sourceUrl?: string;
  bestEffortNotice?: string;
  summary?: string;
  citationIntents: string[];
  targetReferenceMatch?: {
    matched?: boolean;
    matchedBy?: string;
    marker?: string;
    referenceText?: string;
    confidence?: number;
  };
  citingPaper?: {
    id?: string;
    title?: string;
    venue?: string;
    year?: number | null;
    doi?: string;
  };
  contexts: CitationContextItem[];
}
