import { ExternalLink, FileText, Search, Filter, BookOpen } from 'lucide-react';
import { useState, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

interface Paper {
  title?: string;
  arxiv_id?: string;
  abs_url?: string;
  pdf_url?: string;
  authors?: string[];
  abstract?: string;
  analysis?: string;
}

interface PaperListProps {
  searchedPapers?: Paper[] | any[];
  selectedPapers?: Paper[] | any[];
  analyzedPapers?: Paper[] | any[];
}

/** 统一后端返回的论文对象格式（兼容 snake_case / 异常结构） */
function normalizePaper(p: any, index: number): Paper {
  if (!p || typeof p !== 'object') {
    return { title: `论文 ${index + 1}`, authors: [], abstract: '', analysis: '' };
  }
  return {
    title: p.title ?? p.Title ?? '',
    arxiv_id: p.arxiv_id ?? p.arxiv_id ?? '',
    abs_url: p.abs_url ?? p.abs_url ?? '',
    pdf_url: p.pdf_url ?? p.pdf_url ?? '',
    authors: Array.isArray(p.authors) ? p.authors : Array.isArray(p.Authors) ? p.Authors : [],
    abstract: p.abstract ?? p.Abstract ?? '',
    analysis: p.analysis ?? p.Analysis ?? '',
  };
}

function normalizePapers(list: any[]): Paper[] {
  return (list || []).map((p, i) => normalizePaper(p, i));
}

const PaperList = ({ searchedPapers = [], selectedPapers = [], analyzedPapers = [] }: PaperListProps) => {
  const { t } = useTranslation(['paper2rebuttal']);
  const searched = useMemo(() => normalizePapers(searchedPapers), [searchedPapers]);
  const selected = useMemo(() => normalizePapers(selectedPapers), [selectedPapers]);
  const analyzed = useMemo(() => normalizePapers(analyzedPapers), [analyzedPapers]);

  const hasSearched = searched.length > 0;
  const hasSelected = selected.length > 0;
  const hasAnalyzed = analyzed.length > 0;
  const defaultTab: 'searched' | 'selected' | 'analyzed' = hasAnalyzed ? 'analyzed' : hasSelected ? 'selected' : 'searched';
  const [activeTab, setActiveTab] = useState<'searched' | 'selected' | 'analyzed'>(defaultTab);
  const [expandedPaper, setExpandedPaper] = useState<string | null>(null);

  const tabs = [
    { id: 'searched' as const, label: t('paper2rebuttal:papers.searched'), count: searched.length, icon: Search },
    { id: 'selected' as const, label: t('paper2rebuttal:papers.selected'), count: selected.length, icon: Filter },
    { id: 'analyzed' as const, label: t('paper2rebuttal:papers.analyzed'), count: analyzed.length, icon: BookOpen },
  ];

  const getCurrentPapers = (): Paper[] => {
    switch (activeTab) {
      case 'searched':
        return searched;
      case 'selected':
        return selected;
      case 'analyzed':
        return analyzed;
      default:
        return [];
    }
  };

  const papers = getCurrentPapers();
  const hasAnyPapers = hasSearched || hasSelected || hasAnalyzed;

  return (
    <div className="space-y-4">
      {/* Tabs */}
      <div className="flex gap-2 border-b border-primary-100">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-4 py-2 flex items-center gap-2 border-b-2 transition-colors ${
                activeTab === tab.id
                  ? 'border-blue-500 text-blue-400'
                  : 'border-transparent text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
              }`}
            >
              <Icon className="w-4 h-4" />
              <span>{tab.label}</span>
              {tab.count > 0 && (
                <span className={`px-2 py-0.5 rounded text-xs ${
                  activeTab === tab.id ? 'bg-blue-500/20' : 'bg-gray-500/20'
                }`}>
                  {tab.count}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* Papers list */}
      <div className="space-y-3 max-h-96 overflow-y-auto">
        {papers.length === 0 ? (
          <div className="py-8 text-center text-sm text-[var(--text-secondary)]">
            {t('paper2rebuttal:papers.noPapers')}
          </div>
        ) : (
          papers.map((paper, index) => {
            const paperId = `${activeTab}-${index}`;
            const isExpanded = expandedPaper === paperId;

            return (
              <div
                key={index}
                className="rounded-lg border border-primary-100 bg-white/82 p-4 transition-colors hover:bg-white"
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    <h4 className="mb-1 line-clamp-2 font-semibold text-[var(--text-primary)]">
                      {paper.title || t('paper2rebuttal:papers.noTitle')}
                    </h4>
                    
                    {paper.authors && paper.authors.length > 0 && (
                      <div className="mb-2 text-sm text-[var(--text-secondary)]">
                        {paper.authors.slice(0, 3).join(', ')}
                        {paper.authors.length > 3 && ' et al.'}
                      </div>
                    )}
                    
                    {paper.abstract && (
                      <p className="mb-2 line-clamp-2 text-sm text-[var(--text-secondary)]">
                        {paper.abstract}
                      </p>
                    )}
                    
                    {paper.analysis && (
                      <div className="mt-2">
                        <button
                          onClick={() => setExpandedPaper(isExpanded ? null : paperId)}
                          className="text-sm text-blue-400 hover:text-blue-300"
                        >
                          {isExpanded ? t('paper2rebuttal:papers.hideAnalysis') : t('paper2rebuttal:papers.showAnalysis')}
                        </button>
                        {isExpanded && (
                          <div className="mt-2 rounded border border-blue-200 bg-blue-50 p-3 text-sm text-[var(--text-secondary)] whitespace-pre-wrap">
                            {paper.analysis}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                  
                  <div className="flex-shrink-0 flex gap-2">
                    {paper.abs_url && (
                      <a
                        href={paper.abs_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="p-2 bg-blue-500/20 hover:bg-blue-500/30 rounded transition-colors"
                        title={t('paper2rebuttal:papers.viewAbstract')}
                      >
                        <FileText className="w-4 h-4 text-blue-300" />
                      </a>
                    )}
                    {paper.pdf_url && (
                      <a
                        href={paper.pdf_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="p-2 bg-green-500/20 hover:bg-green-500/30 rounded transition-colors"
                        title={t('paper2rebuttal:papers.downloadPdf')}
                      >
                        <ExternalLink className="w-4 h-4 text-green-300" />
                      </a>
                    )}
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};

export default PaperList;
