import { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { Upload, FileText, CheckCircle, RefreshCw, Download, ArrowRight, Loader2, History, ChevronLeft, Clock } from 'lucide-react';
import { useAuthStore } from '../../stores/authStore';
import { getApiSettings, saveApiSettings } from '../../services/apiSettingsService';
import { API_KEY, DEFAULT_LLM_API_URL, API_URL_OPTIONS, getPurchaseUrl } from '../../config/api';
import { DEFAULT_PAPER2REBUTTAL_MODEL, PAPER2REBUTTAL_MODELS, withModelOptions } from '../../config/models';
import ReactMarkdown from 'react-markdown';
import Timeline from './Timeline';
import TodoList from './TodoList';
import PaperList from './PaperList';
import QRCodeTooltip from '../QRCodeTooltip';

interface TodoItem {
  id: number;
  title: string;
  description: string;
  type: 'experiment' | 'analysis' | 'clarification' | 'comparison' | 'ablation';
  status: 'pending' | 'completed' | 'in_progress';
  related_papers?: string[];
}

interface HistoryItem {
  timestamp: string;
  revision: number;
  strategy_text?: string;
  todo_list?: TodoItem[];
  draft_response?: string;
  feedback?: string;
}

interface Question {
  question_id: number;
  question_text: string;
  strategy: string;
  strategy_text?: string;
  todo_list?: TodoItem[];
  draft_response?: string;
  revision_count: number;
  is_satisfied: boolean;
  feedback_history: Array<{ feedback: string; timestamp: string }>;
  searched_papers?: any[];
  selected_papers?: any[];
  analyzed_papers?: any[];
  history?: HistoryItem[];
}

interface Session {
  session_id: string;
  questions: Question[];
  final_rebuttal: string;
}

interface ParsedReviewItem {
  id: string;
  content: string;
}

interface HistorySession {
  session_id: string;
  created_at?: string;
  updated_at?: string;
  status?: string;
  total_questions?: number;
  processed_questions?: number;
  satisfied_questions?: number;
  has_final?: boolean;
}

const REVIEW_TEXT_EXAMPLES = [
  {
    title: '示例 1 · Review 1/2',
    text: `Review 1:
1) 缺少消融实验，请补充模块 A/B 的对比。
2) 数据集划分未说明，请补充训练/验证/测试比例。

Review 2:
- 运行时间对比不充分，请补充与 baseline 的耗时。
- 图 2 标注不清，请增加图例与轴标题。`,
  },
  {
    title: '示例 2 · Q1/Q2 结构',
    text: `Q1: 方法在小样本设置下表现如何？请补充实验结果。
Q2: 参数选择依据是什么？建议补充敏感性分析。
Q3: 与方法 X 的差异和优势需要更明确的讨论。`,
  },
  {
    title: '示例 3 · Major/Minor',
    text: `Major comments:
1. 结果表缺少显著性检验，请补充 p-value 或置信区间。
2. 讨论部分未覆盖局限性，请增加对潜在失败场景的说明。

Minor comments:
- 第 3 节存在拼写错误，请统一术语。`,
  },
];

const Paper2RebuttalPage = () => {
  const { t } = useTranslation(['common', 'paper2rebuttal']);
  const { user } = useAuthStore();
  const [step, setStep] = useState<'upload' | 'review_check' | 'processing' | 'review' | 'generating' | 'result'>('upload');
  const [session, setSession] = useState<Session | null>(null);
  const [currentQuestionIdx, setCurrentQuestionIdx] = useState(0);
  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const [reviewFile, setReviewFile] = useState<File | null>(null);
  const [reviewInputMode, setReviewInputMode] = useState<'file' | 'text'>('file');
  const [reviewTextDirect, setReviewTextDirect] = useState('');
  const [parsedReviews, setParsedReviews] = useState<ParsedReviewItem[]>([]);
  const [reviewTextForStart, setReviewTextForStart] = useState('');
  const [llmApiUrl, setLlmApiUrl] = useState(DEFAULT_LLM_API_URL);
  const [apiKey, setApiKey] = useState('');
  const [model, setModel] = useState(DEFAULT_PAPER2REBUTTAL_MODEL);
  const modelOptions = withModelOptions(PAPER2REBUTTAL_MODELS, model);
  const [feedback, setFeedback] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [unsatisfiedQuestionIds, setUnsatisfiedQuestionIds] = useState<number[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [selectedHistoryIndex, setSelectedHistoryIndex] = useState<number | null>(null);
  const [showPapers, setShowPapers] = useState(false);
  const [canGoBack, setCanGoBack] = useState(false);
  const [exportingZip, setExportingZip] = useState(false);
  const [historySessions, setHistorySessions] = useState<HistorySession[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState('');

  // 加载保存的 API 设置
  useEffect(() => {
    if (user?.id) {
      const savedSettings = getApiSettings(user.id);
      if (savedSettings) {
        setLlmApiUrl(savedSettings.apiUrl || DEFAULT_LLM_API_URL);
        setApiKey(savedSettings.apiKey || '');
      }
    }
  }, [user?.id]);

  useEffect(() => {
    if (step === 'upload') {
      fetchHistory();
    }
  }, [step, user?.id, user?.email]);

  const addLog = (message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => {
      const newLogs = [...prev, `${timestamp}: ${message}`];
      // Keep last 100 logs
      return newLogs.slice(-100);
    });
  };

  const fetchHistory = async () => {
    setHistoryLoading(true);
    setHistoryError('');
    try {
      const params = new URLSearchParams();
      if (user?.email || user?.id) {
        params.append('email', user?.email || user?.id || '');
      }
      const url = params.toString()
        ? `/api/v1/paper2rebuttal/history?${params.toString()}`
        : '/api/v1/paper2rebuttal/history';
      const response = await fetch(url, {
        headers: { 'X-API-Key': API_KEY },
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data?.detail || t('paper2rebuttal:errors.fetchSessionFailed'));
      }
      setHistorySessions(data.sessions || []);
    } catch (err: any) {
      setHistoryError(err.message || t('paper2rebuttal:errors.fetchSessionFailed'));
    } finally {
      setHistoryLoading(false);
    }
  };

  const handleLoadHistorySession = async (targetSessionId: string) => {
    setLoading(true);
    setError('');
    try {
      const response = await fetch(`/api/v1/paper2rebuttal/session/${targetSessionId}`, {
        headers: { 'X-API-Key': API_KEY },
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data?.detail || t('paper2rebuttal:errors.fetchSessionFailed'));
      }
      setSession(data);
      const questions = data?.questions || [];
      const firstPendingIdx = questions.findIndex((q: any) => !q.is_satisfied);
      const nextIdx = firstPendingIdx === -1 ? 0 : firstPendingIdx;
      setCurrentQuestionIdx(nextIdx);
      setCanGoBack(nextIdx > 0);
      setFeedback('');
      setSelectedHistoryIndex(null);
      setLogs([]);
      setStep(data.final_rebuttal ? 'result' : 'review');
    } catch (err: any) {
      setError(err.message || t('paper2rebuttal:errors.fetchSessionFailed'));
    } finally {
      setLoading(false);
    }
  };

  const handleParseReview = async () => {
    const hasFile = reviewInputMode === 'file' && reviewFile;
    const hasText = reviewInputMode === 'text' && reviewTextDirect.trim();
    if (!hasFile && !hasText) {
      setError(t('paper2rebuttal:errors.uploadReview'));
      return;
    }
    setLoading(true);
    setError('');
    try {
      const parseJson = async (response: Response) => {
        const text = await response.text();
        if (!text) return { data: null as any, raw: '' };
        try {
          return { data: JSON.parse(text), raw: text };
        } catch {
          return { data: null as any, raw: text };
        }
      };
      const formData = new FormData();
      if (reviewInputMode === 'file' && reviewFile) {
        formData.append('review_file', reviewFile);
      } else {
        formData.append('review_text', reviewTextDirect.trim());
      }
      // 所有形式的输入都传 API 配置，后端统一用 LLM 形式化为 review-1, review-2... 供 check
      if (apiKey && llmApiUrl) {
        formData.append('chat_api_url', llmApiUrl.trim());
        formData.append('api_key', apiKey);
        formData.append('model', model);
      }
      const response = await fetch('/api/v1/paper2rebuttal/parse-review', {
        method: 'POST',
        headers: { 'X-API-Key': API_KEY },
        body: formData,
      });
      const { data, raw } = await parseJson(response);
      if (!response.ok) {
        const message = data?.detail || data?.message || raw || `HTTP ${response.status}: ${t('paper2rebuttal:errors.parseReviewFailed')}`;
        throw new Error(message);
      }
      if (!data) {
        throw new Error(t('paper2rebuttal:errors.parseReviewEmpty'));
      }
      setParsedReviews(data.reviews || []);
      setReviewTextForStart(data.review_text || '');
      setStep('review_check');
    } catch (err: any) {
      setError(err.message || t('paper2rebuttal:errors.parseReviewFailed'));
    } finally {
      setLoading(false);
    }
  };

  const handleStartAnalysis = async () => {
    // 所有形式的评审输入都必须先经过「解析/预览评审」，得到形式化 review 后再开始分析
    if (!pdfFile || !reviewTextForStart.trim() || !apiKey || !llmApiUrl) {
      setError(t('paper2rebuttal:errors.needParsedReview'));
      return;
    }

    setLoading(true);
    setError('');
    setStep('processing');
    setLogs([]);
    addLog(t('paper2rebuttal:logs.startAnalysis'));

    // 保存 API 设置
    if (user?.id) {
      saveApiSettings(user.id, { apiUrl: llmApiUrl, apiKey });
    }

    try {
      const formData = new FormData();
      formData.append('pdf_file', pdfFile);
      if (reviewTextForStart.trim()) {
        formData.append('review_text', reviewTextForStart.trim());
      } else if (reviewFile) {
        formData.append('review_file', reviewFile);
      }
      if (user?.email || user?.id) {
        formData.append('email', user?.email || user?.id || '');
      }
      formData.append('chat_api_url', llmApiUrl.trim());
      formData.append('api_key', apiKey);
      formData.append('model', model);

      // Start the analysis (non-blocking)
      const response = await fetch('/api/v1/paper2rebuttal/start', {
        method: 'POST',
        headers: { 'X-API-Key': API_KEY },
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || '分析失败');
      }

      const data = await response.json();
      const sessionId = data.session_id;

      // Start listening to progress stream
      addLog(t('paper2rebuttal:logs.connectingProgress'));
      const progressUrl = `/api/v1/paper2rebuttal/progress/${sessionId}?x_api_key=${encodeURIComponent(API_KEY)}`;
      let eventSource: EventSource | null = null;
      let completed = false;
      
      try {
        console.log('[SSE] Creating EventSource:', progressUrl);
        eventSource = new EventSource(progressUrl);
        console.log('[SSE] EventSource created, readyState:', eventSource.readyState);
        
        eventSource.onmessage = (event) => {
          console.log('[SSE] Received message:', event.data);
          try {
            const progressData = JSON.parse(event.data);
            
            if (progressData.type === 'progress') {
              addLog(progressData.message);
            } else if (progressData.type === 'complete') {
              if (!completed) {
                completed = true;
                addLog(progressData.message);
                eventSource?.close();
                
                // Wait a bit for final data to be saved, then fetch
                // Increase wait time to ensure data is fully saved
                setTimeout(() => {
                  fetchSessionData(sessionId);
                }, 2000);
              }
            } else if (progressData.type === 'error') {
              setError(progressData.message);
              addLog(progressData.message);
              eventSource?.close();
              setLoading(false);
            } else if (progressData.type === 'timeout') {
              addLog(progressData.message);
              eventSource?.close();
              // SSE timeout: switch to polling, wait for all_questions_processed
              addLog('⚠️ SSE 超时，改用轮询等待全部问题处理完成...');
              const pollInterval = setInterval(() => {
                fetch(`/api/v1/paper2rebuttal/session/${sessionId}`, {
                  headers: { 'X-API-Key': API_KEY },
                })
                  .then((res) => res.json())
                  .then((data) => {
                    const hasQuestions = data.questions && data.questions.length > 0;
                    const allProcessed = data.all_questions_processed === true;
                    if (hasQuestions && allProcessed) {
                      clearInterval(pollInterval);
                      fetchSessionData(sessionId);
                    }
                  })
                  .catch(() => {});
              }, 3000);
              setTimeout(() => clearInterval(pollInterval), 30 * 60 * 1000);
            }
          } catch (e) {
            console.error('Failed to parse progress data:', e);
          }
        };
        
        eventSource.onerror = (error) => {
          console.error('[SSE] EventSource error:', error, 'readyState:', eventSource?.readyState);
          // ReadyState: 0=CONNECTING, 1=OPEN, 2=CLOSED
          if (eventSource?.readyState === EventSource.CLOSED && !completed) {
            eventSource.close();
            addLog(t('paper2rebuttal:logs.connectionClosed'));
            // Do NOT fetch immediately: backend may still be processing. Poll until all_questions_processed.
            console.log('[Polling] Starting polling due to SSE onerror (CLOSED)');
            const pollInterval = setInterval(() => {
              fetch(`/api/v1/paper2rebuttal/session/${sessionId}`, {
                headers: { 'X-API-Key': API_KEY },
              })
                .then((res) => res.json())
                .then((data) => {
                  const hasQuestions = data.questions && data.questions.length > 0;
                  const allProcessed = data.all_questions_processed === true;
                  console.log('[Polling] Check (onerror):', {
                    questions_count: data.questions?.length || 0,
                    all_questions_processed: allProcessed,
                    hasQuestions,
                  });
                  if (hasQuestions && allProcessed) {
                    console.log('[Polling] All processed, calling fetchSessionData');
                    clearInterval(pollInterval);
                    fetchSessionData(sessionId);
                  }
                })
                .catch(() => {});
            }, 3000);
            setTimeout(() => clearInterval(pollInterval), 30 * 60 * 1000);
          }
        };
      } catch (err) {
        console.error('[SSE] Failed to create EventSource:', err);
        addLog(t('paper2rebuttal:logs.cannotConnect'));
        console.log('[Polling] Starting fallback polling due to EventSource creation failure');
        // Fallback: poll for session data; only switch when ALL questions have strategy (avoid incomplete data)
        const pollInterval = setInterval(() => {
          fetch(`/api/v1/paper2rebuttal/session/${sessionId}`, {
            headers: { 'X-API-Key': API_KEY },
          })
            .then(res => res.json())
            .then(data => {
              const hasQuestions = data.questions && data.questions.length > 0;
              const allProcessed = data.all_questions_processed === true;
              console.log('[Polling] Check (catch):', {
                questions_count: data.questions?.length || 0,
                all_questions_processed: allProcessed,
                hasQuestions,
              });
              if (hasQuestions && allProcessed) {
                console.log('[Polling] All processed, calling fetchSessionData');
                clearInterval(pollInterval);
                fetchSessionData(sessionId);
              }
            })
            .catch(() => {});
        }, 3000);
        
        // Stop polling after 30 minutes
        setTimeout(() => clearInterval(pollInterval), 30 * 60 * 1000);
      }
      
    } catch (err: any) {
      setError(err.message || t('paper2rebuttal:errors.analysisFailed'));
      setStep('upload');
      addLog(t('paper2rebuttal:logs.error', { message: err.message }));
      setLoading(false);
    }
  };

  const fetchSessionData = async (sessionId: string, retryCount = 0) => {
    const maxRetries = 2;
    const retryDelayMs = 1500;
    try {
      setLoading(true);
      console.log(`[fetchSessionData] Called with sessionId=${sessionId}, retryCount=${retryCount}`);
      const response = await fetch(`/api/v1/paper2rebuttal/session/${sessionId}`, {
        headers: { 'X-API-Key': API_KEY },
      });

      if (!response.ok) {
        throw new Error(t('paper2rebuttal:errors.fetchSessionFailed'));
      }

      const data = await response.json();
      
      console.log('[fetchSessionData] Fetched session data:', {
        session_id: data.session_id,
        questions_count: data.questions?.length || 0,
        has_questions: !!data.questions,
        all_questions_processed: data.all_questions_processed,
      });
      
      // Validate data
      if (!data.questions || !Array.isArray(data.questions) || data.questions.length === 0) {
        console.error('[fetchSessionData] Invalid questions data:', data);
        throw new Error(t('paper2rebuttal:errors.noQuestionData'));
      }

      // If backend says not all questions processed yet (e.g. race after SSE "complete"), retry a few times
      if (data.all_questions_processed === false && retryCount < maxRetries) {
        console.log(`[fetchSessionData] Not all processed (${retryCount + 1}/${maxRetries}), retrying...`);
        addLog(t('paper2rebuttal:logs.retrying', { seconds: retryDelayMs / 1000, current: retryCount + 1, max: maxRetries }));
        await new Promise((r) => setTimeout(r, retryDelayMs));
        return fetchSessionData(sessionId, retryCount + 1);
      }
      
      // If still not processed after max retries, log warning but continue
      if (data.all_questions_processed === false) {
        console.warn(`[fetchSessionData] Not all questions processed after ${maxRetries} retries. Proceeding anyway.`);
        addLog(t('paper2rebuttal:logs.partialProcessing'));
      }

      addLog(t('paper2rebuttal:logs.analysisComplete', { count: data.questions.length }));
      
      // Ensure all questions have required fields
      const questionsWithDefaults = data.questions.map((q: any, idx: number) => {
        const question = {
          ...q,
          question_id: q.question_id || idx + 1,
          question_text: q.question_text || '',
          strategy_text: q.strategy_text || q.strategy || '',
          todo_list: q.todo_list || [],
          draft_response: q.draft_response || '',
          searched_papers: q.searched_papers || [],
          selected_papers: q.selected_papers || [],
          analyzed_papers: q.analyzed_papers || [],
          history: q.history || [],
          revision_count: q.revision_count || 0,
          is_satisfied: q.is_satisfied || false,
        };
        console.log(`Question ${idx + 1}:`, {
          question_id: question.question_id,
          has_text: !!question.question_text,
          has_strategy: !!question.strategy_text
        });
        return question;
      });
      
      setSession({
        session_id: data.session_id,
        questions: questionsWithDefaults,
        final_rebuttal: data.final_rebuttal || '',
      });
      setCurrentQuestionIdx(0);
      setSelectedHistoryIndex(null);
      setCanGoBack(false);
      setError(''); // Clear any previous errors
      setShowPapers(true); // 进入 review 时默认展示相关论文
      // Set step after a small delay to ensure state is updated
      setTimeout(() => {
        setStep('review');
      }, 100);
    } catch (err: any) {
      const errorMsg = err.message || t('paper2rebuttal:errors.fetchSessionFailed');
      setError(errorMsg);
      addLog(t('paper2rebuttal:logs.error', { message: errorMsg }));
      console.error('fetchSessionData error:', err);
      console.error('Session state:', { session, currentQuestionIdx, step });
      // Don't change step if there's an error, stay on processing or go back to upload
      if (step === 'processing') {
        // Keep showing processing screen with error
      } else {
        // If we're already on review but data is invalid, show error state
        if (step === 'review') {
          // Stay on review to show error message
        } else {
          setStep('upload');
        }
      }
    } finally {
      setLoading(false);
    }
  };

  const handleRevise = async () => {
    if (!session || !feedback.trim()) {
      setError(t('paper2rebuttal:errors.needFeedback'));
      return;
    }

    setLoading(true);
    setError('');

    try {
      const formData = new FormData();
      formData.append('session_id', session.session_id);
      formData.append('question_idx', currentQuestionIdx.toString());
      formData.append('feedback', feedback);
      formData.append('chat_api_url', llmApiUrl.trim());
      formData.append('api_key', apiKey);
      formData.append('model', model);

      const response = await fetch('/api/v1/paper2rebuttal/revise', {
        method: 'POST',
        headers: { 'X-API-Key': API_KEY },
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || t('paper2rebuttal:errors.revisionFailed'));
      }

      const data = await response.json();
      
      // Update session
      const updatedQuestions = [...session.questions];
      updatedQuestions[currentQuestionIdx] = {
        ...updatedQuestions[currentQuestionIdx],
        strategy: data.strategy,
        strategy_text: data.strategy_text || '',
        todo_list: data.todo_list || [],
        draft_response: data.draft_response || '',
        revision_count: data.revision_count,
      };
      setSession({ ...session, questions: updatedQuestions });
      setFeedback('');
      setSelectedHistoryIndex(null);
      addLog(t('paper2rebuttal:logs.strategyRevised', { count: data.revision_count }));
    } catch (err: any) {
      setError(err.message || t('paper2rebuttal:errors.revisionFailed'));
    } finally {
      setLoading(false);
    }
  };

  const handleNextQuestion = async () => {
    if (!session) return;

    setLoading(true);
    setError('');

    try {
      const formData = new FormData();
      formData.append('session_id', session.session_id);
      formData.append('question_idx', currentQuestionIdx.toString());

      const response = await fetch('/api/v1/paper2rebuttal/mark-satisfied', {
        method: 'POST',
        headers: { 'X-API-Key': API_KEY },
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || t('paper2rebuttal:errors.operationFailed'));
      }

      const updatedQuestions = [...session.questions];
      updatedQuestions[currentQuestionIdx] = {
        ...updatedQuestions[currentQuestionIdx],
        is_satisfied: true,
      };
      setSession({ ...session, questions: updatedQuestions });
      setUnsatisfiedQuestionIds(prev => prev.filter(id => id !== updatedQuestions[currentQuestionIdx].question_id));

      if (currentQuestionIdx + 1 < session.questions.length) {
        setCurrentQuestionIdx(currentQuestionIdx + 1);
        setFeedback('');
        setSelectedHistoryIndex(null);
        setCanGoBack(true);
      } else {
        // All questions done, generate final rebuttal
        await generateFinalRebuttal();
      }
    } catch (err: any) {
      setError(err.message || t('paper2rebuttal:errors.operationFailed'));
    } finally {
      setLoading(false);
    }
  };

  const handlePreviousQuestion = () => {
    if (currentQuestionIdx > 0) {
      setCurrentQuestionIdx(currentQuestionIdx - 1);
      setFeedback('');
      setSelectedHistoryIndex(null);
      setCanGoBack(currentQuestionIdx > 1);
    }
  };

  const generateFinalRebuttal = async () => {
    if (!session) return;

    setLoading(true);
    setError('');
    // Switch to generating step to show loading UI
    setStep('generating');
    setLogs([]); // Clear previous logs

    try {
      const formData = new FormData();
      formData.append('session_id', session.session_id);
      formData.append('chat_api_url', llmApiUrl.trim());
      formData.append('api_key', apiKey);
      formData.append('model', model);
      if (user?.email || user?.id) {
        formData.append('email', user?.email || user?.id || '');
      }

      addLog(t('paper2rebuttal:logs.startGenerating'));
      addLog(t('paper2rebuttal:logs.integrating'));
      
      const response = await fetch('/api/v1/paper2rebuttal/generate-final', {
        method: 'POST',
        headers: { 'X-API-Key': API_KEY },
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        const detail = errorData?.detail;
        if (detail && typeof detail === 'object') {
          const ids = Array.isArray(detail.unsatisfied_question_ids) ? detail.unsatisfied_question_ids : [];
          if (ids.length > 0) {
            setUnsatisfiedQuestionIds(ids);
          }
          const msg = detail.message || t('paper2rebuttal:errors.generateFailed');
          throw new Error(msg);
        }
        throw new Error(detail || t('paper2rebuttal:errors.generateFailed'));
      }

      addLog(t('paper2rebuttal:logs.generatingContent'));
      
      const data = await response.json();
      setSession({ ...session, final_rebuttal: data.final_rebuttal });
      setUnsatisfiedQuestionIds([]);
      addLog(t('paper2rebuttal:logs.generateComplete'));
      
      // Small delay to show success message
      setTimeout(() => {
        setStep('result');
      }, 500);
    } catch (err: any) {
      const errorMsg = err.message || t('paper2rebuttal:errors.generateFailed');
      setError(errorMsg);
      addLog(t('paper2rebuttal:logs.error', { message: errorMsg }));
      // Go back to review step on error
      setStep('review');
    } finally {
      setLoading(false);
    }
  };

  const handleExportZip = async () => {
    if (!session) return;
    setExportingZip(true);
    setError('');
    try {
      const response = await fetch('/api/v1/paper2rebuttal/export-zip', {
        method: 'POST',
        headers: {
          'X-API-Key': API_KEY,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: session.session_id,
          email: user?.email || user?.id || '',
          include_root_dir: true,
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || t('paper2rebuttal:errors.exportFailed'));
      }
      if (data.zip_path) {
        window.open(data.zip_path, '_blank');
      }
    } catch (err: any) {
      setError(err.message || t('paper2rebuttal:errors.exportFailed'));
    } finally {
      setExportingZip(false);
    }
  };

  const currentQuestion = session?.questions?.[currentQuestionIdx] || null;

  // Build global timeline nodes - always include upload step
  const getGlobalTimelineNodes = () => {
    const nodes: Array<{
      id: string;
      title: string;
      description: string;
      status: 'completed' | 'current' | 'pending';
    }> = [
      {
        id: 'upload',
        title: t('paper2rebuttal:steps.upload'),
        description: t('paper2rebuttal:steps.uploadDesc'),
        status: step === 'upload' ? 'current' : (step === 'review_check' || step === 'processing' || step === 'review' || step === 'generating' || step === 'result') ? 'completed' : 'pending',
      },
    ];

    if (step === 'review_check' || parsedReviews.length > 0) {
      nodes.push({
        id: 'review_check',
        title: t('paper2rebuttal:steps.reviewCheck'),
        description: t('paper2rebuttal:steps.reviewCheckDesc'),
        status: step === 'review_check' ? 'current' : (step === 'processing' || step === 'review' || step === 'generating' || step === 'result') ? 'completed' : 'pending',
      });
    }

    // Add analysis step when we have session or are past review_check
    if (session || step === 'processing' || step === 'review' || step === 'generating' || step === 'result') {
      nodes.push({
        id: 'analysis',
        title: t('paper2rebuttal:steps.analysis'),
        description: t('paper2rebuttal:steps.analysisDesc'),
        status: step === 'processing' ? 'current' : (step === 'review' || step === 'result') ? 'completed' : 'pending',
      });
    }

    // Add question nodes
    if (session?.questions) {
      session.questions.forEach((q, idx) => {
        const isCurrent = step === 'review' && currentQuestionIdx === idx;
        const isCompleted = (step === 'review' && currentQuestionIdx > idx) || step === 'generating' || step === 'result' || q.is_satisfied;
        nodes.push({
          id: `question-${q.question_id}`,
          title: t('paper2rebuttal:review.questionTitle', { current: q.question_id, total: session.questions.length }),
          description: q.is_satisfied ? t('paper2rebuttal:review.questionCompleted') : isCurrent ? t('paper2rebuttal:review.questionProcessing') : t('paper2rebuttal:review.questionPending'),
          status: isCurrent ? 'current' : isCompleted ? 'completed' : 'pending',
        });
      });
    }

    // Add generating step
    if (step === 'generating' || step === 'result') {
      nodes.push({
        id: 'generating',
        title: t('paper2rebuttal:steps.generating'),
        description: t('paper2rebuttal:steps.generatingDesc'),
        status: step === 'generating' ? 'current' : 'completed',
      });
    }

    // Add final result node
    nodes.push({
      id: 'result',
      title: t('paper2rebuttal:steps.result'),
      description: t('paper2rebuttal:steps.resultDesc'),
      status: step === 'result' ? 'current' : (step === 'generating' ? 'pending' : 'pending'),
    });

    return nodes;
  };

  const globalTimelineNodes = getGlobalTimelineNodes();
  const currentTimelineIndex = globalTimelineNodes.findIndex(n => n.status === 'current');

  return (
    <div className="w-full h-full overflow-y-auto p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center space-y-3">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/5 border border-white/10 text-[11px] text-slate-300">
            <span className="w-2 h-2 rounded-full bg-[#0A84FF] shadow-[0_0_10px_rgba(10,132,255,0.6)]" />
            · {t('paper2rebuttal:header.badge')}
          </div>
          <h1 className="text-3xl md:text-4xl font-bold text-white">{t('paper2rebuttal:header.title')}</h1>
          <p className="text-gray-400">{t('paper2rebuttal:header.description')}</p>
        </div>

        {/* Global Timeline - Always show at top, horizontal */}
        {globalTimelineNodes.length > 0 && (
          <div className="glass-dark rounded-2xl p-6">
            <Timeline
              nodes={globalTimelineNodes}
              currentIndex={currentTimelineIndex >= 0 ? currentTimelineIndex : 0}
              horizontal={true}
            />
          </div>
        )}

        {/* Upload Step */}
        {step === 'upload' && (
          <div className="glass-dark rounded-3xl p-6 md:p-8 space-y-6 shadow-[0_20px_60px_rgba(0,0,0,0.35)]">
            <div className="flex items-center justify-between flex-wrap gap-3">
              <h2 className="text-xl font-bold text-white">{t('paper2rebuttal:upload.title')}</h2>
              <span className="text-xs text-gray-500">{t('paper2rebuttal:upload.supportedFormats')}</span>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  {t('paper2rebuttal:upload.apiConfig')}
                </label>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-4 rounded-2xl bg-white/5 border border-white/10">
                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <label className="block text-xs text-gray-400">{t('paper2rebuttal:upload.apiUrl')}</label>
                      <QRCodeTooltip>
                        <a
                          href={getPurchaseUrl(llmApiUrl)}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-[10px] text-purple-300 hover:text-purple-200 hover:underline"
                        >
                          {t('paper2rebuttal:upload.buyLink')}
                        </a>
                      </QRCodeTooltip>
                    </div>
                    {API_URL_OPTIONS.length > 1 ? (
                      <select
                        value={llmApiUrl}
                        onChange={(e) => setLlmApiUrl(e.target.value)}
                        className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white"
                      >
                        {API_URL_OPTIONS.map((url: string) => (
                          <option key={url} value={url}>
                            {url}
                          </option>
                        ))}
                      </select>
                    ) : (
                      <input
                        type="text"
                        value={llmApiUrl}
                        onChange={(e) => setLlmApiUrl(e.target.value)}
                        className="w-full px-3 py-2.5 bg-black/20 border border-white/10 rounded-xl text-white focus:outline-none focus:border-[#0A84FF]/60 focus:bg-black/30 transition"
                        placeholder="https://api.apiyi.com/v1"
                      />
                    )}
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">{t('paper2rebuttal:upload.model')}</label>
                    <select
                      value={model}
                      onChange={(e) => setModel(e.target.value)}
                      className="w-full px-3 py-2.5 bg-black/20 border border-white/10 rounded-xl text-white focus:outline-none focus:border-[#0A84FF]/60 focus:bg-black/30 transition"
                    >
                      {modelOptions.map((option) => (
                        <option key={option} value={option} className="bg-slate-900">
                          {option}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">{t('paper2rebuttal:upload.apiKey')}</label>
                    <input
                      type="password"
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                      className="w-full px-3 py-2.5 bg-black/20 border border-white/10 rounded-xl text-white focus:outline-none focus:border-[#0A84FF]/60 focus:bg-black/30 transition"
                      placeholder={t('paper2rebuttal:upload.apiKeyPlaceholder')}
                    />
                  </div>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  <FileText className="inline mr-2" size={16} />
                  {t('paper2rebuttal:upload.paperPdf')}
                </label>
                <input
                  type="file"
                  accept=".pdf"
                  onChange={(e) => setPdfFile(e.target.files?.[0] || null)}
                  className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-2xl text-white file:mr-4 file:py-2 file:px-4 file:rounded-xl file:border-0 file:text-sm file:font-semibold file:bg-[#0A84FF] file:text-white hover:file:bg-[#0974E0]"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">{t('paper2rebuttal:upload.reviewContent')}</label>
                <div className="inline-flex items-center gap-1 p-1 rounded-full bg-black/30 border border-white/10 mb-3">
                  <button
                    type="button"
                    onClick={() => setReviewInputMode('file')}
                    className={`px-4 py-2 text-xs rounded-full transition ${
                      reviewInputMode === 'file'
                        ? 'bg-white/15 text-white shadow-[0_6px_16px_rgba(0,0,0,0.25)]'
                        : 'text-gray-400 hover:text-white'
                    }`}
                  >
                    {t('paper2rebuttal:upload.uploadFile')}
                  </button>
                  <button
                    type="button"
                    onClick={() => setReviewInputMode('text')}
                    className={`px-4 py-2 text-xs rounded-full transition ${
                      reviewInputMode === 'text'
                        ? 'bg-white/15 text-white shadow-[0_6px_16px_rgba(0,0,0,0.25)]'
                        : 'text-gray-400 hover:text-white'
                    }`}
                  >
                    {t('paper2rebuttal:upload.directInput')}
                  </button>
                  <span className="text-[11px] text-gray-500 px-2">
                    {t('paper2rebuttal:upload.reviewFormats')}
                  </span>
                </div>
                {reviewInputMode === 'file' ? (
                  <input
                    type="file"
                    accept=".pdf,.txt,.md"
                    onChange={(e) => setReviewFile(e.target.files?.[0] || null)}
                    className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-2xl text-white file:mr-4 file:py-2 file:px-4 file:rounded-xl file:border-0 file:text-sm file:font-semibold file:bg-[#0A84FF] file:text-white hover:file:bg-[#0974E0]"
                  />
                ) : (
                  <div className="space-y-3">
                    <textarea
                      value={reviewTextDirect}
                      onChange={(e) => setReviewTextDirect(e.target.value)}
                      placeholder={t('paper2rebuttal:upload.reviewPlaceholder')}
                      className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-2xl text-white placeholder-gray-500 min-h-[140px] focus:outline-none focus:border-[#0A84FF]/60 focus:bg-black/30 transition"
                    />
                    <div className="flex items-center justify-between">
                      <span className="text-[11px] text-gray-500">{t('paper2rebuttal:upload.exampleHint')}</span>
                      {reviewTextDirect.trim() && (
                        <button
                          type="button"
                          onClick={() => setReviewTextDirect('')}
                          className="text-[11px] text-gray-400 hover:text-gray-200 transition"
                        >
                          {t('paper2rebuttal:upload.clearInput')}
                        </button>
                      )}
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                      {REVIEW_TEXT_EXAMPLES.map((example) => (
                        <button
                          key={example.title}
                          type="button"
                          onClick={() => setReviewTextDirect(example.text)}
                          className="text-left p-3 rounded-2xl bg-white/5 border border-white/10 hover:border-[#0A84FF]/50 hover:bg-white/10 transition"
                        >
                          <div className="text-xs font-semibold text-white">{example.title}</div>
                          <div className="mt-2 text-[11px] text-gray-400 line-clamp-4 whitespace-pre-wrap">
                            {example.text}
                          </div>
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {error && (
                <div className="p-3 bg-red-500/20 border border-red-500/50 rounded-lg text-red-300 text-sm">
                  {error}
                </div>
              )}

              <button
                onClick={handleParseReview}
                disabled={!pdfFile || (reviewInputMode === 'file' ? !reviewFile : !reviewTextDirect.trim()) || loading}
                className="w-full px-6 py-3 bg-gradient-to-r from-[#0A84FF] via-[#5AC8FA] to-[#34C759] text-white rounded-2xl font-semibold hover:from-[#0974E0] hover:via-[#4AB7EA] hover:to-[#2FB85A] disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 shadow-[0_12px_30px_rgba(10,132,255,0.25)]"
              >
                {loading ? (
                  <>
                    <Loader2 className="animate-spin" size={20} />
                    {t('paper2rebuttal:upload.parsing')}
                  </>
                ) : (
                  <>
                    <FileText size={20} />
                    {t('paper2rebuttal:upload.parseButton')}
                  </>
                )}
              </button>

              <div className="rounded-2xl p-4 bg-white/5 border border-white/10 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-sm text-white">
                    <History size={16} className="text-[#0A84FF]" />
                    {t('paper2rebuttal:history.title')}
                  </div>
                  <button
                    type="button"
                    onClick={fetchHistory}
                    className="text-xs text-gray-400 hover:text-gray-200 flex items-center gap-1"
                  >
                    <RefreshCw size={12} />
                    {t('paper2rebuttal:history.refresh')}
                  </button>
                </div>

                {historyLoading && (
                  <div className="text-xs text-gray-400">{t('paper2rebuttal:history.loading')}</div>
                )}

                {historyError && (
                  <div className="text-xs text-red-300 bg-red-500/10 border border-red-500/30 rounded-lg px-3 py-2">
                    {historyError}
                  </div>
                )}

                {!historyLoading && historySessions.length === 0 && (
                  <div className="text-xs text-gray-500">{t('paper2rebuttal:history.empty')}</div>
                )}

                {!historyLoading && historySessions.length > 0 && (
                  <div className="space-y-2">
                    {historySessions.map((item) => {
                      const statusLabel = item.status === 'completed'
                        ? t('paper2rebuttal:history.statusCompleted')
                        : item.status === 'ready'
                          ? t('paper2rebuttal:history.statusReady')
                          : t('paper2rebuttal:history.statusProcessing');
                      const progressText = typeof item.processed_questions === 'number' && typeof item.total_questions === 'number'
                        ? t('paper2rebuttal:history.questions', { processed: item.processed_questions, total: item.total_questions })
                        : '';
                      return (
                        <div
                          key={item.session_id}
                          className="flex flex-col md:flex-row md:items-center gap-3 p-3 rounded-xl bg-black/30 border border-white/5"
                        >
                          <div className="flex-1">
                            <div className="text-sm font-semibold text-white">{item.session_id}</div>
                            <div className="mt-1 flex flex-wrap items-center gap-2 text-[11px] text-gray-400">
                              <span className="px-2 py-0.5 rounded-full bg-white/10 text-gray-300">{statusLabel}</span>
                              {progressText && <span>{progressText}</span>}
                              {item.updated_at && <span>{t('paper2rebuttal:history.updatedAt', { time: item.updated_at })}</span>}
                            </div>
                          </div>
                          <button
                            type="button"
                            onClick={() => handleLoadHistorySession(item.session_id)}
                            className="px-3 py-2 text-xs rounded-full bg-[#0A84FF]/20 text-[#7FD0FF] hover:bg-[#0A84FF]/30"
                          >
                            {t('paper2rebuttal:history.load')}
                          </button>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Review check step: 展示解析出的 review-1, review-2... */}
        {step === 'review_check' && (
          <div className="glass-dark rounded-3xl p-6 md:p-8 space-y-6 shadow-[0_20px_60px_rgba(0,0,0,0.35)]">
            <h2 className="text-xl font-bold text-white">{t('paper2rebuttal:reviewCheck.title')}</h2>
            <p className="text-gray-400 text-sm">{t('paper2rebuttal:reviewCheck.description')}</p>
            <div className="space-y-4 max-h-[60vh] overflow-y-auto">
              {parsedReviews.length === 0 ? (
                <div className="text-gray-400 py-4">{t('paper2rebuttal:reviewCheck.noResults')}</div>
              ) : (
                parsedReviews.map((item, idx) => (
                  <div key={`${item.id}-${idx}`} className="p-4 bg-white/5 border border-white/10 rounded-lg">
                    <h3 className="text-sm font-semibold text-blue-300 mb-2">{item.id}</h3>
                    <div className="text-gray-300 text-sm [&_ul]:list-disc [&_ul]:pl-5 [&_ol]:list-decimal [&_ol]:pl-5 [&_p]:my-1 [&_h1]:font-bold [&_h2]:font-bold [&_strong]:font-semibold">
                      <ReactMarkdown>{item.content}</ReactMarkdown>
                    </div>
                  </div>
                ))
              )}
            </div>
            <div className="flex gap-4">
              <button
                onClick={() => { setStep('upload'); setError(''); }}
                className="px-4 py-2 bg-gray-500/20 text-gray-300 rounded-lg hover:bg-gray-500/30"
              >
                {t('paper2rebuttal:reviewCheck.backButton')}
              </button>
              <button
                onClick={handleStartAnalysis}
                disabled={!pdfFile || !reviewTextForStart.trim() || !apiKey || !llmApiUrl || loading}
                className="flex-1 px-6 py-3 bg-gradient-to-r from-[#0A84FF] to-[#AF52DE] text-white rounded-2xl font-semibold hover:from-[#0974E0] hover:to-[#9E44CE] disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 shadow-[0_12px_30px_rgba(175,82,222,0.25)]"
              >
                {loading ? (
                  <>
                    <Loader2 className="animate-spin" size={20} />
                    {t('paper2rebuttal:reviewCheck.processing')}
                  </>
                ) : (
                  <>
                    <Upload size={20} />
                    {t('paper2rebuttal:reviewCheck.confirmButton')}
                  </>
                )}
              </button>
            </div>
          </div>
        )}

        {/* Processing Step */}
        {step === 'processing' && (
          <div className="glass-dark rounded-3xl p-6 md:p-8 space-y-4 shadow-[0_20px_60px_rgba(0,0,0,0.35)]">
            <div className="flex items-center gap-3">
              <Loader2 className="animate-spin text-blue-400" size={24} />
              <h2 className="text-xl font-bold text-white">{t('paper2rebuttal:processing.title')}</h2>
            </div>
            <div className="space-y-2">
              {logs.length > 0 && (
                <div className="bg-black/30 rounded-lg p-4 max-h-96 overflow-y-auto">
                  <div className="space-y-1">
                    {logs.map((log, idx) => (
                      <div key={idx} className="text-sm text-gray-300 font-mono hover:bg-white/5 p-1 rounded transition-colors">
                        {log}
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {logs.length === 0 && (
                <div className="text-center py-8 text-gray-400">
                  <Loader2 className="animate-spin mx-auto mb-2" size={32} />
                  <p>{t('paper2rebuttal:processing.initializing')}</p>
                </div>
              )}
              {logs.length > 0 && (
                <div className="mt-4 text-xs text-gray-500 text-center">
                  {t('paper2rebuttal:processing.logCount', { count: logs.length })}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Review Step - Show only if we have valid data */}
        {step === 'review' && !loading && session && session.questions && session.questions.length > 0 && currentQuestionIdx >= 0 && currentQuestionIdx < session.questions.length && currentQuestion && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left Column: Question Navigation */}
            <div className="lg:col-span-1">
              <div className="glass-dark rounded-3xl p-6 space-y-4 sticky top-6 shadow-[0_20px_60px_rgba(0,0,0,0.35)]">
                <h3 className="text-lg font-bold text-white mb-4">{t('paper2rebuttal:review.questionList')}</h3>
                
                <div className="space-y-2">
                  {unsatisfiedQuestionIds.length > 0 && (
                    <div className="text-xs text-red-300 bg-red-500/10 border border-red-500/30 rounded-md p-2">
                      还有未确认的问题：{unsatisfiedQuestionIds.join(', ')}
                    </div>
                  )}
                  {session.questions.map((q, idx) => {
                    const isCurrent = idx === currentQuestionIdx;
                    const isCompleted = q.is_satisfied;
                    const isUnsatisfied = unsatisfiedQuestionIds.includes(q.question_id);
                    
                    return (
                      <button
                        key={q.question_id}
                        onClick={() => {
                          setCurrentQuestionIdx(idx);
                          setSelectedHistoryIndex(null);
                          setFeedback('');
                          setCanGoBack(idx > 0);
                        }}
                        className={`w-full text-left p-3 rounded-lg transition-colors ${
                          isCurrent
                            ? 'bg-blue-500/30 border-2 border-blue-500'
                            : isCompleted
                            ? 'bg-green-500/20 border border-green-500/30 hover:bg-green-500/30'
                            : isUnsatisfied
                            ? 'bg-red-500/10 border border-red-500/30 hover:bg-red-500/20'
                            : 'bg-white/5 border border-white/10 hover:bg-white/10'
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <span className={`font-semibold ${
                            isCurrent ? 'text-blue-300' : isCompleted ? 'text-green-300' : 'text-gray-300'
                          }`}>
                            问题 {q.question_id}
                          </span>
                          {isCompleted && <CheckCircle className="w-4 h-4 text-green-400" />}
                          {isCurrent && <Clock className="w-4 h-4 text-blue-400 animate-pulse" />}
                        </div>
                        {q.question_text && (
                          <div className="text-xs text-gray-400 mt-1 line-clamp-2">
                            {q.question_text.substring(0, 60)}...
                          </div>
                        )}
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Middle Column: Main Content */}
            <div className="lg:col-span-2 space-y-6">
              <div className="glass-dark rounded-3xl p-6 md:p-8 space-y-6 shadow-[0_20px_60px_rgba(0,0,0,0.35)]">
                <div className="flex items-center justify-between">
                  <h2 className="text-xl font-bold text-white">
                    {t('paper2rebuttal:review.questionTitle', { current: currentQuestionIdx + 1, total: session?.questions.length })}
                  </h2>
                  <div className="flex items-center gap-4">
                    <div className="text-sm text-gray-400">
                      {t('paper2rebuttal:review.revisionCount', { count: currentQuestion.revision_count })}
                    </div>
                    <button
                      onClick={() => setShowPapers(!showPapers)}
                      className="px-3 py-1.5 bg-[#0A84FF]/20 hover:bg-[#0A84FF]/30 text-[#7FD0FF] rounded-full text-sm transition-colors"
                    >
                      {showPapers ? t('paper2rebuttal:review.hidePapers') : t('paper2rebuttal:review.showPapers')}
                    </button>
                  </div>
                </div>

                <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">
                        {t('paper2rebuttal:review.reviewQuestion')}
                      </label>
                      <div className="p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-2xl text-white">
                        {currentQuestion.question_text}
                      </div>
                    </div>

                  {/* Strategy Text */}
                  {(() => {
                    const displayHistory = selectedHistoryIndex !== null
                      ? currentQuestion.history?.[selectedHistoryIndex]
                      : null;
                    const strategyText = displayHistory?.strategy_text || currentQuestion.strategy_text || currentQuestion.strategy;

                    return strategyText ? (
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                          {t('paper2rebuttal:review.strategy')}
                        </label>
                        <div className="p-4 bg-white/5 border border-white/10 rounded-2xl text-white whitespace-pre-wrap">
                          {strategyText}
                        </div>
                      </div>
                    ) : null;
                  })()}

                  {/* Todo List */}
                  {(() => {
                    const displayHistory = selectedHistoryIndex !== null
                      ? currentQuestion.history?.[selectedHistoryIndex]
                      : null;
                    const todoList = displayHistory?.todo_list || currentQuestion.todo_list || [];

                    return todoList.length > 0 ? (
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                          {t('paper2rebuttal:review.todoList')}
                        </label>
                        <TodoList todos={todoList} />
                      </div>
                    ) : null;
                  })()}

                  {/* Draft Response */}
                  {(() => {
                    const displayHistory = selectedHistoryIndex !== null
                      ? currentQuestion.history?.[selectedHistoryIndex]
                      : null;
                    const draftResponse = displayHistory?.draft_response || currentQuestion.draft_response;

                    return draftResponse ? (
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                          {t('paper2rebuttal:review.draftResponse')}
                        </label>
                        <div className="p-4 bg-white/5 border border-white/10 rounded-2xl text-white whitespace-pre-wrap max-h-64 overflow-y-auto">
                          {draftResponse}
                        </div>
                      </div>
                    ) : null;
                  })()}

                  {/* Papers List */}
                  {showPapers && (
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">
                        {t('paper2rebuttal:review.relatedPapers')}
                      </label>
                      <PaperList
                        key={`q-${currentQuestionIdx}-${currentQuestion.question_id}`}
                        searchedPapers={currentQuestion.searched_papers}
                        selectedPapers={currentQuestion.selected_papers}
                        analyzedPapers={currentQuestion.analyzed_papers}
                      />
                    </div>
                  )}

                  {/* Feedback Section */}
                  {selectedHistoryIndex === null && (
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">
                        {t('paper2rebuttal:review.feedback')}
                      </label>
                      <textarea
                        value={feedback}
                        onChange={(e) => setFeedback(e.target.value)}
                        placeholder={t('paper2rebuttal:review.feedbackPlaceholder')}
                        className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-2xl text-white placeholder-gray-500 min-h-[100px] focus:outline-none focus:border-[#0A84FF]/60 focus:bg-black/30 transition"
                      />
                      <button
                        onClick={handleRevise}
                        disabled={!feedback.trim() || loading}
                        className="mt-2 px-4 py-2 bg-[#0A84FF] text-white rounded-full hover:bg-[#0974E0] disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 shadow-[0_8px_20px_rgba(10,132,255,0.25)]"
                      >
                        <RefreshCw size={16} />
                        {t('paper2rebuttal:review.regenerateStrategy')}
                      </button>
                    </div>
                  )}

                  {error && (
                    <div className="p-3 bg-red-500/20 border border-red-500/50 rounded-lg text-red-300 text-sm">
                      {error}
                    </div>
                  )}

                  <div className="flex gap-4">
                    {selectedHistoryIndex !== null && (
                      <button
                        onClick={() => setSelectedHistoryIndex(null)}
                        className="px-4 py-2 bg-gray-500/20 text-gray-300 rounded-lg hover:bg-gray-500/30 flex items-center gap-2"
                      >
                        <ChevronLeft size={16} />
                        {t('paper2rebuttal:review.backToCurrent')}
                      </button>
                    )}
                    {canGoBack && currentQuestionIdx > 0 && (
                      <button
                        onClick={handlePreviousQuestion}
                        disabled={loading}
                        className="px-4 py-2 bg-gray-500/20 text-gray-300 rounded-lg hover:bg-gray-500/30 flex items-center gap-2"
                      >
                        <ChevronLeft size={16} />
                        {t('paper2rebuttal:review.previousQuestion')}
                      </button>
                    )}
                    <button
                      onClick={handleNextQuestion}
                      disabled={loading}
                      className="flex-1 px-6 py-3 bg-gradient-to-r from-[#34C759] to-[#30D158] text-white rounded-2xl font-semibold hover:from-[#2FB85A] hover:to-[#27C34E] disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 shadow-[0_12px_30px_rgba(52,199,89,0.25)]"
                    >
                      {currentQuestionIdx + 1 === session?.questions.length ? (
                        <>
                          <CheckCircle size={20} />
                          {t('paper2rebuttal:review.generateFinal')}
                        </>
                      ) : (
                        <>
                          <ArrowRight size={20} />
                          {t('paper2rebuttal:review.nextQuestion')}
                        </>
                      )}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Generating Step - Show while generating final rebuttal */}
        {step === 'generating' && (
          <div className="glass-dark rounded-3xl p-6 md:p-8 space-y-6 shadow-[0_20px_60px_rgba(0,0,0,0.35)]">
            <div className="flex items-center gap-3">
              <Loader2 className="animate-spin text-purple-400" size={32} />
              <h2 className="text-2xl font-bold text-white">{t('paper2rebuttal:generating.title')}</h2>
            </div>

            <div className="space-y-4">
              <div className="p-4 bg-purple-500/10 border border-purple-500/30 rounded-lg">
                <p className="text-purple-300 mb-2">
                  ✨ {t('paper2rebuttal:generating.message')}
                </p>
                <p className="text-sm text-gray-400">
                  {t('paper2rebuttal:generating.wait')}
                </p>
              </div>

              {logs.length > 0 && (
                <div className="p-4 bg-black/30 rounded-lg max-h-96 overflow-y-auto">
                  <div className="space-y-2 text-sm">
                    {logs.map((log, idx) => (
                      <div key={idx} className="text-gray-300 font-mono">
                        {log}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {logs.length === 0 && (
                <div className="text-center py-8 text-gray-400">
                  <Loader2 className="animate-spin mx-auto mb-2" size={32} />
                  <p>{t('paper2rebuttal:generating.initializing')}</p>
                </div>
              )}

              {error && (
                <div className="p-4 bg-red-500/20 border border-red-500/50 rounded-lg text-red-300">
                  {error}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Empty State - Show if step is review but no valid data */}
        {step === 'review' && (!session || !session.questions || session.questions.length === 0 || !currentQuestion) && !loading && (
          <div className="glass-dark rounded-3xl p-6 md:p-8 space-y-6 shadow-[0_20px_60px_rgba(0,0,0,0.35)]">
            <h2 className="text-xl font-bold text-white">
              {error ? t('paper2rebuttal:errors.dataLoadFailed') : t('paper2rebuttal:errors.dataLoading')}
            </h2>
            {error && (
              <div className="p-4 bg-red-500/20 border border-red-500/50 rounded-lg text-red-300">
                <p className="mb-2">{error}</p>
                <p className="text-sm text-red-200">
                  {t('paper2rebuttal:emptyState.sessionId', { id: session?.session_id || 'N/A' })}
                </p>
                <p className="text-sm text-red-200">
                  {t('paper2rebuttal:emptyState.questionCount', { count: session?.questions?.length || 0 })}
                </p>
                <p className="text-sm text-red-200">
                  {t('paper2rebuttal:emptyState.currentIndex', { index: currentQuestionIdx })}
                </p>
              </div>
            )}
            {!error && (
              <div className="text-gray-400">
                <div className="flex items-center gap-2 mb-4">
                  <Loader2 className="animate-spin" size={20} />
                  <p>{t('paper2rebuttal:emptyState.loading')}</p>
                </div>
                <button
                  onClick={() => {
                    if (session?.session_id) {
                      fetchSessionData(session.session_id);
                    } else {
                      setStep('upload');
                    }
                  }}
                  className="mt-4 px-4 py-2 bg-blue-500/20 hover:bg-blue-500/30 text-blue-300 rounded-lg"
                >
                  {session?.session_id ? t('paper2rebuttal:emptyState.reload') : t('paper2rebuttal:emptyState.backToUpload')}
                </button>
              </div>
            )}
          </div>
        )}

        {/* Result Step */}
        {step === 'result' && session && (
          <div className="glass-dark rounded-3xl p-6 md:p-8 space-y-6 shadow-[0_20px_60px_rgba(0,0,0,0.35)]">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-bold text-white">{t('paper2rebuttal:result.title')}</h2>
              <button
                onClick={() => {
                  setStep('review');
                  setCurrentQuestionIdx(0);
                  setCanGoBack(false);
                }}
                className="px-4 py-2 bg-gray-500/20 text-gray-300 rounded-lg hover:bg-gray-500/30 flex items-center gap-2"
              >
                <ChevronLeft size={16} />
                {t('paper2rebuttal:result.backToQuestions')}
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  {t('paper2rebuttal:result.finalRebuttal')}
                </label>
                <div className="p-4 bg-white/5 border border-white/10 rounded-lg text-white whitespace-pre-wrap max-h-96 overflow-y-auto">
                  {session.final_rebuttal || t('paper2rebuttal:result.generating')}
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <button
                  onClick={async () => {
                    try {
                      const response = await fetch(`/api/v1/paper2rebuttal/summary/${session.session_id}`, {
                        headers: { 'X-API-Key': API_KEY },
                      });
                      const data = await response.json();
                      const blob = new Blob([data.markdown], { type: 'text/markdown' });
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement('a');
                      a.href = url;
                      a.download = `rebuttal_summary_${session.session_id}.md`;
                      a.click();
                      URL.revokeObjectURL(url);
                    } catch (err) {
                      setError(t('paper2rebuttal:errors.downloadReportFailed'));
                    }
                  }}
                  className="px-6 py-3 bg-gradient-to-r from-[#AF52DE] to-[#FF2D55] text-white rounded-2xl font-semibold hover:from-[#9E44CE] hover:to-[#E0264C] flex items-center justify-center gap-2 shadow-[0_12px_30px_rgba(175,82,222,0.25)]"
                >
                  <Download size={20} />
                  {t('paper2rebuttal:result.downloadReport')}
                </button>

                <button
                  onClick={() => {
                    const blob = new Blob([session.final_rebuttal], { type: 'text/markdown' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'rebuttal.md';
                    a.click();
                    URL.revokeObjectURL(url);
                  }}
                  className="px-6 py-3 bg-gradient-to-r from-[#0A84FF] to-[#64D2FF] text-white rounded-2xl font-semibold hover:from-[#0974E0] hover:to-[#54C2F0] flex items-center justify-center gap-2 shadow-[0_12px_30px_rgba(10,132,255,0.25)]"
                >
                  <Download size={20} />
                  {t('paper2rebuttal:result.downloadRebuttal')}
                </button>

                <button
                  onClick={handleExportZip}
                  disabled={exportingZip}
                  className="px-6 py-3 bg-gradient-to-r from-[#34C759] to-[#30D158] text-white rounded-2xl font-semibold hover:from-[#2FB85A] hover:to-[#27C34E] disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 shadow-[0_12px_30px_rgba(52,199,89,0.25)]"
                >
                  <Download size={20} />
                  {exportingZip ? t('paper2rebuttal:result.exporting') : t('paper2rebuttal:result.exportZip')}
                </button>
              </div>

              <button
                onClick={() => {
                  setStep('upload');
                  setSession(null);
                  setCurrentQuestionIdx(0);
                  setPdfFile(null);
                  setReviewFile(null);
                  setFeedback('');
                  setError('');
                  setLogs([]);
                  setCanGoBack(false);
                }}
                className="w-full px-6 py-3 bg-white/10 text-white rounded-lg font-semibold hover:bg-white/20"
              >
                {t('paper2rebuttal:result.restart')}
              </button>
            </div>
          </div>
        )}

        {/* Feishu Doc */}
        <div className="pt-4">
          <div className="glass-dark rounded-2xl px-6 py-4 border border-white/10 flex flex-col md:flex-row items-center justify-between gap-3">
            <div className="text-sm text-gray-300">
              {t('paper2rebuttal:feishu.title')}
            </div>
            <a
              href="https://wcny4qa9krto.feishu.cn/wiki/VXKiwYndwiWAVmkFU6kcqsTenWh"
              target="_blank"
              rel="noopener noreferrer"
              className="group relative inline-flex items-center gap-2 px-4 py-2 rounded-full bg-black/50 border border-white/10 text-xs font-medium text-white overflow-hidden transition-all hover:border-white/30 hover:shadow-[0_0_15px_rgba(10,132,255,0.4)]"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-[#0A84FF]/20 via-[#5AC8FA]/20 to-[#AF52DE]/20 opacity-0 group-hover:opacity-100 transition-opacity" />
              <span className="bg-gradient-to-r from-[#7FD0FF] via-[#AF52DE] to-[#FF9F0A] bg-clip-text text-transparent">
                {t('paper2rebuttal:feishu.link')}
              </span>
            </a>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Paper2RebuttalPage;
