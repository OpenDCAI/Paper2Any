import { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { Upload, FileText, CheckCircle, RefreshCw, Download, ArrowRight, Loader2, History, ChevronLeft, Clock } from 'lucide-react';
import { useAuthStore } from '../../stores/authStore';
import { getApiSettings, saveApiSettings } from '../../services/apiSettingsService';
import { API_KEY, DEFAULT_LLM_API_URL, API_URL_OPTIONS } from '../../config/api';
import ReactMarkdown from 'react-markdown';
import Timeline from './Timeline';
import TodoList from './TodoList';
import PaperList from './PaperList';

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
  const [model, setModel] = useState('gpt-5.1');
  const [feedback, setFeedback] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [logs, setLogs] = useState<string[]>([]);
  const [selectedHistoryIndex, setSelectedHistoryIndex] = useState<number | null>(null);
  const [showPapers, setShowPapers] = useState(false);
  const [canGoBack, setCanGoBack] = useState(false);

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

  const addLog = (message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => {
      const newLogs = [...prev, `${timestamp}: ${message}`];
      // Keep last 100 logs
      return newLogs.slice(-100);
    });
  };

  const handleParseReview = async () => {
    const hasFile = reviewInputMode === 'file' && reviewFile;
    const hasText = reviewInputMode === 'text' && reviewTextDirect.trim();
    if (!hasFile && !hasText) {
      setError('请上传评审文件（PDF/txt/md）或直接输入评审内容');
      return;
    }
    setLoading(true);
    setError('');
    try {
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
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || '解析评审失败');
      }
      const data = await response.json();
      setParsedReviews(data.reviews || []);
      setReviewTextForStart(data.review_text || '');
      setStep('review_check');
    } catch (err: any) {
      setError(err.message || '解析评审失败');
    } finally {
      setLoading(false);
    }
  };

  const handleStartAnalysis = async () => {
    // 所有形式的评审输入都必须先经过「解析/预览评审」，得到形式化 review 后再开始分析
    if (!pdfFile || !reviewTextForStart.trim() || !apiKey || !llmApiUrl) {
      setError('请先点击「解析/预览评审」得到形式化 review 并确认后再开始分析，并配置 API');
      return;
    }

    setLoading(true);
    setError('');
    setStep('processing');
    setLogs([]);
    addLog('🚀 开始分析...');

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
      addLog('📡 正在连接进度流...');
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
            addLog('⚠️ 连接已关闭，改用轮询等待全部问题处理完成...');
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
        addLog('⚠️ 无法连接进度流，将使用轮询方式获取进度...');
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
      setError(err.message || '分析失败');
      setStep('upload');
      addLog(`❌ 错误: ${err.message}`);
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
        throw new Error('获取会话数据失败');
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
        throw new Error('未找到问题数据，请检查分析是否完成');
      }

      // If backend says not all questions processed yet (e.g. race after SSE "complete"), retry a few times
      if (data.all_questions_processed === false && retryCount < maxRetries) {
        console.log(`[fetchSessionData] Not all processed (${retryCount + 1}/${maxRetries}), retrying...`);
        addLog(`⏳ 数据仍在写入，${retryDelayMs / 1000} 秒后重试 (${retryCount + 1}/${maxRetries})...`);
        await new Promise((r) => setTimeout(r, retryDelayMs));
        return fetchSessionData(sessionId, retryCount + 1);
      }
      
      // If still not processed after max retries, log warning but continue
      if (data.all_questions_processed === false) {
        console.warn(`[fetchSessionData] Not all questions processed after ${maxRetries} retries. Proceeding anyway.`);
        addLog('⚠️ 部分问题可能仍在处理中，请稍后刷新查看最新数据');
      }
      
      addLog(`✅ 分析完成！提取了 ${data.questions.length} 个问题`);
      
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
      const errorMsg = err.message || '获取数据失败';
      setError(errorMsg);
      addLog(`❌ 错误: ${errorMsg}`);
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
      setError('请输入反馈意见');
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
        throw new Error(errorData.detail || '修订失败');
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
      addLog(`策略已修订 (第 ${data.revision_count} 次)`);
    } catch (err: any) {
      setError(err.message || '修订失败');
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
        throw new Error(errorData.detail || '操作失败');
      }

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
      setError(err.message || '操作失败');
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

      addLog('🚀 开始生成最终反驳信...');
      addLog('📝 正在整合所有问题的策略和回复...');
      
      const response = await fetch('/api/v1/paper2rebuttal/generate-final', {
        method: 'POST',
        headers: { 'X-API-Key': API_KEY },
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || '生成失败');
      }

      addLog('✨ 正在生成最终反驳信内容...');
      
      const data = await response.json();
      setSession({ ...session, final_rebuttal: data.final_rebuttal });
      addLog('✅ 最终反驳信生成完成！');
      
      // Small delay to show success message
      setTimeout(() => {
        setStep('result');
      }, 500);
    } catch (err: any) {
      const errorMsg = err.message || '生成失败';
      setError(errorMsg);
      addLog(`❌ 错误: ${errorMsg}`);
      // Go back to review step on error
      setStep('review');
    } finally {
      setLoading(false);
    }
  };

  const currentQuestion = session?.questions?.[currentQuestionIdx] || null;

  // Build global timeline nodes - always include upload step
  const getGlobalTimelineNodes = () => {
    const nodes = [
      {
        id: 'upload',
        title: '上传 / 输入',
        description: '上传论文 PDF 与评审（文件或直接输入）',
        status: step === 'upload' ? 'current' : (step === 'review_check' || step === 'processing' || step === 'review' || step === 'generating' || step === 'result') ? 'completed' : 'pending',
      },
    ];

    if (step === 'review_check' || parsedReviews.length > 0) {
      nodes.push({
        id: 'review_check',
        title: 'Review check',
        description: '解析并确认评审条目',
        status: step === 'review_check' ? 'current' : (step === 'processing' || step === 'review' || step === 'generating' || step === 'result') ? 'completed' : 'pending',
      });
    }

    // Add analysis step when we have session or are past review_check
    if (session || step === 'processing' || step === 'review' || step === 'generating' || step === 'result') {
      nodes.push({
        id: 'analysis',
        title: '分析处理',
        description: '提取问题并生成策略',
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
          title: `问题 ${q.question_id}`,
          description: q.is_satisfied ? '已完成' : isCurrent ? '处理中...' : '待处理',
          status: isCurrent ? 'current' : isCompleted ? 'completed' : 'pending',
        });
      });
    }

    // Add generating step
    if (step === 'generating' || step === 'result') {
      nodes.push({
        id: 'generating',
        title: '生成中',
        description: '正在生成最终反驳信',
        status: step === 'generating' ? 'current' : 'completed',
      });
    }

    // Add final result node
    nodes.push({
      id: 'result',
      title: '最终反驳信',
      description: '生成最终反驳信',
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
        <div className="text-center space-y-2">
          <h1 className="text-3xl font-bold text-white">Paper2Rebuttal</h1>
          <p className="text-gray-400">AI辅助学术论文反驳信生成工具</p>
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
          <div className="glass-dark rounded-2xl p-6 space-y-6">
            <h2 className="text-xl font-bold text-white">上传文件</h2>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  API配置
                </label>
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">API URL</label>
                    {API_URL_OPTIONS.length > 1 ? (
                      <select
                        value={llmApiUrl}
                        onChange={(e) => setLlmApiUrl(e.target.value)}
                        className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white"
                      >
                        {API_URL_OPTIONS.map((url) => (
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
                        className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white"
                        placeholder="https://api.apiyi.com/v1"
                      />
                    )}
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Model</label>
                    <input
                      type="text"
                      value={model}
                      onChange={(e) => setModel(e.target.value)}
                      className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white"
                      placeholder="gpt-5.1"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">API Key</label>
                    <input
                      type="password"
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                      className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white"
                      placeholder="输入API Key"
                    />
                  </div>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  <FileText className="inline mr-2" size={16} />
                  论文 PDF
                </label>
                <input
                  type="file"
                  accept=".pdf"
                  onChange={(e) => setPdfFile(e.target.files?.[0] || null)}
                  className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-primary-500 file:text-white hover:file:bg-primary-600"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">评审内容</label>
                <div className="flex gap-4 mb-2">
                  <label className="flex items-center gap-2 text-gray-300 cursor-pointer">
                    <input
                      type="radio"
                      name="reviewMode"
                      checked={reviewInputMode === 'file'}
                      onChange={() => setReviewInputMode('file')}
                      className="text-primary-500"
                    />
                    上传文件（PDF / .txt / .md）
                  </label>
                  <label className="flex items-center gap-2 text-gray-300 cursor-pointer">
                    <input
                      type="radio"
                      name="reviewMode"
                      checked={reviewInputMode === 'text'}
                      onChange={() => setReviewInputMode('text')}
                      className="text-primary-500"
                    />
                    直接输入
                  </label>
                </div>
                {reviewInputMode === 'file' ? (
                  <input
                    type="file"
                    accept=".pdf,.txt,.md"
                    onChange={(e) => setReviewFile(e.target.files?.[0] || null)}
                    className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-primary-500 file:text-white hover:file:bg-primary-600"
                  />
                ) : (
                  <textarea
                    value={reviewTextDirect}
                    onChange={(e) => setReviewTextDirect(e.target.value)}
                    placeholder="粘贴评审意见全文（支持按 Review 1 / Q1. / [q1] 等格式分段解析）"
                    className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-500 min-h-[120px]"
                  />
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
                className="w-full px-6 py-3 bg-gradient-to-r from-amber-500 to-orange-500 text-white rounded-lg font-semibold hover:from-amber-600 hover:to-orange-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <Loader2 className="animate-spin" size={20} />
                    解析中...
                  </>
                ) : (
                  <>
                    <FileText size={20} />
                    解析 / 预览评审（Review check）
                  </>
                )}
              </button>
            </div>
          </div>
        )}

        {/* Review check step: 展示解析出的 review-1, review-2... */}
        {step === 'review_check' && (
          <div className="glass-dark rounded-2xl p-6 space-y-6">
            <h2 className="text-xl font-bold text-white">评审预览（Review check）</h2>
            <p className="text-gray-400 text-sm">请确认下方解析出的评审条目无误后，点击「确认并开始分析」。</p>
            <div className="space-y-4 max-h-[60vh] overflow-y-auto">
              {parsedReviews.length === 0 ? (
                <div className="text-gray-400 py-4">暂无解析结果</div>
              ) : (
                parsedReviews.map((item) => (
                  <div key={item.id} className="p-4 bg-white/5 border border-white/10 rounded-lg">
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
                返回修改
              </button>
              <button
                onClick={handleStartAnalysis}
                disabled={!pdfFile || !reviewTextForStart.trim() || !apiKey || !llmApiUrl || loading}
                className="flex-1 px-6 py-3 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-lg font-semibold hover:from-purple-600 hover:to-pink-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <Loader2 className="animate-spin" size={20} />
                    处理中...
                  </>
                ) : (
                  <>
                    <Upload size={20} />
                    确认并开始分析
                  </>
                )}
              </button>
            </div>
          </div>
        )}

        {/* Processing Step */}
        {step === 'processing' && (
          <div className="glass-dark rounded-2xl p-6 space-y-4">
            <div className="flex items-center gap-3">
              <Loader2 className="animate-spin text-blue-400" size={24} />
              <h2 className="text-xl font-bold text-white">分析中...</h2>
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
                  <p>正在初始化分析流程...</p>
                </div>
              )}
              {logs.length > 0 && (
                <div className="mt-4 text-xs text-gray-500 text-center">
                  共 {logs.length} 条进度信息
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
              <div className="glass-dark rounded-2xl p-6 space-y-4 sticky top-6">
                <h3 className="text-lg font-bold text-white mb-4">问题列表</h3>
                
                <div className="space-y-2">
                  {session.questions.map((q, idx) => {
                    const isCurrent = idx === currentQuestionIdx;
                    const isCompleted = q.is_satisfied;
                    
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
              <div className="glass-dark rounded-2xl p-6 space-y-6">
                <div className="flex items-center justify-between">
                  <h2 className="text-xl font-bold text-white">
                    问题 {currentQuestionIdx + 1} / {session?.questions.length}
                  </h2>
                  <div className="flex items-center gap-4">
                    <div className="text-sm text-gray-400">
                      已修订 {currentQuestion.revision_count} 次
                    </div>
                    <button
                      onClick={() => setShowPapers(!showPapers)}
                      className="px-3 py-1.5 bg-blue-500/20 hover:bg-blue-500/30 text-blue-300 rounded-lg text-sm transition-colors"
                    >
                      {showPapers ? '隐藏论文' : '查看论文'}
                    </button>
                  </div>
                </div>

                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      评审问题
                    </label>
                    <div className="p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-lg text-white">
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
                          反驳策略
                        </label>
                        <div className="p-4 bg-white/5 border border-white/10 rounded-lg text-white whitespace-pre-wrap">
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
                          待办事项列表
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
                          草稿片段
                        </label>
                        <div className="p-4 bg-white/5 border border-white/10 rounded-lg text-white whitespace-pre-wrap max-h-64 overflow-y-auto">
                          {draftResponse}
                        </div>
                      </div>
                    ) : null;
                  })()}

                  {/* Papers List */}
                  {showPapers && (
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">
                        相关论文
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
                        反馈意见
                      </label>
                      <textarea
                        value={feedback}
                        onChange={(e) => setFeedback(e.target.value)}
                        placeholder="输入您的反馈意见，AI将根据反馈调整策略..."
                        className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-500 min-h-[100px]"
                      />
                      <button
                        onClick={handleRevise}
                        disabled={!feedback.trim() || loading}
                        className="mt-2 px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                      >
                        <RefreshCw size={16} />
                        重新生成策略
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
                        返回当前版本
                      </button>
                    )}
                    {canGoBack && currentQuestionIdx > 0 && (
                      <button
                        onClick={handlePreviousQuestion}
                        disabled={loading}
                        className="px-4 py-2 bg-gray-500/20 text-gray-300 rounded-lg hover:bg-gray-500/30 flex items-center gap-2"
                      >
                        <ChevronLeft size={16} />
                        上一个问题
                      </button>
                    )}
                    <button
                      onClick={handleNextQuestion}
                      disabled={loading}
                      className="flex-1 px-6 py-3 bg-gradient-to-r from-green-500 to-emerald-500 text-white rounded-lg font-semibold hover:from-green-600 hover:to-emerald-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                    >
                      {currentQuestionIdx + 1 === session?.questions.length ? (
                        <>
                          <CheckCircle size={20} />
                          生成最终反驳信
                        </>
                      ) : (
                        <>
                          <ArrowRight size={20} />
                          下一个问题
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
          <div className="glass-dark rounded-2xl p-6 space-y-6">
            <div className="flex items-center gap-3">
              <Loader2 className="animate-spin text-purple-400" size={32} />
              <h2 className="text-2xl font-bold text-white">正在生成最终反驳信</h2>
            </div>
            
            <div className="space-y-4">
              <div className="p-4 bg-purple-500/10 border border-purple-500/30 rounded-lg">
                <p className="text-purple-300 mb-2">
                  ✨ 正在整合所有问题的策略和回复，生成完整的反驳信...
                </p>
                <p className="text-sm text-gray-400">
                  这可能需要几分钟时间，请稍候...
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
                  <p>正在初始化生成流程...</p>
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
          <div className="glass-dark rounded-2xl p-6 space-y-6">
            <h2 className="text-xl font-bold text-white">
              {error ? '数据加载失败' : '数据加载中...'}
            </h2>
            {error && (
              <div className="p-4 bg-red-500/20 border border-red-500/50 rounded-lg text-red-300">
                <p className="mb-2">{error}</p>
                <p className="text-sm text-red-200">
                  Session ID: {session?.session_id || 'N/A'}
                </p>
                <p className="text-sm text-red-200">
                  问题数量: {session?.questions?.length || 0}
                </p>
                <p className="text-sm text-red-200">
                  当前问题索引: {currentQuestionIdx}
                </p>
              </div>
            )}
            {!error && (
              <div className="text-gray-400">
                <div className="flex items-center gap-2 mb-4">
                  <Loader2 className="animate-spin" size={20} />
                  <p>正在加载问题数据，请稍候...</p>
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
                  {session?.session_id ? '重新加载' : '返回上传'}
                </button>
              </div>
            )}
          </div>
        )}

        {/* Result Step */}
        {step === 'result' && session && (
          <div className="glass-dark rounded-2xl p-6 space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-bold text-white">完成！</h2>
              <button
                onClick={() => {
                  setStep('review');
                  setCurrentQuestionIdx(0);
                  setCanGoBack(false);
                }}
                className="px-4 py-2 bg-gray-500/20 text-gray-300 rounded-lg hover:bg-gray-500/30 flex items-center gap-2"
              >
                <ChevronLeft size={16} />
                返回问题列表
              </button>
            </div>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  最终反驳信
                </label>
                <div className="p-4 bg-white/5 border border-white/10 rounded-lg text-white whitespace-pre-wrap max-h-96 overflow-y-auto">
                  {session.final_rebuttal || '生成中...'}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
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
                      setError('下载总结报告失败');
                    }
                  }}
                  className="px-6 py-3 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-lg font-semibold hover:from-purple-600 hover:to-pink-600 flex items-center justify-center gap-2"
                >
                  <Download size={20} />
                  下载完整报告 (MD)
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
                  className="px-6 py-3 bg-gradient-to-r from-blue-500 to-cyan-500 text-white rounded-lg font-semibold hover:from-blue-600 hover:to-cyan-600 flex items-center justify-center gap-2"
                >
                  <Download size={20} />
                  下载反驳信
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
                重新开始
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Paper2RebuttalPage;
