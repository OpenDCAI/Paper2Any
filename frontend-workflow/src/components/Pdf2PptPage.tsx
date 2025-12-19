import { useState, ChangeEvent } from 'react';
import { 
  UploadCloud, Settings2, Download, Loader2, CheckCircle2, 
  AlertCircle, ChevronDown, ChevronUp, Github, Star, X, Sparkles,
  ArrowRight, ArrowLeft, GripVertical, Trash2, Edit3, Check, RotateCcw,
  SkipForward, MessageSquare, RefreshCw, FileText
} from 'lucide-react';

// ============== 类型定义 ==============
type Step = 'upload' | 'outline' | 'generate' | 'complete';

// 前端使用的 Slide 数据结构
interface SlideOutline {
  id: string;
  pageNum: number;
  title: string;
  layout_description: string;
  key_points: string[];
  asset_ref: string | null;
}

interface GenerateResult {
  slideId: string;
  afterImage: string;
  status: 'pending' | 'processing' | 'done' | 'skipped';
  userPrompt?: string;
}

// ============== 假数据模拟 ==============
const MOCK_OUTLINE: SlideOutline[] = [
  { 
    id: '1', pageNum: 1, 
    title: 'Multimodal DeepResearcher：从零生成文本‑图表交织报告的框架概览', 
    layout_description: '标题置顶居中，下方左侧为论文基本信息（作者、单位、场景），右侧放置论文提供的生成示例截图作为引入。底部一行给出演讲提纲要点。',
    key_points: [
      '研究目标：自动从一个主题出发，生成高质量的文本‑图表交织（text‑chart interleaved）研究报告。',
      '核心创新：提出Formal Description of Visualization (FDV) 和 Multimodal DeepResearcher 代理式框架。',
      '实验结果：在相同模型（Claude 3.7 Sonnet）条件下，对基线方法整体胜率达 82%。',
      '汇报结构：背景与动机 → 方法框架 → FDV 表示 → 实验与评估 → 分析与展望。'
    ],
    asset_ref: 'images/ced6b7ce492d7889aa0186544fc8fad7c725d1deb19765e339e806907251963f.jpg'
  },
  { 
    id: '2', pageNum: 2, 
    title: '研究动机：从文本报告到多模态报告', 
    layout_description: '左侧用要点阐述现有 deep research 框架的局限，右侧以两栏对比示意：上为"纯文本报告"示意，下为"文本+图表交织报告"示意。',
    key_points: [
      '当前 deep research 框架主要输出长篇文本报告，忽略可视化在沟通中的关键作用。',
      '仅文本形式难以有效传递复杂数据洞见，降低可读性与实用性。',
      '真实世界的研究报告与演示文稿通常由专家精心设计多种图表，并与文本紧密交织。',
      '本工作提出一种系统化框架，使 LLM 能"像专家一样"规划、生成并整合多种可视化。'
    ],
    asset_ref: null
  },
];

const MOCK_AFTER_IMAGES = [
  '/ppe2more_2.jpg',
  '/ppe2more_2.jpg',
];

// ============== 主组件 ==============
const Pdf2PptPage = () => {
  const [currentStep, setCurrentStep] = useState<Step>('upload');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [stylePreset, setStylePreset] = useState<'modern' | 'business' | 'academic' | 'creative'>('modern');
  const [globalPrompt, setGlobalPrompt] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [outlineData, setOutlineData] = useState<SlideOutline[]>([]);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editContent, setEditContent] = useState<{
    title: string;
    layout_description: string;
    key_points: string[];
  }>({ title: '', layout_description: '', key_points: [] });
  const [currentSlideIndex, setCurrentSlideIndex] = useState(0);
  const [generateResults, setGenerateResults] = useState<GenerateResult[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [slidePrompt, setSlidePrompt] = useState('');
  const [isGeneratingFinal, setIsGeneratingFinal] = useState(false);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showBanner, setShowBanner] = useState(true);

  const validateDocFile = (file: File): boolean => {
    const ext = file.name.split('.').pop()?.toLowerCase();
    if (ext !== 'pdf') {
      setError('仅支持 PDF 格式');
      return false;
    }
    return true;
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !validateDocFile(file)) return;
    setSelectedFile(file);
    setError(null);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (!file || !validateDocFile(file)) return;
    setSelectedFile(file);
    setError(null);
  };

  const handleUploadAndParse = async () => {
    if (!selectedFile) {
      setError('请先选择 PDF 文件');
      return;
    }
    setIsUploading(true);
    setError(null);
    try {
      await new Promise(resolve => setTimeout(resolve, 2000));
      setOutlineData(MOCK_OUTLINE);
    } catch (err) {
      setError('解析失败，请重试');
    } finally {
      setIsUploading(false);
    }
    setCurrentStep('outline');
  };

  const handleEditStart = (slide: SlideOutline) => {
    setEditingId(slide.id);
    setEditContent({ 
      title: slide.title, 
      layout_description: slide.layout_description,
      key_points: [...slide.key_points]
    });
  };

  const handleEditSave = () => {
    if (!editingId) return;
    setOutlineData(prev => prev.map(s => 
      s.id === editingId 
        ? { ...s, title: editContent.title, layout_description: editContent.layout_description, key_points: editContent.key_points }
        : s
    ));
    setEditingId(null);
  };

  const handleKeyPointChange = (index: number, value: string) => {
    setEditContent(prev => {
      const newKeyPoints = [...prev.key_points];
      newKeyPoints[index] = value;
      return { ...prev, key_points: newKeyPoints };
    });
  };

  const handleAddKeyPoint = () => {
    setEditContent(prev => ({ ...prev, key_points: [...prev.key_points, ''] }));
  };

  const handleRemoveKeyPoint = (index: number) => {
    setEditContent(prev => ({ ...prev, key_points: prev.key_points.filter((_, i) => i !== index) }));
  };

  const handleEditCancel = () => setEditingId(null);
  const handleDeleteSlide = (id: string) => setOutlineData(prev => prev.filter(s => s.id !== id).map((s, i) => ({ ...s, pageNum: i + 1 })));
  const handleMoveSlide = (index: number, direction: 'up' | 'down') => {
    const newData = [...outlineData];
    const targetIndex = direction === 'up' ? index - 1 : index + 1;
    if (targetIndex < 0 || targetIndex >= newData.length) return;
    [newData[index], newData[targetIndex]] = [newData[targetIndex], newData[index]];
    setOutlineData(newData.map((s, i) => ({ ...s, pageNum: i + 1 })));
  };

  const handleConfirmOutline = () => {
    const results: GenerateResult[] = outlineData.map((slide, index) => ({
      slideId: slide.id,
      afterImage: MOCK_AFTER_IMAGES[index % MOCK_AFTER_IMAGES.length],
      status: 'pending',
    }));
    setGenerateResults(results);
    setCurrentSlideIndex(0);
    setCurrentStep('generate');
    startGenerateCurrentSlide(results, 0);
  };

  const startGenerateCurrentSlide = async (results: GenerateResult[], index: number) => {
    setIsGenerating(true);
    const updatedResults = [...results];
    updatedResults[index] = { ...updatedResults[index], status: 'processing' };
    setGenerateResults(updatedResults);
    await new Promise(resolve => setTimeout(resolve, 2500));
    updatedResults[index] = { ...updatedResults[index], status: 'done' };
    setGenerateResults(updatedResults);
    setIsGenerating(false);
  };

  const handleConfirmSlide = () => {
    if (currentSlideIndex < outlineData.length - 1) {
      const nextIndex = currentSlideIndex + 1;
      setCurrentSlideIndex(nextIndex);
      setSlidePrompt('');
      startGenerateCurrentSlide(generateResults, nextIndex);
    } else {
      setCurrentStep('complete');
    }
  };

  const handleSkipSlide = () => {
    const updatedResults = [...generateResults];
    updatedResults[currentSlideIndex] = { ...updatedResults[currentSlideIndex], status: 'skipped' };
    setGenerateResults(updatedResults);
    if (currentSlideIndex < outlineData.length - 1) {
      const nextIndex = currentSlideIndex + 1;
      setCurrentSlideIndex(nextIndex);
      setSlidePrompt('');
      startGenerateCurrentSlide(updatedResults, nextIndex);
    } else {
      setCurrentStep('complete');
    }
  };

  const handleRegenerateSlide = async () => {
    const updatedResults = [...generateResults];
    updatedResults[currentSlideIndex] = { ...updatedResults[currentSlideIndex], userPrompt: slidePrompt, status: 'pending' };
    setGenerateResults(updatedResults);
    await startGenerateCurrentSlide(updatedResults, currentSlideIndex);
  };

  const handleGenerateFinal = async () => {
    setIsGeneratingFinal(true);
    await new Promise(resolve => setTimeout(resolve, 3000));
    setDownloadUrl('/mock-generated.pptx');
    setIsGeneratingFinal(false);
  };

  const renderStepIndicator = () => {
    const steps = [
      { key: 'upload', label: '上传 PDF', num: 1 },
      { key: 'outline', label: '大纲确认', num: 2 },
      { key: 'generate', label: '逐页生成', num: 3 },
      { key: 'complete', label: '完成下载', num: 4 },
    ];
    const currentIndex = steps.findIndex(s => s.key === currentStep);
    return (
      <div className="flex items-center justify-center gap-2 mb-8">
        {steps.map((step, index) => (
          <div key={step.key} className="flex items-center">
            <div className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-all ${index === currentIndex ? 'bg-gradient-to-r from-orange-500 to-red-500 text-white shadow-lg' : index < currentIndex ? 'bg-orange-500/20 text-orange-300 border border-orange-500/40' : 'bg-white/5 text-gray-500 border border-white/10'}`}>
              <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs ${index < currentIndex ? 'bg-orange-400 text-white' : ''}`}>{index < currentIndex ? <Check size={14} /> : step.num}</span>
              <span className="hidden sm:inline">{step.label}</span>
            </div>
            {index < steps.length - 1 && <ArrowRight size={16} className={`mx-2 ${index < currentIndex ? 'text-orange-400' : 'text-gray-600'}`} />}
          </div>
        ))}
      </div>
    );
  };

  const renderUploadStep = () => (
    <div className="max-w-6xl mx-auto">
      <div className="mb-10 text-center">
        <p className="text-xs uppercase tracking-[0.2em] text-orange-300 mb-3 font-semibold">PDF → PPTX</p>
        <h1 className="text-4xl md:text-5xl font-bold mb-4"><span className="bg-gradient-to-r from-orange-400 via-red-400 to-pink-400 bg-clip-text text-transparent">Pdf2PPT</span></h1>
        <p className="text-base text-gray-300 max-w-2xl mx-auto leading-relaxed">上传 PDF 文档，AI 智能总结内容并生成精美幻灯片。<br /><span className="text-orange-400">支持逐页编辑和微调，打造完美演示文稿！</span></p>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="glass rounded-xl border border-white/10 p-6">
          <h3 className="text-white font-semibold flex items-center gap-2 mb-4"><FileText size={18} className="text-orange-400" /> 上传 PDF</h3>
          <div className={`border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center text-center gap-4 transition-all ${isDragOver ? 'border-orange-500 bg-orange-500/10' : 'border-white/20 hover:border-orange-400'}`} onDragOver={e => { e.preventDefault(); setIsDragOver(true); }} onDragLeave={e => { e.preventDefault(); setIsDragOver(false); }} onDrop={handleDrop}>
            <div className="w-16 h-16 rounded-full bg-gradient-to-br from-orange-500/20 to-red-500/20 flex items-center justify-center"><UploadCloud size={32} className="text-orange-400" /></div>
            <div><p className="text-white font-medium mb-1">拖拽 PDF 文件到此处</p><p className="text-sm text-gray-400">仅支持 PDF 格式</p></div>
            <label className="px-6 py-2.5 rounded-full bg-gradient-to-r from-orange-600 to-red-600 text-white text-sm font-medium cursor-pointer hover:from-orange-700 hover:to-red-700 transition-all">选择文件<input type="file" accept=".pdf" className="hidden" onChange={handleFileChange} /></label>
            {selectedFile && <div className="px-4 py-2 bg-orange-500/20 border border-orange-500/40 rounded-lg"><p className="text-sm text-orange-300">✓ {selectedFile.name}</p><p className="text-xs text-gray-400 mt-1">✨ 将根据 PDF 内容生成 PPT</p></div>}
          </div>
        </div>
        <div className="glass rounded-xl border border-white/10 p-6 space-y-5">
          <h3 className="text-white font-semibold flex items-center gap-2"><Settings2 size={18} className="text-orange-400" /> 风格配置</h3>
          <div>
            <label className="block text-sm text-gray-300 mb-2">选择风格</label>
            <select value={stylePreset} onChange={e => setStylePreset(e.target.value as typeof stylePreset)} className="w-full rounded-lg border border-white/20 bg-black/40 px-4 py-2.5 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-orange-500"><option value="modern">现代简约</option><option value="business">商务专业</option><option value="academic">学术报告</option><option value="creative">创意设计</option></select>
          </div>
          <div>
            <label className="block text-sm text-gray-300 mb-2">风格提示词（可选）</label>
            <textarea value={globalPrompt} onChange={e => setGlobalPrompt(e.target.value)} placeholder="例如：使用橙色系配色，保持简洁风格..." rows={3} className="w-full rounded-lg border border-white/20 bg-black/40 px-4 py-2.5 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-orange-500 resize-none" />
          </div>
          <button onClick={handleUploadAndParse} disabled={!selectedFile || isUploading} className="w-full py-3 rounded-lg bg-gradient-to-r from-orange-600 to-red-600 hover:from-orange-700 hover:to-red-700 disabled:from-gray-600 text-white font-semibold flex items-center justify-center gap-2 transition-all">{isUploading ? <><Loader2 size={18} className="animate-spin" /> 解析中...</> : <><ArrowRight size={18} /> 开始解析</>}</button>
        </div>
      </div>
      {error && <div className="mt-4 flex items-center gap-2 text-sm text-red-300 bg-red-500/10 border border-red-500/40 rounded-lg px-4 py-3"><AlertCircle size={16} /> {error}</div>}
    </div>
  );

  const renderOutlineStep = () => (
    <div className="max-w-5xl mx-auto">
      <div className="text-center mb-8"><h2 className="text-2xl font-bold text-white mb-2">确认大纲</h2><p className="text-gray-400">检查从 PDF 提取的内容结构，可编辑、排序或删除</p></div>
      <div className="glass rounded-xl border border-white/10 p-6 mb-6">
        <div className="space-y-3">
          {outlineData.map((slide, index) => (
            <div key={slide.id} className={`flex items-start gap-4 p-4 rounded-lg border transition-all ${editingId === slide.id ? 'bg-orange-500/10 border-orange-500/40' : 'bg-white/5 border-white/10 hover:border-white/20'}`}>
              <div className="flex items-center gap-2 pt-1"><GripVertical size={16} className="text-gray-500" /><span className="w-8 h-8 rounded-full bg-orange-500/20 text-orange-300 text-sm font-medium flex items-center justify-center">{slide.pageNum}</span></div>
              <div className="flex-1">
                {editingId === slide.id ? (
                  <div className="space-y-3">
                    <input type="text" value={editContent.title} onChange={e => setEditContent(p => ({ ...p, title: e.target.value }))} className="w-full px-3 py-2 rounded-lg bg-black/40 border border-white/20 text-white text-sm outline-none focus:ring-2 focus:ring-orange-500" placeholder="标题" />
                    <textarea value={editContent.layout_description} onChange={e => setEditContent(p => ({ ...p, layout_description: e.target.value }))} rows={2} className="w-full px-3 py-2 rounded-lg bg-black/40 border border-white/20 text-white text-sm outline-none focus:ring-2 focus:ring-orange-500 resize-none" placeholder="布局描述" />
                    <div className="space-y-2">{editContent.key_points.map((p, i) => (<div key={i} className="flex gap-2"><input type="text" value={p} onChange={e => handleKeyPointChange(i, e.target.value)} className="flex-1 px-3 py-2 rounded-lg bg-black/40 border border-white/20 text-white text-sm" placeholder={`要点 ${i + 1}`} /><button onClick={() => handleRemoveKeyPoint(i)} className="p-2 text-gray-400 hover:text-red-400"><Trash2 size={14} /></button></div>))}<button onClick={handleAddKeyPoint} className="px-3 py-1.5 rounded-lg bg-white/5 border border-dashed border-white/20 text-gray-400 text-sm w-full hover:text-orange-400 hover:border-orange-400">+ 添加要点</button></div>
                    <div className="flex gap-2 pt-2"><button onClick={handleEditSave} className="px-3 py-1.5 rounded-lg bg-orange-500 text-white text-sm flex items-center gap-1"><Check size={14} /> 保存</button><button onClick={handleEditCancel} className="px-3 py-1.5 rounded-lg bg-white/10 text-gray-300 text-sm">取消</button></div>
                  </div>
                ) : (
                  <><div className="mb-2"><h4 className="text-white font-medium">{slide.title}</h4></div><p className="text-xs text-orange-400/70 mb-2 italic">📐 {slide.layout_description}</p><ul className="space-y-1">{slide.key_points.map((p, i) => (<li key={i} className="text-sm text-gray-400 flex items-start gap-2"><span className="text-orange-400 mt-0.5">•</span><span>{p}</span></li>))}</ul></>
                )}
              </div>
              {editingId !== slide.id && (
                <div className="flex items-center gap-1">
                  <button onClick={() => handleMoveSlide(index, 'up')} disabled={index === 0} className="p-2 text-gray-400 hover:text-white disabled:opacity-30"><ChevronUp size={16} /></button>
                  <button onClick={() => handleMoveSlide(index, 'down')} disabled={index === outlineData.length - 1} className="p-2 text-gray-400 hover:text-white disabled:opacity-30"><ChevronDown size={16} /></button>
                  <button onClick={() => handleEditStart(slide)} className="p-2 text-gray-400 hover:text-orange-400"><Edit3 size={16} /></button>
                  <button onClick={() => handleDeleteSlide(slide.id)} className="p-2 text-gray-400 hover:text-red-400"><Trash2 size={16} /></button>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
      <div className="flex justify-between"><button onClick={() => setCurrentStep('upload')} className="px-6 py-2.5 rounded-lg border border-white/20 text-gray-300 hover:bg-white/10 flex items-center gap-2"><ArrowLeft size={18} /> 返回上传</button><button onClick={handleConfirmOutline} className="px-6 py-2.5 rounded-lg bg-gradient-to-r from-orange-600 to-red-600 text-white font-semibold flex items-center gap-2 transition-all">确认并开始生成 <ArrowRight size={18} /></button></div>
    </div>
  );

  const renderGenerateStep = () => {
    const currentSlide = outlineData[currentSlideIndex];
    const currentResult = generateResults[currentSlideIndex];
    return (
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-6"><h2 className="text-2xl font-bold text-white mb-2">逐页生成</h2><p className="text-gray-400">第 {currentSlideIndex + 1} / {outlineData.length} 页：{currentSlide?.title}</p><p className="text-xs text-gray-500 mt-1">✨ 生成模式 - 根据 PDF 内容创建 PPT</p></div>
        <div className="mb-6"><div className="flex gap-1">{generateResults.map((result, index) => (<div key={result.slideId} className={`flex-1 h-2 rounded-full transition-all ${result.status === 'done' ? 'bg-orange-400' : result.status === 'skipped' ? 'bg-yellow-400' : result.status === 'processing' ? 'bg-gradient-to-r from-orange-400 to-red-400 animate-pulse' : index === currentSlideIndex ? 'bg-orange-400/50' : 'bg-white/10'}`} />))}</div></div>
        
        {/* 当前页面内容信息 */}
        {currentSlide && (
          <div className="glass rounded-xl border border-white/10 p-4 mb-4">
            <div className="mb-3"><h4 className="text-sm text-gray-400 mb-2 flex items-center gap-2"><FileText size={14} className="text-orange-400" /> 布局描述</h4><p className="text-xs text-orange-400/80 italic">{currentSlide.layout_description}</p></div>
            <div className="pt-3 border-t border-white/10"><h4 className="text-sm text-gray-400 mb-2">要点内容</h4><ul className="grid grid-cols-1 md:grid-cols-2 gap-1">{currentSlide.key_points.slice(0, 4).map((point, idx) => (<li key={idx} className="text-xs text-gray-400 flex items-start gap-1"><span className="text-orange-400">•</span><span className="line-clamp-1">{point}</span></li>))}{currentSlide.key_points.length > 4 && (<li className="text-xs text-gray-500 italic">...还有 {currentSlide.key_points.length - 4} 条</li>)}</ul></div>
          </div>
        )}

        <div className="glass rounded-xl border border-white/10 p-6 mb-6">
          <div className="max-w-3xl mx-auto">
            <h4 className="text-sm text-gray-400 mb-3 flex items-center justify-center gap-2"><Sparkles size={14} className="text-orange-400" /> AI 生成结果</h4>
            <div className="rounded-lg overflow-hidden border border-orange-500/30 min-h-[400px] bg-gradient-to-br from-orange-500/10 to-red-500/10 flex items-center justify-center p-4">{isGenerating ? <div className="text-center"><Loader2 size={40} className="text-orange-400 animate-spin mx-auto mb-3" /><p className="text-base text-orange-300">正在根据 PDF 内容生成 PPT...</p><p className="text-xs text-gray-500 mt-1">AI 正在分析文档并创建精美幻灯片</p></div> : currentResult?.afterImage ? <img src={currentResult.afterImage} alt="Generated" className="max-w-full max-h-[500px] object-contain rounded" /> : <div className="text-center"><FileText size={32} className="text-gray-500 mx-auto mb-2" /><span className="text-gray-500">等待生成</span></div>}</div>
            <p className="text-center text-xs text-gray-500 mt-3">📄 基于 PDF 内容智能生成</p>
          </div>
        </div>
        <div className="glass rounded-xl border border-white/10 p-4 mb-6"><div className="flex items-center gap-3"><MessageSquare size={18} className="text-orange-400" /><input type="text" value={slidePrompt} onChange={e => setSlidePrompt(e.target.value)} placeholder="输入微调 Prompt，然后点击重新生成..." className="flex-1 bg-transparent outline-none text-white text-sm placeholder:text-gray-500" /><button onClick={handleRegenerateSlide} disabled={isGenerating || !slidePrompt.trim()} className="px-4 py-2 rounded-lg bg-white/10 hover:bg-white/20 text-gray-300 text-sm flex items-center gap-2 disabled:opacity-50"><RefreshCw size={14} /> 重新生成</button></div></div>
        <div className="flex justify-between"><button onClick={() => setCurrentStep('outline')} className="px-6 py-2.5 rounded-lg border border-white/20 text-gray-300 hover:bg-white/10 flex items-center gap-2"><ArrowLeft size={18} /> 返回大纲</button><div className="flex gap-3"><button onClick={handleSkipSlide} disabled={isGenerating} className="px-5 py-2.5 rounded-lg bg-yellow-500/20 border border-yellow-500/40 text-yellow-300 hover:bg-yellow-500/30 flex items-center gap-2 disabled:opacity-50"><SkipForward size={18} /> 跳过此页</button><button onClick={handleConfirmSlide} disabled={isGenerating} className="px-6 py-2.5 rounded-lg bg-gradient-to-r from-orange-600 to-red-600 text-white font-semibold flex items-center gap-2 disabled:opacity-50"><CheckCircle2 size={18} /> 确认并继续</button></div></div>
      </div>
    );
  };

  const renderCompleteStep = () => {
    const doneCount = generateResults.filter(r => r.status === 'done').length;
    const skippedCount = generateResults.filter(r => r.status === 'skipped').length;
    return (
      <div className="max-w-2xl mx-auto text-center">
        <div className="mb-8"><div className="w-20 h-20 rounded-full bg-gradient-to-br from-orange-500 to-red-500 flex items-center justify-center mx-auto mb-4"><CheckCircle2 size={40} className="text-white" /></div><h2 className="text-2xl font-bold text-white mb-2">生成完成！</h2><p className="text-gray-400">共处理 {outlineData.length} 页，生成 {doneCount} 页，跳过 {skippedCount} 页</p></div>
        <div className="glass rounded-xl border border-white/10 p-6 mb-6"><h3 className="text-white font-semibold mb-4">处理结果概览</h3><div className="grid grid-cols-4 gap-2">{generateResults.map((result, index) => (<div key={result.slideId} className={`p-3 rounded-lg border ${result.status === 'done' ? 'bg-orange-500/20 border-orange-500/40' : 'bg-yellow-500/20 border-yellow-500/40'}`}><p className="text-sm text-white">第 {index + 1} 页</p><p className={`text-xs ${result.status === 'done' ? 'text-orange-300' : 'text-yellow-300'}`}>{result.status === 'done' ? '已生成' : '已跳过'}</p></div>))}</div></div>
        {!downloadUrl ? <button onClick={handleGenerateFinal} disabled={isGeneratingFinal} className="px-8 py-3 rounded-lg bg-gradient-to-r from-orange-600 to-red-600 text-white font-semibold flex items-center justify-center gap-2 mx-auto transition-all">{isGeneratingFinal ? <><Loader2 size={18} className="animate-spin" /> 正在生成最终 PPT...</> : <><Sparkles size={18} /> 生成最终 PPT</>}</button> : <div className="space-y-4"><button onClick={() => alert('下载功能对接中')} className="px-8 py-3 rounded-lg bg-gradient-to-r from-emerald-500 to-teal-500 text-white font-semibold flex items-center gap-2 mx-auto transition-all"><Download size={18} /> 下载生成的 PPT</button><button onClick={() => { setCurrentStep('upload'); setSelectedFile(null); setOutlineData([]); setGenerateResults([]); setDownloadUrl(null); }} className="text-sm text-gray-400 hover:text-white transition-colors"><RotateCcw size={14} className="inline mr-1" /> 处理新的 PDF</button></div>}
      </div>
    );
  };

  return (
    <div className="w-full h-screen flex flex-col bg-[#050512] overflow-hidden">
      {showBanner && (<div className="w-full bg-gradient-to-r from-orange-600 via-red-600 to-pink-500 relative flex-shrink-0"><div className="absolute inset-0 bg-black opacity-20"></div><div className="relative max-w-7xl mx-auto px-4 py-2.5 flex items-center justify-between"><div className="flex items-center gap-3"><Star size={14} className="text-yellow-300 fill-yellow-300" /><span className="text-sm text-white">✨ Pdf2PPT - PDF 智能转 PPT</span></div><div className="flex items-center gap-2"><a href="https://github.com/OpenDCAI/DataFlow-Agent" target="_blank" rel="noopener noreferrer" className="px-3 py-1 bg-white/90 text-gray-900 rounded-full text-xs font-medium flex items-center gap-1"><Github size={12} /> GitHub</a><button onClick={() => setShowBanner(false)} className="p-1 hover:bg-white/20 rounded-full"><X size={14} className="text-white" /></button></div></div></div>)}
      <div className="flex-1 overflow-auto"><div className="max-w-7xl mx-auto px-6 py-8 pb-24">{renderStepIndicator()}{currentStep === 'upload' && renderUploadStep()}{currentStep === 'outline' && renderOutlineStep()}{currentStep === 'generate' && renderGenerateStep()}{currentStep === 'complete' && renderCompleteStep()}</div></div>
      <style>{`.glass { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(10px); }`}</style>
    </div>
  );
};

export default Pdf2PptPage;

