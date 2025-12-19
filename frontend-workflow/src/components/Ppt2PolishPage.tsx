import { useState, ChangeEvent } from 'react';
import { 
  Presentation, UploadCloud, Settings2, Download, Loader2, CheckCircle2, 
  AlertCircle, ChevronDown, ChevronUp, Github, Star, X, Sparkles,
  ArrowRight, ArrowLeft, GripVertical, Trash2, Edit3, Check, RotateCcw,
  SkipForward, MessageSquare, Eye, RefreshCw, FileText, Image as ImageIcon
} from 'lucide-react';

// ============== 类型定义 ==============
type Step = 'upload' | 'outline' | 'beautify' | 'complete';

// 后端返回的原始数据结构（TODO: 待真实 API 对接时使用）
/*
interface BackendSlideData {
  title: string;
  layout_description: string;
  key_points: string[];
  asset_ref: string | null;
}
*/

// 前端使用的 Slide 数据结构（在后端数据基础上添加 id 和 pageNum）
interface SlideOutline {
  id: string;
  pageNum: number;
  title: string;
  layout_description: string;  // 布局描述
  key_points: string[];        // 要点数组
  asset_ref: string | null;    // 资源引用（图片路径或 null）
}

interface BeautifyResult {
  slideId: string;
  beforeImage: string;
  afterImage: string;
  status: 'pending' | 'processing' | 'done' | 'skipped';
  userPrompt?: string;
}

// ============== 假数据模拟 ==============
// 模拟后端返回的数据（转换为前端格式）
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
      '当前 deep research 框架（OpenResearcher、Search‑o1 等）主要输出长篇文本报告，忽略可视化在沟通中的关键作用。',
      '仅文本形式难以有效传递复杂数据洞见，降低可读性与实用性。',
      '真实世界的研究报告与演示文稿通常由专家精心设计多种图表，并与文本紧密交织。',
      '缺乏标准化的文本‑图表混排格式，使得基于示例的 in‑context learning 难以应用。',
      '本工作提出一种系统化框架，使 LLM 能"像专家一样"规划、生成并整合多种可视化。'
    ],
    asset_ref: null
  },
  { 
    id: '3', pageNum: 3, 
    title: '整体框架：Multimodal DeepResearcher 四阶段流程', 
    layout_description: '整页采用"上图下文"布局：上半部分居中大图展示框架流程图，下半部分分两栏简要解释每个阶段的功能。',
    key_points: [
      '将"从主题到多模态报告"的复杂任务拆解为四个阶段的代理式流程。',
      '阶段 1 Researching：迭代式检索 + 推理，构建高质量 learnings 与引用。',
      '阶段 2 Exemplar Textualization：将人类专家多模态报告转成仅文本形式，并用 FDV 编码图表。',
      '阶段 3 Planning：基于 learnings 与示例生成报告大纲 O 与可视化风格指南 G。',
      '阶段 4 Multimodal Report Generation：先生成含 FDV 的文本草稿，再自动写代码、渲染并迭代优化图表。'
    ],
    asset_ref: 'images/98925d41396b1c5db17882d7a83faf7af0d896c6f655d6ca0e3838fc7c65d1ab.jpg'
  },
  { 
    id: '4', pageNum: 4, 
    title: '关键设计一：Formal Description of Visualization (FDV)', 
    layout_description: '左文右图：左侧用分点解释 FDV 的四个部分及作用；右侧展示三联图（原图 → FDV 文本 → 重建图）。',
    key_points: [
      'FDV 是受 Grammar of Graphics 启发的结构化文本表示，可对任意可视化进行高保真描述。',
      '四个视角：整体布局（Part‑A）、坐标与编码尺度（Part‑B）、底层数据与文本（Part‑C）、图形标记及样式（Part‑D）。',
      '借助 FDV，可将专家报告中的图表"文本化"，用于 LLM 的 in‑context 学习。',
      '同一 FDV 可被代码自动"反向生成"为对应图表，实现图表的可逆描述与重构。'
    ],
    asset_ref: 'images/46f46d81324259498bf3cd7e63831f7074eac0f0b7dd8b6bd0350debf22344e7.jpg'
  },
];

// 辅助函数：将后端返回的数据转换为前端格式（TODO: 待真实 API 对接时使用）
// const convertBackendDataToSlides = (backendData: BackendSlideData[]): SlideOutline[] => {
//   return backendData.map((item, index) => ({
//     id: String(index + 1),
//     pageNum: index + 1,
//     title: item.title,
//     layout_description: item.layout_description,
//     key_points: item.key_points,
//     asset_ref: item.asset_ref,
//   }));
// };

const MOCK_BEFORE_IMAGES = [
  '/ppe2more_1.jpg',
  '/ppe2more_1.jpg',
  '/ppe2more_1.jpg',
  '/ppe2more_1.jpg',
  '/ppe2more_1.jpg',
  '/ppe2more_1.jpg',
  '/ppe2more_1.jpg',
  '/ppe2more_1.jpg',
];

const MOCK_AFTER_IMAGES = [
  '/ppe2more_2.jpg',
  '/ppe2more_2.jpg',
  '/ppe2more_2.jpg',
  '/ppe2more_2.jpg',
  '/ppe2more_2.jpg',
  '/ppe2more_2.jpg',
  '/ppe2more_2.jpg',
  '/ppe2more_2.jpg',
];

// ============== 主组件 ==============
const Ppt2PolishPage = () => {
  // 步骤状态
  const [currentStep, setCurrentStep] = useState<Step>('upload');
  
  // Step 1: 上传相关状态
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [styleMode, setStyleMode] = useState<'preset' | 'reference'>('preset');
  const [stylePreset, setStylePreset] = useState<'modern' | 'business' | 'academic' | 'creative'>('modern');
  const [globalPrompt, setGlobalPrompt] = useState('');
  const [referenceImage, setReferenceImage] = useState<File | null>(null);
  const [referenceImagePreview, setReferenceImagePreview] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  
  // Step 2: Outline 相关状态
  const [outlineData, setOutlineData] = useState<SlideOutline[]>([]);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editContent, setEditContent] = useState<{
    title: string;
    layout_description: string;
    key_points: string[];
  }>({ title: '', layout_description: '', key_points: [] });
  
  // Step 3: 美化相关状态
  const [currentSlideIndex, setCurrentSlideIndex] = useState(0);
  const [beautifyResults, setBeautifyResults] = useState<BeautifyResult[]>([]);
  const [isBeautifying, setIsBeautifying] = useState(false);
  const [slidePrompt, setSlidePrompt] = useState('');
  
  // Step 4: 完成状态
  const [isGeneratingFinal, setIsGeneratingFinal] = useState(false);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  
  // 通用状态
  const [error, setError] = useState<string | null>(null);
  const [showBanner, setShowBanner] = useState(true);

  // ============== Step 1: 上传处理 ==============
  const validateDocFile = (file: File): boolean => {
    const ext = file.name.split('.').pop()?.toLowerCase();
    if (ext !== 'ppt' && ext !== 'pptx') {
      setError('仅支持 PPT/PPTX 格式');
      return false;
    }
    return true;
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!validateDocFile(file)) return;
    setSelectedFile(file);
    setError(null);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (!file) return;
    if (!validateDocFile(file)) return;
    setSelectedFile(file);
    setError(null);
  };

  const handleReferenceImageChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const ext = file.name.split('.').pop()?.toLowerCase();
    if (!['jpg', 'jpeg', 'png', 'webp', 'gif'].includes(ext || '')) {
      setError('参考图片仅支持 JPG/PNG/WEBP/GIF 格式');
      return;
    }
    setReferenceImage(file);
    setReferenceImagePreview(URL.createObjectURL(file));
    setError(null);
  };

  const handleRemoveReferenceImage = () => {
    if (referenceImagePreview) {
      URL.revokeObjectURL(referenceImagePreview);
    }
    setReferenceImage(null);
    setReferenceImagePreview(null);
  };

  const handleUploadAndParse = async () => {
    if (!selectedFile) {
      setError('请先选择 PPT 文件');
      return;
    }
    
    if (styleMode === 'reference' && !referenceImage) {
      setError('请上传参考风格图片');
      return;
    }
    
    setIsUploading(true);
    setError(null);
    
    try {
      // 模拟后端解析延迟
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // 使用假数据
      setOutlineData(MOCK_OUTLINE);
    } catch (err) {
      setError('解析失败，请重试');
      console.error(err);
    } finally {
      setIsUploading(false);
    }
    
    setCurrentStep('outline');
  };

  // ============== Step 2: Outline 编辑处理 ==============
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
        ? { 
            ...s, 
            title: editContent.title, 
            layout_description: editContent.layout_description,
            key_points: editContent.key_points 
          }
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
    setEditContent(prev => ({
      ...prev,
      key_points: [...prev.key_points, '']
    }));
  };

  const handleRemoveKeyPoint = (index: number) => {
    setEditContent(prev => ({
      ...prev,
      key_points: prev.key_points.filter((_, i) => i !== index)
    }));
  };

  const handleEditCancel = () => {
    setEditingId(null);
  };

  const handleDeleteSlide = (id: string) => {
    setOutlineData(prev => prev.filter(s => s.id !== id).map((s, i) => ({ ...s, pageNum: i + 1 })));
  };

  const handleMoveSlide = (index: number, direction: 'up' | 'down') => {
    const newData = [...outlineData];
    const targetIndex = direction === 'up' ? index - 1 : index + 1;
    if (targetIndex < 0 || targetIndex >= newData.length) return;
    [newData[index], newData[targetIndex]] = [newData[targetIndex], newData[index]];
    setOutlineData(newData.map((s, i) => ({ ...s, pageNum: i + 1 })));
  };

  const handleConfirmOutline = () => {
    const results: BeautifyResult[] = outlineData.map((slide, index) => ({
      slideId: slide.id,
      beforeImage: MOCK_BEFORE_IMAGES[index % MOCK_BEFORE_IMAGES.length],
      afterImage: MOCK_AFTER_IMAGES[index % MOCK_AFTER_IMAGES.length],
      status: 'pending',
    }));
    setBeautifyResults(results);
    setCurrentSlideIndex(0);
    setCurrentStep('beautify');
    startBeautifyCurrentSlide(results, 0);
  };

  // ============== Step 3: 逐页美化处理 ==============
  const startBeautifyCurrentSlide = async (results: BeautifyResult[], index: number) => {
    setIsBeautifying(true);
    const updatedResults = [...results];
    updatedResults[index] = { ...updatedResults[index], status: 'processing' };
    setBeautifyResults(updatedResults);
    await new Promise(resolve => setTimeout(resolve, 2500));
    updatedResults[index] = { ...updatedResults[index], status: 'done' };
    setBeautifyResults(updatedResults);
    setIsBeautifying(false);
  };

  const handleConfirmSlide = () => {
    if (currentSlideIndex < outlineData.length - 1) {
      const nextIndex = currentSlideIndex + 1;
      setCurrentSlideIndex(nextIndex);
      setSlidePrompt('');
      startBeautifyCurrentSlide(beautifyResults, nextIndex);
    } else {
      setCurrentStep('complete');
    }
  };

  const handleSkipSlide = () => {
    const updatedResults = [...beautifyResults];
    updatedResults[currentSlideIndex] = { ...updatedResults[currentSlideIndex], status: 'skipped' };
    setBeautifyResults(updatedResults);
    if (currentSlideIndex < outlineData.length - 1) {
      const nextIndex = currentSlideIndex + 1;
      setCurrentSlideIndex(nextIndex);
      setSlidePrompt('');
      startBeautifyCurrentSlide(updatedResults, nextIndex);
    } else {
      setCurrentStep('complete');
    }
  };

  const handleRegenerateSlide = async () => {
    const updatedResults = [...beautifyResults];
    updatedResults[currentSlideIndex] = { 
      ...updatedResults[currentSlideIndex], 
      userPrompt: slidePrompt,
      status: 'pending'
    };
    setBeautifyResults(updatedResults);
    await startBeautifyCurrentSlide(updatedResults, currentSlideIndex);
  };

  // ============== Step 4: 完成下载处理 ==============
  const handleGenerateFinal = async () => {
    setIsGeneratingFinal(true);
    await new Promise(resolve => setTimeout(resolve, 3000));
    setDownloadUrl('/mock-beautified.pptx');
    setIsGeneratingFinal(false);
  };

  const handleDownload = () => {
    alert('下载功能将在后端对接后启用');
  };

  // ============== 渲染步骤指示器 ==============
  const renderStepIndicator = () => {
    const steps = [
      { key: 'upload', label: '上传 PPT', num: 1 },
      { key: 'outline', label: 'Outline 确认', num: 2 },
      { key: 'beautify', label: '逐页美化', num: 3 },
      { key: 'complete', label: '完成下载', num: 4 },
    ];
    
    const currentIndex = steps.findIndex(s => s.key === currentStep);
    
    return (
      <div className="flex items-center justify-center gap-2 mb-8">
        {steps.map((step, index) => (
          <div key={step.key} className="flex items-center">
            <div className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-all ${
              index === currentIndex 
                ? 'bg-gradient-to-r from-cyan-500 to-teal-500 text-white shadow-lg' 
                : index < currentIndex 
                  ? 'bg-teal-500/20 text-teal-300 border border-teal-500/40'
                  : 'bg-white/5 text-gray-500 border border-white/10'
            }`}>
              <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs ${
                index < currentIndex ? 'bg-teal-400 text-white' : ''
              }`}>
                {index < currentIndex ? <Check size={14} /> : step.num}
              </span>
              <span className="hidden sm:inline">{step.label}</span>
            </div>
            {index < steps.length - 1 && (
              <ArrowRight size={16} className={`mx-2 ${index < currentIndex ? 'text-teal-400' : 'text-gray-600'}`} />
            )}
          </div>
        ))}
      </div>
    );
  };

  // ============== Step 1: 上传界面 ==============
  const renderUploadStep = () => (
    <div className="max-w-6xl mx-auto">
      <div className="mb-10 text-center">
        <p className="text-xs uppercase tracking-[0.2em] text-teal-300 mb-3 font-semibold">
          PPT → BEAUTIFIED PPT
        </p>
        <h1 className="text-4xl md:text-5xl font-bold mb-4">
          <span className="bg-gradient-to-r from-cyan-400 via-teal-400 to-emerald-400 bg-clip-text text-transparent">
            Ppt2Polish
          </span>
        </h1>
        <p className="text-base text-gray-300 max-w-2xl mx-auto leading-relaxed">
          上传原始 PPT 文件，AI 智能分析内容结构，一键美化生成专业演示文稿。
          <br />
          <span className="text-teal-400">通过左右对比，实时掌控美化效果！</span>
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="glass rounded-xl border border-white/10 p-6">
          <h3 className="text-white font-semibold flex items-center gap-2 mb-4">
            <FileText size={18} className="text-teal-400" />
            上传 PPT
          </h3>
          <div
            className={`border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center text-center gap-4 transition-all ${
              isDragOver ? 'border-teal-500 bg-teal-500/10' : 'border-white/20 hover:border-teal-400'
            }`}
            onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
            onDragLeave={(e) => { e.preventDefault(); setIsDragOver(false); }}
            onDrop={handleDrop}
          >
            <div className="w-16 h-16 rounded-full bg-gradient-to-br from-cyan-500/20 to-teal-500/20 flex items-center justify-center">
              <UploadCloud size={32} className="text-teal-400" />
            </div>
            <div>
              <p className="text-white font-medium mb-1">拖拽 PPT 文件到此处</p>
              <p className="text-sm text-gray-400">支持 PPT / PPTX</p>
            </div>
            <label className="px-6 py-2.5 rounded-full bg-gradient-to-r from-cyan-600 to-teal-600 text-white text-sm font-medium cursor-pointer hover:from-cyan-700 hover:to-teal-700 transition-all">
              <Presentation size={16} className="inline mr-2" />
              选择文件
              <input type="file" accept=".ppt,.pptx" className="hidden" onChange={handleFileChange} />
            </label>
            {selectedFile && (
              <div className="px-4 py-2 bg-teal-500/20 border border-teal-500/40 rounded-lg">
                <p className="text-sm text-teal-300">✓ {selectedFile.name}</p>
                <p className="text-xs text-gray-400 mt-1">🎨 美化模式：将优化原有 PPT 样式</p>
              </div>
            )}
          </div>
        </div>

        <div className="glass rounded-xl border border-white/10 p-6 space-y-5">
          <h3 className="text-white font-semibold flex items-center gap-2">
            <Settings2 size={18} className="text-teal-400" />
            风格配置
          </h3>
          <div className="flex gap-2">
            <button onClick={() => setStyleMode('preset')} className={`flex-1 py-2.5 px-4 rounded-lg text-sm font-medium flex items-center justify-center gap-2 transition-all ${styleMode === 'preset' ? 'bg-gradient-to-r from-cyan-500 to-teal-500 text-white' : 'bg-white/5 text-gray-400 border border-white/10 hover:bg-white/10'}`}>
              <Sparkles size={16} /> 预设风格
            </button>
            <button onClick={() => setStyleMode('reference')} className={`flex-1 py-2.5 px-4 rounded-lg text-sm font-medium flex items-center justify-center gap-2 transition-all ${styleMode === 'reference' ? 'bg-gradient-to-r from-cyan-500 to-teal-500 text-white' : 'bg-white/5 text-gray-400 border border-white/10 hover:bg-white/10'}`}>
              <ImageIcon size={16} /> 参考图片
            </button>
          </div>
          {styleMode === 'preset' && (
            <>
              <div>
                <label className="block text-sm text-gray-300 mb-2">选择风格</label>
                <select value={stylePreset} onChange={(e) => setStylePreset(e.target.value as typeof stylePreset)} className="w-full rounded-lg border border-white/20 bg-black/40 px-4 py-2.5 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-teal-500">
                  <option value="modern">现代简约</option>
                  <option value="business">商务专业</option>
                  <option value="academic">学术报告</option>
                  <option value="creative">创意设计</option>
                </select>
              </div>
              <div>
                <label className="block text-sm text-gray-300 mb-2">风格提示词（可选）</label>
                <textarea value={globalPrompt} onChange={(e) => setGlobalPrompt(e.target.value)} placeholder="例如：使用蓝色系配色，保持简洁风格..." rows={3} className="w-full rounded-lg border border-white/20 bg-black/40 px-4 py-2.5 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-teal-500 placeholder:text-gray-500 resize-none" />
              </div>
            </>
          )}
          {styleMode === 'reference' && (
            <div>
              <label className="block text-sm text-gray-300 mb-2">上传参考风格图片</label>
              {referenceImagePreview ? (
                <div className="relative">
                  <img src={referenceImagePreview} alt="参考风格" className="w-full h-40 object-cover rounded-lg border border-white/20" />
                  <button onClick={handleRemoveReferenceImage} className="absolute top-2 right-2 p-1.5 rounded-full bg-black/60 text-white hover:bg-red-500 transition-colors"><X size={14} /></button>
                  <p className="text-xs text-teal-300 mt-2">✓ 已上传参考图片</p>
                </div>
              ) : (
                <label className="border-2 border-dashed border-white/20 rounded-lg p-6 flex flex-col items-center justify-center text-center gap-2 cursor-pointer hover:border-teal-400 transition-all">
                  <div className="w-12 h-12 rounded-full bg-white/5 flex items-center justify-center"><ImageIcon size={24} className="text-gray-400" /></div>
                  <p className="text-sm text-gray-400">点击上传参考图片</p>
                  <input type="file" accept="image/*" className="hidden" onChange={handleReferenceImageChange} />
                </label>
              )}
            </div>
          )}
          <button onClick={handleUploadAndParse} disabled={!selectedFile || isUploading} className="w-full py-3 rounded-lg bg-gradient-to-r from-cyan-600 to-teal-600 hover:from-cyan-700 hover:to-teal-700 disabled:from-gray-600 disabled:to-gray-700 text-white font-semibold flex items-center justify-center gap-2 transition-all">
            {isUploading ? <><Loader2 size={18} className="animate-spin" /> 解析中...</> : <><ArrowRight size={18} /> 开始解析</>}
          </button>
        </div>
      </div>
      {error && <div className="mt-4 flex items-center gap-2 text-sm text-red-300 bg-red-500/10 border border-red-500/40 rounded-lg px-4 py-3"><AlertCircle size={16} /> {error}</div>}
    </div>
  );

  // ============== Step 2: Outline 编辑界面 ==============
  const renderOutlineStep = () => (
    <div className="max-w-5xl mx-auto">
      <div className="text-center mb-8">
        <h2 className="text-2xl font-bold text-white mb-2">确认 Outline</h2>
        <p className="text-gray-400">检查并调整页面结构，可编辑、排序或删除页面</p>
      </div>
      <div className="glass rounded-xl border border-white/10 p-6 mb-6">
        <div className="space-y-3">
          {outlineData.map((slide, index) => (
            <div key={slide.id} className={`flex items-start gap-4 p-4 rounded-lg border transition-all ${editingId === slide.id ? 'bg-teal-500/10 border-teal-500/40' : 'bg-white/5 border-white/10 hover:border-white/20'}`}>
              <div className="flex items-center gap-2 pt-1">
                <GripVertical size={16} className="text-gray-500 cursor-grab" />
                <span className="w-8 h-8 rounded-full bg-teal-500/20 text-teal-300 text-sm font-medium flex items-center justify-center">{slide.pageNum}</span>
              </div>
              <div className="flex-1">
                {editingId === slide.id ? (
                  <div className="space-y-3">
                    <input type="text" value={editContent.title} onChange={(e) => setEditContent(prev => ({ ...prev, title: e.target.value }))} className="w-full px-3 py-2 rounded-lg bg-black/40 border border-white/20 text-white text-sm outline-none focus:ring-2 focus:ring-teal-500" placeholder="页面标题" />
                    <textarea value={editContent.layout_description} onChange={(e) => setEditContent(prev => ({ ...prev, layout_description: e.target.value }))} rows={2} className="w-full px-3 py-2 rounded-lg bg-black/40 border border-white/20 text-white text-sm outline-none focus:ring-2 focus:ring-teal-500 resize-none" placeholder="布局描述" />
                    <div className="space-y-2">
                      {editContent.key_points.map((point, idx) => (
                        <div key={idx} className="flex gap-2">
                          <input type="text" value={point} onChange={(e) => handleKeyPointChange(idx, e.target.value)} className="flex-1 px-3 py-2 rounded-lg bg-black/40 border border-white/20 text-white text-sm outline-none focus:ring-2 focus:ring-teal-500" placeholder={`要点 ${idx + 1}`} />
                          <button onClick={() => handleRemoveKeyPoint(idx)} className="p-2 rounded-lg hover:bg-red-500/20 text-gray-400 hover:text-red-400"><Trash2 size={14} /></button>
                        </div>
                      ))}
                      <button onClick={handleAddKeyPoint} className="px-3 py-1.5 rounded-lg bg-white/5 border border-dashed border-white/20 text-gray-400 hover:text-teal-400 hover:border-teal-400 text-sm w-full">+ 添加要点</button>
                    </div>
                    <div className="flex gap-2 pt-2">
                      <button onClick={handleEditSave} className="px-3 py-1.5 rounded-lg bg-teal-500 text-white text-sm flex items-center gap-1"><Check size={14} /> 保存</button>
                      <button onClick={handleEditCancel} className="px-3 py-1.5 rounded-lg bg-white/10 text-gray-300 text-sm">取消</button>
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="mb-2"><h4 className="text-white font-medium">{slide.title}</h4></div>
                    <p className="text-xs text-cyan-400/70 mb-2 italic">📐 {slide.layout_description}</p>
                    <ul className="space-y-1">{slide.key_points.map((point, idx) => (<li key={idx} className="text-sm text-gray-400 flex items-start gap-2"><span className="text-teal-400 mt-0.5">•</span><span>{point}</span></li>))}</ul>
                  </>
                )}
              </div>
              {editingId !== slide.id && (
                <div className="flex items-center gap-1">
                  <button onClick={() => handleMoveSlide(index, 'up')} disabled={index === 0} className="p-2 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white disabled:opacity-30"><ChevronUp size={16} /></button>
                  <button onClick={() => handleMoveSlide(index, 'down')} disabled={index === outlineData.length - 1} className="p-2 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white disabled:opacity-30"><ChevronDown size={16} /></button>
                  <button onClick={() => handleEditStart(slide)} className="p-2 rounded-lg hover:bg-white/10 text-gray-400 hover:text-teal-400"><Edit3 size={16} /></button>
                  <button onClick={() => handleDeleteSlide(slide.id)} className="p-2 rounded-lg hover:bg-red-500/20 text-gray-400 hover:text-red-400"><Trash2 size={16} /></button>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
      <div className="flex justify-between">
        <button onClick={() => setCurrentStep('upload')} className="px-6 py-2.5 rounded-lg border border-white/20 text-gray-300 hover:bg-white/10 flex items-center gap-2 transition-all"><ArrowLeft size={18} /> 返回上传</button>
        <button onClick={handleConfirmOutline} className="px-6 py-2.5 rounded-lg bg-gradient-to-r from-cyan-600 to-teal-600 hover:from-cyan-700 hover:to-teal-700 text-white font-semibold flex items-center gap-2 transition-all">确认并开始美化 <ArrowRight size={18} /></button>
      </div>
    </div>
  );

  // ============== Step 3: 逐页美化界面 ==============
  const renderBeautifyStep = () => {
    const currentSlide = outlineData[currentSlideIndex];
    const currentResult = beautifyResults[currentSlideIndex];
    return (
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-6">
          <h2 className="text-2xl font-bold text-white mb-2">逐页美化</h2>
          <p className="text-gray-400">第 {currentSlideIndex + 1} / {outlineData.length} 页：{currentSlide?.title}</p>
          <p className="text-xs text-gray-500 mt-1">🎨 美化模式 - 优化原有 PPT 样式</p>
        </div>
        <div className="mb-6">
          <div className="flex gap-1">{beautifyResults.map((result, index) => (<div key={result.slideId} className={`flex-1 h-2 rounded-full transition-all ${result.status === 'done' ? 'bg-teal-400' : result.status === 'skipped' ? 'bg-yellow-400' : result.status === 'processing' ? 'bg-gradient-to-r from-cyan-400 to-teal-400 animate-pulse' : index === currentSlideIndex ? 'bg-teal-400/50' : 'bg-white/10'}`} />))}</div>
        </div>
        <div className="glass rounded-xl border border-white/10 p-6 mb-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="text-sm text-gray-400 mb-3 flex items-center gap-2"><Eye size={14} /> 原始 PPT 渲染</h4>
              <div className="rounded-lg overflow-hidden border border-white/10 aspect-[4/3] bg-white/5 flex items-center justify-center">{currentResult?.beforeImage ? <img src={currentResult.beforeImage} alt="Before" className="w-full h-full object-cover" /> : <Loader2 size={24} className="text-gray-500 animate-spin" />}</div>
            </div>
            <div>
              <h4 className="text-sm text-gray-400 mb-3 flex items-center gap-2"><Sparkles size={14} className="text-teal-400" /> 美化结果</h4>
              <div className="rounded-lg overflow-hidden border border-teal-500/30 aspect-[4/3] bg-gradient-to-br from-cyan-500/10 to-teal-500/10 flex items-center justify-center">{isBeautifying ? <div className="text-center"><Loader2 size={32} className="text-teal-400 animate-spin mx-auto mb-2" /><p className="text-sm text-teal-300">正在美化中...</p></div> : currentResult?.afterImage ? <img src={currentResult.afterImage} alt="After" className="w-full h-full object-cover" /> : <span className="text-gray-500">等待生成</span>}</div>
            </div>
          </div>
        </div>
        <div className="glass rounded-xl border border-white/10 p-4 mb-6">
          <div className="flex items-center gap-3"><MessageSquare size={18} className="text-teal-400" /><input type="text" value={slidePrompt} onChange={(e) => setSlidePrompt(e.target.value)} placeholder="输入微调 Prompt，然后点击重新生成..." className="flex-1 bg-transparent border-none outline-none text-white text-sm placeholder:text-gray-500" /><button onClick={handleRegenerateSlide} disabled={isBeautifying || !slidePrompt.trim()} className="px-4 py-2 rounded-lg bg-white/10 hover:bg-white/20 text-gray-300 text-sm flex items-center gap-2 disabled:opacity-50 transition-all"><RefreshCw size={14} /> 重新生成</button></div>
        </div>
        <div className="flex justify-between">
          <button onClick={() => setCurrentStep('outline')} className="px-6 py-2.5 rounded-lg border border-white/20 text-gray-300 hover:bg-white/10 flex items-center gap-2 transition-all"><ArrowLeft size={18} /> 返回 Outline</button>
          <div className="flex gap-3"><button onClick={handleSkipSlide} disabled={isBeautifying} className="px-5 py-2.5 rounded-lg bg-yellow-500/20 border border-yellow-500/40 text-yellow-300 hover:bg-yellow-500/30 flex items-center gap-2 transition-all"><SkipForward size={18} /> 跳过此页</button><button onClick={handleConfirmSlide} disabled={isBeautifying} className="px-6 py-2.5 rounded-lg bg-gradient-to-r from-cyan-600 to-teal-600 hover:from-cyan-700 hover:to-teal-700 text-white font-semibold flex items-center gap-2 transition-all"><CheckCircle2 size={18} /> 确认并继续</button></div>
        </div>
      </div>
    );
  };

  // ============== Step 4: 完成下载界面 ==============
  const renderCompleteStep = () => (
    <div className="max-w-2xl mx-auto text-center">
      <div className="mb-8"><div className="w-20 h-20 rounded-full bg-gradient-to-br from-cyan-500 to-teal-500 flex items-center justify-center mx-auto mb-4"><CheckCircle2 size={40} className="text-white" /></div><h2 className="text-2xl font-bold text-white mb-2">美化完成！</h2></div>
      <div className="glass rounded-xl border border-white/10 p-6 mb-6">
        <h3 className="text-white font-semibold mb-4">处理结果概览</h3>
        <div className="grid grid-cols-4 gap-2">{beautifyResults.map((result, index) => (<div key={result.slideId} className={`p-3 rounded-lg border ${result.status === 'done' ? 'bg-teal-500/20 border-teal-500/40' : 'bg-yellow-500/20 border-yellow-500/40'}`}><p className="text-sm text-white">第 {index + 1} 页</p><p className={`text-xs ${result.status === 'done' ? 'text-teal-300' : 'text-yellow-300'}`}>{result.status === 'done' ? '已美化' : '已跳过'}</p></div>))}</div>
      </div>
      {!downloadUrl ? <button onClick={handleGenerateFinal} disabled={isGeneratingFinal} className="px-8 py-3 rounded-lg bg-gradient-to-r from-cyan-600 to-teal-600 hover:from-cyan-700 hover:to-teal-700 text-white font-semibold flex items-center justify-center gap-2 mx-auto transition-all">{isGeneratingFinal ? <><Loader2 size={18} className="animate-spin" /> 正在生成最终 PPT...</> : <><Sparkles size={18} /> 生成最终 PPT</>}</button> : <div className="space-y-4"><button onClick={handleDownload} className="px-8 py-3 rounded-lg bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-600 hover:to-teal-600 text-white font-semibold flex items-center justify-center gap-2 mx-auto transition-all"><Download size={18} /> 下载美化后的 PPT</button><button onClick={() => { setCurrentStep('upload'); setSelectedFile(null); setOutlineData([]); setBeautifyResults([]); setDownloadUrl(null); }} className="text-sm text-gray-400 hover:text-white transition-colors"><RotateCcw size={14} className="inline mr-1" /> 处理新的文档</button></div>}
    </div>
  );

  return (
    <div className="w-full h-screen flex flex-col bg-[#050512] overflow-hidden">
      {showBanner && (<div className="w-full bg-gradient-to-r from-cyan-600 via-teal-600 to-emerald-500 relative overflow-hidden flex-shrink-0"><div className="absolute inset-0 bg-black opacity-20"></div><div className="relative max-w-7xl mx-auto px-4 py-2.5 flex items-center justify-between"><div className="flex items-center gap-3"><Star size={14} className="text-yellow-300 fill-yellow-300" /><span className="text-sm text-white">✨ Ppt2Polish - 智能 PPT 美化工具</span></div><div className="flex items-center gap-2"><a href="https://github.com/OpenDCAI/DataFlow-Agent" target="_blank" rel="noopener noreferrer" className="px-3 py-1 bg-white/90 text-gray-900 rounded-full text-xs font-medium hover:bg-white transition-all flex items-center gap-1"><Github size={12} /> GitHub</a><button onClick={() => setShowBanner(false)} className="p-1 hover:bg-white/20 rounded-full"><X size={14} className="text-white" /></button></div></div></div>)}
      <div className="flex-1 w-full overflow-auto"><div className="max-w-7xl mx-auto px-6 py-8 pb-24">{renderStepIndicator()}{currentStep === 'upload' && renderUploadStep()}{currentStep === 'outline' && renderOutlineStep()}{currentStep === 'beautify' && renderBeautifyStep()}{currentStep === 'complete' && renderCompleteStep()}</div></div>
      <style>{`.glass { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(10px); }`}</style>
    </div>
  );
};

export default Ppt2PolishPage;

