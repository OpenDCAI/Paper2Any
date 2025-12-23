import { useState, ChangeEvent } from 'react';
import { 
  UploadCloud, Settings2, Download, Loader2, CheckCircle2, 
  AlertCircle, ChevronDown, ChevronUp, Github, Star, X, Sparkles,
  ArrowRight, ArrowLeft, GripVertical, Trash2, Edit3, Check, RotateCcw,
  MessageSquare, RefreshCw, FileText, Key, Globe, Cpu
} from 'lucide-react';

// ============== 类型定义 ==============
type Step = 'upload' | 'outline' | 'generate' | 'complete';

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
  beforeImage: string;
  afterImage: string;
  status: 'pending' | 'processing' | 'done';
  userPrompt?: string;
}

// ============== 主组件 ==============
const Paper2PptPage = () => {
  // Step 状态
  const [currentStep, setCurrentStep] = useState<Step>('upload');
  
  // Step 1: 上传相关状态
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [stylePreset, setStylePreset] = useState<'modern' | 'business' | 'academic' | 'creative'>('modern');
  const [globalPrompt, setGlobalPrompt] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [pageCount, setPageCount] = useState(6);
  
  // Step 2: Outline 相关状态
  const [outlineData, setOutlineData] = useState<SlideOutline[]>([]);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editContent, setEditContent] = useState<{
    title: string;
    layout_description: string;
    key_points: string[];
  }>({ title: '', layout_description: '', key_points: [] });
  
  // Step 3: 生成相关状态
  const [currentSlideIndex, setCurrentSlideIndex] = useState(0);
  const [generateResults, setGenerateResults] = useState<GenerateResult[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [slidePrompt, setSlidePrompt] = useState('');
  
  // Step 4: 完成状态
  const [isGeneratingFinal, setIsGeneratingFinal] = useState(false);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [pdfPreviewUrl, setPdfPreviewUrl] = useState<string | null>(null);
  
  // 通用状态
  const [error, setError] = useState<string | null>(null);
  const [showBanner, setShowBanner] = useState(true);
  
  // API 配置状态
  const [inviteCode, setInviteCode] = useState('');
  const [llmApiUrl, setLlmApiUrl] = useState('https://api.apiyi.com/v1');
  const [apiKey, setApiKey] = useState('');
  const [model, setModel] = useState('gpt-5.1');
  const [genFigModel, setGenFigModel] = useState('gemini-2.5-flash-image');
  const [language, setLanguage] = useState<'zh' | 'en'>('zh');
  const [resultPath, setResultPath] = useState<string | null>(null);

  // ============== Step 1: 上传处理 ==============
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
    // if (!inviteCode.trim()) {
    //   setError('请输入邀请码');
    //   return;
    // }
    if (!apiKey.trim()) {
      setError('请输入 API Key');
      return;
    }
    
    setIsUploading(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('input_type', 'pdf');
      formData.append('invite_code', inviteCode.trim());
      formData.append('chat_api_url', llmApiUrl.trim());
      formData.append('api_key', apiKey.trim());
      formData.append('model', model);
      formData.append('language', language);
      formData.append('style', globalPrompt || getStyleDescription(stylePreset));
      formData.append('gen_fig_model', genFigModel);
      formData.append('page_count', String(pageCount));
      
      console.log('Sending request to /api/paper2ppt/pagecontent_json with input_type=pdf');
      
      const res = await fetch('/api/paper2ppt/pagecontent_json', {
        method: 'POST',
        body: formData,
      });
      
      if (!res.ok) {
        let msg = '解析 PDF 失败';
        if (res.status === 403) {
          msg = '邀请码不正确或已失效';
        } else {
          try {
            const errorData = await res.json();
            msg = errorData.detail || errorData.message || msg;
          } catch {
            try {
              const text = await res.text();
              if (text) msg = text;
            } catch {
              // ignore
            }
          }
        }
        throw new Error(msg);
      }
      
      const data = await res.json();
      console.log('API Response:', JSON.stringify(data, null, 2));
      
      if (!data.success) {
        throw new Error(data.message || '解析失败');
      }
      
      const currentResultPath = data.result_path || '';
      if (currentResultPath) {
        setResultPath(currentResultPath);
      } else {
        throw new Error('后端未返回 result_path');
      }
      
      if (!data.pagecontent || data.pagecontent.length === 0) {
        throw new Error('解析结果为空，请检查 PDF 文件是否正确');
      }
      
      const convertedSlides: SlideOutline[] = data.pagecontent.map((item: any, index: number) => ({
        id: String(index + 1),
        pageNum: index + 1,
        title: item.title || `第 ${index + 1} 页`,
        layout_description: item.layout_description || '',
        key_points: item.key_points || [],
        asset_ref: item.asset_ref || null,
      }));
      
      setOutlineData(convertedSlides);
      setCurrentStep('outline');
      
    } catch (err) {
      const message = err instanceof Error ? err.message : '解析失败，请重试';
      setError(message);
      console.error(err);
    } finally {
      setIsUploading(false);
    }
  };

  const getStyleDescription = (preset: string): string => {
    const styles: Record<string, string> = {
      modern: '现代简约风格，使用干净的线条和充足的留白',
      business: '商务专业风格，稳重大气，适合企业演示',
      academic: '学术报告风格，清晰的层次结构，适合论文汇报',
      creative: '创意设计风格，活泼生动，色彩丰富',
    };
    return styles[preset] || styles.modern;
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
  
  const handleDeleteSlide = (id: string) => {
    setOutlineData(prev => prev.filter(s => s.id !== id).map((s, i) => ({ ...s, pageNum: i + 1, id: String(i + 1) })));
  };
  
  const handleMoveSlide = (index: number, direction: 'up' | 'down') => {
    const newData = [...outlineData];
    const targetIndex = direction === 'up' ? index - 1 : index + 1;
    if (targetIndex < 0 || targetIndex >= newData.length) return;
    [newData[index], newData[targetIndex]] = [newData[targetIndex], newData[index]];
    setOutlineData(newData.map((s, i) => ({ ...s, pageNum: i + 1 })));
  };

  const handleConfirmOutline = async () => {
    setCurrentStep('generate');
    setCurrentSlideIndex(0);
    setIsGenerating(true);
    setError(null);
    
    const results: GenerateResult[] = outlineData.map((slide) => ({
      slideId: slide.id,
      beforeImage: '',
      afterImage: '',
      status: 'processing' as const,
    }));
    setGenerateResults(results);
    
    try {
      const formData = new FormData();
      formData.append('img_gen_model_name', genFigModel);
      formData.append('chat_api_url', llmApiUrl.trim());
      formData.append('api_key', apiKey.trim());
      formData.append('model', model);
      formData.append('language', language);
      formData.append('style', globalPrompt || getStyleDescription(stylePreset));
      formData.append('aspect_ratio', '16:9');
      formData.append('invite_code', inviteCode.trim());
      formData.append('result_path', resultPath || '');
      formData.append('get_down', 'false');
      
      const pagecontent = outlineData.map((slide) => ({
        title: slide.title,
        layout_description: slide.layout_description,
        key_points: slide.key_points,
        asset_ref: slide.asset_ref,
      }));
      formData.append('pagecontent', JSON.stringify(pagecontent));
      
      const res = await fetch('/api/paper2ppt/ppt_json', {
        method: 'POST',
        body: formData,
      });
      
      if (!res.ok) {
        let msg = '生成失败';
        try {
          const errorData = await res.json();
          msg = errorData.detail || errorData.message || msg;
        } catch {
          // ignore
        }
        throw new Error(msg);
      }
      
      const data = await res.json();
      
      if (!data.success) {
        throw new Error(data.message || '生成失败');
      }
      
      const updatedResults = results.map((result, index) => {
        const pageNumStr = String(index).padStart(3, '0');
        let afterImage = '';
        
        if (data.all_output_files && Array.isArray(data.all_output_files)) {
          const pageImg = data.all_output_files.find((url: string) => 
            url.includes(`ppt_pages/page_${pageNumStr}.png`)
          );
          if (pageImg) {
            afterImage = pageImg;
          }
        }
        
        return {
          ...result,
          afterImage,
          status: 'done' as const,
        };
      });
      
      // 预加载所有图片到浏览器缓存，避免切换页面时延迟
      if (data.all_output_files && Array.isArray(data.all_output_files)) {
        console.log('预加载所有生成的图片...');
        data.all_output_files.forEach((url: string) => {
          if (url.endsWith('.png') || url.endsWith('.jpg') || url.endsWith('.jpeg')) {
            const img = new Image();
            img.src = url;
          }
        });
      }
      
      setGenerateResults(updatedResults);
      
    } catch (err) {
      const message = err instanceof Error ? err.message : '生成失败';
      setError(message);
      setGenerateResults(results.map(r => ({ ...r, status: 'pending' as const })));
    } finally {
      setIsGenerating(false);
    }
  };

  // ============== Step 3: 重新生成单页 ==============
  const handleRegenerateSlide = async () => {
    if (!resultPath) {
      setError('缺少 result_path，请重新上传文件');
      return;
    }
    
    if (!slidePrompt.trim()) {
      setError('请输入重新生成的提示词');
      return;
    }
    
    setIsGenerating(true);
    setError(null);
    
    const updatedResults = [...generateResults];
    updatedResults[currentSlideIndex] = { 
      ...updatedResults[currentSlideIndex], 
      status: 'processing',
      userPrompt: slidePrompt,
    };
    setGenerateResults(updatedResults);
    
    try {
      const formData = new FormData();
      formData.append('img_gen_model_name', genFigModel);
      formData.append('chat_api_url', llmApiUrl.trim());
      formData.append('api_key', apiKey.trim());
      formData.append('model', model);
      formData.append('language', language);
      formData.append('style', globalPrompt || getStyleDescription(stylePreset));
      formData.append('aspect_ratio', '16:9');
      formData.append('invite_code', inviteCode.trim());
      formData.append('result_path', resultPath);
      formData.append('get_down', 'true');
      formData.append('page_id', String(currentSlideIndex));
      formData.append('edit_prompt', slidePrompt);
      
      const pagecontent = outlineData.map((slide, idx) => {
        const result = generateResults[idx];
        let generatedPath = '';
        if (result?.afterImage) {
          // 直接使用 URL，后端会自动处理
          generatedPath = result.afterImage;
        }
        return {
          title: slide.title,
          layout_description: slide.layout_description,
          key_points: slide.key_points,
          asset_ref: slide.asset_ref,
          generated_img_path: generatedPath || undefined,
        };
      });
      formData.append('pagecontent', JSON.stringify(pagecontent));
      
      const res = await fetch('/api/paper2ppt/ppt_json', {
        method: 'POST',
        body: formData,
      });
      
      if (!res.ok) {
        let msg = '重新生成失败';
        try {
          const errorData = await res.json();
          msg = errorData.detail || errorData.message || msg;
        } catch {
          // ignore
        }
        throw new Error(msg);
      }
      
      const data = await res.json();
      
      if (!data.success) {
        throw new Error(data.message || '重新生成失败');
      }
      
      const pageNumStr = String(currentSlideIndex).padStart(3, '0');
      let afterImage = updatedResults[currentSlideIndex].afterImage;
      
      if (data.all_output_files && Array.isArray(data.all_output_files)) {
        const pageImg = data.all_output_files.find((url: string) => 
          url.includes(`ppt_pages/page_${pageNumStr}.png`)
        );
        if (pageImg) {
          afterImage = pageImg + '?t=' + Date.now();
        }
      }
      
      updatedResults[currentSlideIndex] = { 
        ...updatedResults[currentSlideIndex], 
        afterImage,
        status: 'done',
      };
      setGenerateResults([...updatedResults]);
      setSlidePrompt('');
      
    } catch (err) {
      const message = err instanceof Error ? err.message : '重新生成失败';
      setError(message);
      updatedResults[currentSlideIndex] = { 
        ...updatedResults[currentSlideIndex], 
        status: 'done',
      };
      setGenerateResults([...updatedResults]);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleConfirmSlide = () => {
    setError(null);
    if (currentSlideIndex < outlineData.length - 1) {
      const nextIndex = currentSlideIndex + 1;
      setCurrentSlideIndex(nextIndex);
      setSlidePrompt('');
    } else {
      setCurrentStep('complete');
    }
  };

  // ============== Step 4: 完成处理 ==============
  const handleGenerateFinal = async () => {
    if (!resultPath) {
      setError('缺少 result_path');
      return;
    }
    
    setIsGeneratingFinal(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('img_gen_model_name', genFigModel);
      formData.append('chat_api_url', llmApiUrl.trim());
      formData.append('api_key', apiKey.trim());
      formData.append('model', model);
      formData.append('language', language);
      formData.append('style', globalPrompt || getStyleDescription(stylePreset));
      formData.append('aspect_ratio', '16:9');
      formData.append('invite_code', inviteCode.trim());
      formData.append('result_path', resultPath);
      formData.append('get_down', 'false');
      formData.append('all_edited_down', 'true');
      
      const pagecontent = outlineData.map((slide) => ({
        title: slide.title,
        layout_description: slide.layout_description,
        key_points: slide.key_points,
        asset_ref: slide.asset_ref,
      }));
      formData.append('pagecontent', JSON.stringify(pagecontent));
      
      const res = await fetch('/api/paper2ppt/ppt_json', {
        method: 'POST',
        body: formData,
      });
      
      if (!res.ok) {
        let msg = '生成最终 PPT 失败';
        try {
          const errorData = await res.json();
          msg = errorData.detail || errorData.message || msg;
        } catch {
          // ignore
        }
        throw new Error(msg);
      }
      
      const data = await res.json();
      
      if (!data.success) {
        throw new Error(data.message || '生成失败');
      }
      
      // 优先使用后端直接返回的路径
      if (data.ppt_pptx_path) {
        setDownloadUrl(data.ppt_pptx_path);
      }
      if (data.ppt_pdf_path) {
        setPdfPreviewUrl(data.ppt_pdf_path);
      }
      
      // 备选：从 all_output_files 中查找
      if (data.all_output_files && Array.isArray(data.all_output_files)) {
        if (!data.ppt_pptx_path) {
          const pptxFile = data.all_output_files.find((url: string) => 
            url.endsWith('.pptx') || url.includes('editable.pptx')
          );
          if (pptxFile) {
            setDownloadUrl(pptxFile);
          }
        }
        if (!data.ppt_pdf_path) {
          const pdfFile = data.all_output_files.find((url: string) => 
            url.endsWith('.pdf') && !url.includes('input')
          );
          if (pdfFile) {
            setPdfPreviewUrl(pdfFile);
          }
        }
      }
      
    } catch (err) {
      const message = err instanceof Error ? err.message : '生成失败';
      setError(message);
    } finally {
      setIsGeneratingFinal(false);
    }
  };

  const handleDownload = () => {
    if (!downloadUrl) return;
    const a = document.createElement('a');
    a.href = downloadUrl;
    a.download = 'paper2ppt_result.pptx';
    a.click();
  };

  const handleDownloadPdf = () => {
    if (!pdfPreviewUrl) return;
    const a = document.createElement('a');
    a.href = pdfPreviewUrl;
    a.download = 'paper2ppt_result.pdf';
    a.click();
  };

  // ============== 渲染函数 ==============
  const renderStepIndicator = () => {
    const steps = [
      { key: 'upload', label: '上传论文', num: 1 },
      { key: 'outline', label: '大纲确认', num: 2 },
      { key: 'generate', label: '逐页生成', num: 3 },
      { key: 'complete', label: '完成下载', num: 4 },
    ];
    const currentIndex = steps.findIndex(s => s.key === currentStep);
    return (
      <div className="flex items-center justify-center gap-2 mb-8">
        {steps.map((step, index) => (
          <div key={step.key} className="flex items-center">
            <div className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-all ${
              index === currentIndex 
                ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white shadow-lg' 
                : index < currentIndex 
                  ? 'bg-purple-500/20 text-purple-300 border border-purple-500/40' 
                  : 'bg-white/5 text-gray-500 border border-white/10'
            }`}>
              <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs ${
                index < currentIndex ? 'bg-purple-400 text-white' : ''
              }`}>
                {index < currentIndex ? <Check size={14} /> : step.num}
              </span>
              <span className="hidden sm:inline">{step.label}</span>
            </div>
            {index < steps.length - 1 && (
              <ArrowRight size={16} className={`mx-2 ${index < currentIndex ? 'text-purple-400' : 'text-gray-600'}`} />
            )}
          </div>
        ))}
      </div>
    );
  };

  const renderUploadStep = () => (
    <div className="max-w-6xl mx-auto">
      <div className="mb-10 text-center">
        <p className="text-xs uppercase tracking-[0.2em] text-purple-300 mb-3 font-semibold">PAPER → PPT</p>
        <h1 className="text-4xl md:text-5xl font-bold mb-4">
          <span className="bg-gradient-to-r from-purple-400 via-pink-400 to-rose-400 bg-clip-text text-transparent">
            Paper2PPT
          </span>
        </h1>
        <p className="text-base text-gray-300 max-w-2xl mx-auto leading-relaxed">
          上传论文 PDF，AI 智能分析内容并生成精美幻灯片。<br />
          <span className="text-purple-400">支持逐页编辑、重新生成，打造完美演示文稿！</span>
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 左侧：上传区域 */}
        <div className="glass rounded-xl border border-white/10 p-6">
          <h3 className="text-white font-semibold flex items-center gap-2 mb-4">
            <FileText size={18} className="text-purple-400" /> 上传论文 PDF
          </h3>
          <div 
            className={`border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center text-center gap-4 transition-all ${
              isDragOver ? 'border-purple-500 bg-purple-500/10' : 'border-white/20 hover:border-purple-400'
            }`} 
            onDragOver={e => { e.preventDefault(); setIsDragOver(true); }} 
            onDragLeave={e => { e.preventDefault(); setIsDragOver(false); }} 
            onDrop={handleDrop}
          >
            <div className="w-16 h-16 rounded-full bg-gradient-to-br from-purple-500/20 to-pink-500/20 flex items-center justify-center">
              <UploadCloud size={32} className="text-purple-400" />
            </div>
            <div>
              <p className="text-white font-medium mb-1">拖拽论文 PDF 到此处</p>
              <p className="text-sm text-gray-400">仅支持 PDF 格式</p>
            </div>
            <label className="px-6 py-2.5 rounded-full bg-gradient-to-r from-purple-600 to-pink-600 text-white text-sm font-medium cursor-pointer hover:from-purple-700 hover:to-pink-700 transition-all">
              选择文件
              <input type="file" accept=".pdf" className="hidden" onChange={handleFileChange} />
            </label>
            {selectedFile && (
              <div className="px-4 py-2 bg-purple-500/20 border border-purple-500/40 rounded-lg">
                <p className="text-sm text-purple-300">✓ {selectedFile.name}</p>
                <p className="text-xs text-gray-400 mt-1">✨ 将分析论文内容生成 PPT</p>
              </div>
            )}
          </div>
        </div>

        {/* 右侧：配置区域 */}
        <div className="glass rounded-xl border border-white/10 p-6 space-y-4">
          <h3 className="text-white font-semibold flex items-center gap-2">
            <Settings2 size={18} className="text-purple-400" /> 配置
          </h3>
          
          {/* API 配置 */}
          <div className="grid grid-cols-2 gap-3">
            {/* <div>
              <label className="block text-xs text-gray-400 mb-1 flex items-center gap-1">
                <Key size={12} /> 邀请码 *
              </label>
              <input 
                type="text" 
                value={inviteCode} 
                onChange={e => setInviteCode(e.target.value)}
                placeholder="xxx-xxx"
                className="w-full rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-purple-500"
              />
            </div> */}
            <div>
              <label className="block text-xs text-gray-400 mb-1 flex items-center gap-1">
                <Key size={12} /> API Key *
              </label>
              <input 
                type="password" 
                value={apiKey} 
                onChange={e => setApiKey(e.target.value)}
                placeholder="sk-..."
                className="w-full rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-purple-500"
              />
            </div>
          </div>
          
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-gray-400 mb-1 flex items-center gap-1">
                <Globe size={12} /> API URL
              </label>
              <input 
                type="text" 
                value={llmApiUrl} 
                onChange={e => setLlmApiUrl(e.target.value)}
                className="w-full rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-purple-500"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1 flex items-center gap-1">
                <Cpu size={12} /> 模型
              </label>
              <input 
                type="text" 
                value={model} 
                onChange={e => setModel(e.target.value)}
                className="w-full rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-purple-500"
              />
            </div>
          </div>
          
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-gray-400 mb-1">图像生成模型（中文使用3 pro）</label>
              <select
                value={genFigModel}
                onChange={e => setGenFigModel(e.target.value)}
                className="w-full rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-purple-500"
              >
                <option value="gemini-2.5-flash-image">Gemini 2.5 (Flash Image)</option>
                <option value="gemini-3-pro-image-preview">Gemini 3 Pro (中文推荐)</option>
              </select>
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">生成页数</label>
              <input 
                type="number" 
                value={pageCount} 
                onChange={e => setPageCount(parseInt(e.target.value) || 6)}
                min={1}
                max={20}
                className="w-full rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-purple-500"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-gray-400 mb-1">选择风格</label>
              <select 
                value={stylePreset} 
                onChange={e => setStylePreset(e.target.value as typeof stylePreset)} 
                className="w-full rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-purple-500"
              >
                <option value="modern">现代简约</option>
                <option value="business">商务专业</option>
                <option value="academic">学术报告</option>
                <option value="creative">创意设计</option>
              </select>
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">语言</label>
              <select 
                value={language} 
                onChange={e => setLanguage(e.target.value as 'zh' | 'en')} 
                className="w-full rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-purple-500"
              >
                <option value="zh">中文</option>
                <option value="en">English</option>
              </select>
            </div>
          </div>

          <div>
            <label className="block text-xs text-gray-400 mb-1">风格提示词</label>
            <textarea 
              value={globalPrompt} 
              onChange={e => setGlobalPrompt(e.target.value)} 
              placeholder="例如：使用紫色系配色，保持学术风格 / 多啦A梦风格 / 赛博朋克风格 ...... " 
              rows={2} 
              className="w-full rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-purple-500 resize-none" 
            />
          </div>

          <button 
            onClick={handleUploadAndParse} 
            disabled={!selectedFile || isUploading} 
            className="w-full py-3 rounded-lg bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:from-gray-600 disabled:to-gray-700 text-white font-semibold flex items-center justify-center gap-2 transition-all"
          >
            {isUploading ? (
              <><Loader2 size={18} className="animate-spin" /> 解析中...</>
            ) : (
              <><ArrowRight size={18} /> 开始解析</>
            )}
          </button>
        </div>
      </div>

      {error && (
        <div className="mt-4 flex items-center gap-2 text-sm text-red-300 bg-red-500/10 border border-red-500/40 rounded-lg px-4 py-3">
          <AlertCircle size={16} /> {error}
        </div>
      )}

      {/* 示例区 */}
      <div className="space-y-4 mt-8">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-medium text-gray-200">示例：从 Paper 到 PPTX</h3>
          <span className="text-[11px] text-gray-500">
            下方示例展示从 PDF / 图片 / 文本 到可编辑 PPTX 的效果。
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          <DemoCard
            title="论文 PDF → 学术 PPT"
            desc="上传论文 PDF，自动提取关键信息，生成结构化的学术汇报 PPT。"
            inputImg="/paper2ppt/input_1.png"
            outputImg="/paper2ppt/ouput_1.png"
          />
          <DemoCard
            title="论文内容 → 演示文稿"
            desc="智能分析论文内容，生成排版精美、逻辑清晰的演示文稿。"
            inputImg="/paper2ppt/input_3.png"
            outputImg="/paper2ppt/ouput_3.png"
          />
        </div>
      </div>
    </div>
  );

  const renderOutlineStep = () => (
    <div className="max-w-5xl mx-auto">
      <div className="text-center mb-8">
        <h2 className="text-2xl font-bold text-white mb-2">确认大纲</h2>
        <p className="text-gray-400">检查从论文提取的内容结构，可编辑、排序或删除</p>
      </div>

      <div className="glass rounded-xl border border-white/10 p-6 mb-6">
        <div className="space-y-3">
          {outlineData.map((slide, index) => (
            <div 
              key={slide.id} 
              className={`flex items-start gap-4 p-4 rounded-lg border transition-all ${
                editingId === slide.id 
                  ? 'bg-purple-500/10 border-purple-500/40' 
                  : 'bg-white/5 border-white/10 hover:border-white/20'
              }`}
            >
              <div className="flex items-center gap-2 pt-1">
                <GripVertical size={16} className="text-gray-500" />
                <span className="w-8 h-8 rounded-full bg-purple-500/20 text-purple-300 text-sm font-medium flex items-center justify-center">
                  {slide.pageNum}
                </span>
              </div>
              
              <div className="flex-1">
                {editingId === slide.id ? (
                  <div className="space-y-3">
                    <input type="text" value={editContent.title} onChange={e => setEditContent(p => ({ ...p, title: e.target.value }))} className="w-full px-3 py-2 rounded-lg bg-black/40 border border-white/20 text-white text-sm outline-none focus:ring-2 focus:ring-purple-500" placeholder="标题" />
                    <textarea value={editContent.layout_description} onChange={e => setEditContent(p => ({ ...p, layout_description: e.target.value }))} rows={2} className="w-full px-3 py-2 rounded-lg bg-black/40 border border-white/20 text-white text-sm outline-none focus:ring-2 focus:ring-purple-500 resize-none" placeholder="布局描述" />
                    <div className="space-y-2">
                      {editContent.key_points.map((p, i) => (
                        <div key={i} className="flex gap-2">
                          <input type="text" value={p} onChange={e => handleKeyPointChange(i, e.target.value)} className="flex-1 px-3 py-2 rounded-lg bg-black/40 border border-white/20 text-white text-sm" placeholder={`要点 ${i + 1}`} />
                          <button onClick={() => handleRemoveKeyPoint(i)} className="p-2 text-gray-400 hover:text-red-400"><Trash2 size={14} /></button>
                        </div>
                      ))}
                      <button onClick={handleAddKeyPoint} className="px-3 py-1.5 rounded-lg bg-white/5 border border-dashed border-white/20 text-gray-400 text-sm w-full hover:text-purple-400 hover:border-purple-400">+ 添加要点</button>
                    </div>
                    <div className="flex gap-2 pt-2">
                      <button onClick={handleEditSave} className="px-3 py-1.5 rounded-lg bg-purple-500 text-white text-sm flex items-center gap-1"><Check size={14} /> 保存</button>
                      <button onClick={handleEditCancel} className="px-3 py-1.5 rounded-lg bg-white/10 text-gray-300 text-sm">取消</button>
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="mb-2"><h4 className="text-white font-medium">{slide.title}</h4></div>
                    <p className="text-xs text-purple-400/70 mb-2 italic">📐 {slide.layout_description}</p>
                    <ul className="space-y-1">
                      {slide.key_points.map((p, i) => (
                        <li key={i} className="text-sm text-gray-400 flex items-start gap-2">
                          <span className="text-purple-400 mt-0.5">•</span><span>{p}</span>
                        </li>
                      ))}
                    </ul>
                  </>
                )}
              </div>

              {editingId !== slide.id && (
                <div className="flex items-center gap-1">
                  <button onClick={() => handleMoveSlide(index, 'up')} disabled={index === 0} className="p-2 text-gray-400 hover:text-white disabled:opacity-30"><ChevronUp size={16} /></button>
                  <button onClick={() => handleMoveSlide(index, 'down')} disabled={index === outlineData.length - 1} className="p-2 text-gray-400 hover:text-white disabled:opacity-30"><ChevronDown size={16} /></button>
                  <button onClick={() => handleEditStart(slide)} className="p-2 text-gray-400 hover:text-purple-400"><Edit3 size={16} /></button>
                  <button onClick={() => handleDeleteSlide(slide.id)} className="p-2 text-gray-400 hover:text-red-400"><Trash2 size={16} /></button>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      <div className="flex justify-between">
        <button onClick={() => setCurrentStep('upload')} className="px-6 py-2.5 rounded-lg border border-white/20 text-gray-300 hover:bg-white/10 flex items-center gap-2">
          <ArrowLeft size={18} /> 返回上传
        </button>
        <button onClick={handleConfirmOutline} className="px-6 py-2.5 rounded-lg bg-gradient-to-r from-purple-600 to-pink-600 text-white font-semibold flex items-center gap-2 transition-all">
          确认并开始生成 <ArrowRight size={18} />
        </button>
      </div>

      {error && (
        <div className="mt-4 flex items-center gap-2 text-sm text-red-300 bg-red-500/10 border border-red-500/40 rounded-lg px-4 py-3">
          <AlertCircle size={16} /> {error}
        </div>
      )}
    </div>
  );

  const renderGenerateStep = () => {
    const currentSlide = outlineData[currentSlideIndex];
    const currentResult = generateResults[currentSlideIndex];
    
    return (
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-6">
          <h2 className="text-2xl font-bold text-white mb-2">逐页生成</h2>
          <p className="text-gray-400">第 {currentSlideIndex + 1} / {outlineData.length} 页：{currentSlide?.title}</p>
        </div>

        <div className="mb-6">
          <div className="flex gap-1">
            {generateResults.map((result, index) => (
              <div key={result.slideId} className={`flex-1 h-2 rounded-full transition-all ${
                result.status === 'done' ? 'bg-purple-400' : result.status === 'processing' ? 'bg-gradient-to-r from-purple-400 to-pink-400 animate-pulse' : index === currentSlideIndex ? 'bg-purple-400/50' : 'bg-white/10'
              }`} />
            ))}
          </div>
        </div>

        {currentSlide && (
          <div className="glass rounded-xl border border-white/10 p-4 mb-4">
            <div className="mb-3">
              <h4 className="text-sm text-gray-400 mb-2 flex items-center gap-2"><FileText size={14} className="text-purple-400" /> 布局描述</h4>
              <p className="text-xs text-purple-400/80 italic">{currentSlide.layout_description}</p>
            </div>
            <div className="pt-3 border-t border-white/10">
              <h4 className="text-sm text-gray-400 mb-2">要点内容</h4>
              <ul className="grid grid-cols-1 md:grid-cols-2 gap-1">
                {currentSlide.key_points.slice(0, 4).map((point, idx) => (
                  <li key={idx} className="text-xs text-gray-400 flex items-start gap-1"><span className="text-purple-400">•</span><span className="line-clamp-1">{point}</span></li>
                ))}
                {currentSlide.key_points.length > 4 && (<li className="text-xs text-gray-500 italic">...还有 {currentSlide.key_points.length - 4} 条</li>)}
              </ul>
            </div>
          </div>
        )}

        <div className="glass rounded-xl border border-white/10 p-6 mb-6">
          <div className="max-w-3xl mx-auto">
            <h4 className="text-sm text-gray-400 mb-3 flex items-center justify-center gap-2"><Sparkles size={14} className="text-purple-400" /> AI 生成结果</h4>
            <div className="rounded-lg overflow-hidden border border-purple-500/30 aspect-[16/9] bg-gradient-to-br from-purple-500/10 to-pink-500/10 flex items-center justify-center">
              {isGenerating ? (
                <div className="text-center">
                  <Loader2 size={40} className="text-purple-400 animate-spin mx-auto mb-3" />
                  <p className="text-base text-purple-300">{generateResults.every(r => r.status === 'processing') ? '正在批量生成所有页面...' : '正在重新生成当前页...'}</p>
                  <p className="text-xs text-gray-500 mt-1">{generateResults.every(r => r.status === 'processing') ? `共 ${outlineData.length} 页，请稍候` : 'AI 正在根据您的提示重新创建'}</p>
                </div>
              ) : currentResult?.afterImage ? (
                <img src={currentResult.afterImage} alt="Generated" className="w-full h-full object-contain" />
              ) : (
                <div className="text-center"><FileText size={32} className="text-gray-500 mx-auto mb-2" /><span className="text-gray-500">等待生成</span></div>
              )}
            </div>
          </div>
        </div>

        <div className="glass rounded-xl border border-white/10 p-4 mb-6">
          <div className="flex items-center gap-3">
            <MessageSquare size={18} className="text-purple-400" />
            <input type="text" value={slidePrompt} onChange={e => setSlidePrompt(e.target.value)} placeholder="输入微调 Prompt，然后点击重新生成..." className="flex-1 bg-transparent outline-none text-white text-sm placeholder:text-gray-500" />
            <button onClick={handleRegenerateSlide} disabled={isGenerating || !slidePrompt.trim()} className="px-4 py-2 rounded-lg bg-white/10 hover:bg-white/20 text-gray-300 text-sm flex items-center gap-2 disabled:opacity-50">
              <RefreshCw size={14} /> 重新生成
            </button>
          </div>
        </div>

        <div className="flex justify-between">
          <button onClick={() => setCurrentStep('outline')} className="px-6 py-2.5 rounded-lg border border-white/20 text-gray-300 hover:bg-white/10 flex items-center gap-2">
            <ArrowLeft size={18} /> 返回大纲
          </button>
          <button onClick={handleConfirmSlide} disabled={isGenerating || currentResult?.status !== 'done'} className="px-6 py-2.5 rounded-lg bg-gradient-to-r from-purple-600 to-pink-600 text-white font-semibold flex items-center gap-2 disabled:opacity-50">
            <CheckCircle2 size={18} /> {currentSlideIndex < outlineData.length - 1 ? '确认并继续' : '完成生成'}
          </button>
        </div>

        {error && (
          <div className="mt-4 flex items-center gap-2 text-sm text-red-300 bg-red-500/10 border border-red-500/40 rounded-lg px-4 py-3">
            <AlertCircle size={16} /> {error}
          </div>
        )}
      </div>
    );
  };

  const renderCompleteStep = () => {
    const doneCount = generateResults.filter(r => r.status === 'done').length;
    
    return (
      <div className="max-w-2xl mx-auto text-center">
        <div className="mb-8">
          <div className="w-20 h-20 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center mx-auto mb-4">
            <CheckCircle2 size={40} className="text-white" />
          </div>
          <h2 className="text-2xl font-bold text-white mb-2">生成完成！</h2>
          <p className="text-gray-400">共处理 {outlineData.length} 页，成功生成 {doneCount} 页</p>
        </div>

        <div className="glass rounded-xl border border-white/10 p-6 mb-6">
          <h3 className="text-white font-semibold mb-4">生成结果预览</h3>
          <div className="grid grid-cols-4 gap-2">
            {generateResults.map((result, index) => (
              <div key={result.slideId} className="aspect-[16/9] rounded-lg border border-white/20 overflow-hidden bg-white/5">
                {result.afterImage ? (
                  <img src={result.afterImage} alt={`Page ${index + 1}`} className="w-full h-full object-contain" />
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-gray-500 text-xs">第 {index + 1} 页</div>
                )}
              </div>
            ))}
          </div>
        </div>

        {!(downloadUrl || pdfPreviewUrl) ? (
          <button onClick={handleGenerateFinal} disabled={isGeneratingFinal} className="px-8 py-3 rounded-lg bg-gradient-to-r from-purple-600 to-pink-600 text-white font-semibold flex items-center justify-center gap-2 mx-auto transition-all">
            {isGeneratingFinal ? (<><Loader2 size={18} className="animate-spin" /> 正在生成最终文件...</>) : (<><Sparkles size={18} /> 生成最终文件</>)}
          </button>
        ) : (
          <div className="space-y-4">
            <div className="flex gap-4 justify-center">
              {downloadUrl && (
                <button onClick={handleDownload} className="px-6 py-3 rounded-lg bg-gradient-to-r from-emerald-500 to-teal-500 text-white font-semibold flex items-center gap-2 transition-all">
                  <Download size={18} /> 下载 PPT
                </button>
              )}
              {pdfPreviewUrl && (
                <button onClick={handleDownloadPdf} className="px-6 py-3 rounded-lg bg-gradient-to-r from-purple-500 to-pink-500 text-white font-semibold flex items-center gap-2 transition-all">
                  <Download size={18} /> 下载 PDF
                </button>
              )}
            </div>
            <div>
              <button onClick={() => { setCurrentStep('upload'); setSelectedFile(null); setOutlineData([]); setGenerateResults([]); setDownloadUrl(null); setPdfPreviewUrl(null); setResultPath(null); setError(null); }} className="text-sm text-gray-400 hover:text-white transition-colors">
                <RotateCcw size={14} className="inline mr-1" /> 处理新的论文
              </button>
            </div>
          </div>
        )}

        {error && (
          <div className="mt-4 flex items-center gap-2 text-sm text-red-300 bg-red-500/10 border border-red-500/40 rounded-lg px-4 py-3 justify-center">
            <AlertCircle size={16} /> {error}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="w-full h-screen flex flex-col bg-[#050512] overflow-hidden">
      {showBanner && (
        <div className="w-full bg-gradient-to-r from-purple-600 via-pink-600 to-orange-500 relative overflow-hidden flex-shrink-0">
          <div className="absolute inset-0 bg-black opacity-20"></div>
          <div className="absolute inset-0 animate-pulse">
            <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-r from-transparent via-white to-transparent opacity-10 animate-shimmer"></div>
          </div>
          
          <div className="relative max-w-7xl mx-auto px-4 py-3 flex flex-col sm:flex-row items-center justify-between gap-3">
            <div className="flex items-center gap-3 flex-wrap justify-center sm:justify-start">
              <div className="flex items-center gap-2 bg-white/20 backdrop-blur-sm rounded-full px-3 py-1">
                <Star size={16} className="text-yellow-300 fill-yellow-300 animate-pulse" />
                <span className="text-xs font-bold text-white">开源项目</span>
              </div>
              
              <span className="text-sm font-medium text-white">
                🚀 探索更多 AI 数据处理工具
              </span>
            </div>

            <div className="flex items-center gap-2 flex-wrap justify-center">
              <a
                href="https://github.com/OpenDCAI/DataFlow"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-1.5 bg-white/95 hover:bg-white text-gray-900 rounded-full text-xs font-semibold transition-all hover:scale-105 shadow-lg"
              >
                <Github size={14} />
                <span>DataFlow</span>
                <span className="bg-purple-600 text-white px-2 py-0.5 rounded-full text-[10px]">HOT</span>
              </a>

              <a
                href="https://github.com/OpenDCAI/DataFlow-Agent"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-1.5 bg-white/95 hover:bg-white text-gray-900 rounded-full text-xs font-semibold transition-all hover:scale-105 shadow-lg"
              >
                <Github size={14} />
                <span>DataFlow-Agent</span>
                <span className="bg-pink-600 text-white px-2 py-0.5 rounded-full text-[10px]">NEW</span>
              </a>

              <button
                onClick={() => setShowBanner(false)}
                className="p-1 hover:bg-white/20 rounded-full transition-colors"
                aria-label="关闭"
              >
                <X size={16} className="text-white" />
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="flex-1 overflow-auto">
        <div className="max-w-7xl mx-auto px-6 py-8 pb-24">
          {renderStepIndicator()}
          {currentStep === 'upload' && renderUploadStep()}
          {currentStep === 'outline' && renderOutlineStep()}
          {currentStep === 'generate' && renderGenerateStep()}
          {currentStep === 'complete' && renderCompleteStep()}
        </div>
      </div>

      <style>{`
        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
        .animate-shimmer {
          animation: shimmer 3s infinite;
        }
        .glass { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(10px); }
        .demo-input-placeholder {
          min-height: 80px;
        }
        .demo-output-placeholder {
          min-height: 80px;
        }
      `}</style>
    </div>
  );
};

interface DemoCardProps {
  title: string;
  desc: string;
  inputImg?: string;
  outputImg?: string;
}

const DemoCard = ({ title, desc, inputImg, outputImg }: DemoCardProps) => {
  return (
    <div className="glass rounded-lg border border-white/10 p-3 flex flex-col gap-2 hover:bg-white/5 transition-colors">
      <div className="flex gap-2">
        {/* 左侧：输入示例图片 */}
        <div className="flex-1 rounded-md bg-white/5 border border-dashed border-white/10 flex items-center justify-center demo-input-placeholder overflow-hidden">
          {inputImg ? (
            <img
              src={inputImg}
              alt="输入示例图"
              className="w-full h-full object-cover"
            />
          ) : (
            <span className="text-[10px] text-gray-400">输入示例图（待替换）</span>
          )}
        </div>
        {/* 右侧：输出 PPTX 示例图片 */}
        <div className="flex-1 rounded-md bg-primary-500/10 border border-dashed border-primary-300/40 flex items-center justify-center demo-output-placeholder overflow-hidden">
          {outputImg ? (
            <img
              src={outputImg}
              alt="PPTX 示例图"
              className="w-full h-full object-cover"
            />
          ) : (
            <span className="text-[10px] text-primary-200">PPTX 示例图（待替换）</span>
          )}
        </div>
      </div>
      <div>
        <p className="text-[13px] text-white font-medium mb-1">{title}</p>
        <p className="text-[11px] text-gray-400 leading-snug">{desc}</p>
      </div>
    </div>
  );
};

export default Paper2PptPage;
