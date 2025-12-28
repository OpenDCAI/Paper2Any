import { useState, useEffect, ChangeEvent } from 'react';
import {
  Presentation, UploadCloud, Settings2, Download, Loader2, CheckCircle2,
  AlertCircle, ChevronDown, ChevronUp, Github, Star, X, Sparkles,
  ArrowRight, ArrowLeft, GripVertical, Trash2, Edit3, Check, RotateCcw,
  MessageSquare, Eye, RefreshCw, FileText, Image as ImageIcon, Copy
} from 'lucide-react';
import { uploadAndSaveFile } from '../services/fileService';
import { API_KEY } from '../config/api';
import { checkQuota, recordUsage } from '../services/quotaService';
import { useAuthStore } from '../stores/authStore';

// ============== 类型定义 ==============
type Step = 'upload' | 'beautify' | 'complete';

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
  status: 'pending' | 'processing' | 'done';
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
  const { user, refreshQuota } = useAuthStore();
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
  const [progress, setProgress] = useState(0);
  const [progressStatus, setProgressStatus] = useState('');
  
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
  const [isGeneratingInitial, setIsGeneratingInitial] = useState(false);
  const [slidePrompt, setSlidePrompt] = useState('');
  
  // Step 4: 完成状态
  const [isGeneratingFinal, setIsGeneratingFinal] = useState(false);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [pdfDownloadUrl, setPdfDownloadUrl] = useState<string | null>(null);
  
  // 通用状态
  const [error, setError] = useState<string | null>(null);
  const [showBanner, setShowBanner] = useState(true);

  // API 配置状态
  const [inviteCode, setInviteCode] = useState('');
  const [llmApiUrl, setLlmApiUrl] = useState('https://api.apiyi.com/v1');
  const [apiKey, setApiKey] = useState('');
  const [model, setModel] = useState('gpt-5.1');
  const [genFigModel, setGenFigModel] = useState('gemini-2.5-flash-image');
  const [language, setLanguage] = useState<'zh' | 'en'>('en');
  const [resultPath, setResultPath] = useState<string | null>(null);

  // GitHub Stars
  const [stars, setStars] = useState<{dataflow: number | null, agent: number | null}>({ dataflow: null, agent: null });
  const [copySuccess, setCopySuccess] = useState('');

  const shareText = `发现一个超好用的AI工具 DataFlow-Agent！🚀
支持论文转PPT、PDF转PPT、PPT美化等功能，科研打工人的福音！

🔗 在线体验：https://dcai-paper2any.nas.cpolar.cn/
⭐ GitHub Agent：https://github.com/OpenDCAI/DataFlow-Agent
🌟 GitHub Core：https://github.com/OpenDCAI/DataFlow

转发本文案+截图，联系微信群管理员即可获取免费Key！🎁
#AI工具 #PPT制作 #科研效率 #开源项目`;

  const handleCopyShareText = async () => {
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(shareText);
      } else {
        const textArea = document.createElement("textarea");
        textArea.value = shareText;
        textArea.style.position = "fixed";
        textArea.style.left = "-9999px";
        textArea.style.top = "0";
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        try {
          document.execCommand('copy');
        } catch (err) {
          console.error('Fallback: Oops, unable to copy', err);
          throw err;
        } finally {
          document.body.removeChild(textArea);
        }
      }
      setCopySuccess('文案已复制！快去分享吧');
      setTimeout(() => setCopySuccess(''), 2000);
    } catch (err) {
      console.error('复制失败', err);
      setCopySuccess('复制失败，请手动复制');
    }
  };

  useEffect(() => {
    const fetchStars = async () => {
      try {
        const [res1, res2] = await Promise.all([
          fetch('https://api.github.com/repos/OpenDCAI/DataFlow'),
          fetch('https://api.github.com/repos/OpenDCAI/DataFlow-Agent')
        ]);
        const data1 = await res1.json();
        const data2 = await res2.json();
        setStars({
          dataflow: data1.stargazers_count,
          agent: data2.stargazers_count
        });
      } catch (e) {
        console.error('Failed to fetch stars', e);
      }
    };
    fetchStars();
  }, []);

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
    
    // if (!inviteCode.trim()) {
    //   setError('请先输入邀请码');
    //   return;
    // }
    
    if (!llmApiUrl.trim() || !apiKey.trim()) {
      setError('请先配置模型 API URL 和 API Key');
      return;
    }

    if (!globalPrompt.trim()) {
      setError('请输入风格提示词');
      return;
    }

    // Check quota before proceeding
    const quota = await checkQuota(user?.id || null, user?.is_anonymous || false);
    if (quota.remaining <= 0) {
      setError(quota.isAuthenticated
        ? '今日配额已用完（10次/天），请明天再试'
        : '今日配额已用完（5次/天），登录后可获得更多配额');
      return;
    }

    setIsUploading(true);
    setError(null);
    setProgress(0);
    setProgressStatus('正在初始化...');

    // 模拟进度
    const progressInterval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 90) return 90;
        const messages = [
           '正在上传 PPT...',
           '正在分析页面结构...',
           '正在提取内容...',
           '正在识别图片...',
           '正在生成美化方案...'
        ];
        const msgIndex = Math.floor(prev / 20);
        if (msgIndex < messages.length) {
          setProgressStatus(messages[msgIndex]);
        }
        // 调整进度速度，使其在 3 分钟左右达到 90%
        return prev + (Math.random() * 0.6 + 0.2);
      });
    }, 1000);
    
    try {
      // 调用 /paper2ppt/pagecontent_json 接口
      const formData = new FormData();
      formData.append('chat_api_url', llmApiUrl.trim());
      formData.append('api_key', apiKey.trim());
      formData.append('model', model);
      formData.append('language', language);
      formData.append('style', globalPrompt || stylePreset);
      formData.append('gen_fig_model', genFigModel);
      formData.append('page_count', '10'); // 默认值，后端可能会调整
      formData.append('invite_code', inviteCode.trim());
      formData.append('input_type', 'pptx');
      formData.append('file', selectedFile);
      
      if (referenceImage) {
        formData.append('reference_img', referenceImage);
      }
      
      console.log('Sending request to /api/paper2ppt/pagecontent_json'); // 调试信息

      const res = await fetch('/api/paper2ppt/pagecontent_json', {
        method: 'POST',
        headers: { 'X-API-Key': API_KEY },
        body: formData,
      });

      console.log('Response status:', res.status, res.statusText); // 调试信息
      
      if (!res.ok) {
        let msg = '解析 PPT 失败';
        if (res.status === 403) {
          msg = '邀请码不正确或已失效';
        } else {
          try {
            const errorData = await res.json();
            console.error('Error response:', errorData); // 调试信息
            msg = errorData.detail || errorData.message || msg;
          } catch {
            try {
              const text = await res.text();
              console.error('Error text:', text); // 调试信息
              if (text) msg = text;
            } catch {
              // ignore
            }
          }
        }
        throw new Error(msg);
      }
      
      const data = await res.json();
      
      console.log('API Response:', JSON.stringify(data, null, 2)); // 调试信息
      
      if (!data.success) {
        throw new Error(data.message || '解析失败');
      }
      
      // 保存 result_path
      const currentResultPath = data.result_path || '';
      if (currentResultPath) {
        setResultPath(currentResultPath);
      } else {
        throw new Error('后端未返回 result_path');
      }
      
      // 检查 pagecontent 是否为空
      if (!data.pagecontent || data.pagecontent.length === 0) {
        throw new Error('解析结果为空，请检查PPT文件是否正确');
      }
      
      // 转换后端数据为前端格式
      // 对于 pptx 类型，pagecontent 可能只包含 ppt_img_path
      // 对于 pdf/text 类型，pagecontent 包含 title, layout_description, key_points
      const convertedSlides: SlideOutline[] = data.pagecontent.map((item: any, index: number) => {
        // 如果只有 ppt_img_path（pptx 类型），需要从图片URL中提取或使用默认值
        if (item.ppt_img_path && !item.title) {
          // 从 all_output_files 中找到对应的图片URL
          const imgUrl = data.all_output_files?.find((url: string) => 
            url.includes(`slide_${String(index).padStart(3, '0')}.png`) ||
            url.includes(item.ppt_img_path.split('/').pop() || '')
          );
          
          return {
            id: String(index + 1),
            pageNum: index + 1,
            title: `第 ${index + 1} 页`,
            layout_description: '待编辑：请填写此页的布局描述',
            key_points: ['待编辑：请添加要点'],
            asset_ref: imgUrl || item.ppt_img_path || null,
          };
        }
        
        // 标准格式（pdf/text 类型）
        return {
          id: String(index + 1),
          pageNum: index + 1,
          title: item.title || `第 ${index + 1} 页`,
          layout_description: item.layout_description || '',
          key_points: item.key_points || [],
          asset_ref: item.asset_ref || item.ppt_img_path || null,
        };
      });
      
      console.log('Converted Slides:', convertedSlides); // 调试信息
      
      if (convertedSlides.length === 0) {
        throw new Error('转换后的数据为空');
      }
      
      setOutlineData(convertedSlides);
      
      // 初始化美化结果 - 使用原始图片作为 beforeImage
      const results: BeautifyResult[] = convertedSlides.map((slide, index) => ({
        slideId: slide.id,
        beforeImage: slide.asset_ref || '',
        afterImage: '',
        status: 'pending',
      }));
      setBeautifyResults(results);
      setCurrentSlideIndex(0);
      
      // 不再一次性美化所有页面！
      // 直接进入美化步骤，显示原始图片
      // 用户点击"开始美化"时才调用 API 美化当前页
      
      console.log('解析完成，进入美化步骤, results.length:', results.length, 'currentResultPath:', currentResultPath);
      
      clearInterval(progressInterval);
      setProgress(100);
      setProgressStatus('解析完成！');

      // 稍微延迟一下跳转
      setTimeout(() => {
        // 直接进入美化步骤
        setCurrentStep('beautify');
        
        // 触发批量生成 (Cycle Batch Beautify)
        if (results.length > 0) {
          setIsGeneratingInitial(true);
          console.log('开始批量美化所有页面...');
          
          // 异步执行批量生成，不阻塞 UI 渲染（UI 会显示 Loading）
          // 注意：generateInitialPPT 内部会处理错误提示
          generateInitialPPT(convertedSlides, results, currentResultPath)
            .then((updatedResults) => {
              console.log('批量美化完成');
              const finalResults = updatedResults.map(res => ({
                ...res,
                status: 'done' as const
              }));
              setBeautifyResults(finalResults);
            })
            .catch((err) => {
              console.error("Batch generation failed:", err);
            })
            .finally(() => {
              setIsGeneratingInitial(false);
            });
        }
      }, 500);
    } catch (err) {
      clearInterval(progressInterval);
      setProgress(0);
      const message = err instanceof Error ? err.message : '解析失败，请重试';
      setError(message);
      console.error(err);
    } finally {
      if (currentStep !== 'beautify') {
        setIsUploading(false);
      } else {
        // 如果成功跳转，在组件卸载或状态切换前保持 loading 状态防止闪烁
        setIsUploading(false);
      }
    }
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

  const handleConfirmOutline = async () => {
    // 初始化结果状态，使用 Slide 数据中的 asset_ref 作为 beforeImage
    const results: BeautifyResult[] = outlineData.map((slide) => ({
      slideId: slide.id,
      beforeImage: slide.asset_ref || '',  // 确保使用真实的图片路径
      afterImage: '', // 初始为空，等待批量生成
      status: 'pending',
    }));
    setBeautifyResults(results);
    setCurrentSlideIndex(0);
    setCurrentStep('beautify');
    
    // 触发批量生成
    setIsGeneratingInitial(true);
    try {
      // 传入 outlineData，因为 generateInitialPPT 内部需要用它来构建 pagecontent
      const updatedResults = await generateInitialPPT(outlineData, results);
      
      // 更新结果状态，将状态标记为 done
      const finalResults = updatedResults.map(res => ({
        ...res,
        status: 'done' as const // 显式类型断言
      }));
      setBeautifyResults(finalResults);
    } catch (error) {
      console.error("Batch generation failed:", error);
      // 错误已在 generateInitialPPT 中通过 setError 处理，这里只需确保 loading 状态结束
    } finally {
      setIsGeneratingInitial(false);
    }
  };

  // ============== 生成初始 PPT ==============
  const generateInitialPPT = async (slides: SlideOutline[], initialResults: BeautifyResult[], resultPathParam?: string) => {
    // 优先使用传入的参数，其次使用 state
    const currentPath = resultPathParam || resultPath;
    console.log('generateInitialPPT - currentPath:', currentPath);
    
    if (!currentPath) {
      setError('缺少 result_path，请重新上传文件');
      return initialResults; // 返回原始结果，避免 undefined
    }
    
    try {
      // 根据文档 2.2，对于 pptx 类型，需要先传入图片路径格式的 pagecontent
      // 从 all_output_files 中找到对应的图片 URL（后端会自动处理为本地路径）
      const pagecontent = slides.map((slide, index) => {
        const path = slide.asset_ref || '';
        return { ppt_img_path: path };
      }).filter(item => item.ppt_img_path);
      
      const formData = new FormData();
      formData.append('img_gen_model_name', genFigModel);
      formData.append('chat_api_url', llmApiUrl.trim());
      formData.append('api_key', apiKey.trim());
      formData.append('model', model);
      formData.append('language', language);
      formData.append('style', globalPrompt || stylePreset);
      formData.append('aspect_ratio', '16:9');
      formData.append('invite_code', inviteCode.trim());
      formData.append('result_path', currentPath);
      formData.append('get_down', 'false');
      formData.append('pagecontent', JSON.stringify(pagecontent));
      
      console.log('Generating initial PPT with pagecontent:', pagecontent);
      console.log('Request URL: /api/paper2ppt/ppt_json');
      console.log('Request params:', {
        result_path: resultPath,
        get_down: 'false',
        pagecontent_count: pagecontent.length,
      });

      const res = await fetch('/api/paper2ppt/ppt_json', {
        method: 'POST',
        headers: { 'X-API-Key': API_KEY },
        body: formData,
      });

      console.log('Response status:', res.status, res.statusText);
      
      if (!res.ok) {
        let msg = '生成初始 PPT 失败';
        try {
          const errorData = await res.json();
          console.error('Error response:', errorData);
          msg = errorData.detail || errorData.message || msg;
        } catch {
          const text = await res.text();
          console.error('Error text:', text);
          if (text) msg = text;
        }
        throw new Error(msg);
      }
      
      const data = await res.json();
      console.log('Initial PPT generation response:', JSON.stringify(data, null, 2));
      
      if (!data.success) {
        throw new Error(data.message || '生成失败');
      }
      
      // 更新美化结果，使用生成的 ppt_pages/page_*.png 作为 afterImage
      let updatedResults = initialResults;
      if (data.all_output_files) {
        updatedResults = initialResults.map((result, index) => {
          const pageImageUrl = data.all_output_files.find((url: string) => 
            url.includes(`page_${String(index).padStart(3, '0')}.png`)
          );
          return {
            ...result,
            // beforeImage 保持原始 PPT 截图
            afterImage: pageImageUrl || '',
          };
        });
        setBeautifyResults(updatedResults);
        
        // 同时更新 outlineData 的 asset_ref 为生成后的图片路径
        // 这样后续"重新生成"时才能正确传递路径给后端
        setOutlineData(prev => prev.map((slide, index) => {
          const pageImageUrl = data.all_output_files.find((url: string) => 
            url.includes(`page_${String(index).padStart(3, '0')}.png`)
          );
          return {
            ...slide,
            asset_ref: pageImageUrl || slide.asset_ref,
          };
        }));
        
        // 预加载所有图片到浏览器缓存，避免切换页面时延迟
        console.log('预加载所有生成的图片...');
        data.all_output_files.forEach((url: string) => {
          if (url.endsWith('.png') || url.endsWith('.jpg') || url.endsWith('.jpeg')) {
            const img = new Image();
            img.src = url;
          }
        });
      }
      
      // 返回更新后的结果，供调用方使用
      return updatedResults;
    } catch (err) {
      const message = err instanceof Error ? err.message : '生成初始 PPT 失败';
      setError(message);
      console.error(err);
      throw err; // 重新抛出错误
    }
  };

  // ============== Step 3: 逐页美化处理 ==============
  const startBeautifyCurrentSlide = async (
    results: BeautifyResult[] | null, 
    index: number, 
    resultPathParam?: string,
    outlineDataParam?: SlideOutline[]
  ) => {
    // 优先使用传入的参数，其次使用 state
    const currentPath = resultPathParam || resultPath;
    const currentOutlineData = outlineDataParam || outlineData;
    
    console.log('startBeautifyCurrentSlide 被调用, index:', index, 'results:', results?.length || 'null');
    console.log('currentPath:', currentPath);
    console.log('currentOutlineData.length:', currentOutlineData.length);
    console.log('slidePrompt:', slidePrompt);
    
    if (!currentPath) {
      setError('缺少 result_path，请重新上传文件');
      console.error('currentPath 为空');
      return;
    }
    
    // 如果 results 为 null，从 state 中读取
    const currentResults = results || beautifyResults;
    console.log('currentResults.length:', currentResults.length);
    
    if (currentResults.length === 0) {
      setError('没有可美化的页面');
      console.error('currentResults 为空');
      return;
    }
    
    if (currentOutlineData.length === 0) {
      setError('没有 outline 数据');
      console.error('currentOutlineData 为空');
      return;
    }
    
    setIsBeautifying(true);
    const updatedResults = [...currentResults];
    updatedResults[index] = { ...updatedResults[index], status: 'processing' };
    setBeautifyResults(updatedResults);
    
    try {
      // 调用 /paper2ppt/ppt_json 接口进行编辑
      const formData = new FormData();
      formData.append('img_gen_model_name', genFigModel);
      formData.append('chat_api_url', llmApiUrl.trim());
      formData.append('api_key', apiKey.trim());
      formData.append('model', model);
      formData.append('language', language);
      formData.append('style', globalPrompt || stylePreset);
      formData.append('aspect_ratio', '16:9');
      formData.append('invite_code', inviteCode.trim());
      formData.append('result_path', currentPath);
      formData.append('get_down', 'true');
      formData.append('page_id', String(index));
      formData.append('edit_prompt', slidePrompt || '请美化这一页的样式');
      
      // 编辑模式下，必须传递 pagecontent，包含原图路径
      console.log('使用的 outlineData:', currentOutlineData);
      const pagecontent = currentOutlineData.map((slide, i) => {
        // 直接传递 asset_ref（URL），后端会自动转换为本地路径
        const path = slide.asset_ref || '';
        console.log(`slide ${i} asset_ref:`, path);
        return { ppt_img_path: path };
      });
      console.log('pagecontent to send:', pagecontent);
      formData.append('pagecontent', JSON.stringify(pagecontent));

      const res = await fetch('/api/paper2ppt/ppt_json', {
        method: 'POST',
        headers: { 'X-API-Key': API_KEY },
        body: formData,
      });
      
      if (!res.ok) {
        let msg = '美化失败';
        try {
          const errorData = await res.json();
          msg = errorData.detail || errorData.message || msg;
        } catch {
          // ignore
        }
        throw new Error(msg);
      }
      
      const data = await res.json();
      console.log('美化响应:', JSON.stringify(data, null, 2));
      console.log('all_output_files:', data.all_output_files);
      
      if (!data.success) {
        throw new Error(data.message || '美化失败');
      }
      
      // 从 all_output_files 中找到对应的页面图片
      // 优先匹配美化后的图 (ppt_pages/page_xxx.png)，其次才是原图 (ppt_images/slide_xxx.png)
      const pagePattern = `ppt_pages/page_${String(index).padStart(3, '0')}.png`;
      const slidePattern = `ppt_images/slide_${String(index).padStart(3, '0')}.png`;
      console.log('查找美化后图片模式:', pagePattern);
      console.log('查找原图模式:', slidePattern);
      
      // 先找美化后的图
      let pageImageUrl = data.all_output_files?.find((url: string) => url.includes(pagePattern));
      console.log('美化后图片 URL:', pageImageUrl);
      
      // 如果没有美化后的图，再找原图作为 fallback
      if (!pageImageUrl) {
        pageImageUrl = data.all_output_files?.find((url: string) => url.includes(slidePattern));
        console.log('Fallback 到原图 URL:', pageImageUrl);
      }

      // 添加时间戳防止缓存
      if (pageImageUrl) {
        pageImageUrl = `${pageImageUrl}?t=${new Date().getTime()}`;
      }
      
      console.log('最终使用的图片 URL:', pageImageUrl);
      
      updatedResults[index] = { 
        ...updatedResults[index], 
        status: 'done',
        afterImage: pageImageUrl || updatedResults[index].afterImage,
        userPrompt: slidePrompt || undefined,
      };
    setBeautifyResults(updatedResults);
    } catch (err) {
      const message = err instanceof Error ? err.message : '美化失败';
      setError(message);
      updatedResults[index] = { ...updatedResults[index], status: 'pending' };
    setBeautifyResults(updatedResults);
    } finally {
    setIsBeautifying(false);
    }
  };

  const handleConfirmSlide = () => {
    if (currentSlideIndex < outlineData.length - 1) {
      const nextIndex = currentSlideIndex + 1;
      setCurrentSlideIndex(nextIndex);
      setSlidePrompt('');
      // 移除自动美化逻辑，因为现在是预先批量生成好了
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
    if (!resultPath) {
      setError('缺少 result_path，请重新上传文件');
      return;
    }
    
    setIsGeneratingFinal(true);
    setError(null);
    
    try {
      // 调用 /paper2ppt/ppt_json 接口生成最终 PPT
      const formData = new FormData();
      formData.append('img_gen_model_name', genFigModel);
      formData.append('chat_api_url', llmApiUrl.trim());
      formData.append('api_key', apiKey.trim());
      formData.append('model', model);
      formData.append('language', language);
      formData.append('style', globalPrompt || stylePreset);
      formData.append('aspect_ratio', '16:9');
      formData.append('invite_code', inviteCode.trim());
      formData.append('result_path', resultPath);
      formData.append('get_down', 'false');
      formData.append('all_edited_down', 'true');

      // 传递最终的 pagecontent
      const pagecontent = outlineData.map(slide => ({
        title: slide.title,
        layout_description: slide.layout_description,
        key_points: slide.key_points,
        asset_ref: slide.asset_ref,
      }));
      formData.append('pagecontent', JSON.stringify(pagecontent));

      const res = await fetch('/api/paper2ppt/ppt_json', {
        method: 'POST',
        headers: { 'X-API-Key': API_KEY },
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
      
      // 从 all_output_files 中找到 PPTX 和 PDF 文件
      const pptxUrl = data.all_output_files?.find((url: string) => url.endsWith('.pptx')) || data.ppt_pptx_path;
      const pdfUrl = data.all_output_files?.find((url: string) => 
        url.endsWith('.pdf') && !url.includes('input')
      ) || data.ppt_pdf_path;
      
      if (pptxUrl) {
        setDownloadUrl(pptxUrl);
      }
      if (pdfUrl) {
        setPdfDownloadUrl(pdfUrl);
      }
      // 只要有一个文件生成成功即可
      if (!pptxUrl && !pdfUrl) {
        throw new Error('未找到生成的文件');
      }

      // Record usage
      await recordUsage(user?.id || null, 'ppt2polish');
      refreshQuota();

      // Fetch PPT file and upload to Supabase Storage
      if (pptxUrl) {
        try {
          const pptRes = await fetch(pptxUrl);
          if (pptRes.ok) {
            const pptBlob = await pptRes.blob();
            const pptName = pptxUrl.split('/').pop() || 'ppt2polish_result.pptx';
            uploadAndSaveFile(pptBlob, pptName, 'ppt2polish');
          }
        } catch (e) {
          console.warn('[Ppt2PolishPage] Failed to upload file:', e);
        }
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : '生成最终 PPT 失败';
      setError(message);
    } finally {
    setIsGeneratingFinal(false);
    }
  };

  const handleDownload = async () => {
    if (!downloadUrl) {
      setError('下载链接不存在');
      return;
    }
    
    try {
      const res = await fetch(downloadUrl);
      if (!res.ok) {
        throw new Error('下载失败');
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'paper2ppt_editable.pptx';
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      const message = err instanceof Error ? err.message : '下载失败';
      setError(message);
    }
  };

  // ============== 渲染步骤指示器 ==============
  const renderStepIndicator = () => {
    const steps = [
      { key: 'upload', label: '上传 PPT', num: 1 },
      { key: 'beautify', label: '逐页美化', num: 2 },
      { key: 'complete', label: '完成下载', num: 3 },
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
            PptPolish
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
            配置
          </h3>
          
          {/* <div>
            <label className="block text-sm text-gray-300 mb-2">邀请码</label>
            <input
              type="text"
              value={inviteCode}
              onChange={(e) => setInviteCode(e.target.value)}
              placeholder="请输入邀请码"
              className="w-full rounded-lg border border-white/20 bg-black/40 px-4 py-2.5 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-teal-500 placeholder:text-gray-500"
            />
          </div> */}
          
          <div>
            <label className="block text-sm text-gray-300 mb-2">模型 API URL</label>
            <input
              type="text"
              value={llmApiUrl}
              onChange={(e) => setLlmApiUrl(e.target.value)}
              placeholder="https://api.apiyi.com/v1"
              className="w-full rounded-lg border border-white/20 bg-black/40 px-4 py-2.5 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-teal-500 placeholder:text-gray-500"
            />
          </div>
          
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="block text-sm text-gray-300 mb-0">API Key</label>
              <a
                href="https://api.apiyi.com/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-[10px] text-teal-300 hover:text-teal-200 hover:underline"
              >
                点击购买
              </a>
            </div>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="sk-..."
              className="w-full rounded-lg border border-white/20 bg-black/40 px-4 py-2.5 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-teal-500 placeholder:text-gray-500"
            />
          </div>
          
          <div>
            <label className="block text-sm text-gray-300 mb-2">模型名称</label>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="w-full rounded-lg border border-white/20 bg-black/40 px-4 py-2.5 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-teal-500"
            >
              <option value="gpt-4o">gpt-4o</option>
              <option value="gpt-5.1">gpt-5.1</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm text-gray-300 mb-2">图像生成模型</label>
            <select
              value={genFigModel}
              onChange={(e) => setGenFigModel(e.target.value)}
              className="w-full rounded-lg border border-white/20 bg-black/40 px-4 py-2.5 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-teal-500"
            >
              <option value="gemini-3-pro-image-preview">gemini-3-pro-image-preview</option>
              <option value="gemini-2.5-flash-image">gemini-2.5-flash-image</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm text-gray-300 mb-2">生成语言</label>
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value as 'zh' | 'en')}
              className="w-full rounded-lg border border-white/20 bg-black/40 px-4 py-2.5 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-teal-500"
            >
              <option value="zh">中文 (zh)</option>
              <option value="en">英文 (en)</option>
            </select>
          </div>
          
          <div className="border-t border-white/10 pt-4">
            <h4 className="text-sm text-gray-300 mb-3 font-medium">风格配置</h4>
          <div className="flex gap-2">
            <button onClick={() => setStyleMode('preset')} className={`flex-1 py-2.5 px-4 rounded-lg text-sm font-medium flex items-center justify-center gap-2 transition-all ${styleMode === 'preset' ? 'bg-gradient-to-r from-cyan-500 to-teal-500 text-white' : 'bg-white/5 text-gray-400 border border-white/10 hover:bg-white/10'}`}>
              <Sparkles size={16} /> 预设风格
            </button>
            <button disabled className={`flex-1 py-2.5 px-4 rounded-lg text-sm font-medium flex items-center justify-center gap-2 transition-all bg-white/5 text-gray-600 border border-white/5 cursor-not-allowed opacity-50`}>
              <ImageIcon size={16} /> 参考图片 (Coming Soon)
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
                <label className="block text-sm text-gray-300 mb-2">风格提示词（必填）</label>
                <textarea value={globalPrompt} onChange={(e) => setGlobalPrompt(e.target.value)} placeholder="例如：使用紫色系配色，保持学术风格 / 多啦A梦风格 / 赛博朋克风格 ...... "  rows={3} className="w-full rounded-lg border border-white/20 bg-black/40 px-4 py-2.5 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-teal-500 placeholder:text-gray-500 resize-none" />
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
            </div>
          <button onClick={handleUploadAndParse} disabled={!selectedFile || isUploading} className="w-full py-3 rounded-lg bg-gradient-to-r from-cyan-600 to-teal-600 hover:from-cyan-700 hover:to-teal-700 disabled:from-gray-600 disabled:to-gray-700 text-white font-semibold flex items-center justify-center gap-2 transition-all">
            {isUploading ? <><Loader2 size={18} className="animate-spin" /> 解析中...</> : <><ArrowRight size={18} /> 开始解析</>}
          </button>

          {isUploading && (
            <div className="mt-4 animate-in fade-in slide-in-from-top-2">
              <div className="flex justify-between text-xs text-gray-400 mb-1">
                <span>{progressStatus}</span>
                <span>{Math.round(progress)}%</span>
              </div>
              <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-cyan-500 to-teal-500 transition-all duration-300 ease-out"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )}
        </div>
      </div>
      {error && <div className="mt-4 flex items-center gap-2 text-sm text-red-300 bg-red-500/10 border border-red-500/40 rounded-lg px-4 py-3"><AlertCircle size={16} /> {error}</div>}

      {/* 示例区 */}
      {/* 示例区 */}
      <div className="space-y-8 mt-10">
        {/* 第一组：PPT 增色美化 */}
        <div className="space-y-4">
          <div className="flex items-center gap-3">
            <div className="w-1 h-8 bg-gradient-to-b from-cyan-400 to-teal-500 rounded-full"></div>
            <div>
              <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                <Sparkles size={18} className="text-cyan-400" />
                PPT 增色美化
              </h3>
              <p className="text-sm text-gray-400">
                基于原有 PPT 内容，智能调整风格、配色与视觉层次，让演示更具专业感与吸引力
              </p>
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {/* Demo 1 */}
            <div className="glass rounded-xl border border-white/10 p-4 hover:border-cyan-500/30 transition-all">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <p className="text-xs text-gray-500 mb-2 text-center">原始 PPT</p>
                  <div className="rounded-lg overflow-hidden border border-white/10 aspect-[16/9] bg-white/5">
                    <img src="/ppt2polish/paper2ppt_orgin_1.png" alt="原始PPT示例1" className="w-full h-full object-contain" />
                  </div>
                </div>
                <div>
                  <p className="text-xs text-cyan-400 mb-2 text-center">增色后</p>
                  <div className="rounded-lg overflow-hidden border border-cyan-500/30 aspect-[16/9] bg-gradient-to-br from-cyan-500/5 to-teal-500/5">
                    <img src="/ppt2polish/paper2ppt_polish_1.png" alt="美化后PPT示例1" className="w-full h-full object-contain" />
                  </div>
                </div>
              </div>
            </div>
            {/* Demo 2 */}
            <div className="glass rounded-xl border border-white/10 p-4 hover:border-cyan-500/30 transition-all">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <p className="text-xs text-gray-500 mb-2 text-center">原始 PPT</p>
                  <div className="rounded-lg overflow-hidden border border-white/10 aspect-[16/9] bg-white/5">
                    <img src="/ppt2polish/paper2ppt_orgin_2.png" alt="原始PPT示例2" className="w-full h-full object-contain" />
                  </div>
                </div>
                <div>
                  <p className="text-xs text-cyan-400 mb-2 text-center">增色后</p>
                  <div className="rounded-lg overflow-hidden border border-cyan-500/30 aspect-[16/9] bg-gradient-to-br from-cyan-500/5 to-teal-500/5">
                    <img src="/ppt2polish/paper2ppt_polish_2.png" alt="美化后PPT示例2" className="w-full h-full object-contain" />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* 第二组：PPT 润色拓展 */}
        <div className="space-y-4">
          <div className="flex items-center gap-3">
            <div className="w-1 h-8 bg-gradient-to-b from-purple-400 to-pink-500 rounded-full"></div>
            <div>
              <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                <Edit3 size={18} className="text-purple-400" />
                PPT 润色拓展
              </h3>
              <p className="text-sm text-gray-400">
                将纯文字或简易空白 PPT 智能润色拓展，自动生成精美排版与视觉元素，一键变身专业演示
              </p>
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {/* Demo 3 */}
            <div className="glass rounded-xl border border-white/10 p-4 hover:border-purple-500/30 transition-all">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <p className="text-xs text-gray-500 mb-2 text-center">原始 PPT</p>
                  <div className="rounded-lg overflow-hidden border border-white/10 aspect-[16/9] bg-white/5">
                    <img src="/ppt2polish/orgin_3.png" alt="原始PPT示例3" className="w-full h-full object-contain" />
                  </div>
                </div>
                <div>
                  <p className="text-xs text-purple-400 mb-2 text-center">润色后</p>
                  <div className="rounded-lg overflow-hidden border border-purple-500/30 aspect-[16/9] bg-gradient-to-br from-purple-500/5 to-pink-500/5">
                    <img src="/ppt2polish/polish_3.png" alt="美化后PPT示例3" className="w-full h-full object-contain" />
                  </div>
                </div>
              </div>
            </div>
            {/* Demo 4 */}
            <div className="glass rounded-xl border border-white/10 p-4 hover:border-purple-500/30 transition-all">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <p className="text-xs text-gray-500 mb-2 text-center">原始 PPT</p>
                  <div className="rounded-lg overflow-hidden border border-white/10 aspect-[16/9] bg-white/5">
                    <img src="/ppt2polish/orgin_4.png" alt="原始PPT示例4" className="w-full h-full object-contain" />
                  </div>
                </div>
                <div>
                  <p className="text-xs text-purple-400 mb-2 text-center">润色后</p>
                  <div className="rounded-lg overflow-hidden border border-purple-500/30 aspect-[16/9] bg-gradient-to-br from-purple-500/5 to-pink-500/5">
                    <img src="/ppt2polish/polish_4.png" alt="美化后PPT示例4" className="w-full h-full object-contain" />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
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
    
    // 如果正在生成初始 PPT，显示加载状态
    if (isGeneratingInitial) {
      return (
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-6">
            <h2 className="text-2xl font-bold text-white mb-2">正在生成初始 PPT</h2>
            <p className="text-gray-400">请稍候，正在处理您的 PPT 文件...</p>
          </div>
          <div className="glass rounded-xl border border-white/10 p-12 flex flex-col items-center justify-center">
            <Loader2 size={48} className="text-teal-400 animate-spin mb-4" />
            <p className="text-teal-300 text-lg font-medium mb-2">正在生成初始 PPT 和预览图</p>
            <p className="text-gray-400 text-sm">这可能需要几分钟时间，请耐心等待...</p>
          </div>
        </div>
      );
    }
    
    return (
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-6">
          <h2 className="text-2xl font-bold text-white mb-2">逐页美化</h2>
          <p className="text-gray-400">第 {currentSlideIndex + 1} / {outlineData.length} 页：{currentSlide?.title}</p>
          <p className="text-xs text-gray-500 mt-1">🎨 美化模式 - 优化原有 PPT 样式</p>
        </div>
        <div className="mb-6">
          <div className="flex gap-1">{beautifyResults.map((result, index) => (<div key={result.slideId} className={`flex-1 h-2 rounded-full transition-all ${result.status === 'done' ? 'bg-teal-400' : result.status === 'processing' ? 'bg-gradient-to-r from-cyan-400 to-teal-400 animate-pulse' : index === currentSlideIndex ? 'bg-teal-400/50' : 'bg-white/10'}`} />))}</div>
        </div>
        <div className="glass rounded-xl border border-white/10 p-6 mb-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="text-sm text-gray-400 mb-3 flex items-center gap-2"><Eye size={14} /> 原始 PPT 渲染</h4>
              <div className="rounded-lg overflow-hidden border border-white/10 aspect-[16/9] bg-white/5 flex items-center justify-center">{currentResult?.beforeImage ? <img src={currentResult.beforeImage} alt="Before" className="max-w-full max-h-full object-contain" /> : <Loader2 size={24} className="text-gray-500 animate-spin" />}</div>
            </div>
            <div>
              <h4 className="text-sm text-gray-400 mb-3 flex items-center gap-2"><Sparkles size={14} className="text-teal-400" /> 美化结果</h4>
              <div className="rounded-lg overflow-hidden border border-teal-500/30 aspect-[16/9] bg-gradient-to-br from-cyan-500/10 to-teal-500/10 flex items-center justify-center">{isBeautifying ? <div className="text-center"><Loader2 size={32} className="text-teal-400 animate-spin mx-auto mb-2" /><p className="text-sm text-teal-300">正在美化中...</p></div> : currentResult?.afterImage ? <img src={currentResult.afterImage} alt="After" className="max-w-full max-h-full object-contain" /> : <span className="text-gray-500">等待生成</span>}</div>
            </div>
          </div>
        </div>
        <div className="glass rounded-xl border border-white/10 p-4 mb-6">
          <div className="flex items-center gap-3"><MessageSquare size={18} className="text-teal-400" /><input type="text" value={slidePrompt} onChange={(e) => setSlidePrompt(e.target.value)} placeholder="输入微调 Prompt，然后点击重新生成..." className="flex-1 bg-transparent border-none outline-none text-white text-sm placeholder:text-gray-500" /><button onClick={handleRegenerateSlide} disabled={isBeautifying || !slidePrompt.trim()} className="px-4 py-2 rounded-lg bg-white/10 hover:bg-white/20 text-gray-300 text-sm flex items-center gap-2 disabled:opacity-50 transition-all"><RefreshCw size={14} /> 重新生成</button></div>
        </div>
        <div className="flex justify-between">
          <button onClick={() => setCurrentStep('upload')} className="px-6 py-2.5 rounded-lg border border-white/20 text-gray-300 hover:bg-white/10 flex items-center gap-2 transition-all"><ArrowLeft size={18} /> 返回上传</button>
          <div className="flex gap-3"><button onClick={handleConfirmSlide} disabled={isBeautifying || !currentResult?.afterImage} className="px-6 py-2.5 rounded-lg bg-gradient-to-r from-cyan-600 to-teal-600 hover:from-cyan-700 hover:to-teal-700 text-white font-semibold flex items-center gap-2 transition-all disabled:opacity-50"><CheckCircle2 size={18} /> 确认并继续</button></div>
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
        <div className="grid grid-cols-4 gap-2">{beautifyResults.map((result, index) => (<div key={result.slideId} className="p-3 rounded-lg border bg-teal-500/20 border-teal-500/40"><p className="text-sm text-white">第 {index + 1} 页</p><p className="text-xs text-teal-300">已美化</p></div>))}</div>
      </div>
      {!(downloadUrl || pdfDownloadUrl) ? (
        <button onClick={handleGenerateFinal} disabled={isGeneratingFinal} className="px-8 py-3 rounded-lg bg-gradient-to-r from-cyan-600 to-teal-600 hover:from-cyan-700 hover:to-teal-700 text-white font-semibold flex items-center justify-center gap-2 mx-auto transition-all">
          {isGeneratingFinal ? <><Loader2 size={18} className="animate-spin" /> 正在生成最终文件...</> : <><Sparkles size={18} /> 生成最终文件</>}
        </button>
      ) : (
        <div className="space-y-4">
          <div className="flex gap-4 justify-center">
            {/* 已移除 PPTX 下载按钮 */}
            {pdfDownloadUrl && (
              <a href={pdfDownloadUrl} target="_blank" rel="noopener noreferrer" className="px-6 py-3 rounded-lg bg-gradient-to-r from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600 text-white font-semibold flex items-center gap-2 transition-all">
                <Download size={18} /> 下载 PDF
              </a>
            )}
          </div>

          {/* 引导去 PDF2PPT */}
          <div className="text-center text-sm text-gray-400 bg-white/5 border border-white/10 rounded-lg p-3">
            如果需要继续 PDF 转可编辑 PPTX，请前往 <a href="/pdf2ppt" className="text-teal-400 hover:text-teal-300 hover:underline font-medium transition-colors">PDF2PPT 页面</a>
          </div>

          <div>
            <button onClick={() => { setCurrentStep('upload'); setSelectedFile(null); setOutlineData([]); setBeautifyResults([]); setDownloadUrl(null); setPdfDownloadUrl(null); }} className="text-sm text-gray-400 hover:text-white transition-colors">
              <RotateCcw size={14} className="inline mr-1" /> 处理新的文档
            </button>
          </div>

          {/* 分享与交流群区域 */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-8 text-left">
            {/* 获取免费 Key */}
            <div className="glass rounded-xl border border-white/10 p-5 flex flex-col items-center text-center hover:bg-white/5 transition-colors">
              <div className="w-12 h-12 rounded-full bg-yellow-500/20 text-yellow-300 flex items-center justify-center mb-3">
                <Star size={24} />
              </div>
              <h4 className="text-white font-semibold mb-2">获取免费 API Key</h4>
              <p className="text-xs text-gray-400 mb-4 leading-relaxed">
                点击下方平台图标复制推广文案<br/>
                分享至朋友圈/小红书/推特，截图联系微信群管理员领 Key！
              </p>
              
              {/* 分享按钮组 */}
              <div className="flex items-center justify-center gap-4 mb-5 w-full">
                <button onClick={handleCopyShareText} className="flex flex-col items-center gap-1 group">
                  <div className="w-10 h-10 rounded-full bg-[#00C300]/20 text-[#00C300] flex items-center justify-center border border-[#00C300]/30 group-hover:scale-110 transition-transform">
                    <MessageSquare size={18} />
                  </div>
                  <span className="text-[10px] text-gray-400">微信</span>
                </button>
                <button onClick={handleCopyShareText} className="flex flex-col items-center gap-1 group">
                  <div className="w-10 h-10 rounded-full bg-[#FF2442]/20 text-[#FF2442] flex items-center justify-center border border-[#FF2442]/30 group-hover:scale-110 transition-transform">
                    <span className="font-bold text-xs">小红书</span>
                  </div>
                  <span className="text-[10px] text-gray-400">小红书</span>
                </button>
                <button onClick={handleCopyShareText} className="flex flex-col items-center gap-1 group">
                  <div className="w-10 h-10 rounded-full bg-white/10 text-white flex items-center justify-center border border-white/20 group-hover:scale-110 transition-transform">
                    <span className="font-bold text-lg">𝕏</span>
                  </div>
                  <span className="text-[10px] text-gray-400">Twitter</span>
                </button>
                <button onClick={handleCopyShareText} className="flex flex-col items-center gap-1 group">
                  <div className="w-10 h-10 rounded-full bg-purple-500/20 text-purple-300 flex items-center justify-center border border-purple-500/30 group-hover:scale-110 transition-transform">
                    <Copy size={18} />
                  </div>
                  <span className="text-[10px] text-gray-400">复制</span>
                </button>
              </div>

              {copySuccess && (
                <div className="mb-4 px-3 py-1 bg-green-500/20 text-green-300 text-xs rounded-full animate-in fade-in zoom-in">
                  ✨ {copySuccess}
                </div>
              )}

              <div className="w-full space-y-2">
                 <a href="https://dcai-paper2any.nas.cpolar.cn/" target="_blank" rel="noopener noreferrer" className="block w-full py-1.5 px-3 rounded bg-white/5 hover:bg-white/10 text-xs text-teal-300 truncate transition-colors border border-white/5 text-center">
                   🌐 如果项目对你有帮助的话，可以点个star嘛~
                 </a>
                 <div className="flex gap-2">
                   <a href="https://github.com/OpenDCAI/DataFlow-Agent" target="_blank" rel="noopener noreferrer" className="flex-1 py-1.5 px-3 rounded bg-white/5 hover:bg-white/10 text-xs text-gray-300 truncate transition-colors border border-white/5 flex items-center justify-center gap-1">
                     <Github size={12} /> Agent
                   </a>
                   <a href="https://github.com/OpenDCAI/DataFlow" target="_blank" rel="noopener noreferrer" className="flex-1 py-1.5 px-3 rounded bg-white/5 hover:bg-white/10 text-xs text-gray-300 truncate transition-colors border border-white/5 flex items-center justify-center gap-1">
                     <Github size={12} /> Core
                   </a>
                 </div>
              </div>
              <p className="text-[10px] text-gray-500">点亮 Star ⭐ 支持开源开发</p>
            </div>

            {/* 交流群 */}
            <div className="glass rounded-xl border border-white/10 p-5 flex flex-col items-center text-center hover:bg-white/5 transition-colors">
              <div className="w-12 h-12 rounded-full bg-green-500/20 text-green-300 flex items-center justify-center mb-3">
                <MessageSquare size={24} />
              </div>
              <h4 className="text-white font-semibold mb-2">加入交流群</h4>
              <p className="text-xs text-gray-400 mb-4">
                效果满意？遇到问题？<br/>欢迎扫码加入交流群反馈与讨论
              </p>
              <div className="w-32 h-32 bg-white p-1 rounded-lg mb-2">
                <img src="/wechat.png" alt="交流群二维码" className="w-full h-full object-contain" />
              </div>
              <p className="text-[10px] text-gray-500">扫码加入微信交流群</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );

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
              <a
                href="https://github.com/OpenDCAI"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 bg-white/20 backdrop-blur-sm rounded-full px-3 py-1 hover:bg-white/30 transition-colors"
              >
                <Star size={16} className="text-yellow-300 fill-yellow-300 animate-pulse" />
                <span className="text-xs font-bold text-white">GitHub开源项目</span>
              </a>
              
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
                <span className="bg-gray-200 text-gray-800 px-1.5 py-0.5 rounded-full text-[10px] flex items-center gap-0.5"><Star size={8} fill="currentColor" /> {stars.dataflow || 'Star'}</span>
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
                <span className="bg-gray-200 text-gray-800 px-1.5 py-0.5 rounded-full text-[10px] flex items-center gap-0.5"><Star size={8} fill="currentColor" /> {stars.agent || 'Star'}</span>
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
      <div className="flex-1 w-full overflow-auto"><div className="max-w-7xl mx-auto px-6 py-8 pb-24">{renderStepIndicator()}{currentStep === 'upload' && renderUploadStep()}{currentStep === 'beautify' && renderBeautifyStep()}{currentStep === 'complete' && renderCompleteStep()}</div></div>
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

export default Ppt2PolishPage;
