import { useState, useEffect, ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';
import {
  Presentation, UploadCloud, Settings2, Download, Loader2, CheckCircle2,
  AlertCircle, ChevronDown, ChevronUp, Github, Star, X, Sparkles,
  ArrowRight, ArrowLeft, GripVertical, Trash2, Edit3, Check, RotateCcw,
  MessageSquare, Eye, RefreshCw, FileText, Image as ImageIcon, Copy, Info
} from 'lucide-react';
import { uploadAndSaveFile } from '../services/fileService';
import { API_KEY, API_URL_OPTIONS, DEFAULT_LLM_API_URL, getPurchaseUrl } from '../config/api';
import {
  DEFAULT_PPT2POLISH_GEN_FIG_MODEL,
  DEFAULT_PPT2POLISH_MODEL,
  PPT2POLISH_GEN_FIG_MODELS,
  PPT2POLISH_MODELS,
  withModelOptions,
} from '../config/models';
import { checkQuota, recordUsage } from '../services/quotaService';
import { verifyLlmConnection } from '../services/llmService';
import { useAuthStore } from '../stores/authStore';
import { getApiSettings, saveApiSettings } from '../services/apiSettingsService';
import QRCodeTooltip from './QRCodeTooltip';
import VersionHistory from './paper2ppt/VersionHistory';

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

// 版本历史类型定义
interface ImageVersion {
  versionNumber: number;
  imageUrl: string;
  prompt: string;
  timestamp: number;
  isCurrentVersion: boolean;
}

interface BeautifyResult {
  slideId: string;
  beforeImage: string;
  afterImage: string;
  status: 'pending' | 'processing' | 'done';
  userPrompt?: string;
  versionHistory: ImageVersion[];
  currentVersionIndex: number;
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

const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB
const STORAGE_KEY = 'pptpolish-storage';

// ============== 主组件 ==============
const Ppt2PolishPage = () => {
  const { t, i18n } = useTranslation(['pptPolish', 'common']);
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
  const [isValidating, setIsValidating] = useState(false);
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
  const [llmApiUrl, setLlmApiUrl] = useState(DEFAULT_LLM_API_URL);
  const [apiKey, setApiKey] = useState('');
  const [model, setModel] = useState(DEFAULT_PPT2POLISH_MODEL);
  const [genFigModel, setGenFigModel] = useState(DEFAULT_PPT2POLISH_GEN_FIG_MODEL);
  const [language, setLanguage] = useState<'zh' | 'en'>('en');
  const [renderResolution, setRenderResolution] = useState<'auto' | '1080p' | '2k' | '4k'>('2k');
  const [resultPath, setResultPath] = useState<string | null>(null);

  // GitHub Stars
  const [stars, setStars] = useState<{dataflow: number | null, agent: number | null, dataflex: number | null}>({
    dataflow: null,
    agent: null,
    dataflex: null,
  });
  const [copySuccess, setCopySuccess] = useState('');

  const shareText = `发现一个超好用的AI工具 DataFlow-Agent！🚀
支持论文转PPT、PDF转PPT、PPT美化等功能，科研打工人的福音！

🔗 在线体验：https://dcai-paper2any.nas.cpolar.cn/
⭐ GitHub Agent：https://github.com/OpenDCAI/Paper2Any
🌟 GitHub Core：https://github.com/OpenDCAI/DataFlow

转发本文案+截图，联系微信群管理员即可获取免费Key！🎁
#AI工具 #PPT制作 #科研效率 #开源项目`;

  const modelOptions = withModelOptions(PPT2POLISH_MODELS, model);
  const genFigModelOptions = withModelOptions(PPT2POLISH_GEN_FIG_MODELS, genFigModel);
  const renderDpiMap = { '1080p': 144, '2k': 192, '4k': 288 } as const;
  const getRenderDpi = () => (renderResolution === 'auto' ? null : renderDpiMap[renderResolution]);
  const imageResolutionMap = { '1080p': '1K', '2k': '2K', '4k': '4K' } as const;
  const getImageResolution = () => (renderResolution === 'auto' ? null : imageResolutionMap[renderResolution]);
  const uiLang = i18n.language?.startsWith('zh') ? 'zh' : 'en';
  const stylePromptCards = uiLang === 'zh'
    ? [
        {
          title: '手绘卡通信息图',
          text: '手绘卡通风格的信息图。线条：素描感、粗糙笔触、卡通简化\n禁止写实、禁止照片级明暗、禁止 3D 渲染\n效果参考：涂鸦 / 蜡笔 / 马克笔 / 粉彩',
        },
        {
          title: '极简专业商务',
          text: '极简商务风格。大留白、清晰对比、2~3 色主辅配色\n强调对齐与网格、轻阴影、扁平图标\n禁止复杂纹理、禁止炫光、禁止杂乱背景',
        },
        {
          title: '科技蓝紫渐变',
          text: '科技感视觉：深蓝到青色渐变背景，发光线条/节点\n图表与关键数字高亮，模块卡片玻璃拟态\n禁止复古元素、禁止卡通元素',
        },
        {
          title: '学术论文风',
          text: '学术报告风格：白底、严谨排版、稳重配色（蓝/灰/黑）\n图表优先、标题清晰、关键结论加粗\n禁止花哨装饰、禁止大面积高饱和色',
        },
        {
          title: '品牌宣传风',
          text: '品牌宣传风格：高质感图片占比高，统一品牌色系\n标题大、层级分明，口号式短句\n禁止密集文字、禁止表格式排版',
        },
        {
          title: '自然柔和插画',
          text: '自然柔和插画风：米白背景、低饱和配色、柔和阴影\n插画/贴纸元素点缀，整体温暖亲和\n禁止强对比、禁止金属质感、禁止赛博霓虹',
        },
      ]
    : [
        {
          title: 'Hand-drawn Infographic',
          text: 'Hand-drawn cartoon infographic. Lines: sketchy, rough strokes, simplified shapes.\nNo realism, no photographic lighting, no 3D rendering.\nLook & feel: doodle / crayon / marker / pastel.',
        },
        {
          title: 'Minimal Business',
          text: 'Minimal business style. Spacious layout, strong contrast, 2–3 color palette.\nStrict alignment/grid, subtle shadows, flat icons.\nNo heavy textures, no glow effects, no busy backgrounds.',
        },
        {
          title: 'Tech Gradient',
          text: 'Futuristic tech look: deep blue to cyan gradients, glowing lines/nodes.\nHighlight charts and key numbers, glassmorphism cards.\nNo retro elements, no cartoon elements.',
        },
        {
          title: 'Academic Report',
          text: 'Academic report style: white background, rigorous layout, sober colors (blue/gray/black).\nChart-first, clear titles, bold key findings.\nNo fancy decorations, no highly saturated blocks.',
        },
        {
          title: 'Brand Promo',
          text: 'Brand promo style: high-quality visuals, consistent brand colors.\nBig titles, clear hierarchy, slogan-like short phrases.\nNo dense text, no table-like layouts.',
        },
        {
          title: 'Soft Illustration',
          text: 'Soft illustration style: off-white background, low-saturation palette, gentle shadows.\nLight stickers/illustrations as accents, warm and friendly tone.\nNo harsh contrast, no metallic textures, no cyber neon.',
        },
      ];

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
        const [res1, res2, res3] = await Promise.all([
          fetch('https://api.github.com/repos/OpenDCAI/DataFlow'),
          fetch('https://api.github.com/repos/OpenDCAI/Paper2Any'),
          fetch('https://api.github.com/repos/OpenDCAI/DataFlex')
        ]);
        const data1 = await res1.json();
        const data2 = await res2.json();
        const data3 = await res3.json();
        setStars({
          dataflow: data1.stargazers_count,
          agent: data2.stargazers_count,
          dataflex: data3.stargazers_count,
        });
      } catch (e) {
        console.error('Failed to fetch stars', e);
      }
    };
    fetchStars();
  }, []);

  // 从 localStorage 恢复配置
  useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const saved = JSON.parse(raw);
        
        if (saved.styleMode) setStyleMode(saved.styleMode);
        if (saved.stylePreset) setStylePreset(saved.stylePreset);
        if (saved.globalPrompt) setGlobalPrompt(saved.globalPrompt);
        if (saved.model) setModel(saved.model);
        if (saved.genFigModel) setGenFigModel(saved.genFigModel);
        if (saved.language) setLanguage(saved.language);
        if (saved.renderResolution) setRenderResolution(saved.renderResolution);

        // API settings: prioritize user-specific settings from apiSettingsService
        const userApiSettings = getApiSettings(user?.id || null);
        if (userApiSettings) {
          if (userApiSettings.apiUrl) setLlmApiUrl(userApiSettings.apiUrl);
          if (userApiSettings.apiKey) setApiKey(userApiSettings.apiKey);
        } else {
          // Fallback to legacy localStorage
          if (saved.llmApiUrl) setLlmApiUrl(saved.llmApiUrl);
          if (saved.apiKey) setApiKey(saved.apiKey);
        }
      }
    } catch (e) {
      console.error('Failed to restore pptpolish config', e);
    }
  }, [user?.id]);

  // 将配置写入 localStorage
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const data = {
      styleMode,
      stylePreset,
      globalPrompt,
      llmApiUrl,
      apiKey,
      model,
      genFigModel,
      language,
      renderResolution
    };
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
      // Also save API settings to user-specific storage
      if (user?.id && llmApiUrl && apiKey) {
        saveApiSettings(user.id, { apiUrl: llmApiUrl, apiKey });
      }
    } catch (e) {
      console.error('Failed to persist pptpolish config', e);
    }
  }, [
    styleMode, stylePreset, globalPrompt,
    llmApiUrl, apiKey, model, genFigModel, language, renderResolution, user?.id
  ]);

  // 自动加载版本历史
  useEffect(() => {
    if (currentStep === 'beautify' && currentSlideIndex >= 0 && beautifyResults[currentSlideIndex]) {
      const currentResult = beautifyResults[currentSlideIndex];
      // 如果版本历史为空且页面已生成，则自动加载版本历史
      if (currentResult.versionHistory.length === 0 && currentResult.afterImage) {
        console.log(`[Ppt2PolishPage] 自动加载页面 ${currentSlideIndex} 的版本历史`);
        fetchVersionHistory(currentSlideIndex);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentStep, currentSlideIndex]); // 移除 beautifyResults 依赖，避免无限循环

  // ============== Step 1: 上传处理 ==============
  const validateDocFile = (file: File): boolean => {
    const ext = file.name.split('.').pop()?.toLowerCase();
    if (ext !== 'ppt' && ext !== 'pptx' && ext !== 'pdf') {
      setError(t('errors.format'));
      return false;
    }
    return true;
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!validateDocFile(file)) return;
    if (file.size > MAX_FILE_SIZE) {
      setError(t('errors.size'));
      return;
    }
    setSelectedFile(file);
    setError(null);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (!file) return;
    if (!validateDocFile(file)) return;
    if (file.size > MAX_FILE_SIZE) {
      setError(t('errors.size'));
      return;
    }
    setSelectedFile(file);
    setError(null);
  };

  const handleReferenceImageChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const ext = file.name.split('.').pop()?.toLowerCase();
    if (!['jpg', 'jpeg', 'png', 'webp', 'gif'].includes(ext || '')) {
      setError(t('errors.imageFormat')); // Assuming I added this key, wait, I didn't add imageFormat to pptPolish.json. I'll use hardcoded or add it.
      // I missed imageFormat in pptPolish.json. I'll use a generic error or keep it hardcoded for now.
      // Actually I can use 'errors.format' but that says PPT/PPTX.
      // Let's keep it hardcoded for now to avoid error.
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
      setError(t('errors.selectFile'));
      return;
    }
    
    if (!llmApiUrl.trim() || !apiKey.trim()) {
      setError(t('errors.config'));
      return;
    }

    if (styleMode === 'preset' && !globalPrompt.trim()) {
      setError(t('errors.prompt'));
      return;
    }

    if (styleMode === 'reference' && !referenceImage) {
      setError(t('errors.reference'));
      return;
    }

    // Check quota before proceeding
    const quota = await checkQuota(user?.id || null, user?.is_anonymous || false);
    if (quota.remaining <= 0) {
      setError(t('errors.quota'));
      return;
    }

    try {
        // Step 0: Verify LLM Connection first
        setIsValidating(true);
        setError(null);
        await verifyLlmConnection(llmApiUrl, apiKey, import.meta.env.VITE_DEFAULT_LLM_MODEL || 'deepseek-v3.2');
        setIsValidating(false);
    } catch (err) {
        setIsValidating(false);
        const message = err instanceof Error ? err.message : 'API 验证失败';
        setError(message);
        return; // Stop execution if validation fails
    }

    setIsUploading(true);
    setError(null);
    setProgress(0);
    setProgressStatus(t('progress.init'));

    // 模拟进度
    const progressInterval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 90) return 90;
        const messages = [
           t('progress.uploading'),
           t('progress.analyzing'),
           t('progress.extracting'),
           t('progress.identifying'),
           t('progress.planning')
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
      formData.append('email', user?.id || user?.email || '');
      const ext = selectedFile.name.split('.').pop()?.toLowerCase();
      const isPdf = ext === 'pdf';
      formData.append('input_type', isPdf ? 'pdf' : 'pptx');
      if (isPdf) {
        formData.append('pdf_as_slides', 'true');
      }
      const renderDpi = getRenderDpi();
      if (renderDpi) {
        formData.append('render_dpi', String(renderDpi));
      }
      formData.append('file', selectedFile);
      
      if (referenceImage) {
        formData.append('reference_img', referenceImage);
      }
      
      console.log('Sending request to /api/v1/paper2ppt/page-content'); // 调试信息
      
      const res = await fetch('/api/v1/paper2ppt/page-content', {
        method: 'POST',
        headers: { 'X-API-Key': API_KEY },
        body: formData,
      });

      console.log('Response status:', res.status, res.statusText); // 调试信息
      
      if (!res.ok) {
        let msg = t('errors.serverBusy');
        if (res.status === 403) {
          msg = '邀请码不正确或已失效';
        } else if (res.status === 429) {
          msg = '请求过于频繁，请稍后再试';
        } else {
          try {
            const errBody = await res.json();
            if (errBody?.error) msg = errBody.error;
          } catch { /* ignore parse error */ }
        }
        throw new Error(msg);
      }

      const data = await res.json();

      console.log('API Response:', JSON.stringify(data, null, 2)); // 调试信息

      if (!data.success) {
        throw new Error(data.error || t('errors.serverBusy'));
      }
      
      // 保存 result_path
      const currentResultPath = data.result_path || '';
      if (currentResultPath) {
        setResultPath(currentResultPath);
      } else {
        throw new Error(t('errors.noResultPath'));
      }
      
      // 检查 pagecontent 是否为空
      if (!data.pagecontent || data.pagecontent.length === 0) {
        throw new Error(t('errors.emptyResult'));
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
        versionHistory: [],
        currentVersionIndex: 0,
      }));
      setBeautifyResults(results);
      setCurrentSlideIndex(0);
      
      // 不再一次性美化所有页面！
      // 直接进入美化步骤，显示原始图片
      // 用户点击"开始美化"时才调用 API 美化当前页
      
      console.log('解析完成，进入美化步骤, results.length:', results.length, 'currentResultPath:', currentResultPath);
      
      clearInterval(progressInterval);
      setProgress(100);
      setProgressStatus(t('progress.done'));

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
      const message = err instanceof Error ? err.message : t('errors.serverBusy');
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
      versionHistory: [],
      currentVersionIndex: 0,
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
      const imageResolution = getImageResolution();
      if (imageResolution) {
        formData.append('image_resolution', imageResolution);
      }
      formData.append('email', user?.id || user?.email || '');
      formData.append('result_path', currentPath);
      formData.append('get_down', 'false');
      formData.append('pagecontent', JSON.stringify(pagecontent));
      
      console.log('Generating initial PPT with pagecontent:', pagecontent);
      console.log('Request URL: /api/v1/paper2ppt/generate');
      console.log('Request params:', {
        img_gen_model_name: genFigModel,
        chat_api_url: llmApiUrl,
        // ... 其他参数
      });

      const res = await fetch('/api/v1/paper2ppt/generate', {
        method: 'POST',
        headers: { 'X-API-Key': API_KEY },
        body: formData,
      });

      console.log('Response status:', res.status, res.statusText);
      
      if (!res.ok) {
        let msg = '服务器繁忙，请稍后再试';
        if (res.status === 429) {
          msg = '请求过于频繁，请稍后再试';
        } else {
          try {
            const errBody = await res.json();
            if (errBody?.error) msg = errBody.error;
          } catch { /* ignore parse error */ }
        }
        throw new Error(msg);
      }

      const data = await res.json();
      console.log('Initial PPT generation response:', JSON.stringify(data, null, 2));

      if (!data.success) {
        throw new Error(data.error || '服务器繁忙，请稍后再试');
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
      const message = err instanceof Error ? err.message : '服务器繁忙，请稍后再试';
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
      const imageResolution = getImageResolution();
      if (imageResolution) {
        formData.append('image_resolution', imageResolution);
      }
      formData.append('email', user?.id || user?.email || '');
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

      const res = await fetch('/api/v1/paper2ppt/generate', {
        method: 'POST',
        headers: { 'X-API-Key': API_KEY },
        body: formData,
      });
      
      if (!res.ok) {
        let msg = '服务器繁忙，请稍后再试';
        if (res.status === 429) {
          msg = '请求过于频繁，请稍后再试';
        } else {
          try {
            const errBody = await res.json();
            if (errBody?.error) msg = errBody.error;
          } catch { /* ignore parse error */ }
        }
        throw new Error(msg);
      }

      const data = await res.json();
      console.log('美化响应:', JSON.stringify(data, null, 2));
      console.log('all_output_files:', data.all_output_files);

      if (!data.success) {
        throw new Error(data.error || '服务器繁忙，请稍后再试');
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

    // 获取更新的版本历史
    await fetchVersionHistory(index);
    } catch (err) {
      const message = err instanceof Error ? err.message : '服务器繁忙，请稍后再试';
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

  // ============== 版本历史管理 ==============
  const fetchVersionHistory = async (pageIndex: number) => {
    if (!resultPath) return;

    try {
      const encodedPath = btoa(resultPath);
      const res = await fetch(
        `/api/v1/paper2ppt/version-history/${encodedPath}/${pageIndex}`,
        { headers: { 'X-API-Key': API_KEY } }
      );

      if (!res.ok) return;

      const data = await res.json();
      if (data.success && data.versions) {
        setBeautifyResults(prev => prev.map((result, idx) =>
          idx === pageIndex
            ? {
                ...result,
                versionHistory: data.versions.map((v: any) => ({
                  versionNumber: v.version,
                  imageUrl: v.imageUrl,
                  prompt: v.prompt,
                  timestamp: v.timestamp,
                })),
                currentVersionIndex: data.versions.length - 1
              }
            : result
        ));
      }
    } catch (err) {
      console.error('Failed to fetch version history:', err);
    }
  };

  const handleRevertToVersion = async (versionNumber: number) => {
    if (!resultPath) {
      setError('缺少 result_path');
      return;
    }

    setIsBeautifying(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('result_path', resultPath);
      formData.append('page_id', String(currentSlideIndex));
      formData.append('target_version', String(versionNumber));

      const res = await fetch('/api/v1/paper2ppt/revert-version', {
        method: 'POST',
        headers: { 'X-API-Key': API_KEY },
        body: formData,
      });

      if (!res.ok) throw new Error('恢复版本失败');

      const data = await res.json();

      if (data.success) {
        const updatedResults = [...beautifyResults];
        updatedResults[currentSlideIndex] = {
          ...updatedResults[currentSlideIndex],
          afterImage: data.currentImageUrl + '?t=' + Date.now(),
          currentVersionIndex: versionNumber - 1,
        };
        setBeautifyResults(updatedResults);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : '恢复版本失败';
      setError(message);
    } finally {
      setIsBeautifying(false);
    }
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
      const imageResolution = getImageResolution();
      if (imageResolution) {
        formData.append('image_resolution', imageResolution);
      }
      formData.append('email', user?.id || user?.email || '');
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

      const res = await fetch('/api/v1/paper2ppt/generate', {
        method: 'POST',
        headers: { 'X-API-Key': API_KEY },
        body: formData,
      });
      
      if (!res.ok) {
        let msg = '服务器繁忙，请稍后再试';
        if (res.status === 429) {
          msg = '请求过于频繁，请稍后再试';
        } else {
          try {
            const errBody = await res.json();
            if (errBody?.error) msg = errBody.error;
          } catch { /* ignore parse error */ }
        }
        throw new Error(msg);
      }

      const data = await res.json();

      if (!data.success) {
        throw new Error(data.error || '服务器繁忙，请稍后再试');
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
      await recordUsage(user?.id || null, 'ppt2polish', { isAnonymous: user?.is_anonymous || false });
      refreshQuota();

      // Upload generated file to Supabase Storage (either PPTX or PDF)
      // Prefer PPTX, fallback to PDF
      let fileUrl = pptxUrl;
      let defaultName = 'ppt2polish_result.pptx';

      if (!fileUrl && pdfUrl) {
        fileUrl = pdfUrl;
        defaultName = 'ppt2polish_result.pdf';
      }

      if (fileUrl) {
        try {
          // Fix Mixed Content issue: upgrade http to https if current page is https
          let fetchUrl = fileUrl;
          if (window.location.protocol === 'https:' && fileUrl.startsWith('http:')) {
            fetchUrl = fileUrl.replace('http:', 'https:');
          }

          const fileRes = await fetch(fetchUrl);
          if (fileRes.ok) {
            const fileBlob = await fileRes.blob();
            // Use defaultName instead of extracting from URL to avoid reserved keywords and special chars
            const fileName = defaultName;
            console.log('[Ppt2PolishPage] Uploading file to storage:', fileName);
            await uploadAndSaveFile(fileBlob, fileName, 'ppt2polish');
            console.log('[Ppt2PolishPage] File uploaded successfully');
          }
        } catch (e) {
          console.error('[Ppt2PolishPage] Failed to upload file:', e);
        }
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : '服务器繁忙，请稍后再试';
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
      const message = err instanceof Error ? err.message : '服务器繁忙，请稍后再试';
      setError(message);
    }
  };

  // ============== 渲染步骤指示器 ==============
  const renderStepIndicator = () => {
    const steps = [
      { key: 'upload', label: t('steps.upload'), num: 1 },
      { key: 'beautify', label: t('steps.beautify'), num: 2 },
      { key: 'complete', label: t('steps.complete'), num: 3 },
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
          {t('subtitle')}
        </p>
        <h1 className="text-4xl md:text-5xl font-bold mb-4">
          <span className="bg-gradient-to-r from-cyan-400 via-teal-400 to-emerald-400 bg-clip-text text-transparent">
            {t('title')}
          </span>
        </h1>
        <p className="text-base text-gray-300 max-w-2xl mx-auto leading-relaxed">
          {t('desc')}
          <br />
          <span className="text-teal-400">{t('descHighlight')}</span>
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="glass rounded-xl border border-white/10 p-6 flex flex-col h-full">
          <h3 className="text-white font-semibold flex items-center gap-2 mb-4">
            <FileText size={18} className="text-teal-400" />
            {t('upload.title')}
          </h3>
          <div
            className={`border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center text-center gap-4 transition-all flex-1 ${
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
              <p className="text-white font-medium mb-1">{t('upload.dragText')}</p>
              <p className="text-sm text-gray-400">{t('upload.supportText')}</p>
            </div>
            <label className="group relative px-6 py-2.5 rounded-full bg-gradient-to-r from-cyan-600 to-teal-600 text-white text-sm font-medium cursor-pointer hover:from-cyan-700 hover:to-teal-700 transition-all">
              <Presentation size={16} className="inline mr-2" />
              {t('upload.button')}
              <input type="file" accept=".ppt,.pptx,.pdf" className="hidden" onChange={handleFileChange} />
              <div className="pointer-events-none absolute left-1/2 top-full z-10 mt-2 w-72 -translate-x-1/2 rounded-xl border border-white/15 bg-black/70 px-3 py-2 text-[11px] text-gray-200 shadow-lg opacity-0 backdrop-blur transition-all duration-200 group-hover:opacity-100">
                <div className="absolute -top-1 left-1/2 h-2 w-2 -translate-x-1/2 rotate-45 border border-white/15 bg-black/70"></div>
                {t('upload.fileTip')}
              </div>
            </label>
            {selectedFile && (
              <div className="px-4 py-2 bg-teal-500/20 border border-teal-500/40 rounded-lg">
                <p className="text-sm text-teal-300">{t('upload.fileInfo', { name: selectedFile.name })}</p>
                <p className="text-xs text-gray-400 mt-1">{t('upload.modeInfo')}</p>
              </div>
            )}
          </div>
        </div>

        <div className="glass rounded-xl border border-white/10 p-6 space-y-5">
          <h3 className="text-white font-semibold flex items-center gap-2">
            <Settings2 size={18} className="text-teal-400" />
            {t('upload.config.title')}
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
            <label className="block text-sm text-gray-300 mb-2">{t('upload.config.apiUrl')}</label>
            <div className="flex items-center gap-2">
                        <select 
                          value={llmApiUrl} 
                          onChange={e => {
                            const val = e.target.value;
                            setLlmApiUrl(val);
                            if (val.includes('123.129.219.111')) {
                              setGenFigModel('gemini-3-pro-image-preview');
                            }
                          }}
                          className="flex-1 rounded-lg border border-white/20 bg-black/40 px-4 py-2.5 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-teal-500"
                        >
                          {API_URL_OPTIONS.map((url: string) => (
                            <option key={url} value={url}>{url}</option>
                          ))}
                        </select>
              <QRCodeTooltip>
                <a
                  href={getPurchaseUrl(llmApiUrl)}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="whitespace-nowrap text-[10px] text-teal-300 hover:text-teal-200 hover:underline px-1"
                >
                  {t('upload.config.buyLink')}
                </a>
              </QRCodeTooltip>
            </div>
          </div>
          
          <div>
            <label className="block text-sm text-gray-300 mb-2">{t('upload.config.apiKey')}</label>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder={t('upload.config.apiKeyPlaceholder')}
              className="w-full rounded-lg border border-white/20 bg-black/40 px-4 py-2.5 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-teal-500 placeholder:text-gray-500"
            />
          </div>
          
          <div>
            <label className="block text-sm text-gray-300 mb-2">{t('upload.config.model')}</label>
            <div className="grid grid-cols-2 gap-2">
              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className="w-full rounded-lg border border-white/20 bg-black/40 px-4 py-2.5 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-teal-500"
              >
                {modelOptions.map((option) => (
                  <option key={option} value={option}>{option}</option>
                ))}
              </select>
              <div className="relative group">
                <input
                  type="text"
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  placeholder="自定义模型"
                  className="w-full rounded-lg border border-white/20 bg-black/40 px-4 py-2.5 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-teal-500"
                />
                <div className="pointer-events-none absolute left-full top-1/2 z-20 ml-2 w-56 -translate-y-1/2 rounded-md border border-white/10 bg-black/80 px-2 py-1.5 text-[10px] text-gray-100 opacity-0 shadow-lg transition group-hover:opacity-100">
                  {t('upload.config.customModelTip')}
                </div>
              </div>
            </div>
          </div>
          
          <div>
            <label className="block text-sm text-gray-300 mb-2">{t('upload.config.genModel')}</label>
            <select
              value={genFigModel}
              onChange={(e) => setGenFigModel(e.target.value)}
              disabled={llmApiUrl === 'http://123.129.219.111:3000/v1'}
              className="w-full rounded-lg border border-white/20 bg-black/40 px-4 py-2.5 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-teal-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {genFigModelOptions.map((option) => (
                <option key={option} value={option}>{option}</option>
              ))}
            </select>
            {llmApiUrl === 'http://123.129.219.111:3000/v1' && (
               <p className="text-[10px] text-gray-500 mt-1">此源仅支持 gemini-3-pro</p>
            )}
          </div>
          
          <div>
            <label className="block text-sm text-gray-300 mb-2">{t('upload.config.language')}</label>
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value as 'zh' | 'en')}
              className="w-full rounded-lg border border-white/20 bg-black/40 px-4 py-2.5 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-teal-500"
            >
              <option value="zh">中文 (zh)</option>
              <option value="en">英文 (en)</option>
            </select>
          </div>

          <div>
            <label className="block text-sm text-gray-300 mb-2">{t('upload.config.renderTitle')}</label>
            <select
              value={renderResolution}
              onChange={(e) => setRenderResolution(e.target.value as typeof renderResolution)}
              className="w-full rounded-lg border border-white/20 bg-black/40 px-4 py-2.5 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-teal-500"
            >
              <option value="auto">{t('upload.config.renderOptions.auto')}</option>
              <option value="1080p">{t('upload.config.renderOptions.1080p')}</option>
              <option value="2k">{t('upload.config.renderOptions.2k')}</option>
              <option value="4k">{t('upload.config.renderOptions.4k')}</option>
            </select>
            <p className="text-[11px] text-gray-500 mt-1">{t('upload.config.renderTip')}</p>
          </div>
          
          <div className="border-t border-white/10 pt-4">
            <h4 className="text-sm text-gray-300 mb-3 font-medium">{t('upload.config.styleTitle')}</h4>
          <div className="flex gap-2">
            <button onClick={() => setStyleMode('preset')} className={`flex-1 py-2.5 px-4 rounded-lg text-sm font-medium flex items-center justify-center gap-2 transition-all ${styleMode === 'preset' ? 'bg-gradient-to-r from-cyan-500 to-teal-500 text-white' : 'bg-white/5 text-gray-400 border border-white/10 hover:bg-white/10'}`}>
              <Sparkles size={16} /> {t('upload.config.styleMode.preset')}
            </button>
            <button onClick={() => setStyleMode('reference')} className={`flex-1 py-2.5 px-4 rounded-lg text-sm font-medium flex items-center justify-center gap-2 transition-all ${styleMode === 'reference' ? 'bg-gradient-to-r from-cyan-500 to-teal-500 text-white' : 'bg-white/5 text-gray-400 border border-white/10 hover:bg-white/10'}`}>
              <ImageIcon size={16} /> {t('upload.config.styleMode.reference')}
            </button>
          </div>
          {styleMode === 'preset' && (
            <>
              <div>
                <label className="block text-sm text-gray-300 mb-2">{t('upload.config.stylePreset')}</label>
                <select value={stylePreset} onChange={(e) => setStylePreset(e.target.value as typeof stylePreset)} className="w-full rounded-lg border border-white/20 bg-black/40 px-4 py-2.5 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-teal-500">
                  <option value="modern">{t('upload.config.presets.modern')}</option>
                  <option value="business">{t('upload.config.presets.business')}</option>
                  <option value="academic">{t('upload.config.presets.academic')}</option>
                  <option value="creative">{t('upload.config.presets.creative')}</option>
                </select>
              </div>
              <div>
                <label className="block text-sm text-gray-300 mb-2">{t('upload.config.promptLabel')}</label>
                <textarea value={globalPrompt} onChange={(e) => setGlobalPrompt(e.target.value)} placeholder={t('upload.config.promptPlaceholder')}  rows={3} className="w-full rounded-lg border border-white/20 bg-black/40 px-4 py-2.5 text-sm text-gray-100 outline-none focus:ring-2 focus:ring-teal-500 placeholder:text-gray-500 resize-none" />
              </div>
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="block text-sm text-gray-300">{t('upload.config.promptCardsTitle')}</label>
                  <span className="text-[11px] text-gray-500">{t('upload.config.promptCardsTip')}</span>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {stylePromptCards.map((card) => (
                    <button
                      key={card.title}
                      type="button"
                      onClick={() => {
                        setStyleMode('preset');
                        setGlobalPrompt(card.text);
                      }}
                      className="group text-left rounded-2xl border border-white/15 bg-white/5 px-4 py-3 shadow-[0_10px_30px_rgba(0,0,0,0.25)] backdrop-blur transition-all hover:-translate-y-0.5 hover:border-teal-400/60 hover:bg-white/10"
                    >
                      <div className="text-sm font-semibold text-white mb-1">{card.title}</div>
                      <div className="text-[11px] leading-relaxed text-gray-300 whitespace-pre-line line-clamp-4">
                        {card.text}
                      </div>
                      <div className="mt-2 text-[10px] text-teal-300 opacity-0 transition-opacity group-hover:opacity-100">
                        {t('upload.config.promptCardsUse')}
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            </>
          )}
          {styleMode === 'reference' && (
            <>
              <div>
                <label className="block text-sm text-gray-300 mb-2">{t('upload.config.referenceLabel')}</label>
                {referenceImagePreview ? (
                  <div className="relative">
                    <img src={referenceImagePreview} alt="参考风格" className="w-full h-40 object-cover rounded-lg border border-white/20" />
                    <button onClick={handleRemoveReferenceImage} className="absolute top-2 right-2 p-1.5 rounded-full bg-black/60 text-white hover:bg-red-500 transition-colors"><X size={14} /></button>
                    <p className="text-xs text-teal-300 mt-2">✓ {t('upload.config.referenceUploaded')}</p>
                  </div>
                ) : (
                  <label className="border-2 border-dashed border-white/20 rounded-lg p-6 flex flex-col items-center justify-center text-center gap-2 cursor-pointer hover:border-teal-400 transition-all">
                    <div className="w-12 h-12 rounded-full bg-white/5 flex items-center justify-center"><ImageIcon size={24} className="text-gray-400" /></div>
                    <p className="text-sm text-gray-400">{t('upload.config.referenceUpload')}</p>
                    <input type="file" accept="image/*" className="hidden" onChange={handleReferenceImageChange} />
                  </label>
                )}
              </div>
            </>
          )}
            </div>
          <button onClick={handleUploadAndParse} disabled={!selectedFile || isUploading} className="w-full py-3 rounded-lg bg-gradient-to-r from-cyan-600 to-teal-600 hover:from-cyan-700 hover:to-teal-700 disabled:from-gray-600 disabled:to-gray-700 text-white font-semibold flex items-center justify-center gap-2 transition-all">
            {isUploading ? <><Loader2 size={18} className="animate-spin" /> {t('upload.config.parsing')}</> : <><ArrowRight size={18} /> {t('upload.config.start')}</>}
          </button>

          <div className="flex items-start gap-2 text-xs text-gray-500 mt-3 px-1">
            <Info size={14} className="mt-0.5 text-gray-400 flex-shrink-0" />
            <p>{t('upload.config.tip')}</p>
          </div>

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

      {isValidating && (
        <div className="mt-4 flex items-center gap-2 text-sm text-cyan-300 bg-cyan-500/10 border border-cyan-500/40 rounded-lg px-4 py-3 animate-pulse">
            <Loader2 size={16} className="animate-spin" />
            <p>{t('errors.validating')}</p>
        </div>
      )}

      {error && <div className="mt-4 flex items-center gap-2 text-sm text-red-300 bg-red-500/10 border border-red-500/40 rounded-lg px-4 py-3"><AlertCircle size={16} /> {error}</div>}

      {/* 示例区 */}
      {/* 示例区 */}
      <div className="space-y-8 mt-10">
        <div className="flex items-center justify-end">
            <a
              href="https://wcny4qa9krto.feishu.cn/wiki/VXKiwYndwiWAVmkFU6kcqsTenWh"
              target="_blank"
              rel="noopener noreferrer"
              className="group relative inline-flex items-center gap-2 px-3 py-1 rounded-full bg-black/50 border border-white/10 text-xs font-medium text-white overflow-hidden transition-all hover:border-white/30 hover:shadow-[0_0_15px_rgba(45,212,191,0.5)]"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/20 via-teal-500/20 to-emerald-500/20 opacity-0 group-hover:opacity-100 transition-opacity" />
              <Sparkles size={12} className="text-teal-300 animate-pulse" />
              <span className="bg-gradient-to-r from-cyan-300 via-teal-300 to-emerald-300 bg-clip-text text-transparent group-hover:from-cyan-200 group-hover:via-teal-200 group-hover:to-emerald-200">
                常见问题与更多案例
              </span>
            </a>
        </div>

        {/* 第一组：PPT 增色美化 */}
        <div className="space-y-4">
          <div className="flex items-center gap-3">
            <div className="w-1 h-8 bg-gradient-to-b from-cyan-400 to-teal-500 rounded-full"></div>
            <div>
              <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                <Sparkles size={18} className="text-cyan-400" />
                {t('demo.group1.title')}
              </h3>
              <p className="text-sm text-gray-400">
                {t('demo.group1.desc')}
              </p>
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {/* Demo 1 */}
            <div className="glass rounded-xl border border-white/10 p-4 hover:border-cyan-500/30 transition-all">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <p className="text-xs text-gray-500 mb-2 text-center">{t('demo.group1.original')}</p>
                  <div className="rounded-lg overflow-hidden border border-white/10 aspect-[16/9] bg-white/5">
                    <img src="/ppt2polish/paper2ppt_orgin_1.png" alt="原始PPT示例1" className="w-full h-full object-contain" />
                  </div>
                </div>
                <div>
                  <p className="text-xs text-cyan-400 mb-2 text-center">{t('demo.group1.result')}</p>
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
                  <p className="text-xs text-gray-500 mb-2 text-center">{t('demo.group1.original')}</p>
                  <div className="rounded-lg overflow-hidden border border-white/10 aspect-[16/9] bg-white/5">
                    <img src="/ppt2polish/paper2ppt_orgin_2.png" alt="原始PPT示例2" className="w-full h-full object-contain" />
                  </div>
                </div>
                <div>
                  <p className="text-xs text-cyan-400 mb-2 text-center">{t('demo.group1.result')}</p>
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
                {t('demo.group2.title')}
              </h3>
              <p className="text-sm text-gray-400">
                {t('demo.group2.desc')}
              </p>
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {/* Demo 3 */}
            <div className="glass rounded-xl border border-white/10 p-4 hover:border-purple-500/30 transition-all">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <p className="text-xs text-gray-500 mb-2 text-center">{t('demo.group2.original')}</p>
                  <div className="rounded-lg overflow-hidden border border-white/10 aspect-[16/9] bg-white/5">
                    <img src="/ppt2polish/orgin_3.png" alt="原始PPT示例3" className="w-full h-full object-contain" />
                  </div>
                </div>
                <div>
                  <p className="text-xs text-purple-400 mb-2 text-center">{t('demo.group2.result')}</p>
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
                  <p className="text-xs text-gray-500 mb-2 text-center">{t('demo.group2.original')}</p>
                  <div className="rounded-lg overflow-hidden border border-white/10 aspect-[16/9] bg-white/5">
                    <img src="/ppt2polish/orgin_4.png" alt="原始PPT示例4" className="w-full h-full object-contain" />
                  </div>
                </div>
                <div>
                  <p className="text-xs text-purple-400 mb-2 text-center">{t('demo.group2.result')}</p>
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
        <h2 className="text-2xl font-bold text-white mb-2">{t('outline.title')}</h2>
        <p className="text-gray-400">{t('outline.subtitle')}</p>
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
                    <input type="text" value={editContent.title} onChange={(e) => setEditContent(prev => ({ ...prev, title: e.target.value }))} className="w-full px-3 py-2 rounded-lg bg-black/40 border border-white/20 text-white text-sm outline-none focus:ring-2 focus:ring-teal-500" placeholder={t('outline.edit.titlePlaceholder')} />
                    <textarea value={editContent.layout_description} onChange={(e) => setEditContent(prev => ({ ...prev, layout_description: e.target.value }))} rows={2} className="w-full px-3 py-2 rounded-lg bg-black/40 border border-white/20 text-white text-sm outline-none focus:ring-2 focus:ring-teal-500 resize-none" placeholder={t('outline.edit.layoutPlaceholder')} />
                    <div className="space-y-2">
                      {editContent.key_points.map((point, idx) => (
                        <div key={idx} className="flex gap-2">
                          <input type="text" value={point} onChange={(e) => handleKeyPointChange(idx, e.target.value)} className="flex-1 px-3 py-2 rounded-lg bg-black/40 border border-white/20 text-white text-sm outline-none focus:ring-2 focus:ring-teal-500" placeholder={`${t('outline.edit.pointPlaceholder')} ${idx + 1}`} />
                          <button onClick={() => handleRemoveKeyPoint(idx)} className="p-2 rounded-lg hover:bg-red-500/20 text-gray-400 hover:text-red-400"><Trash2 size={14} /></button>
                        </div>
                      ))}
                      <button onClick={handleAddKeyPoint} className="px-3 py-1.5 rounded-lg bg-white/5 border border-dashed border-white/20 text-gray-400 hover:text-teal-400 hover:border-teal-400 text-sm w-full">{t('outline.edit.addPoint')}</button>
                    </div>
                    <div className="flex gap-2 pt-2">
                      <button onClick={handleEditSave} className="px-3 py-1.5 rounded-lg bg-teal-500 text-white text-sm flex items-center gap-1"><Check size={14} /> {t('outline.edit.save')}</button>
                      <button onClick={handleEditCancel} className="px-3 py-1.5 rounded-lg bg-white/10 text-gray-300 text-sm">{t('outline.edit.cancel')}</button>
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
        <button onClick={() => setCurrentStep('upload')} className="px-6 py-2.5 rounded-lg border border-white/20 text-gray-300 hover:bg-white/10 flex items-center gap-2 transition-all"><ArrowLeft size={18} /> {t('outline.back')}</button>
        <button onClick={handleConfirmOutline} className="px-6 py-2.5 rounded-lg bg-gradient-to-r from-cyan-600 to-teal-600 hover:from-cyan-700 hover:to-teal-700 text-white font-semibold flex items-center gap-2 transition-all">{t('outline.confirm')} <ArrowRight size={18} /></button>
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
            <h2 className="text-2xl font-bold text-white mb-2">{t('beautify.initTitle')}</h2>
            <p className="text-gray-400">{t('beautify.initDesc')}</p>
          </div>
          <div className="glass rounded-xl border border-white/10 p-12 flex flex-col items-center justify-center">
            <Loader2 size={48} className="text-teal-400 animate-spin mb-4" />
            <p className="text-teal-300 text-lg font-medium mb-2">{t('beautify.loadingTitle')}</p>
            <p className="text-gray-400 text-sm">{t('beautify.loadingDesc')}</p>
          </div>
        </div>
      );
    }
    
    return (
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-6">
          <h2 className="text-2xl font-bold text-white mb-2">{t('beautify.title')}</h2>
          <p className="text-gray-400">{t('beautify.pageInfo', { current: currentSlideIndex + 1, total: outlineData.length, title: currentSlide?.title })}</p>
          <p className="text-xs text-gray-500 mt-1">{t('beautify.modeInfo')}</p>
        </div>
        <div className="mb-6">
          <div className="flex gap-1">{beautifyResults.map((result, index) => (<div key={result.slideId} className={`flex-1 h-2 rounded-full transition-all ${result.status === 'done' ? 'bg-teal-400' : result.status === 'processing' ? 'bg-gradient-to-r from-cyan-400 to-teal-400 animate-pulse' : index === currentSlideIndex ? 'bg-teal-400/50' : 'bg-white/10'}`} />))}</div>
        </div>
        <div className="glass rounded-xl border border-white/10 p-6 mb-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="text-sm text-gray-400 mb-3 flex items-center gap-2"><Eye size={14} /> {t('beautify.original')}</h4>
              <div className="rounded-lg overflow-hidden border border-white/10 aspect-[16/9] bg-white/5 flex items-center justify-center">{currentResult?.beforeImage ? <img src={currentResult.beforeImage} alt="Before" className="max-w-full max-h-full object-contain" /> : <Loader2 size={24} className="text-gray-500 animate-spin" />}</div>
            </div>
            <div>
              <h4 className="text-sm text-gray-400 mb-3 flex items-center gap-2"><Sparkles size={14} className="text-teal-400" /> {t('beautify.result')}</h4>
              <div className="rounded-lg overflow-hidden border border-teal-500/30 aspect-[16/9] bg-gradient-to-br from-cyan-500/10 to-teal-500/10 flex items-center justify-center">{isBeautifying ? <div className="text-center"><Loader2 size={32} className="text-teal-400 animate-spin mx-auto mb-2" /><p className="text-sm text-teal-300">{t('beautify.processing')}</p></div> : currentResult?.afterImage ? <img src={currentResult.afterImage} alt="After" className="max-w-full max-h-full object-contain" /> : <span className="text-gray-500">{t('beautify.waiting')}</span>}</div>
            </div>
          </div>
        </div>

        {/* 版本历史组件 */}
        {currentResult?.versionHistory && currentResult.versionHistory.length > 0 && (
          <VersionHistory
            versions={currentResult.versionHistory}
            currentVersionIndex={currentResult.currentVersionIndex}
            onRevert={handleRevertToVersion}
            isGenerating={isBeautifying}
          />
        )}

        <div className="glass rounded-xl border border-white/10 p-4 mb-6">
          <div className="flex items-center gap-3"><MessageSquare size={18} className="text-teal-400" /><input type="text" value={slidePrompt} onChange={(e) => setSlidePrompt(e.target.value)} placeholder={t('beautify.regeneratePlaceholder')} className="flex-1 bg-transparent border-none outline-none text-white text-sm placeholder:text-gray-500" /><button onClick={handleRegenerateSlide} disabled={isBeautifying || !slidePrompt.trim()} className="px-4 py-2 rounded-lg bg-white/10 hover:bg-white/20 text-gray-300 text-sm flex items-center gap-2 disabled:opacity-50 transition-all"><RefreshCw size={14} /> {t('beautify.regenerate')}</button></div>
        </div>
        <div className="flex justify-between">
          <button onClick={() => setCurrentStep('upload')} className="px-6 py-2.5 rounded-lg border border-white/20 text-gray-300 hover:bg-white/10 flex items-center gap-2 transition-all"><ArrowLeft size={18} /> {t('beautify.back')}</button>
          <div className="flex gap-3">
            <button 
              onClick={() => {
                if (currentSlideIndex > 0) {
                  setCurrentSlideIndex(currentSlideIndex - 1);
                  setSlidePrompt('');
                }
              }}
              disabled={currentSlideIndex === 0 || isBeautifying}
              className="px-6 py-2.5 rounded-lg border border-white/20 text-gray-300 hover:bg-white/10 flex items-center gap-2 transition-all disabled:opacity-30"
            >
              <ArrowLeft size={18} /> {t('beautify.prev')}
            </button>
            <button onClick={handleConfirmSlide} disabled={isBeautifying || !currentResult?.afterImage} className="px-6 py-2.5 rounded-lg bg-gradient-to-r from-cyan-600 to-teal-600 hover:from-cyan-700 hover:to-teal-700 text-white font-semibold flex items-center gap-2 transition-all disabled:opacity-50"><CheckCircle2 size={18} /> {t('beautify.next')}</button>
          </div>
        </div>
      </div>
    );
  };

  // ============== Step 4: 完成下载界面 ==============
  const renderCompleteStep = () => (
    <div className="max-w-2xl mx-auto text-center">
      <div className="mb-8"><div className="w-20 h-20 rounded-full bg-gradient-to-br from-cyan-500 to-teal-500 flex items-center justify-center mx-auto mb-4"><CheckCircle2 size={40} className="text-white" /></div><h2 className="text-2xl font-bold text-white mb-2">{t('complete.title')}</h2></div>
      <div className="glass rounded-xl border border-white/10 p-6 mb-6">
        <h3 className="text-white font-semibold mb-4">{t('complete.overview')}</h3>
        <div className="grid grid-cols-4 gap-2">{beautifyResults.map((result, index) => (<div key={result.slideId} className="p-3 rounded-lg border bg-teal-500/20 border-teal-500/40"><p className="text-sm text-white">{t('complete.page', { index: index + 1 })}</p><p className="text-xs text-teal-300">{t('complete.status')}</p></div>))}</div>
      </div>
      {!(downloadUrl || pdfDownloadUrl) ? (
        <button onClick={handleGenerateFinal} disabled={isGeneratingFinal} className="px-8 py-3 rounded-lg bg-gradient-to-r from-cyan-600 to-teal-600 hover:from-cyan-700 hover:to-teal-700 text-white font-semibold flex items-center justify-center gap-2 mx-auto transition-all">
          {isGeneratingFinal ? <><Loader2 size={18} className="animate-spin" /> {t('complete.generating')}</> : <><Sparkles size={18} /> {t('complete.generateFinal')}</>}
        </button>
      ) : (
        <div className="space-y-4">
          <div className="flex gap-4 justify-center">
            {downloadUrl && (
              <button onClick={handleDownload} className="px-6 py-3 rounded-lg bg-gradient-to-r from-cyan-500 to-teal-500 hover:from-cyan-600 hover:to-teal-600 text-white font-semibold flex items-center gap-2 transition-all">
                <Download size={18} /> {t('complete.downloadPptx')}
              </button>
            )}
            {pdfDownloadUrl && (
              <a href={pdfDownloadUrl} target="_blank" rel="noopener noreferrer" className="px-6 py-3 rounded-lg bg-gradient-to-r from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600 text-white font-semibold flex items-center gap-2 transition-all">
                <Download size={18} /> {t('complete.downloadPdf')}
              </a>
            )}
          </div>

          <div>
            <button onClick={() => { setCurrentStep('upload'); setSelectedFile(null); setOutlineData([]); setBeautifyResults([]); setDownloadUrl(null); setPdfDownloadUrl(null); }} className="text-sm text-gray-400 hover:text-white transition-colors">
              <RotateCcw size={14} className="inline mr-1" /> {t('complete.new')}
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
               <a href="https://github.com/OpenDCAI/Paper2Any" target="_blank" rel="noopener noreferrer" className="block w-full py-1.5 px-3 rounded bg-white/5 hover:bg-white/10 text-xs text-teal-300 truncate transition-colors border border-white/5 text-center">
                 ✨如果本项目对你有帮助，可以点个star嘛～
               </a>
               <div className="flex gap-2">
                 <a href="https://github.com/OpenDCAI/Paper2Any" target="_blank" rel="noopener noreferrer" className="flex-1 inline-flex items-center justify-center gap-1 px-2 py-1.5 bg-white/95 hover:bg-white text-gray-900 rounded-full text-[10px] font-semibold transition-all hover:scale-105 shadow-lg">
                   <Github size={10} />
                   <span>Agent</span>
                   <span className="bg-gray-200 text-gray-800 px-1 py-0.5 rounded-full text-[9px] flex items-center gap-0.5"><Star size={7} fill="currentColor" /> {stars.agent || 'Star'}</span>
                 </a>
                 <a href="https://github.com/OpenDCAI/DataFlow" target="_blank" rel="noopener noreferrer" className="flex-1 inline-flex items-center justify-center gap-1 px-2 py-1.5 bg-white/95 hover:bg-white text-gray-900 rounded-full text-[10px] font-semibold transition-all hover:scale-105 shadow-lg">
                   <Github size={10} />
                   <span>Core</span>
                   <span className="bg-gray-200 text-gray-800 px-1 py-0.5 rounded-full text-[9px] flex items-center gap-0.5"><Star size={7} fill="currentColor" /> {stars.dataflow || 'Star'}</span>
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
                <span className="text-xs font-bold text-white">{t('app.githubProject', { ns: 'common' })}</span>
              </a>
              
              <span className="text-sm font-medium text-white">
                {t('app.exploreMore', { ns: 'common' })}
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
                href="https://github.com/OpenDCAI/Paper2Any"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-1.5 bg-white/95 hover:bg-white text-gray-900 rounded-full text-xs font-semibold transition-all hover:scale-105 shadow-lg"
              >
                <Github size={14} />
                <span>Paper2Any</span>
                <span className="bg-gray-200 text-gray-800 px-1.5 py-0.5 rounded-full text-[10px] flex items-center gap-0.5"><Star size={8} fill="currentColor" /> {stars.agent || 'Star'}</span>
                <span className="bg-pink-600 text-white px-2 py-0.5 rounded-full text-[10px]">NEW</span>
              </a>

              <a
                href="https://github.com/OpenDCAI/DataFlex"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-1.5 bg-white/95 hover:bg-white text-gray-900 rounded-full text-xs font-semibold transition-all hover:scale-105 shadow-lg"
              >
                <Github size={14} />
                <span>DataFlex</span>
                <span className="bg-gray-200 text-gray-800 px-1.5 py-0.5 rounded-full text-[10px] flex items-center gap-0.5"><Star size={8} fill="currentColor" /> {stars.dataflex || 'Star'}</span>
                <span className="bg-sky-600 text-white px-2 py-0.5 rounded-full text-[10px]">NEW</span>
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
