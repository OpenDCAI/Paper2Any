import React from 'react';
import { ImageIcon, MessageSquare, Loader2, RotateCcw, Download, ExternalLink } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { GraphType, FigureComplex, Language } from './types';
import { API_KEY } from '../../config/api';
import { JSON_API } from './constants';

interface PreviewSectionProps {
  graphType: GraphType;
  graphStep: 'input' | 'preview' | 'done';
  previewImgUrl: string | null;
  setPreviewImgUrl: (url: string | null) => void;
  pptUrl?: string | null; // 新增
  setPptUrl: (url: string | null) => void;
  setGraphStep: (step: 'input' | 'preview' | 'done') => void;
  editPrompt: string;
  setEditPrompt: (prompt: string) => void;
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  model: string;
  llmApiUrl: string;
  apiKey: string;
  email: string;
  figureComplex: FigureComplex;
  language: Language;
  showDrawioButton?: boolean;
  drawioLoading?: boolean;
  onConvertToDrawio?: () => void;
  drawioLabel?: string;
  onReset?: () => void;
}

const PreviewSection: React.FC<PreviewSectionProps> = ({
  graphType,
  graphStep,
  previewImgUrl,
  setPreviewImgUrl,
  pptUrl,
  setPptUrl,
  setGraphStep,
  editPrompt,
  setEditPrompt,
  isLoading,
  setIsLoading,
  setError,
  model,
  llmApiUrl,
  apiKey,
  email,
  figureComplex,
  language,
  showDrawioButton,
  drawioLoading,
  onConvertToDrawio,
  drawioLabel,
  onReset,
}) => {
  const [imgError, setImgError] = React.useState(false);

  // 当 previewImgUrl 改变时重置错误状态
  React.useEffect(() => {
    setImgError(false);
  }, [previewImgUrl]);

  // 允许 graphStep 为 'done' 时显示，只要 previewImgUrl 存在
  if (graphType !== 'model_arch' || graphStep === 'input' || !previewImgUrl) return null;
  return (
    <div className="portal-panel-dark relative mb-8 overflow-hidden rounded-xl border border-primary-100 p-6 animate-fade-in">
      {/* 装饰光效 */}
      <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-primary-500/70 via-primary-400/45 to-amber-500/55"></div>
      
      <div className="flex justify-between items-center mb-4">
        <h3 className="flex items-center gap-2 text-lg font-semibold text-[var(--text-primary)]">
          <ImageIcon size={20} className="text-primary-400" />
          模型结构图预览
        </h3>
        
        {/* 新增：下载图片按钮 */}
        <a
          href={previewImgUrl}
          download={`model_arch_preview_${Date.now()}.png`}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 rounded-lg border border-primary-100 bg-white/82 px-3 py-1.5 text-xs text-primary-800 transition-colors hover:bg-white"
        >
          <Download size={14} />
          下载图片
        </a>
      </div>
      
      <div className="portal-preview-canvas mb-6 flex min-h-[300px] w-full items-center justify-center overflow-hidden rounded-xl p-4">
        {imgError ? (
          <div className="flex flex-col items-center justify-center p-4 text-[var(--text-secondary)]">
            <ImageIcon size={48} className="mb-4 opacity-50" />
            <p className="mb-2 font-medium">图片加载失败</p>
            <p className="max-w-md break-all text-center text-xs text-[#8b726b]">{previewImgUrl}</p>
            <a 
              href={previewImgUrl} 
              target="_blank" 
              rel="noopener noreferrer"
              className="mt-4 rounded-lg border border-primary-100 bg-white/86 px-4 py-2 text-sm text-primary-800 transition-colors hover:bg-white"
            >
              尝试在新标签页打开
            </a>
          </div>
        ) : (
          <img
            src={previewImgUrl}
            alt="模型结构图预览"
            className="max-w-full h-auto object-contain max-h-[600px] rounded-lg shadow-2xl"
            onError={() => setImgError(true)}
          />
        )}
      </div>
      
      <div className="flex flex-col md:flex-row gap-4 items-end">
        <div className="flex-1 w-full">
          <label className="mb-2 flex items-center gap-2 text-sm text-[var(--text-secondary)]">
            <MessageSquare size={16} />
            不满意？输入提示词微调重绘
          </label>
          <div className="relative">
            <input
              type="text"
              value={editPrompt}
              onChange={e => setEditPrompt(e.target.value)}
              placeholder="例如：把背景改成深色，增加一些连接线..."
              className="portal-input-soft w-full rounded-xl px-4 py-3 pr-24 text-sm outline-none focus:ring-2 focus:ring-primary-500"
            />
            <button
              type="button"
              onClick={async () => {
                if (!editPrompt.trim() || !previewImgUrl) return;
                
                try {
                  setIsLoading(true);
                  setError(null);
                  
                  const formData = new FormData();
                  formData.append('img_gen_model_name', model);
                  formData.append('chat_api_url', llmApiUrl.trim());
                  formData.append('api_key', apiKey.trim());
                  formData.append('input_type', 'FIGURE'); // 使用 FIGURE 模式触发
                  formData.append('email', email);
                  formData.append('graph_type', 'model_arch');
                  formData.append('figure_complex', figureComplex);
                  formData.append('language', language);
                  
                  // 传入上一次的图片路径作为 prev_image
                  // 注意：后端 wa_paper2figure 会在 input_type=FIGURE 且有 edit_prompt 时进入 paper2fig_image_only
                  // 此时 input_content (即这里的 text/file) 会被当作 prev_image 使用
                  formData.append('text', previewImgUrl); 
                  
                  // 传入修改提示词
                  formData.append('edit_prompt', editPrompt.trim());
                  
                  const res = await fetch(JSON_API, {
                    method: 'POST',
                    headers: { 'X-API-Key': API_KEY },
                    body: formData,
                  });
                  
                  if (!res.ok) throw new Error('重绘失败');
                  
                  const data = await res.json();
                  if (!data.success) throw new Error('重绘失败');
                  
                  // 更新预览图
                  let newImg = null;
                  const files = data.all_output_files ?? [];
                  const figPngs = files.filter((f: string) => /fig_/i.test(f) && /\.(png|jpg)$/i.test(f));
                  if (figPngs.length > 0) {
                    newImg = figPngs[0];
                  } else {
                    // 如果没找到 fig_ 前缀的，找任意 png
                    const pngs = files.filter((f: string) => /\.(png|jpg)$/i.test(f));
                    if (pngs.length > 0) newImg = pngs[0];
                  }
                  
                  if (newImg) {
                    // 添加时间戳防止缓存
                    setPreviewImgUrl(`${newImg}?t=${Date.now()}`);
                    setEditPrompt(''); // 清空输入框
                  } else {
                    throw new Error('未获取到新生成的图片');
                  }
                  
                } catch (e) {
                  const msg = e instanceof Error ? e.message : '重绘失败';
                  setError(msg);
                } finally {
                  setIsLoading(false);
                }
              }}
              disabled={isLoading || !editPrompt.trim()}
              className="absolute right-2 top-1.5 rounded-lg bg-primary-500 px-3 py-1.5 text-xs text-white transition-colors hover:bg-primary-600 disabled:opacity-50"
            >
              {isLoading ? <Loader2 size={12} className="animate-spin" /> : '重新生成'}
            </button>
          </div>
        </div>
        
        <div className="flex flex-wrap gap-3 w-full md:w-auto">
          <button
            type="button"
            onClick={() => {
              setGraphStep('input');
              setPreviewImgUrl(null);
              setPptUrl(null);
              setEditPrompt('');
              onReset?.();
            }}
            className="flex items-center justify-center gap-2 rounded-xl border border-primary-100 bg-white/78 px-5 py-3 text-sm text-[var(--text-primary)] transition-all hover:bg-white"
          >
            <RotateCcw size={16} />
            放弃
          </button>
          {showDrawioButton && (
            <button
              type="button"
              onClick={onConvertToDrawio}
              disabled={drawioLoading || isLoading}
              className="px-6 py-3 rounded-xl bg-gradient-to-r from-primary-700 to-amber-600 hover:from-primary-600 hover:to-amber-500 text-white font-semibold flex items-center justify-center gap-2 shadow-lg shadow-primary-900/25 transition-all disabled:opacity-60 disabled:cursor-not-allowed min-w-[200px]"
            >
              {drawioLoading ? <Loader2 size={18} className="animate-spin" /> : <ExternalLink size={18} />}
              {drawioLabel || '转成 DrawIO 在线编辑'}
            </button>
          )}
          {graphStep === 'done' && pptUrl ? (
            <button
              type="button"
              onClick={async () => {
                if (!pptUrl) return;
                try {
                  // 如果当前页面是 HTTPS，但资源是 HTTP，尝试升级协议以避免 Mixed Content 错误
                  let fetchUrl = pptUrl;
                  if (window.location.protocol === 'https:' && fetchUrl.startsWith('http:')) {
                    fetchUrl = fetchUrl.replace(/^http:/, 'https:');
                  }

                  // 简单的加载提示
                  const btn = document.activeElement as HTMLButtonElement;
                  const originalText = btn.innerText;
                  if (btn) btn.innerText = '下载中...';

                  const res = await fetch(fetchUrl);
                  if (!res.ok) throw new Error('下载失败');
                  
                  const blob = await res.blob();
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement('a');
                  a.href = url;
                  a.download = `paper2figure_ppt_${Date.now()}.pptx`;
                  document.body.appendChild(a);
                  a.click();
                  document.body.removeChild(a);
                  URL.revokeObjectURL(url);
                  
                  if (btn) btn.innerText = originalText;
                } catch (e) {
                  console.error('Download failed', e);
                  alert('下载失败，请尝试手动复制链接或检查网络。');
                  // 恢复按钮文本
                  const btn = document.activeElement as HTMLButtonElement;
                  if (btn && btn.innerText === '下载中...') btn.innerText = '下载 PPT';
                }
              }}
              className="px-6 py-3 rounded-xl bg-gradient-to-r from-primary-700 to-amber-600 hover:from-primary-600 hover:to-amber-500 text-white font-semibold flex items-center justify-center gap-2 shadow-lg shadow-primary-900/25 transition-all min-w-[180px]"
            >
              <Download size={18} />
              下载 PPT
            </button>
          ) : (
            <button
              type="button"
              onClick={async () => {
                  if (!previewImgUrl) return;
                  
                  // 触发 Step 2：将图片转 PPT
                  try {
                    setIsLoading(true);
                    setError(null);
                    
                    const formData = new FormData();
                    formData.append('img_gen_model_name', model);
                    formData.append('chat_api_url', llmApiUrl.trim());
                    formData.append('api_key', apiKey.trim());
                    formData.append('input_type', 'FIGURE'); 
                    formData.append('email', email);
                    formData.append('graph_type', 'model_arch');
                    formData.append('figure_complex', figureComplex);
                    formData.append('language', language);
                    formData.append('text', previewImgUrl); // 复用 text 传路径
                    
                    const res = await fetch(JSON_API, {
                      method: 'POST',
                      headers: { 'X-API-Key': API_KEY },
                      body: formData,
                    });
                    
                    if (!res.ok) throw new Error('PPT 生成失败');
                    
                    const data = await res.json();
                    if (!data.success) throw new Error('PPT 生成失败');
                    
                    let finalPpt = data.ppt_filename;
                    if (!finalPpt && data.all_output_files) {
                      finalPpt = data.all_output_files.find((f: string) => /\.pptx$/i.test(f));
                    }
                    
                    if (finalPpt) {
                      setPptUrl(finalPpt);
                      // window.open(finalPpt, '_blank'); // 用户可能会被拦截，改为显示下载按钮
                      setGraphStep('done');
                    } else {
                      throw new Error('未找到生成的 PPT 文件');
                    }
                    
                  } catch (e) {
                    const msg = e instanceof Error ? e.message : '生成失败';
                    setError(msg);
                  } finally {
                    setIsLoading(false);
                  }
              }}
              disabled={isLoading}
              className="px-6 py-3 rounded-xl bg-gradient-to-r from-primary-700 to-primary-500 hover:from-primary-600 hover:to-primary-400 text-white font-semibold flex items-center justify-center gap-2 shadow-lg shadow-primary-900/25 transition-all disabled:opacity-50 disabled:cursor-not-allowed min-w-[180px]"
            >
              {isLoading ? <Loader2 size={18} className="animate-spin" /> : <Download size={18} />}
              确认并转 PPT
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default PreviewSection;
