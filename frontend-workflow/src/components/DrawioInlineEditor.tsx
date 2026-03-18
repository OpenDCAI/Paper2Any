import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Download } from 'lucide-react';

const DRAWIO_ORIGINS = new Set(['https://embed.diagrams.net', 'https://app.diagrams.net']);
const DRAWIO_EXPORT_TIMEOUT_MS = 5000;
const DRAWIO_ANIMATE_STEP_MS = 60;
const DRAWIO_ANIMATE_MAX_CELLS = 240;
const DRAWIO_ANIMATE_LARGE_BATCH = 5;

interface DrawioInlineEditorProps {
  title?: string;
  subtitle?: string;
  xmlContent: string;
  onXmlChange?: (xml: string) => void;
  height?: string;
  loadingLabel?: string;
}

const DrawioInlineEditor: React.FC<DrawioInlineEditorProps> = ({
  title = 'DrawIO 在线编辑 / Editor',
  subtitle = '在线编辑后可复制或下载 .drawio / Edit online and download .drawio',
  xmlContent,
  onXmlChange,
  height = '560px',
  loadingLabel,
}) => {
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const lastLoadedXmlRef = useRef('');
  const [drawioReady, setDrawioReady] = useState(false);
  const [exportFormat, setExportFormat] = useState<'drawio' | 'png' | 'svg'>('drawio');
  const [exportFilename, setExportFilename] = useState('diagram');
  const [isExporting, setIsExporting] = useState(false);
  const isAnimatingRef = useRef(false);
  const animationTokenRef = useRef(0);
  const pendingExportRef = useRef<{
    resolve: ((data: string) => void) | null;
    reject: ((error: Error) => void) | null;
    format: 'xml' | 'png' | 'svg' | null;
  }>({ resolve: null, reject: null, format: null });

  const postToDrawio = useCallback((payload: Record<string, unknown>) => {
    const frame = iframeRef.current?.contentWindow;
    if (!frame) return;
    frame.postMessage(JSON.stringify(payload), '*');
  }, []);

  const requestDrawioFit = useCallback(() => {
    postToDrawio({ action: 'zoom', zoom: 'fit' });
  }, [postToDrawio]);

  const parseXmlForAnimation = useCallback((xml: string) => {
    try {
      const parser = new DOMParser();
      const doc = parser.parseFromString(xml, 'text/xml');
      if (doc.querySelector('parsererror')) return null;
      const root =
        doc.querySelector('mxGraphModel > root') ||
        doc.querySelector('root');
      if (!root) return null;

      const rootCells = Array.from(root.children).filter(
        node => node.nodeName === 'mxCell'
      ) as Element[];
      if (!rootCells.length) return null;

      const baseCells = rootCells.filter(cell => {
        const id = cell.getAttribute('id');
        return id === '0' || id === '1';
      });
      const normalCells = rootCells.filter(cell => {
        const id = cell.getAttribute('id');
        return id !== '0' && id !== '1';
      });
      const nonEdges = normalCells.filter(cell => cell.getAttribute('edge') !== '1');
      const edges = normalCells.filter(cell => cell.getAttribute('edge') === '1');
      const orderedCells = [...nonEdges, ...edges];

      return { doc, baseCells, orderedCells };
    } catch {
      return null;
    }
  }, []);

  const buildXmlWithCells = useCallback((sourceDoc: Document, cells: Element[]) => {
    const docClone = sourceDoc.cloneNode(true) as Document;
    const root =
      docClone.querySelector('mxGraphModel > root') ||
      docClone.querySelector('root');
    if (!root) return '';
    while (root.firstChild) {
      root.removeChild(root.firstChild);
    }
    for (const cell of cells) {
      root.appendChild(docClone.importNode(cell, true));
    }
    return new XMLSerializer().serializeToString(docClone);
  }, []);

  const animateDrawioLoad = useCallback(
    async (xml: string) => {
      const parsed = parseXmlForAnimation(xml);
      if (!parsed) {
        postToDrawio({ action: 'load', xml, autosave: 1 });
        lastLoadedXmlRef.current = xml;
        setTimeout(() => requestDrawioFit(), 120);
        return;
      }

      const { doc, baseCells, orderedCells } = parsed;
      const total = orderedCells.length;
      const batchSize =
        total > DRAWIO_ANIMATE_MAX_CELLS ? DRAWIO_ANIMATE_LARGE_BATCH : 1;
      const token = ++animationTokenRef.current;
      isAnimatingRef.current = true;

      for (let i = 0; i < total; i += batchSize) {
        if (animationTokenRef.current !== token) return;
        const subset = orderedCells.slice(0, Math.min(i + batchSize, total));
        const autosave = i + batchSize >= total ? 1 : 0;
        const partialXml = buildXmlWithCells(doc, [...baseCells, ...subset]);
        if (!partialXml) break;
        postToDrawio({ action: 'load', xml: partialXml, autosave });
        setTimeout(() => requestDrawioFit(), 80);
        await new Promise(resolve => setTimeout(resolve, DRAWIO_ANIMATE_STEP_MS));
      }

      if (animationTokenRef.current === token) {
        lastLoadedXmlRef.current = xml;
        isAnimatingRef.current = false;
        setTimeout(() => requestDrawioFit(), 120);
      }
    },
    [buildXmlWithCells, parseXmlForAnimation, postToDrawio, requestDrawioFit]
  );

  const requestDrawioExport = useCallback(
    (format: 'xml' | 'png' | 'svg') => {
      if (!drawioReady) {
        return Promise.reject(new Error('Draw.io not ready'));
      }

      return new Promise<string>((resolve, reject) => {
        pendingExportRef.current = { resolve, reject, format };
        postToDrawio({ action: 'export', format });
        window.setTimeout(() => {
          if (pendingExportRef.current.resolve === resolve) {
            pendingExportRef.current = { resolve: null, reject: null, format: null };
            reject(new Error('Export timeout'));
          }
        }, DRAWIO_EXPORT_TIMEOUT_MS);
      });
    },
    [drawioReady, postToDrawio],
  );

  const syncXmlFromDrawio = useCallback(async () => {
    if (!drawioReady) return xmlContent;
    try {
      const exported = await requestDrawioExport('xml');
      if (exported && exported.includes('<mxfile')) {
        return exported;
      }
    } catch (e) {
      console.warn('Failed to sync XML from draw.io:', e);
    }
    return xmlContent;
  }, [drawioReady, xmlContent, requestDrawioExport]);

  const downloadXmlFile = useCallback((xml: string, filename: string) => {
    const blob = new Blob([xml], { type: 'application/xml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 100);
  }, []);

  const downloadExportData = useCallback((data: string, format: 'png' | 'svg', filename: string) => {
    let url = '';
    let shouldRevoke = false;
    const trimmed = data.trim();

    if (trimmed.startsWith('data:')) {
      url = trimmed;
    } else if (format === 'png') {
      url = `data:image/png;base64,${trimmed}`;
    } else if (trimmed.startsWith('<svg')) {
      const blob = new Blob([trimmed], { type: 'image/svg+xml' });
      url = URL.createObjectURL(blob);
      shouldRevoke = true;
    } else {
      const blob = new Blob([trimmed], { type: 'image/svg+xml' });
      url = URL.createObjectURL(blob);
      shouldRevoke = true;
    }

    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    if (shouldRevoke) {
      setTimeout(() => URL.revokeObjectURL(url), 100);
    }
  }, []);

  const handleExport = useCallback(async () => {
    if (!xmlContent || isExporting) return;
    setIsExporting(true);

    const trimmedName = exportFilename.trim();
    const safeName = (trimmedName || 'diagram').replace(/[\\/:*?"<>|]/g, '_');

    if (exportFormat === 'drawio') {
      const latestXml = await syncXmlFromDrawio();
      if (latestXml && latestXml.includes('<mxfile')) {
        downloadXmlFile(latestXml, `${safeName}.drawio`);
      }
      setIsExporting(false);
      return;
    }

    try {
      const exportData = await requestDrawioExport(exportFormat);
      if (exportData) {
        downloadExportData(exportData, exportFormat, `${safeName}.${exportFormat}`);
      }
    } catch (e) {
      console.warn('Export failed:', e);
    } finally {
      setIsExporting(false);
    }
  }, [
    xmlContent,
    isExporting,
    exportFormat,
    exportFilename,
    syncXmlFromDrawio,
    downloadXmlFile,
    downloadExportData,
    requestDrawioExport,
  ]);

  useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      if (!DRAWIO_ORIGINS.has(event.origin) || typeof event.data !== 'string') return;
      let message: { event?: string; xml?: string; data?: string } = {};
      try {
        message = JSON.parse(event.data) as { event?: string; xml?: string; data?: string };
      } catch {
        return;
      }

      if (message.event === 'init' || message.event === 'ready') {
        setDrawioReady(true);
        postToDrawio({
          action: 'configure',
          config: {
            sidebar: false,
            format: false,
            layers: false,
            menubar: false,
            toolbar: false,
            status: false,
          },
        });
        return;
      }

      if ((message.event === 'save' || message.event === 'autosave') && typeof message.xml === 'string') {
        if (isAnimatingRef.current) return;
        lastLoadedXmlRef.current = message.xml;
        if (onXmlChange) onXmlChange(message.xml);
        return;
      }

      if (message.event === 'export' && pendingExportRef.current.resolve && typeof message.data === 'string') {
        const resolver = pendingExportRef.current.resolve;
        pendingExportRef.current = { resolve: null, reject: null, format: null };
        resolver(message.data);
      }
    };

    window.addEventListener('message', handleMessage);
    return () => window.removeEventListener('message', handleMessage);
  }, [onXmlChange, postToDrawio]);

  useEffect(() => {
    if (!drawioReady || !xmlContent) return;
    if (xmlContent === lastLoadedXmlRef.current) return;
    animateDrawioLoad(xmlContent);
  }, [drawioReady, xmlContent, animateDrawioLoad]);

  return (
    <div className="rounded-2xl border border-white/12 bg-white/8 p-4 backdrop-blur-xl shadow-[0_20px_60px_rgba(56,16,28,0.28)]">
      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div>
          <h3 className="text-sm font-semibold text-[#fff3ee]">{title}</h3>
          <p className="text-xs text-[#cfb0aa]">{subtitle}</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className={`text-[11px] ${drawioReady ? 'text-amber-300' : 'text-[#c7a7a0]'}`}>
            {drawioReady ? '就绪 / Ready' : (loadingLabel ?? '加载中 / Loading')}
          </span>
          <select
            value={exportFormat}
            onChange={(event) => setExportFormat(event.target.value as 'drawio' | 'png' | 'svg')}
            className="rounded-lg border border-white/12 bg-white/12 px-2 py-1 text-[11px] text-[#fff5f0] outline-none focus:ring-2 focus:ring-primary-500/20"
          >
            <option value="drawio">.drawio</option>
            <option value="png">.png</option>
            <option value="svg">.svg</option>
          </select>
          <div className="relative flex items-center">
            <input
              value={exportFilename}
              onChange={(event) => setExportFilename(event.target.value)}
              className="rounded-lg border border-white/12 bg-white/12 px-2 py-1 pr-10 text-[11px] text-[#fff5f0] outline-none focus:ring-2 focus:ring-primary-500/20"
              placeholder="diagram"
            />
            <span className="pointer-events-none absolute right-2 text-[10px] text-[#c7a7a0]">.{exportFormat}</span>
          </div>
          <button
            type="button"
            onClick={handleExport}
            disabled={isExporting || !xmlContent}
            className="inline-flex items-center gap-2 rounded-xl border border-white/12 bg-white/12 px-3 py-2 text-xs text-[#fff5f0] hover:bg-white/18 transition-all disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isExporting ? <span className="h-3 w-3 animate-spin rounded-full border border-white/30 border-t-white" /> : <Download size={14} />}
            导出 / Export
          </button>
        </div>
      </div>
      <div className="mt-4 overflow-hidden rounded-2xl border border-white/12 bg-[#2f1620]" style={{ height }}>
        <iframe
          ref={iframeRef}
          src="https://embed.diagrams.net/?embed=1&spin=1&proto=json&autosave=1&saveAndExit=0&noSaveBtn=1&noExitBtn=1&sidebar=0&layers=0&toolbar=0&menubar=0&status=0&format=0"
          className="h-full w-full border-0"
          title="draw.io editor"
        />
      </div>
    </div>
  );
};

export default DrawioInlineEditor;
