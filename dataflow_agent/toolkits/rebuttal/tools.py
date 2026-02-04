import sys
import os

os.environ["OMP_NUM_THREADS"] = "4"

DOCLING_DEVICE = os.environ.get("DOCLING_DEVICE", "cpu").lower()

import re
import json
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
import urllib.parse
import threading
from dataflow_agent.toolkits.rebuttal.arxiv import _fetch_metadata_by_id
from dataflow_agent.logger import get_logger

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ARXIV_DIRECT_OPENER = urllib.request.build_opener()
ARXIV_HOSTS = {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}

PDF_CONVERT_LOCK = threading.Lock()

from pathlib import Path
import shutil

_docling_converter = None
_docling_current_device = None

log = get_logger(__name__)


def _read_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def _pdf_to_md_mineru(pdf_path: str, output_path: str) -> str | None:
    """Convert PDF to Markdown using MinerU CLI."""
    try:
        from dataflow_agent.toolkits.multimodaltool.mineru_tool import run_mineru_pdf_extract
    except Exception as e:
        log.warning(f"[PDF2MD-MinerU] MinerU import failed: {e}")
        return None

    try:
        paths = Path(pdf_path)
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        source = os.environ.get("MINERU_SOURCE", "modelscope")
        log.info(f"[PDF2MD-MinerU] Running MinerU on {pdf_path} (source={source})")
        run_mineru_pdf_extract(str(paths), str(output_dir), source)

        pdf_stem = paths.stem
        auto_dir = (output_dir / pdf_stem / "auto").resolve()
        hybrid_dir = (output_dir / pdf_stem / "hybrid_auto").resolve()
        candidates = [
            auto_dir / f"{pdf_stem}.md",
            hybrid_dir / f"{pdf_stem}.md",
        ]
        markdown_path = next((p for p in candidates if p.exists()), None)
        if not markdown_path:
            # Fallback: pick the first .md under auto/hybrid_auto if filename differs
            for base_dir in (auto_dir, hybrid_dir):
                if base_dir.exists():
                    md_files = list(base_dir.glob("*.md"))
                    if md_files:
                        markdown_path = md_files[0]
                        break
        if not markdown_path or not markdown_path.exists():
            log.error(
                "[PDF2MD-MinerU] Markdown not found. "
                f"checked auto={auto_dir}, hybrid_auto={hybrid_dir}"
            )
            return None

        md_content = markdown_path.read_text(encoding="utf-8", errors="replace")
        target_md = output_dir / f"{pdf_stem}.md"
        target_md.write_text(md_content, encoding="utf-8")
        log.info(f"[PDF2MD-MinerU] Markdown file saved to: {target_md}")
        return str(target_md)
    except Exception as e:
        log.exception(f"[PDF2MD-MinerU] Failed: {type(e).__name__}: {e}")
        return None


def _fix_json_escapes(json_str: str) -> str:
    """Fix unescaped backslashes in JSON strings (e.g., backslashes in LaTeX formulas)"""
    json_str = json_str.replace('\\\\', '\x00ESCAPED_BACKSLASH\x00')
    json_str = json_str.replace('\\n', '\x00ESCAPED_N\x00')
    json_str = json_str.replace('\\t', '\x00ESCAPED_T\x00')
    json_str = json_str.replace('\\"', '\x00ESCAPED_QUOTE\x00')
    json_str = json_str.replace('\\/', '\x00ESCAPED_SLASH\x00')
    json_str = json_str.replace('\\r', '\x00ESCAPED_R\x00')
    json_str = json_str.replace('\\b', '\x00ESCAPED_B\x00')
    json_str = json_str.replace('\\f', '\x00ESCAPED_F\x00')
    # Escape remaining single backslashes
    json_str = json_str.replace('\\', '\\\\')
    # Restore protected content
    json_str = json_str.replace('\x00ESCAPED_BACKSLASH\x00', '\\\\')
    json_str = json_str.replace('\x00ESCAPED_N\x00', '\\n')
    json_str = json_str.replace('\x00ESCAPED_T\x00', '\\t')
    json_str = json_str.replace('\x00ESCAPED_QUOTE\x00', '\\"')
    json_str = json_str.replace('\x00ESCAPED_SLASH\x00', '\\/')
    json_str = json_str.replace('\x00ESCAPED_R\x00', '\\r')
    json_str = json_str.replace('\x00ESCAPED_B\x00', '\\b')
    json_str = json_str.replace('\x00ESCAPED_F\x00', '\\f')
    return json_str


def pdf_to_md(pdf_path: str, output_path: str, parser: str | None = None) -> str | None:
    """Convert PDF to Markdown.

    Uses a global lock to protect docling calls, ensuring only one PDF is converted at a time.
    Returns the generated file path, or None on failure.

    Note: docling is imported lazily to avoid triggering CUDA initialization errors
    in HF Spaces Stateless GPU environment.
    """
    global _docling_converter

    try:
        paths = Path(pdf_path)

        parser_choice = (parser or os.environ.get("REBUTTAL_PDF_PARSER", "mineru")).strip().lower()
        if parser_choice in {"mineru", "miner-u"}:
            mineru_md = _pdf_to_md_mineru(pdf_path, output_path)
            if mineru_md:
                return mineru_md
            log.error("[PDF2MD] MinerU failed; no fallback is allowed for PDF parsing.")
            return None

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        log.info(f"[PDF2MD] Preparing to convert PDF: {pdf_path}")
        log.debug("[PDF2MD] Waiting to acquire docling conversion lock...")
        with PDF_CONVERT_LOCK:
            log.debug("[PDF2MD] Lock acquired, starting docling conversion...")

            if _docling_converter is None:
                device_str = DOCLING_DEVICE
                log.info(
                    "[PDF2MD] Initializing docling DocumentConverter\n"
                    f"device={device_str.upper()}"
                )
                from docling.document_converter import DocumentConverter, PdfFormatOption
                from docling.datamodel.pipeline_options import PdfPipelineOptions
                from docling.datamodel.base_models import InputFormat
                try:
                    from docling.datamodel.pipeline_options import AcceleratorDevice
                    if device_str == "cuda":
                        accelerator_device = AcceleratorDevice.CUDA
                    else:
                        accelerator_device = AcceleratorDevice.CPU
                except ImportError:
                    accelerator_device = device_str

                pipeline_options = PdfPipelineOptions()
                pipeline_options.accelerator_options.device = accelerator_device

                _docling_converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                    }
                )
                log.info(f"[PDF2MD] docling initialization complete ({device_str.upper()} mode)")

            converter = _docling_converter

            log.debug("[PDF2MD] Calling docling converter.convert()...")
            raw_result = converter.convert(pdf_path)

            if hasattr(raw_result, 'document'):
                md_content = raw_result.document.export_to_markdown()
            else:
                md_content = raw_result.export_to_markdown()

            log.debug("[PDF2MD] docling conversion complete, releasing lock")

        target_md = os.path.join(output_path, paths.stem + ".md")
        with open(target_md, 'w', encoding='utf-8') as f:
            f.write(md_content)

        log.info(f"[PDF2MD] Markdown file saved to: {target_md}")
        log.debug(f"[PDF2MD] Markdown file size: {len(md_content)} characters")

        return target_md

    except Exception as e:
        log.exception(f"[ERROR] pdf_to_md failed: {type(e).__name__}: {e}")
        return None


def _safe_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]+', '_', name)[:100]


def download_pdf_and_convert_md(paper: dict, output_dir: str) -> str | None:
    """Download PDF and convert to Markdown.

    If download or conversion fails, creates a file containing paper metadata.
    Returns the file path.

    Args:
        paper: Dictionary containing paper metadata (title, arxiv_id, pdf_url, etc.)
        output_dir: Directory to save downloaded PDFs and converted Markdown files
    """
    papers_dir = output_dir

    if not os.path.exists(papers_dir):
        os.makedirs(papers_dir)

    def create_fallback_markdown_file(paper: dict, safe_name: str, error_msg: str = "") -> str:
        """Create a Markdown file containing basic paper information"""
        title = paper.get('title', 'Unknown Paper')
        abstract = paper.get('abstract', 'No abstract available')
        arxiv_id = paper.get('arxiv_id', 'N/A')
        pdf_url = paper.get('pdf_url', '')
        abs_url = paper.get('abs_url', '')
        authors = paper.get('authors', [])

        authors_str = ', '.join(authors) if authors else 'Unknown'

        md_content = f"""# {title}

**arXiv ID**: {arxiv_id}
**Authors**: {authors_str}
**PDF URL**: {pdf_url}
**Abstract URL**: {abs_url}

---

**Note**: PDF download or conversion failed. Only metadata is available.
{f"**Error**: {error_msg}" if error_msg else ""}

---

## Abstract

{abstract}

---

**Full text is not available. Please refer to the original paper via the URLs above.**
"""

        md_path = os.path.join(papers_dir, f"{safe_name}.md")
        try:
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
            log.info(f"[INFO] Created fallback Markdown file (metadata only): {md_path}")
            return md_path
        except Exception as e:
            log.error(f"[ERROR] Failed to create fallback Markdown: {e}")
            return None

    try:
        log.info(f"[PDF] Starting paper download: {paper.get('title', 'Unknown')[:60]}...")
        log.debug(f"[PDF] arXiv ID: {paper.get('arxiv_id', 'N/A')}")

        title = paper.get('title') or paper.get('arxiv_id') or 'paper'
        arxiv_id = paper.get('arxiv_id') or ''
        base_name = f"{arxiv_id}_{title[:50]}" if arxiv_id else title[:50]
        safe = _safe_filename(base_name)

        pdf_url = paper.get('pdf_url') or ''
        if not pdf_url:
            abs_url = paper.get('abs_url') or ''
            if abs_url:
                pdf_url = abs_url.replace('/abs/', '/pdf/')
        if pdf_url and not pdf_url.endswith('.pdf'):
            pdf_url = pdf_url + '.pdf'

        if not pdf_url:
            log.warning("[PDF] Unable to get PDF URL, creating fallback Markdown")
            return create_fallback_markdown_file(paper, safe, "Unable to get PDF URL")

        log.debug(f"[PDF] PDF URL: {pdf_url}")
        pdf_path = os.path.join(papers_dir, f"{safe}.pdf")

        try:
            if not os.path.exists(pdf_path):
                log.info(f"[PDF] Starting download to: {pdf_path}")

                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'application/pdf,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                }

                parsed = urllib.parse.urlparse(pdf_url)
                host = (parsed.hostname or '').lower()
                use_direct = (host in ARXIV_HOSTS) or any(host.endswith('.' + h) for h in ARXIV_HOSTS)

                log.debug(
                    "[PDF] Using proxy mode: "
                    f"{'DIRECT_OPENER' if use_direct else 'urlopen (with proxy)'}"
                )

                opener_open = ARXIV_DIRECT_OPENER.open if use_direct else urllib.request.urlopen

                max_retries = 3
                retry_delay = 2

                for attempt in range(max_retries):
                    try:
                        if attempt > 0:
                            wait_time = retry_delay * (attempt + 1) + random.uniform(0.5, 2.0)
                            log.info(
                                "[PDF] Retry wait\n"
                                f"attempt={attempt + 1}/{max_retries}\n"
                                f"wait={wait_time:.1f}s"
                            )
                            time.sleep(wait_time)

                        log.debug(f"[PDF] Sending HTTP request (attempt {attempt + 1}/{max_retries})...")
                        req = urllib.request.Request(pdf_url, headers=headers)

                        with opener_open(req, timeout=60) as resp:
                            pdf_data = resp.read()
                            log.debug(f"[PDF] Download complete, size: {len(pdf_data)} bytes")

                            if len(pdf_data) < 100:
                                log.warning(
                                    f"[PDF] Downloaded file is too small ({len(pdf_data)} bytes), may not be a valid PDF"
                                )
                                if attempt < max_retries - 1:
                                    continue
                                raise ValueError(f"Downloaded file is too small: {len(pdf_data)} bytes")

                            if not pdf_data.startswith(b'%PDF-'):
                                log.warning(
                                    f"[PDF] Downloaded file is not a valid PDF (header: {pdf_data[:20]})"
                                )
                                if attempt < max_retries - 1:
                                    continue
                                raise ValueError("Downloaded file is not a valid PDF format")

                            with open(pdf_path, 'wb') as f:
                                f.write(pdf_data)
                            log.info(f"[PDF] Download successful: {pdf_path}")
                            break

                    except urllib.error.HTTPError as e:
                        error_msg = f"HTTP Error {e.code}: {e.reason}"
                        log.warning(f"[PDF] {error_msg}")

                        if e.code == 403:
                            log.warning(
                                "[PDF] 403 error usually indicates:\n"
                                "1. Request identified as bot (using real browser User-Agent)\n"
                                "2. IP temporarily rate-limited (retrying...)\n"
                                "3. Longer request interval needed"
                            )

                        if attempt == max_retries - 1:
                            raise

                    except Exception as e:
                        log.warning(f"[PDF] Download exception: {type(e).__name__}: {e}")
                        if attempt == max_retries - 1:
                            raise

                time.sleep(random.uniform(1.0, 3.0))

            else:
                log.debug(f"[PDF] Already exists, skipping download: {pdf_path}")

        except Exception as e:
            log.error(f"[PDF] Download failed (all retries exhausted): {type(e).__name__}: {e}")
            log.info("[PDF] Creating fallback Markdown (metadata only)")
            return create_fallback_markdown_file(paper, safe, f"PDF download failed: {str(e)}")

        log.debug("[PDF] Starting PDF to Markdown conversion (arXiv paper, using docling/CPU)...")
        md_path = pdf_to_md(pdf_path, papers_dir, parser="docling")

        if md_path and os.path.isfile(md_path):
            log.info(f"[PDF] Markdown conversion successful: {md_path}")
            return md_path
        else:
            log.warning("[PDF] Markdown conversion failed, creating fallback Markdown")
            return create_fallback_markdown_file(paper, safe, "PDF conversion failed")

    except Exception as e:
        log.exception(f"[WARNING] download_pdf_and_convert_md exception: {type(e).__name__}: {e}")

        try:
            title = paper.get('title') or paper.get('arxiv_id') or 'paper'
            arxiv_id = paper.get('arxiv_id') or ''
            base_name = f"{arxiv_id}_{title[:50]}" if arxiv_id else title[:50]
            safe = _safe_filename(base_name)
            return create_fallback_markdown_file(paper, safe, f"Processing exception: {str(e)}")
        except Exception:
            return None


# Rebuttal prompts 已迁移到 dataflow_agent/promptstemplates/resources/pt_rebuttal_repo.py
# 通过 PromptsTemplateGenerator 统一加载（扫描 pt_*.py）
_rebuttal_ptg = None


def _get_rebuttal_ptg():
    """Lazy init PromptsTemplateGenerator，用于加载 resources 下的 pt_rebuttal_repo 等模板。"""
    global _rebuttal_ptg
    if _rebuttal_ptg is None:
        try:
            from dataflow_agent.promptstemplates.prompt_template import PromptsTemplateGenerator
            from dataflow_agent.utils import get_project_root
            _rebuttal_ptg = PromptsTemplateGenerator(
                output_language="en",
                python_modules=None,
                template_dirs=[f"{get_project_root()}/dataflow_agent/promptstemplates/resources"],
            )
        except Exception as e:
            log.warning(f"PromptsTemplateGenerator init failed for rebuttal: {e}")
    return _rebuttal_ptg


# Mapping: 旧 prompt 名 -> PromptsTemplateGenerator.templates 中的 key（即 pt_rebuttal_repo 中的属性名）
PROMPT_KEY_MAPPING = {
    "0.txt": "system_prompt_for_review_extractor",
    "1.txt": "system_prompt_for_semantic_encoder",
    "2.txt": "system_prompt_for_issue_extractor",
    "2_c.txt": "system_prompt_for_issue_extractor_checker",
    "3.txt": "system_prompt_for_literature_retrieval",
    "4.txt": "system_prompt_for_reference_filter",
    "5.txt": "system_prompt_for_reference_analyzer",
    "6.txt": "system_prompt_for_strategy_generator",
    "7.txt": "system_prompt_for_strategy_reviewer",
    "7_h.txt": "system_prompt_for_strategy_human_refinement",
    "8.txt": "system_prompt_for_rebuttal_writer",
    "9.txt": "system_prompt_for_rebuttal_reviewer",
}


def load_prompt(name: str) -> str:
    """Load prompt from dataflow_agent promptstemplates (pt_rebuttal_repo in resources).

    Priority order:
    1. PromptsTemplateGenerator (dataflow_agent/promptstemplates/resources/pt_rebuttal_repo.py)

    Args:
        name: Prompt identifier (e.g., "1.txt", "2.txt")

    Returns:
        The prompt text content

    Raises:
        FileNotFoundError: If prompt cannot be found in any format
    """
    if name in PROMPT_KEY_MAPPING:
        try:
            ptg = _get_rebuttal_ptg()
            if ptg:
                key = PROMPT_KEY_MAPPING[name]
                prompt_text = ptg.templates.get(key)
                if prompt_text:
                    return prompt_text
        except Exception as e:
            log.warning(f"Failed to load prompt from PromptsTemplateGenerator: {e}, falling back to YAML")

    raise FileNotFoundError(f"Prompt not found in pt_rebuttal_repo for name: {name}")
