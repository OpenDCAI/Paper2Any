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
from fastapi_app.services.rebuttal.arxiv import _fetch_metadata_by_id

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

def _read_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

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

def pdf_to_md(pdf_path: str, output_path: str) -> str | None:
    """Convert PDF to Markdown.
    
    Uses a global lock to protect docling calls, ensuring only one PDF is converted at a time.
    Returns the generated file path, or None on failure.
    
    Note: docling is imported lazily to avoid triggering CUDA initialization errors
    in HF Spaces Stateless GPU environment.
    """
    global _docling_converter
    
    try:
        paths = Path(pdf_path)
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        print(f"[DEBUG] Preparing to convert PDF: {pdf_path}")
        print(f"[DEBUG] Waiting to acquire docling conversion lock...")
        with PDF_CONVERT_LOCK:
            print(f"[DEBUG] Lock acquired, starting docling conversion...")
            

            if _docling_converter is None:
                device_str = DOCLING_DEVICE
                print(f"[DEBUG] First use, importing and initializing docling DocumentConverter ({device_str.upper()} mode)...")
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
                print(f"[DEBUG] docling DocumentConverter initialization complete ({device_str.upper()} mode)")
        
            converter = _docling_converter
        
            print(f"[DEBUG] Calling docling converter.convert()...")
            raw_result = converter.convert(pdf_path)
        
            if hasattr(raw_result, 'document'):
                md_content = raw_result.document.export_to_markdown()
            else:
                md_content = raw_result.export_to_markdown()
            
            print(f"[DEBUG] docling conversion complete, releasing lock")
        
        target_md = os.path.join(output_path, paths.stem + ".md")
        with open(target_md, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"[SUCCESS] Markdown file saved to: {target_md}")
        print(f"[DEBUG] Markdown file size: {len(md_content)} characters")
        
        return target_md
        
    except Exception as e:
        print(f"[ERROR] pdf_to_md failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
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
            print(f"[INFO] Created fallback Markdown file (metadata only): {md_path}")
            return md_path
        except Exception as e:
            print(f"[ERROR] Failed to create fallback Markdown: {e}")
            return None
    
    try:
        print(f"[DEBUG] Starting paper download: {paper.get('title', 'Unknown')[:60]}...")
        print(f"[DEBUG] arXiv ID: {paper.get('arxiv_id', 'N/A')}")
        
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
            print(f"[WARNING] Unable to get PDF URL, creating fallback Markdown")
            return create_fallback_markdown_file(paper, safe, "Unable to get PDF URL")
        
        print(f"[DEBUG] PDF URL: {pdf_url}")
        pdf_path = os.path.join(papers_dir, f"{safe}.pdf")
    
        try:
            if not os.path.exists(pdf_path):
                print(f"[DEBUG] Starting PDF download to: {pdf_path}")
                
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
                
                print(f"[DEBUG] Using proxy mode: {'DIRECT_OPENER' if use_direct else 'urlopen (with proxy)'}")
                
                opener_open = ARXIV_DIRECT_OPENER.open if use_direct else urllib.request.urlopen
                
                max_retries = 3
                retry_delay = 2 
                
                for attempt in range(max_retries):
                    try:
                        if attempt > 0:
                            wait_time = retry_delay * (attempt + 1) + random.uniform(0.5, 2.0)
                            print(f"[INFO] Waiting {wait_time:.1f} seconds before retry (attempt {attempt + 1}/{max_retries})...")
                            time.sleep(wait_time)
                        
                        print(f"[DEBUG] Sending HTTP request (attempt {attempt + 1}/{max_retries})...")
                        req = urllib.request.Request(pdf_url, headers=headers)
                        
                        with opener_open(req, timeout=60) as resp:
                            pdf_data = resp.read()
                            print(f"[DEBUG] Download complete, size: {len(pdf_data)} bytes")
                            
                            if len(pdf_data) < 100:
                                print(f"[WARNING] Downloaded file is too small ({len(pdf_data)} bytes), may not be a valid PDF")
                                if attempt < max_retries - 1:
                                    continue
                                raise ValueError(f"Downloaded file is too small: {len(pdf_data)} bytes")
                            
                            if not pdf_data.startswith(b'%PDF-'):
                                print(f"[WARNING] Downloaded file is not a valid PDF (header: {pdf_data[:20]})")
                                if attempt < max_retries - 1:
                                    continue
                                raise ValueError("Downloaded file is not a valid PDF format")
                            
                            with open(pdf_path, 'wb') as f:
                                f.write(pdf_data)
                            print(f"[SUCCESS] PDF download successful: {pdf_path}")
                            break  
                            
                    except urllib.error.HTTPError as e:
                        error_msg = f"HTTP Error {e.code}: {e.reason}"
                        print(f"[WARNING] {error_msg}")
                        
                        if e.code == 403:
                            print(f"[TIP] 403 error usually indicates:")
                            print(f"  1. Request identified as bot (using real browser User-Agent)")
                            print(f"  2. IP temporarily rate-limited (retrying...)")
                            print(f"  3. Longer request interval needed")
                        
                        if attempt == max_retries - 1:
                            raise  
                    
                    except Exception as e:
                        print(f"[WARNING] Download exception: {type(e).__name__}: {e}")
                        if attempt == max_retries - 1:
                            raise  
                
                time.sleep(random.uniform(1.0, 3.0))
                
            else:
                print(f"[DEBUG] PDF already exists, skipping download: {pdf_path}")
                
        except Exception as e:
            print(f"[ERROR] PDF download failed (all retries exhausted): {type(e).__name__}: {e}")
            print(f"[INFO] Creating fallback Markdown (metadata only)")
            return create_fallback_markdown_file(paper, safe, f"PDF download failed: {str(e)}")
        
        print(f"[DEBUG] Starting PDF to Markdown conversion...")
        md_path = pdf_to_md(pdf_path, papers_dir)
        
        if md_path and os.path.isfile(md_path):
            print(f"[SUCCESS] Markdown conversion successful: {md_path}")
            return md_path
        else:
            print(f"[WARNING] Markdown conversion failed, creating fallback Markdown")
            return create_fallback_markdown_file(paper, safe, "PDF conversion failed")
        
    except Exception as e:
        print(f"[WARNING] download_pdf_and_convert_md exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        
        try:
            title = paper.get('title') or paper.get('arxiv_id') or 'paper'
            arxiv_id = paper.get('arxiv_id') or ''
            base_name = f"{arxiv_id}_{title[:50]}" if arxiv_id else title[:50]
            safe = _safe_filename(base_name)
            return create_fallback_markdown_file(paper, safe, f"Processing exception: {str(e)}")
        except:
            return None



_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(_CURRENT_DIR, "prompts")


import yaml

# Mapping from old prompt names to new YAML names
PROMPT_NAME_MAPPING = {
    "1.txt": "semantic_encoder.yaml",
    "2.txt": "issue_extractor.yaml",
    "2_c.txt": "issue_extractor_checker.yaml",
    "3.txt": "literature_retrieval.yaml",
    "4.txt": "reference_filter.yaml",
    "5.txt": "reference_analyzer.yaml",
    "6.txt": "strategy_generator.yaml",
    "7.txt": "strategy_reviewer.yaml",
    "7_h.txt": "strategy_human_refinement.yaml",
    "8.txt": "rebuttal_writer.yaml",
    "9.txt": "rebuttal_reviewer.yaml",
}

def load_prompt(name: str) -> str:
    """Load prompt from YAML or TXT file.
    
    Supports both new YAML format and legacy TXT format.
    For YAML files, extracts the 'prompt' field.
    """
    # Map old names to new names
    mapped_name = PROMPT_NAME_MAPPING.get(name, name)
    prompt_path = os.path.join(PROMPTS_DIR, mapped_name)
    
    # Try YAML first, then fall back to original name
    if not os.path.exists(prompt_path):
        prompt_path = os.path.join(PROMPTS_DIR, name)
    
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    with open(prompt_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # If it's a YAML file, extract the prompt field
    if prompt_path.endswith('.yaml') or prompt_path.endswith('.yml'):
        try:
            data = yaml.safe_load(content)
            if isinstance(data, dict) and 'prompt' in data:
                return data['prompt']
            return content
        except yaml.YAMLError:
            return content
    
    return content