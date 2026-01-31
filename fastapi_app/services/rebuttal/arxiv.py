import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import os
import re
import time
import tarfile
import shutil
import subprocess
from glob import glob
from typing import List, Dict, Optional
from pathlib import Path
import sys

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


DIRECT_OPENER = urllib.request.build_opener()


ARXIV_API = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

class ArxivAgent:
    def __init__(self, max_results: int = 30, pdf_dir: str = "arxiv_papers", 
                 md_dir: str = "arxiv_papers_md", ddl: int = 180, 
                 download_mode: str = "source_first", download_dir: str | None = None):

        self.max_results = max_results
        self.pdf_dir = pdf_dir
        self.md_dir = md_dir
        self.ddl = ddl
        self.download_mode = download_mode
        self.download_dir = download_dir if download_dir is not None else pdf_dir
            
        self.last_query_start_ts: Optional[float] = None
    
    def search_and_analyze(self, query: str) -> List[Dict]:
        self.last_query_start_ts = time.time()
        
        search_query = f'all:{query}'
        
        raw_results = self._search_arxiv(search_query, max_results=self.max_results * 2) 

        final_papers = raw_results[:self.max_results]
        
        for paper in final_papers:
            paper['relevance_score'] = 1.0 
        
        return final_papers
    
    def download_papers(self, papers: List[Dict]) -> List[str]:
        downloaded_files = []
        
        for i, paper in enumerate(papers, 1):
            title = self._clean_filename(paper['title'])
            arxiv_id = paper.get('arxiv_id', '')
            start_time = time.time()
            
            global_start = self.last_query_start_ts if self.last_query_start_ts else start_time
            deadline = global_start + self.ddl
            
            try:
                base_name = f"{arxiv_id}_{title[:50]}" if arxiv_id else title[:50]
                
                existing_md = self._check_existing_markdown(base_name)
                if existing_md:
                    print(f"[{i}/{len(papers)}] Already exists: {title[:50]}... ({existing_md})")
                    downloaded_files.append(existing_md)
                    continue
                
                if time.time() >= deadline:
                    md_path = self._write_abstract_markdown(paper, base_name)
                    if md_path:
                        print(f"[{i}/{len(papers)}] Timeout, saving abstract: {title[:50]}...")
                        downloaded_files.append(md_path)
                    continue
                
                result = None
                if self.download_mode == "source_first":
                    result = self._download_source_first(paper, base_name, deadline)
                elif self.download_mode == "pdf_first":
                    result = self._download_pdf_first(paper, base_name, deadline)
                
                if result:
                    print(f"[{i}/{len(papers)}] {result['status']}: {title[:50]}...")
                    downloaded_files.append(result['path'])
                else:
                    print(f"[{i}/{len(papers)}] Download failed: {title[:50]}...")
                
                time.sleep(1)
                
            except Exception as e:
                print(f"[{i}/{len(papers)}] Processing failed: {str(e)}")
        
        return downloaded_files
    
    def _search_arxiv(self, query: str, max_results: int = 10) -> List[Dict]:
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        
        url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
        print(f"[DEBUG] Requesting arXiv API: {url[:100]}...")
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "arxiv-agent/1.0 (research@example.com)"}
        )
        
        try:
            print(f"[DEBUG] Sending request, timeout set to 30 seconds...")
            with DIRECT_OPENER.open(req, timeout=30) as resp:
                xml_text = resp.read()
                print(f"[DEBUG] Response received, length: {len(xml_text)} bytes")
            
            root = ET.fromstring(xml_text)
            papers = []
            
            for entry in root.findall("atom:entry", ATOM_NS):
                paper = self._parse_entry(entry)
                if paper:
                    papers.append(paper)
            
            print(f"[DEBUG] Parsing complete, found {len(papers)} papers")
            return papers
            
        except Exception as e:
            print(f"[ERROR] Search failed: {type(e).__name__}: {str(e)}")
            return []
    
    def _parse_entry(self, entry) -> Optional[Dict]:
        title = (entry.findtext("atom:title", default="", namespaces=ATOM_NS) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=ATOM_NS) or "").strip()
        published = entry.findtext("atom:published", default="", namespaces=ATOM_NS)
        abs_url = entry.findtext("atom:id", default="", namespaces=ATOM_NS)
        
        authors = []
        for a in entry.findall("atom:author", ATOM_NS):
            name = a.findtext("atom:name", default="", namespaces=ATOM_NS)
            if name:
                authors.append(name)
        
        pdf_url = ""
        source_url = ""
        for link in entry.findall("atom:link", ATOM_NS):
            if link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href", "")
            elif link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href", "")
        
        arxiv_id = abs_url.rsplit("/", 1)[-1] if abs_url else ""
        
        if arxiv_id:
            source_url = f"https://arxiv.org/e-print/{arxiv_id}"
        
        if not all([title, summary, arxiv_id]):
            return None
        
        return {
            "title": " ".join(title.split()),
            "abstract": " ".join(summary.split()),
            "published": published,
            "authors": authors,
            "abs_url": abs_url,
            "pdf_url": pdf_url,
            "source_url": source_url,
            "arxiv_id": arxiv_id,
        }
    
    def _download_pdf(self, paper: Dict, title: str) -> Optional[str]:
        if not paper['pdf_url']:
            return None
        
        filename = f"{paper['arxiv_id']}_{title[:50]}.pdf"
        filepath = os.path.join(self.download_dir, filename)
        
        req = urllib.request.Request(
            paper['pdf_url'],
            headers={"User-Agent": "arxiv-agent/1.0"}
        )
        
        with DIRECT_OPENER.open(req, timeout=60) as response:
            with open(filepath, 'wb') as f:
                f.write(response.read())
        
        return filepath
    
    def _download_source(self, paper: Dict, title: str, deadline: Optional[float] = None) -> Optional[str]:
        if not paper['source_url']:
            return None

        filename = f"{paper['arxiv_id']}_{title[:50]}_source.tar.gz"
        filepath = os.path.join(self.download_dir, filename)

        try:
            req = urllib.request.Request(
                paper['source_url'],
                headers={"User-Agent": "arxiv-agent/1.0"}
            )

            per_read_timeout = 10
            if deadline is not None:
                remaining = max(0.0, deadline - time.time())
                per_read_timeout = max(1.0, min(10.0, remaining))

            with DIRECT_OPENER.open(req, timeout=per_read_timeout) as response:
                with open(filepath, 'wb') as f:
                    while True:
                        if (deadline is not None) and (time.time() >= deadline):
                            try: f.close()
                            except Exception: pass
                            try:
                                if os.path.exists(filepath): os.remove(filepath)
                            except Exception: pass
                            return None
                        try:
                            chunk = response.read(64 * 1024)
                        except Exception:
                            chunk = b""
                        if not chunk:
                            break
                        f.write(chunk)

            return filepath
        except Exception:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception:
                pass
            return None
    
    def _clean_filename(self, filename: str) -> str:
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        return filename[:100]
    
    def _check_existing_markdown(self, base_name: str) -> Optional[str]:
        md_paths = glob(os.path.join(self.download_dir, f"{base_name}*.md"))
        if md_paths:
            return md_paths[0]
        md_paths = glob(os.path.join(self.md_dir, f"{base_name}*.md"))
        if md_paths:
            return md_paths[0]
        return None

    def _download_source_first(self, paper: Dict, base_name: str, deadline: Optional[float] = None) -> Optional[Dict]:
        if paper['source_url'] and time.time() < deadline:
            try:
                print(f"[INFO] Attempting to download source: {paper['arxiv_id']}")
                archive_path = self._download_source(paper, base_name, deadline=deadline)
                
                if archive_path and time.time() < deadline:
                    print(f"[INFO] Attempting to convert source to Markdown: {paper['arxiv_id']}")
                    md_path = self.convert_source_archive_to_markdown(archive_path, deadline=deadline)
                    if md_path:
                        target_md_path = os.path.join(self.md_dir, os.path.basename(md_path))
                        shutil.copy(md_path, target_md_path)
                        return {'path': target_md_path, 'status': 'Source converted to Markdown'}
            except Exception as e:
                print(f"[WARNING] Source processing failed: {e}")
        
        return self._download_pdf_first(paper, base_name, deadline)

    def _download_pdf_first(self, paper: Dict, base_name: str, deadline: Optional[float] = None) -> Optional[Dict]:
        pdf_path = None
        if paper['pdf_url'] and time.time() < deadline:
            try:
                print(f"[INFO] Attempting to download PDF: {paper['arxiv_id']}")
                pdf_path = self._download_pdf(paper, base_name)
            except Exception as e:
                print(f"[WARNING] PDF download failed: {e}")
        
        md_path = self._write_abstract_markdown(paper, base_name)
        if md_path:
            return {'path': md_path, 'status': 'Saved abstract Markdown'}
            
        return None
        
    def convert_source_archive_to_markdown(self, archive_path: str, deadline: Optional[float] = None) -> Optional[str]:
        try:
            if not os.path.isfile(archive_path) or not archive_path.endswith('.tar.gz'):
                return None

            if (deadline is not None) and (time.time() >= deadline):
                return None

            base_dir = os.path.dirname(archive_path)
            base_name = os.path.basename(archive_path)[:-7]
            extract_dir = os.path.join(base_dir, base_name)
            if not os.path.exists(extract_dir):
                os.makedirs(extract_dir)

            with tarfile.open(archive_path, 'r:gz') as tar:
                ok = _safe_extract_tar(tar, path=extract_dir, deadline=deadline)
                if not ok:
                    return None

            if (deadline is not None) and (time.time() >= deadline):
                return None

            main_tex = _guess_main_tex_file(extract_dir, deadline=deadline)
            if not main_tex:
                if (deadline is not None) and (time.time() >= deadline):
                    return None
                tex_files = glob(os.path.join(extract_dir, '**', '*.tex'), recursive=True)
                if not tex_files:
                    return None
                main_tex = tex_files[0]

            out_md = os.path.join(base_dir, f"{base_name}_source.md")
            if shutil.which('pandoc'):
                try:
                    remaining = None
                    if deadline is not None:
                        remaining = max(0.0, deadline - time.time())
                        if remaining <= 1.0:
                            raise TimeoutError("deadline too close for pandoc conversion")
                    subprocess.run(
                        ['pandoc', '-f', 'latex', '-t', 'gfm', main_tex, '-o', out_md],
                        cwd=os.path.dirname(main_tex),
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        shell=False,
                        timeout=remaining if remaining and remaining > 0 else None,
                    )
                    return out_md
                except Exception:
                    pass

            if (deadline is not None) and (time.time() >= deadline):
                return None

            try:
                full_text = _read_tex_with_includes(main_tex, deadline=deadline)
                md_text = _latex_to_markdown_basic(full_text)
                with open(out_md, 'w', encoding='utf-8') as f:
                    f.write(md_text)
                return out_md
            except Exception:
                return None
        except Exception:
            return None

    def _write_abstract_markdown(self, paper: Dict, title_clean: Optional[str] = None) -> Optional[str]:
        try:
            title = paper.get('title') or ''
            authors = paper.get('authors') or []
            published = (paper.get('published') or '')[:10]
            abs_url = paper.get('abs_url') or ''
            pdf_url = paper.get('pdf_url') or ''
            source_url = paper.get('source_url') or ''
            arxiv_id = paper.get('arxiv_id') or ''
            abstract = paper.get('abstract') or ''

            safe_title = self._clean_filename(title if not title_clean else title_clean)
            filename = f"{arxiv_id}_{safe_title[:50]}_abstract.md" if arxiv_id else f"{safe_title[:50]}_abstract.md"
            out_path = os.path.join(self.download_dir, filename)

            lines = []
            if title:
                lines.append(f"# {title}")
                lines.append("")
            meta = []
            if authors:
                meta.append(f"**Authors**: {', '.join(authors)}")
            if published:
                meta.append(f"**Published**: {published}")
            if arxiv_id:
                meta.append(f"**arXiv ID**: {arxiv_id}")
            if meta:
                lines.append("\n".join(meta))
                lines.append("")
            links = []
            if abs_url:
                links.append(f"- [Abstract]({abs_url})")
            if pdf_url:
                links.append(f"- [PDF]({pdf_url})")
            if source_url:
                links.append(f"- [Source]({source_url})")
            if links:
                lines.append("**Links**:")
                lines.extend(links)
                lines.append("")
            lines.append("## Abstract")
            lines.append("")
            lines.append(abstract)
            lines.append("")

            with open(out_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(lines))
            return out_path
        except Exception:
            return None
        

def _safe_extract_tar(tar: tarfile.TarFile, path: str, deadline: Optional[float] = None) -> bool:
    for member in tar.getmembers():
        if (deadline is not None) and (time.time() >= deadline):
            return False
        member_path = os.path.join(path, member.name)
        abs_path = os.path.abspath(member_path)
        abs_base = os.path.abspath(path)
        if not abs_path.startswith(abs_base):
            continue
        try:
            tar.extract(member, path)
        except Exception:
            continue
    return True


def _guess_main_tex_file(root_dir: str, deadline: Optional[float] = None) -> Optional[str]:
    candidates = []
    for tex in glob(os.path.join(root_dir, '**', '*.tex'), recursive=True):
        if (deadline is not None) and (time.time() >= deadline):
            break
        try:
            with open(tex, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(5000)
            if ('\\documentclass' in content) and ('\\begin{document}' in content):
                candidates.append(tex)
        except Exception:
            continue
    if candidates:
        candidates.sort(key=lambda p: (p.count(os.sep), len(p)))
        return candidates[0]
    return None


def _read_tex_with_includes(main_tex_path: str, deadline: Optional[float] = None) -> str:
    visited = set()

    def _read(path: str, base_dir: str) -> str:
        if (deadline is not None) and (time.time() >= deadline):
            return ""
        norm = os.path.normpath(os.path.abspath(path))
        if norm in visited:
            return ""
        visited.add(norm)
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        except Exception:
            return ""

        def _replace_include(m):
            inner = m.group(1).strip()
            if not os.path.splitext(inner)[1]:
                inner = inner + '.tex'
            sub_path = os.path.join(base_dir, inner)
            return _read(sub_path, os.path.dirname(sub_path))

        pattern = re.compile(r"\\(?:input|include)\{([^}]+)\}")
        try:
            text = re.sub(pattern, _replace_include, text)
        except Exception:
            pass
        return text

    return _read(main_tex_path, os.path.dirname(main_tex_path))


def _latex_to_markdown_basic(tex: str) -> str:
    s = tex
    lines = []
    for line in s.splitlines():
        if '%' in line:
            idx = line.find('%')
            if idx >= 0:
                line = line[:idx]
        lines.append(line)
    s = "\n".join(lines)

    def remove_env(name: str, text: str) -> str:
        return re.sub(rf"\\begin\{{{name}\}}[\s\S]*?\\end\{{{name}\}}", "", text, flags=re.DOTALL)
    for env in ["figure", "table", "algorithm", "lstlisting"]:
        s = remove_env(env, s)

    s = re.sub(r"\\section\*?\{([^}]*)\}", r"\n\n## \1\n\n", s)
    s = re.sub(r"\\subsection\*?\{([^}]*)\}", r"\n\n### \1\n\n", s)
    s = re.sub(r"\\subsubsection\*?\{([^}]*)\}", r"\n\n#### \1\n\n", s)

    s = re.sub(r"\\textbf\{([^}]*)\}", r"**\1**", s)
    s = re.sub(r"\\emph\{([^}]*)\}", r"*\1*", s)
    s = re.sub(r"\\textit\{([^}]*)\}", r"*\1*", s)

    s = re.sub(r"\\cite[t|p]?\{[^}]*\}", "", s)
    s = re.sub(r"\\ref\{[^}]*\}", "", s)
    s = re.sub(r"\\label\{[^}]*\}", "", s)

    s = re.sub(r"\\[a-zA-Z]+(?:\[[^\]]*\])?\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+(?:\[[^\]]*\])?", "", s)

    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip() + "\n"




def search_relevant_papers(query: str, max_results: int = 30) -> List[Dict]:
    print(f"[DEBUG] search_relevant_papers called, query: '{query}', max_results={max_results}")
    agent = ArxivAgent(max_results=max_results)
    ranked = agent.search_and_analyze(query)
    print(f"[DEBUG] search_relevant_papers returning {len(ranked)} papers")
    return ranked[:max_results]


def _extract_arxiv_id_from_url(url: str) -> Optional[str]:
    try:
        parsed = urllib.parse.urlparse(url)
        path = parsed.path.strip('/')
        parts = path.split('/')
        if not parts:
            return None

        if parts[0] in {"abs", "pdf", "e-print"} and len(parts) >= 2:
            arxiv_part = parts[1]
            if arxiv_part.endswith('.pdf'):
                arxiv_part = arxiv_part[:-4]
            arxiv_part = re.sub(r"v\d+$", "", arxiv_part)
            if re.match(r"^\d{4}\.\d{4,5}$", arxiv_part):
                return arxiv_part
            if arxiv_part:
                return arxiv_part
        candidate = re.sub(r"v\d+$", "", parts[-1])
        candidate = candidate.replace('.pdf', '')
        if candidate:
            return candidate
        return None
    except Exception:
        return None


def _fetch_metadata_by_id(arxiv_id: str) -> Optional[Dict]:
    try:
        query = f"id_list={arxiv_id}"
        url = f"{ARXIV_API}?{query}"
        req = urllib.request.Request(url, headers={"User-Agent": "arxiv-agent/1.0 (metadata)"})
        with DIRECT_OPENER.open(req, timeout=20) as resp:
            xml_text = resp.read()
        root = ET.fromstring(xml_text)
        entry = root.find("atom:entry", ATOM_NS)
        if entry is None:
            return None
        title = (entry.findtext("atom:title", default="", namespaces=ATOM_NS) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=ATOM_NS) or "").strip()
        published = entry.findtext("atom:published", default="", namespaces=ATOM_NS)
        abs_url = entry.findtext("atom:id", default=f"https://arxiv.org/abs/{arxiv_id}", namespaces=ATOM_NS)
        
        authors = []
        for a in entry.findall("atom:author", ATOM_NS):
            name = a.findtext("atom:name", default="", namespaces=ATOM_NS)
            if name:
                authors.append(name)
        
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        for link in entry.findall("atom:link", ATOM_NS):
            if link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href", pdf_url)
                break
        source_url = f"https://arxiv.org/e-print/{arxiv_id}"
        return {
            "title": " ".join(title.split()),
            "abstract": " ".join(summary.split()),
            "published": published,
            "authors": authors,
            "abs_url": abs_url,
            "pdf_url": pdf_url,
            "source_url": source_url,
            "arxiv_id": arxiv_id,
        }
    except Exception:
        return None
