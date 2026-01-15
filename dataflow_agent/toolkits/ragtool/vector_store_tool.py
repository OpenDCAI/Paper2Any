import os
import json
import pickle
import shutil
import subprocess
import uuid
import httpx
import numpy as np
import faiss
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from PIL import Image

# Import existing tools
from dataflow_agent.toolkits.imtool.mineru_tool import run_mineru_pdf_extract
from dataflow_agent.toolkits.imtool.req_videos import call_video_understanding_async
from dataflow_agent.toolkits.imtool.req_understanding import call_image_understanding_async
import dataflow_agent.utils as utils
from dataflow_agent.logger import get_logger

log = get_logger(__name__)

class VectorStoreManager:
    def __init__(
        self,
        base_dir: str,
        project_name: str = "kb_project",
        embedding_api_url: str = "http://123.129.219.111:3000/v1/embeddings",
        embedding_model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        multimodal_model: str = "gemini-2.5-flash",
        image_model: str = "gemini-2.5-flash",
        video_model: str = "gemini-2.5-flash"
    ):
        """
        Manage Vector Store (Faiss) and File Manifest.
        
        Args:
            base_dir: Root directory for storing processed files and index.
            project_name: Name of the project.
            embedding_api_url: URL for embedding API.
            embedding_model: Model name for embedding.
            api_key: API Key for embedding service.
            multimodal_model: Legacy parameter for multimodal understanding.
            image_model: Model name for image understanding.
            video_model: Model name for video understanding.
        """
        self.base_dir = Path(base_dir)
        self.project_name = project_name
        self.embedding_api_url = embedding_api_url
        self.embedding_model = embedding_model
        self.api_key = api_key or os.getenv("DF_API_KEY")
        
        # Multimodal config
        self.multimodal_model = multimodal_model
        self.image_model = image_model
        self.video_model = video_model
        
        # Fallback to multimodal_model if specific models not provided or default
        if self.image_model == "gemini-2.5-flash" and self.multimodal_model != "gemini-2.5-flash":
             self.image_model = self.multimodal_model
        if self.video_model == "gemini-2.5-flash" and self.multimodal_model != "gemini-2.5-flash":
             self.video_model = self.multimodal_model
        # Assume chat endpoint is at same host, replace /embeddings with nothing (base url)
        if "/embeddings" in self.embedding_api_url:
            self.multimodal_api_url = self.embedding_api_url.replace("/embeddings", "")
        else:
            self.multimodal_api_url = self.embedding_api_url
        
        # Directories
        self.processed_dir = self.base_dir / "processed"
        self.vector_store_dir = self.base_dir / "vector_store"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths
        self.manifest_path = self.base_dir / "knowledge_manifest.json"
        self.faiss_index_path = self.vector_store_dir / f"{project_name}.index"
        self.faiss_meta_path = self.vector_store_dir / f"{project_name}.meta"
        
        # State
        self.manifest = self._load_manifest()
        self.index = None
        self.meta_data = [] # List corresponding to index vectors
        self._load_index()

    def _load_manifest(self) -> Dict[str, Any]:
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "project_name": self.project_name,
            "base_dir": str(self.base_dir),
            "faiss_index_path": str(self.faiss_index_path),
            "faiss_meta_path": str(self.faiss_meta_path),
            "files": []
        }

    def _load_index(self):
        if self.faiss_index_path.exists() and self.faiss_meta_path.exists():
            log.info(f"Loading existing index from {self.faiss_index_path}")
            self.index = faiss.read_index(str(self.faiss_index_path))
            with open(self.faiss_meta_path, 'rb') as f:
                self.meta_data = pickle.load(f)
        else:
            log.info("Initializing new index")
            self.index = None # Will be initialized on first add
            self.meta_data = []

    def save(self):
        """Save Manifest, Index and Meta data to disk."""
        # Save Manifest
        with open(self.manifest_path, 'w', encoding='utf-8') as f:
            json.dump(self.manifest, f, ensure_ascii=False, indent=2)
            
        # Save Index & Meta
        if self.index is not None:
            faiss.write_index(self.index, str(self.faiss_index_path))
            with open(self.faiss_meta_path, 'wb') as f:
                pickle.dump(self.meta_data, f)
        
        log.info(f"Saved vector store to {self.vector_store_dir}")

    def search(self, query: str, top_k: int = 5, file_ids: Optional[List[str]] = None) -> List[Dict]:
        """
        Search knowledge base.
        
        Args:
            query: Query string.
            top_k: Number of results to return.
            file_ids: List of file IDs to filter by. If None, search all files.
                      Uses post-filtering strategy (retrieve more, then filter).
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        # 1. Embed query
        query_vecs = self._call_embedding_api([query])
        if len(query_vecs) == 0:
            return []
            
        # 2. Determine search k (expand if filtering)
        # If filtering by file_ids, we need to retrieve more candidates
        # because many might belong to other files.
        search_k = top_k
        if file_ids:
            # Simple heuristic: fetch more candidates. 
            # In production, might need to be much larger or use iterative search.
            search_k = max(top_k * 20, 100) 
            
        # Cap at total vectors
        search_k = min(search_k, self.index.ntotal)
            
        # 3. Search Faiss
        # D: distances (scores), I: indices
        D, I = self.index.search(query_vecs, search_k)
        
        # 4. Filter and Format Results
        results = []
        target_file_ids = set(file_ids) if file_ids else None
        
        # I[0] contains indices for the first (and only) query
        for rank, idx in enumerate(I[0]):
            if idx < 0 or idx >= len(self.meta_data):
                continue
                
            meta = self.meta_data[idx]
            
            # Post-filtering
            if target_file_ids and meta.get("source_file_id") not in target_file_ids:
                continue
                
            result_item = {
                "score": float(D[0][rank]),
                "content": meta.get("content"),
                "source_file_id": meta.get("source_file_id"),
                "type": meta.get("type"),
                "metadata": meta
            }
            results.append(result_item)
            
            if len(results) >= top_k:
                break
                
        return results

    def _call_embedding_api(self, texts: List[str]) -> np.ndarray:
        """Call Embedding API (OpenAI compatible)."""
        if not texts:
            return np.array([])
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        vecs = []
        # Batch processing to avoid payload limits
        batch_size = 10 
        
        with httpx.Client(timeout=60.0) as client:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                # Replace newlines which can negatively affect performance
                batch = [t.replace("\n", " ") for t in batch]
                
                try:
                    resp = client.post(
                        self.embedding_api_url,
                        headers=headers,
                        json={"model": self.embedding_model, "input": batch},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    # Ensure order is preserved
                    batch_vecs = [item["embedding"] for item in data["data"]]
                    vecs.extend(batch_vecs)
                except Exception as e:
                    log.error(f"Embedding API error: {e}")
                    raise RuntimeError(f"Failed to embed texts: {e}")

        arr = np.asarray(vecs, dtype=np.float32)
        if len(arr) > 0:
            faiss.normalize_L2(arr)
        return arr

    def _add_vectors(self, vectors: np.ndarray, meta_list: List[Dict]):
        """Add vectors and meta data to index."""
        if len(vectors) == 0:
            return
            
        if self.index is None:
            dim = vectors.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            
        self.index.add(vectors)
        self.meta_data.extend(meta_list)

    async def process_file(self, file_path: str, description: Optional[str] = None) -> str:
        """
        Main entry point to process a file.
        Returns the file ID in the manifest.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_id = str(uuid.uuid4())
        ext = file_path.suffix.lower()
        
        file_record = {
            "id": file_id,
            "original_path": str(file_path),
            "file_type": ext.lstrip('.'),
            "status": "processing",
            "chunks_count": 0,
            "media_desc_count": 0
        }
        
        log.info(f"Processing file: {file_path} (ID: {file_id})")

        try:
            if ext == '.pdf':
                await self._process_pdf(file_path, file_record, file_id)
            elif ext in ['.docx', '.doc']:
                await self._process_word(file_path, file_record, file_id)
            elif ext in ['.pptx', '.ppt']:
                await self._process_ppt(file_path, file_record, file_id)
            elif ext in ['.png', '.jpg', '.jpeg', '.mp4', '.avi', '.mov']:
                await self._process_media(file_path, description, file_record, file_id)
            else:
                log.warning(f"Unsupported file type: {ext}")
                file_record["status"] = "skipped"

            if file_record["status"] == "processing":
                 file_record["status"] = "embedded"

        except Exception as e:
            log.error(f"Error processing {file_path}: {e}")
            file_record["status"] = "failed"
            file_record["error"] = str(e)
            
        self.manifest["files"].append(file_record)
        self.save()
        return file_id

    def _convert_to_pdf(self, input_path: Path, output_dir: Path) -> Path:
        """Convert office document to PDF using LibreOffice."""
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "libreoffice",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(output_dir),
            str(input_path)
        ]
        
        log.info(f"Converting {input_path} to PDF...")
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        pdf_name = input_path.with_suffix('.pdf').name
        pdf_path = output_dir / pdf_name
        if not pdf_path.exists():
            raise RuntimeError(f"PDF conversion failed, expected output: {pdf_path}")
            
        return pdf_path

    async def _process_pdf(self, file_path: Path, record: Dict, file_id: str):
        # 1. MinerU Extract
        output_subdir = self.processed_dir / file_id
        # Run synchronous MinerU in a thread to avoid blocking the event loop
        await asyncio.to_thread(run_mineru_pdf_extract, str(file_path), str(output_subdir))
        
        # MinerU output structure: output_subdir / {filename_without_ext} / {filename}.md
        # We need to find the MD file
        mineru_output_folder = output_subdir / file_path.stem
        # Use rglob to find .md file recursively (it might be in 'auto' subdir)
        md_file = next(mineru_output_folder.rglob("*.md"), None)
        
        if not md_file:
            raise RuntimeError(f"MinerU did not generate Markdown file in {mineru_output_folder}")
            
        record["processed_md_path"] = str(md_file)
        # Images are usually in an 'images' folder next to the markdown file
        record["images_dir"] = str(md_file.parent / "images")
        
        # 2. Chunking & Embedding
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Simple chunking by paragraph/headers (can be improved)
        chunks = [c.strip() for c in content.split('\n\n') if c.strip()]
        # Filter very short chunks
        chunks = [c for c in chunks if len(c) > 10]
        
        if chunks:
            vectors = self._call_embedding_api(chunks)
            meta_list = [
                {
                    "source_file_id": file_id,
                    "type": "text_chunk",
                    "content": chunk,
                    "chunk_index": i
                }
                for i, chunk in enumerate(chunks)
            ]
            self._add_vectors(vectors, meta_list)
            record["chunks_count"] = len(chunks)

    async def _process_word(self, file_path: Path, record: Dict, file_id: str):
        # Convert to PDF first
        temp_dir = self.processed_dir / "temp" / file_id
        pdf_path = self._convert_to_pdf(file_path, temp_dir)
        
        # Reuse PDF processing
        await self._process_pdf(pdf_path, record, file_id)
        
        # Cleanup temp PDF
        # shutil.rmtree(temp_dir, ignore_errors=True)

    async def _process_ppt(self, file_path: Path, record: Dict, file_id: str):
        # Same as Word, convert to PDF first
        temp_dir = self.processed_dir / "temp" / file_id
        pdf_path = self._convert_to_pdf(file_path, temp_dir)
        await self._process_pdf(pdf_path, record, file_id)

    async def _process_media(self, file_path: Path, description: Optional[str], record: Dict, file_id: str):
        desc_text = description
        
        # If no description provided, generate one using multimodal API
        if not desc_text:
            log.info(f"No description for {file_path.name}, calling Multimodal API...")
            try:
                ext = file_path.suffix.lower()
                messages = []
                
                # Check file type
                if ext in ['.png', '.jpg', '.jpeg']:
                    # Image Understanding
                    log.info(f"Using image model: {self.image_model}")
                    desc_text = await call_image_understanding_async(
                        model=self.image_model,
                        messages=[{"role": "user", "content": "Please describe this image in detail for knowledge base retrieval."}],
                        api_url=self.multimodal_api_url,
                        api_key=self.api_key,
                        image_path=str(file_path)
                    )
                    log.critical(f'Image Understanding desc_text : {desc_text}')
                elif ext in ['.mp4', '.avi', '.mov']:
                    # Video Understanding
                    log.info(f"Using video model: {self.video_model}")
                    desc_text = await call_video_understanding_async(
                        model=self.video_model,
                        messages=[{"role": "user", "content": "Please analyze this video and provide a detailed description of its content, events, and any text visible, for knowledge base retrieval."}],
                        api_url=self.multimodal_api_url,
                        api_key=self.api_key,
                        video_path=str(file_path)
                    )
                    log.critical(f'Video Understanding desc_text : {desc_text}')
                
                if desc_text:
                    log.info(f"Generated description: {desc_text[:100]}...")
            except Exception as e:
                log.error(f"Failed to generate description: {e}")
                # Fallback or just skip embedding
        
        if desc_text:
            # Save description to file
            desc_path = self.processed_dir / file_id / "description.txt"
            desc_path.parent.mkdir(parents=True, exist_ok=True)
            with open(desc_path, 'w', encoding='utf-8') as f:
                f.write(desc_text)
                
            record["description_text_path"] = str(desc_path)
            
            # Embed description
            vectors = self._call_embedding_api([desc_text])
            meta_list = [{
                "source_file_id": file_id,
                "type": "media_desc",
                "content": desc_text,
                "path": str(file_path)
            }]
            self._add_vectors(vectors, meta_list)
            record["media_desc_count"] = 1
        else:
            log.warning(f"Skipping media {file_path.name} (no description available)")

async def process_knowledge_base_files(
    file_list: List[Dict[str, str]], 
    base_dir: str = "outputs/kb_data/vector_store_project",
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    multimodal_model: Optional[str] = None,
    image_model: Optional[str] = None,
    video_model: Optional[str] = None
):
    """
    Helper function to process a list of files.
    
    Args:
        file_list: List of dicts, each containing 'path' and optional 'description'.
        base_dir: Directory to store the vector store.
        api_url: Custom Embedding API URL.
        api_key: Custom API Key.
        model_name: Custom Model Name.
        multimodal_model: Custom Multimodal Model Name.
        image_model: Custom Image Model Name.
        video_model: Custom Video Model Name.
    """
    # Prepare kwargs
    kwargs = {"base_dir": base_dir}
    if api_url:
        kwargs["embedding_api_url"] = api_url
    if api_key:
        kwargs["api_key"] = api_key
    if model_name:
        kwargs["embedding_model"] = model_name
    if multimodal_model:
        kwargs["multimodal_model"] = multimodal_model
    if image_model:
        kwargs["image_model"] = image_model
    if video_model:
        kwargs["video_model"] = video_model

    manager = VectorStoreManager(**kwargs)
    
    for item in file_list:
        path = item.get("path")
        desc = item.get("description")
        if path:
            try:
                await manager.process_file(path, desc)
            except Exception as e:
                log.error(f"Failed to process {path}: {e}")
                
    manager.save()
    return manager.manifest

if __name__ == "__main__":
    # Test
    # Assuming valid API key is set in env DF_API_KEY
    test_files = [
        {"path": "tests/test.pdf"},
        {"path": "tests/cat_icon.png"} # No description test
    ]
    # process_knowledge_base_files(test_files)
