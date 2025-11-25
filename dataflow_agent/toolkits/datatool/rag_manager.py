# dataflow_agent/toolkits/datatool/rag_manager.py

from __future__ import annotations
import os
import asyncio
import re
import hashlib
from typing import Any, Dict, List, Optional
from datetime import datetime
import shutil

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from dataflow_agent.logger import get_logger

log = get_logger(__name__)


class RAGManager:
    """RAG管理器 - 负责向量存储和检索增强生成"""

    def __init__(
        self,
        api_base_url: str | None,
        api_key: str | None,
        *,
        embed_model: str | None = None,
        persist_directory: str = "./rag_db",
        reset: bool = False,
        collection_name: str = "rag_collection",
    ):
        resolved_api_base = api_base_url or os.getenv("RAG_API_URL")
        resolved_api_key = api_key or os.getenv("RAG_API_KEY")
        resolved_embed_model = embed_model or os.getenv("RAG_EMB_MODEL") or "text-embedding-3-large"

        if not resolved_api_base or not resolved_api_key:
            raise ValueError("RAG 初始化失败：缺少 API 基础地址或 API Key。请在调用阶段传入或设置环境变量。")

        log.info(f"[RAG] 初始化 RAG 管理器，存储目录: {persist_directory}，模型: {resolved_embed_model}")
        self.embeddings = OpenAIEmbeddings(
            openai_api_base=resolved_api_base,
            openai_api_key=resolved_api_key,
            model=resolved_embed_model
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=120,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " ", ""]
        )
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.collection_name = collection_name
        self.document_count = 0
        # 仅在明确要求时重置
        if reset and os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
        os.makedirs(persist_directory, exist_ok=True)
        # 预先初始化一个持久化的空集合，便于后续追加与检索
        try:
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
        except Exception as e:
            log.error(f"[RAG] 初始化向量存储失败: {e}")
            self.vectorstore = None
        # 去重集合，避免重复块污染召回
        self._seen_hashes = set()
    
    async def add_webpage_content(self, url: str, text_content: str, metadata: Dict[str, Any] = None):
        if not text_content or len(text_content.strip()) < 50:
            log.info(f"[RAG] 跳过内容过短的网页: {url}")
            return
        try:
            log.info(f"[RAG] 正在添加网页内容: {url} (长度: {len(text_content)} 字符)")
            # 基础清洗
            cleaned = re.sub(r"\s+", " ", text_content).strip()
            chunks = self.text_splitter.split_text(cleaned)
            log.info(f"[RAG] 文本已分成 {len(chunks)} 个块")
            documents = []
            for i, chunk in enumerate(chunks):
                if not chunk or len(chunk.strip()) < 80:
                    continue
                # 内容去重
                digest = hashlib.sha1(chunk.strip().encode("utf-8")).hexdigest()
                if digest in self._seen_hashes:
                    continue
                self._seen_hashes.add(digest)
                doc_metadata = {
                    "source_url": url,
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "timestamp": datetime.now().isoformat()
                }
                if metadata:
                    doc_metadata.update(metadata)
                documents.append(Document(page_content=chunk, metadata=doc_metadata))
            if not documents:
                log.warning(f"[RAG] 清洗/去重后无有效文档块可添加: {url}")
                return
            if self.vectorstore is None:
                # 兜底：如果之前初始化失败，则在首次添加时创建
                self.vectorstore = await asyncio.to_thread(
                    Chroma.from_documents,
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory,
                )
            else:
                await asyncio.to_thread(self.vectorstore.add_documents, documents)
            # 立即持久化，保证下次运行可用
            try:
                await asyncio.to_thread(self.vectorstore.persist)
            except Exception as e:
                log.error(f"[RAG] 持久化失败: {e}")
            self.document_count += len(documents)
            log.info(f"[RAG] 成功添加 {len(documents)} 个文档块，总计: {self.document_count} 块")
        except Exception as e:
            log.error(f"[RAG] 添加网页内容时出错 ({url}): {e}")
    
    async def get_context_for_single_query(self, query: str, max_chars: int = 18000) -> str:
        """获取单个查询的上下文"""
        if self.vectorstore is None:
            log.warning("[RAG] 向量存储为空，无法检索")
            return ""
        try:
            log.info(f"[RAG] 检索查询: {query[:50]}...")
            mmr_docs = await asyncio.to_thread(
                self.vectorstore.max_marginal_relevance_search,
                query,
                k=15,
                fetch_k=60,
                lambda_mult=0.5
            )
            
            # 构建上下文
            context_parts = []
            total_chars = 0
            seen_urls = set()
            for doc in mmr_docs:
                source_url = doc.metadata.get("source_url", "unknown")
                content = doc.page_content
                if source_url not in seen_urls:
                    header = f"\n--- Source: {source_url} ---\n"
                    context_parts.append(header)
                    total_chars += len(header)
                    seen_urls.add(source_url)
                if total_chars + len(content) > max_chars:
                    remaining = max_chars - total_chars
                    if remaining > 100:
                        context_parts.append(content[:remaining] + "...[truncated]")
                    break
                context_parts.append(content + "\n")
                total_chars += len(content) + 1
            
            context = "".join(context_parts)
            log.info(f"[RAG] 查询检索完成: {len(context)} 字符，来自 {len(seen_urls)} 个不同来源")
            return context
        except Exception as e:
            log.error(f"[RAG] 检索查询 '{query}' 时出错: {e}")
            return ""
    
    async def get_context_for_analysis(self, objective: str, max_chars: int = 20000, queries: List[str] = None) -> str:
        """获取用于分析的上下文，支持多个查询（合并结果）"""
        if self.vectorstore is None:
            log.warning("[RAG] 向量存储为空，无法检索")
            return ""
        try:
            # 如果没有提供查询，使用原始objective
            if queries is None or len(queries) == 0:
                queries = [objective]
            
            log.info(f"[RAG] 使用 {len(queries)} 个查询进行检索")
            all_docs = []
            seen_doc_hashes = set()
            
            # 对每个查询进行检索
            for i, query in enumerate(queries, 1):
                log.info(f"[RAG] 查询 {i}/{len(queries)}: {query[:50]}...")
                try:
                    mmr_docs = await asyncio.to_thread(
                        self.vectorstore.max_marginal_relevance_search,
                        query,
                        k=15,  # 每个查询检索更少的文档，但多个查询会合并
                        fetch_k=60,
                        lambda_mult=0.5
                    )
                    # 去重：使用内容hash避免重复
                    for doc in mmr_docs:
                        doc_hash = hashlib.sha1(doc.page_content.encode("utf-8")).hexdigest()
                        if doc_hash not in seen_doc_hashes:
                            seen_doc_hashes.add(doc_hash)
                            all_docs.append(doc)
                except Exception as e:
                    log.error(f"[RAG] 查询 '{query}' 检索失败: {e}")
            
            log.info(f"[RAG] 合并后共获得 {len(all_docs)} 个去重文档")
            
            # 构建上下文
            context_parts = []
            total_chars = 0
            seen_urls = set()
            for doc in all_docs:
                source_url = doc.metadata.get("source_url", "unknown")
                content = doc.page_content
                if source_url not in seen_urls:
                    header = f"\n--- Source: {source_url} ---\n"
                    context_parts.append(header)
                    total_chars += len(header)
                    seen_urls.add(source_url)
                if total_chars + len(content) > max_chars:
                    remaining = max_chars - total_chars
                    if remaining > 100:
                        context_parts.append(content[:remaining] + "...[truncated]")
                    break
                context_parts.append(content + "\n")
                total_chars += len(content) + 1
            
            context = "".join(context_parts)
            log.info(f"[RAG] 生成分析上下文: {len(context)} 字符，来自 {len(seen_urls)} 个不同来源")
            return context
        except Exception as e:
            log.error(f"[RAG] 检索内容时出错: {e}")
            return ""
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_documents": self.document_count,
            "vectorstore_ready": self.vectorstore is not None
        }


