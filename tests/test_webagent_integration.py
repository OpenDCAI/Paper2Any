"""
WebAgent 集成测试

测试 WebAgent 模块的组件协同工作
需要真实 API 调用，运行前请设置环境变量
"""

from __future__ import annotations

import os
import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path

from dataflow_agent.agentroles.webresearch import (
    ToolManager,
    WebPageReader,
    Executor,
    WebResearchAgent,
)
from dataflow_agent.state import WebCrawlState, WebCrawlRequest
from dataflow_agent.toolkits.datatool.log_manager import LogManager


# 检查环境变量
REQUIRES_API = pytest.mark.skipif(
    not os.getenv("DF_API_KEY") or os.getenv("DF_API_KEY") == "test",
    reason="需要设置 DF_API_KEY 环境变量才能运行集成测试"
)


@pytest.fixture
def temp_dir():
    """创建临时目录"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def test_state(temp_dir):
    """创建测试状态"""
    request = WebCrawlRequest(
        target="测试数据集",
        download_dir=temp_dir,
        model="gpt-4o",
        api_key=os.getenv("DF_API_KEY"),
        chat_api_url=os.getenv("DF_API_URL", "http://123.129.219.111:3000/v1"),
    )
    return WebCrawlState(
        initial_request=request,
        download_dir=temp_dir,
        use_jina_reader=True,
    )


@pytest.fixture
def logger(temp_dir):
    """创建日志管理器"""
    log_dir = os.path.join(temp_dir, "logs")
    return LogManager(log_dir)


# ============ ToolManager 集成测试 ============

@pytest.mark.asyncio
@pytest.mark.integration
@REQUIRES_API
async def test_jina_reader_real_api():
    """测试真实的 Jina Reader API 调用"""
    result = await ToolManager._read_with_jina_reader("https://example.com")
    
    assert "urls" in result
    assert "text" in result
    assert "structured_content" in result
    
    # 验证返回的内容不为空
    assert len(result["text"]) > 0


@pytest.mark.asyncio
@pytest.mark.integration
@REQUIRES_API
async def test_jina_reader_timeout_integration():
    """测试 Jina Reader 超时机制（真实API）"""
    # 使用一个可能较慢的URL测试超时
    result = await ToolManager._read_with_jina_reader("https://httpbin.org/delay/70")
    
    # 应该在65秒超时内返回
    assert result is not None
    assert "urls" in result
    # 如果超时，应该返回错误信息
    if "超时" in result["text"] or "timeout" in result["text"].lower():
        assert result["urls"] == []


@pytest.mark.asyncio
@pytest.mark.integration
@REQUIRES_API
async def test_jina_reader_concurrent_real():
    """测试真实API的并发控制"""
    urls = [
        "https://example.com",
        "https://httpbin.org/get",
        "https://jsonplaceholder.typicode.com/posts/1",
    ]
    
    start_time = asyncio.get_event_loop().time()
    tasks = [ToolManager._read_with_jina_reader(url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = asyncio.get_event_loop().time() - start_time
    
    # 验证所有请求都完成
    assert len(results) == 3
    # 验证没有异常（或异常被正确处理）
    assert all(
        isinstance(r, dict) or isinstance(r, Exception) for r in results
    )
    
    # 验证并发控制：由于信号量限制为3，3个请求应该几乎同时完成
    # 如果顺序执行，时间会明显更长
    print(f"并发请求耗时: {elapsed:.2f}秒")


# ============ WebPageReader 集成测试 ============

@pytest.mark.asyncio
@pytest.mark.integration
@REQUIRES_API
async def test_web_page_reader_integration(test_state, logger, temp_dir):
    """测试 WebPageReader 完整流程"""
    from playwright.async_api import async_playwright
    
    reader = WebPageReader(
        model_name="gpt-4o",
        api_base_url=os.getenv("DF_API_URL", "http://123.129.219.111:3000/v1"),
        api_key=os.getenv("DF_API_KEY"),
    )
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        
        try:
            action_plan = await reader.execute(
                test_state,
                logger,
                page,
                "https://example.com",
                "测试目标",
                is_research=False,
            )
            
            assert "action" in action_plan
            assert action_plan["action"] in ["download", "navigate", "dead_end"]
            assert "discovered_urls" in action_plan
            
        finally:
            await browser.close()


# ============ Executor 集成测试 ============

@pytest.mark.asyncio
@pytest.mark.integration
@REQUIRES_API
async def test_executor_download_flow_integration(test_state, logger, temp_dir):
    """测试 Executor 下载流程集成"""
    from playwright.async_api import async_playwright
    
    executor = Executor()
    
    action_plan = {
        "action": "download",
        "urls": ["https://httpbin.org/robots.txt"],  # 使用小文件测试
    }
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        
        try:
            result_state = await executor.execute(
                test_state,
                action_plan,
                "https://httpbin.org",
                page,
                "download",
            )
            
            # 验证下载状态
            assert hasattr(result_state, "download_successful_for_current_task")
            
        finally:
            await browser.close()


# ============ WebResearchAgent 集成测试 ============

@pytest.mark.asyncio
@pytest.mark.integration
@REQUIRES_API
async def test_web_research_agent_basic(test_state, logger, temp_dir):
    """测试 WebResearchAgent 基本功能"""
    from playwright.async_api import async_playwright
    
    agent = WebResearchAgent(
        model_name="gpt-4o",
        api_base_url=os.getenv("DF_API_URL", "http://123.129.219.111:3000/v1"),
        api_key=os.getenv("DF_API_KEY"),
        concurrent_pages=2,  # 降低并发数用于测试
    )
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        
        try:
            # 测试处理单个URL
            entries = [{"url": "https://example.com", "objective": "测试"}]
            
            results = await agent.process_urls_parallel(
                context,
                entries,
                test_state,
                "测试目标",
                "research",
                is_research_phase=True,
                logger=logger,
            )
            
            assert len(results) == 1
            assert "success" in results[0]
            
        finally:
            await browser.close()


# ============ 端到端集成测试 ============

@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.e2e
@REQUIRES_API
async def test_complete_web_crawl_flow(temp_dir):
    """测试完整的网页爬取流程"""
    from dataflow_agent.workflow.wf_data_collector import WebCrawlOrchestrator
    
    request = WebCrawlRequest(
        target="Python 机器学习数据集",
        download_dir=temp_dir,
        model="gpt-4o",
        api_key=os.getenv("DF_API_KEY"),
        chat_api_url=os.getenv("DF_API_URL", "http://123.129.219.111:3000/v1"),
        max_download_subtasks=2,  # 限制下载任务数量
    )
    
    orchestrator = WebCrawlOrchestrator(
        api_base_url=request.chat_api_url,
        api_key=request.api_key,
        model_name=request.model,
        download_dir=request.download_dir,
        max_crawl_cycles_per_task=2,  # 限制循环次数
        max_crawl_cycles_for_research=1,
        search_engine="tavily",
        use_jina_reader=True,
        enable_rag=False,  # 测试时禁用RAG以加快速度
        concurrent_pages=2,
    )
    
    # 运行爬取流程
    final_state = await orchestrator.run_with_langgraph(request.target)
    
    # 验证结果
    assert final_state is not None
    assert hasattr(final_state, "crawled_data")
    assert hasattr(final_state, "sub_tasks")
    
    print(f"爬取完成，收集到 {len(final_state.crawled_data)} 条数据")
    print(f"剩余子任务: {len(final_state.sub_tasks)}")


# ============ 性能集成测试 ============

@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.performance
@REQUIRES_API
async def test_jina_reader_performance():
    """测试 Jina Reader 性能"""
    import time
    
    urls = [
        "https://example.com",
        "https://httpbin.org/get",
    ]
    
    start_time = time.time()
    tasks = [ToolManager._read_with_jina_reader(url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = time.time() - start_time
    
    # 验证性能：两个请求应该在合理时间内完成
    assert elapsed < 130.0  # 每个请求最多65秒，两个请求并发应该更快
    assert len(results) == 2
    
    print(f"性能测试: 2个并发请求耗时 {elapsed:.2f}秒")


# ============ 错误恢复集成测试 ============

@pytest.mark.asyncio
@pytest.mark.integration
@REQUIRES_API
async def test_error_recovery():
    """测试错误恢复机制"""
    # 测试无效URL
    result1 = await ToolManager._read_with_jina_reader("not-a-valid-url")
    assert result1 is not None
    assert "urls" in result1
    
    # 测试不存在的域名
    result2 = await ToolManager._read_with_jina_reader("https://this-domain-does-not-exist-12345.com")
    assert result2 is not None
    assert "urls" in result2
    # 应该返回错误信息而不是崩溃
    assert "text" in result2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "integration"])


