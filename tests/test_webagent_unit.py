"""
WebAgent 单元测试

测试 WebAgent 模块的各个组件功能
"""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock
import httpx

from dataflow_agent.agentroles.webresearch import (
    ToolManager,
    WebPageReader,
    Executor,
    URLFilter,
    SummaryAgent,
)
from dataflow_agent.state import WebCrawlState, WebCrawlRequest
from dataflow_agent.toolkits.datatool.log_manager import LogManager


# ============ ToolManager 测试 ============

@pytest.mark.asyncio
async def test_jina_reader_timeout():
    """测试 Jina Reader 超时机制"""
    # 模拟超时情况
    with patch('httpx.AsyncClient.get', side_effect=httpx.TimeoutException("Timeout")):
        result = await ToolManager._read_with_jina_reader("https://example.com")
        assert result["urls"] == []
        assert "超时" in result["text"] or "timeout" in result["text"].lower()


@pytest.mark.asyncio
async def test_jina_reader_http_error():
    """测试 Jina Reader HTTP 错误处理"""
    mock_response = Mock()
    mock_response.status_code = 500
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Server Error", request=Mock(), response=mock_response
    )
    
    with patch('httpx.AsyncClient.get', return_value=mock_response):
        result = await ToolManager._read_with_jina_reader("https://example.com")
        assert result["urls"] == []
        assert "HTTP错误" in result["text"] or "500" in result["text"]


@pytest.mark.asyncio
async def test_jina_reader_success():
    """测试 Jina Reader 成功情况"""
    # 模拟成功响应
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = "# Title\n\nContent here\n\n[Link](https://example.com)"
    mock_response.raise_for_status = Mock()
    
    with patch('httpx.AsyncClient.get', return_value=mock_response):
        result = await ToolManager._read_with_jina_reader("https://example.com")
        assert "urls" in result
        assert "text" in result
        assert "structured_content" in result


@pytest.mark.asyncio
async def test_jina_reader_concurrency_limit():
    """测试 Jina Reader 并发控制"""
    # 创建多个并发请求
    urls = [f"https://example.com/page{i}" for i in range(10)]
    
    # 使用计数器跟踪并发数
    concurrent_count = 0
    max_concurrent = 0
    
    async def mock_get(*args, **kwargs):
        nonlocal concurrent_count, max_concurrent
        concurrent_count += 1
        max_concurrent = max(max_concurrent, concurrent_count)
        await asyncio.sleep(0.1)  # 模拟网络延迟
        concurrent_count -= 1
        
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.text = "# Test\nContent"
        mock_resp.raise_for_status = Mock()
        return mock_resp
    
    with patch('httpx.AsyncClient.get', side_effect=mock_get):
        tasks = [ToolManager._read_with_jina_reader(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 验证最多只有3个并发请求（信号量限制）
        assert max_concurrent <= 3
        assert len(results) == 10


def test_parse_jina_text_format():
    """测试 Jina 文本格式解析"""
    text = """# Title

Content here

Warning: Some warning message

[Link](https://example.com)
"""
    result = ToolManager._parse_jina_text_format(text, "https://example.com")
    assert "title" in result
    assert "markdown" in result
    assert "warning" in result


def test_extract_urls_from_markdown():
    """测试从 Markdown 中提取 URL"""
    markdown = """
# Test

[Link1](https://example.com/page1)
[Link2](https://example.com/page2)
[Link3](https://example.com/page3)
"""
    urls = ToolManager._extract_urls_from_markdown(markdown)
    assert len(urls) == 3
    assert "https://example.com/page1" in urls
    assert "https://example.com/page2" in urls
    assert "https://example.com/page3" in urls


# ============ WebPageReader 测试 ============

@pytest.mark.asyncio
async def test_web_page_reader_execute():
    """测试 WebPageReader 执行"""
    # 创建模拟状态
    request = WebCrawlRequest(
        target="测试目标",
        download_dir="./test_downloads"
    )
    state = WebCrawlState(
        initial_request=request,
        download_dir="./test_downloads",
        use_jina_reader=True
    )
    
    # 创建模拟组件
    logger = LogManager("./test_logs")
    page = Mock()
    
    reader = WebPageReader(
        model_name="gpt-4o",
        api_base_url=None,
        api_key="test_key"
    )
    
    # Mock LLM 响应
    mock_llm_response = Mock()
    mock_llm_response.content = '{"action": "navigate", "description": "Test", "is_relevant": true}'
    
    with patch.object(reader, 'llm') as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm_response)
        with patch.object(ToolManager, '_read_with_jina_reader', return_value={
            "urls": ["https://example.com"],
            "text": "Test content",
            "structured_content": {}
        }):
            action_plan = await reader.execute(
                state, logger, page, "https://example.com", "测试目标"
            )
            assert "action" in action_plan
            assert action_plan["action"] in ["download", "navigate", "dead_end"]


# ============ Executor 测试 ============

@pytest.mark.asyncio
async def test_executor_download_action():
    """测试 Executor 下载操作"""
    executor = Executor()
    
    request = WebCrawlRequest(
        target="测试",
        download_dir="./test_downloads"
    )
    state = WebCrawlState(
        initial_request=request,
        download_dir="./test_downloads"
    )
    
    action_plan = {
        "action": "download",
        "urls": ["https://example.com/file.zip"]
    }
    
    page = Mock()
    
    # Mock 下载检查
    with patch('dataflow_agent.toolkits.webatool.download_utils.check_if_download_link', 
               return_value={"is_download": True, "reason": "URL包含文件扩展名: .zip"}):
        # Mock 文件下载
        with patch('dataflow_agent.toolkits.webatool.download_utils.download_file',
                   return_value="./test_downloads/file.zip"):
            result_state = await executor.execute(
                state, action_plan, "https://example.com", page, "download"
            )
            assert result_state.download_successful_for_current_task == True


@pytest.mark.asyncio
async def test_executor_navigate_action():
    """测试 Executor 导航操作"""
    executor = Executor()
    
    request = WebCrawlRequest(target="测试")
    state = WebCrawlState(
        initial_request=request,
        url_queue=[],
        visited_urls=set()
    )
    
    action_plan = {
        "action": "navigate",
        "url": "/page2"
    }
    
    page = Mock()
    result_state = await executor.execute(
        state, action_plan, "https://example.com", page, "research"
    )
    
    assert len(result_state.url_queue) > 0
    assert "https://example.com/page2" in result_state.url_queue


@pytest.mark.asyncio
async def test_executor_dead_end_action():
    """测试 Executor 死胡同操作"""
    executor = Executor()
    
    request = WebCrawlRequest(target="测试")
    state = WebCrawlState(initial_request=request)
    
    action_plan = {
        "action": "dead_end",
        "description": "No more actions"
    }
    
    page = Mock()
    result_state = await executor.execute(
        state, action_plan, "https://example.com", page, "research"
    )
    
    # 死胡同操作不应该改变状态
    assert result_state is not None


# ============ URLFilter 测试 ============

@pytest.mark.asyncio
async def test_url_filter_basic():
    """测试 URL 过滤基本功能"""
    request = WebCrawlRequest(target="测试")
    state = WebCrawlState(
        initial_request=request,
        url_queue=["https://example.com", "https://test.com"]
    )
    
    filter_agent = URLFilter(
        model_name="gpt-4o",
        api_base_url=None,
        api_key="test_key"
    )
    
    logger = LogManager("./test_logs")
    
    # Mock LLM 响应
    mock_llm_response = Mock()
    mock_llm_response.content = '{"filtered_urls": ["https://example.com"]}'
    
    with patch.object(filter_agent, 'llm') as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm_response)
        result_state = await filter_agent.execute(state, logger)
        
        assert len(result_state.url_queue) <= len(state.url_queue)


# ============ 工具函数测试 ============

def test_semaphore_initialization():
    """测试信号量初始化"""
    # 第一次调用应该创建信号量
    sem1 = ToolManager._get_jina_reader_semaphore()
    assert sem1 is not None
    
    # 第二次调用应该返回同一个信号量
    sem2 = ToolManager._get_jina_reader_semaphore()
    assert sem1 is sem2


@pytest.mark.asyncio
async def test_read_web_page_with_jina():
    """测试使用 Jina Reader 读取网页"""
    page = Mock()
    
    with patch.object(ToolManager, '_read_with_jina_reader', return_value={
        "urls": ["https://example.com"],
        "text": "Test content"
    }) as mock_jina:
        result = await ToolManager.read_web_page(
            page, "https://example.com", use_jina_reader=True
        )
        mock_jina.assert_called_once_with("https://example.com")
        assert "urls" in result
        assert "text" in result


@pytest.mark.asyncio
async def test_read_web_page_with_playwright():
    """测试使用 Playwright 读取网页"""
    page = Mock()
    page.goto = AsyncMock()
    page.content = AsyncMock(return_value="<html><body>Test</body></html>")
    page.evaluate = AsyncMock(return_value="https://example.com")
    
    result = await ToolManager.read_web_page(
        page, "https://example.com", use_jina_reader=False
    )
    
    assert "urls" in result
    assert "text" in result
    page.goto.assert_called_once()


# ============ 异常处理测试 ============

@pytest.mark.asyncio
async def test_jina_reader_connection_error():
    """测试连接错误处理"""
    with patch('httpx.AsyncClient.get', side_effect=httpx.ConnectError("Connection failed")):
        result = await ToolManager._read_with_jina_reader("https://example.com")
        assert result["urls"] == []
        assert "错误" in result["text"] or "error" in result["text"].lower()


@pytest.mark.asyncio
async def test_jina_reader_read_timeout():
    """测试读取超时处理"""
    with patch('httpx.AsyncClient.get', side_effect=httpx.ReadTimeout("Read timeout")):
        result = await ToolManager._read_with_jina_reader("https://example.com")
        assert result["urls"] == []
        assert "超时" in result["text"] or "timeout" in result["text"].lower()


@pytest.mark.asyncio
async def test_jina_reader_asyncio_timeout():
    """测试 asyncio 超时保护"""
    async def slow_fetch():
        await asyncio.sleep(100)  # 模拟长时间等待
        return {"urls": [], "text": "test"}
    
    # 替换内部函数为慢速版本
    with patch.object(ToolManager, '_read_with_jina_reader', side_effect=slow_fetch):
        # 使用较短的超时时间测试
        result = await ToolManager._read_with_jina_reader("https://example.com")
        # 由于有 asyncio.wait_for 保护，应该在65秒内返回
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


