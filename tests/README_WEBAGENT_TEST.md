# WebAgent 模块企业级测试方案

## 📋 测试概述

本文档描述了 WebAgent 模块的企业级测试策略，包括测试类型、测试步骤、测试用例设计和最佳实践。

## 🎯 测试目标

1. **功能正确性**: 确保所有功能按预期工作
2. **稳定性**: 确保系统在各种场景下稳定运行
3. **性能**: 确保系统满足性能要求
4. **可靠性**: 确保系统能够处理异常情况
5. **安全性**: 确保系统安全可靠

## 📊 测试金字塔

```
        /\
       /E2E\          ← 端到端测试 (10%)
      /------\
     /Integration\    ← 集成测试 (20%)
    /------------\
   /   Unit Test  \   ← 单元测试 (70%)
  /----------------\
```

## 🧪 测试类型详解

### 1. 单元测试 (Unit Tests)

**目标**: 测试单个组件或函数的独立功能

**测试范围**:
- `ToolManager` 类的各个静态方法
- `WebPageReader` 的页面解析逻辑
- `Executor` 的执行逻辑
- `SummaryAgent` 的总结生成
- URL 过滤和处理逻辑
- Jina Reader 的超时和并发控制

**测试文件**: `tests/test_webagent_unit.py`

**示例测试用例**:
```python
# 测试 Jina Reader 超时机制
async def test_jina_reader_timeout():
    """测试 Jina Reader 在超时情况下能正确返回"""
    result = await ToolManager._read_with_jina_reader("https://invalid-timeout-url.com")
    assert result["urls"] == []
    assert "超时" in result["text"] or "timeout" in result["text"].lower()

# 测试信号量并发控制
async def test_jina_reader_concurrency_limit():
    """测试并发请求数量被正确限制"""
    urls = [f"https://example.com/page{i}" for i in range(10)]
    tasks = [ToolManager._read_with_jina_reader(url) for url in urls]
    # 验证最多只有3个并发请求
    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert len([r for r in results if not isinstance(r, Exception)]) <= 3

# 测试 URL 过滤
def test_url_filtering():
    """测试 URL 过滤逻辑"""
    filter_agent = URLFilter(...)
    urls = ["https://example.com", "javascript:void(0)", "mailto:test@example.com"]
    filtered = filter_agent._filter_urls(urls)
    assert "javascript:" not in filtered
    assert "mailto:" not in filtered
```

### 2. 集成测试 (Integration Tests)

**目标**: 测试多个组件协同工作

**测试范围**:
- WebPageReader + Executor 的完整流程
- SummaryAgent + QueryGenerator 的协同
- WebResearchAgent 的完整工作流
- 与外部服务（Jina API、搜索引擎）的集成

**测试文件**: `tests/test_webagent_integration.py`

**示例测试用例**:
```python
async def test_web_page_reader_integration():
    """测试网页读取的完整流程"""
    state = WebCrawlState(...)
    reader = WebPageReader(...)
    action_plan = await reader.execute(state, logger, page, url, objective)
    assert "action" in action_plan
    assert action_plan["action"] in ["download", "navigate", "dead_end"]

async def test_executor_download_flow():
    """测试下载执行流程"""
    executor = Executor()
    action_plan = {"action": "download", "urls": ["https://example.com/file.zip"]}
    result_state = await executor.execute(state, action_plan, source_url, page, "download")
    assert result_state.download_successful_for_current_task == True
```

### 3. 端到端测试 (E2E Tests)

**目标**: 测试完整的用户场景

**测试范围**:
- 完整的网页爬取流程
- 从搜索到下载的完整链路
- 多任务并发处理
- 错误恢复机制

**测试文件**: `tests/test_webagent_e2e.py`

**示例测试用例**:
```python
async def test_complete_web_crawl_flow():
    """测试完整的网页爬取流程"""
    request = DataCollectionRequest(
        target="测试数据集下载",
        download_dir="./test_downloads"
    )
    state = DataCollectionState(request=request)
    
    orchestrator = WebCrawlOrchestrator(...)
    final_state = await orchestrator.run_with_langgraph(request.target)
    
    assert len(final_state.crawled_data) > 0
    assert os.path.exists(request.download_dir)

async def test_concurrent_download_tasks():
    """测试并发下载任务处理"""
    # 创建多个下载任务
    tasks = [create_download_task(url) for url in test_urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    # 验证所有任务都完成，没有卡死
    assert all(not isinstance(r, Exception) for r in results)
```

### 4. 性能测试 (Performance Tests)

**目标**: 验证系统性能指标

**测试指标**:
- 响应时间
- 吞吐量
- 资源使用（CPU、内存）
- 并发处理能力

**测试文件**: `tests/test_webagent_performance.py`

**示例测试用例**:
```python
@pytest.mark.performance
async def test_jina_reader_response_time():
    """测试 Jina Reader 响应时间"""
    start_time = time.time()
    result = await ToolManager._read_with_jina_reader("https://example.com")
    elapsed = time.time() - start_time
    assert elapsed < 65.0  # 应该在超时时间内完成

@pytest.mark.performance
async def test_concurrent_processing_throughput():
    """测试并发处理吞吐量"""
    urls = generate_test_urls(100)
    start_time = time.time()
    results = await process_urls_parallel(urls, ...)
    elapsed = time.time() - start_time
    throughput = len(results) / elapsed
    assert throughput > 10  # 每秒至少处理10个URL
```

### 5. 压力测试 (Stress Tests)

**目标**: 测试系统在极限条件下的表现

**测试场景**:
- 大量并发请求
- 长时间运行
- 资源耗尽情况
- 网络异常情况

**测试文件**: `tests/test_webagent_stress.py`

**示例测试用例**:
```python
@pytest.mark.stress
async def test_high_concurrency_stress():
    """测试高并发压力"""
    # 创建100个并发任务
    tasks = [create_web_crawl_task() for _ in range(100)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    # 验证系统没有崩溃
    assert len(results) == 100
    # 验证错误率在可接受范围内
    error_rate = sum(1 for r in results if isinstance(r, Exception)) / len(results)
    assert error_rate < 0.1  # 错误率低于10%

@pytest.mark.stress
async def test_long_running_stability():
    """测试长时间运行稳定性"""
    # 运行1小时，验证没有内存泄漏
    start_memory = get_memory_usage()
    for i in range(1000):
        await run_web_crawl_cycle()
        if i % 100 == 0:
            current_memory = get_memory_usage()
            memory_growth = current_memory - start_memory
            assert memory_growth < 500 * 1024 * 1024  # 内存增长小于500MB
```

### 6. 异常测试 (Exception Tests)

**目标**: 测试异常处理和错误恢复

**测试场景**:
- 网络超时
- API 调用失败
- 无效 URL
- 文件下载失败
- 资源不足

**测试文件**: `tests/test_webagent_exceptions.py`

**示例测试用例**:
```python
async def test_network_timeout_handling():
    """测试网络超时处理"""
    result = await ToolManager._read_with_jina_reader("https://httpstat.us/200?sleep=100000")
    assert result["urls"] == []
    assert "超时" in result["text"] or "timeout" in result["text"].lower()

async def test_invalid_url_handling():
    """测试无效URL处理"""
    result = await ToolManager.read_web_page(page, "invalid-url", use_jina_reader=True)
    assert "urls" in result
    assert "text" in result

async def test_api_failure_recovery():
    """测试API失败后的恢复机制"""
    # 模拟API失败
    with patch('httpx.AsyncClient.get', side_effect=httpx.HTTPError("API Error")):
        result = await ToolManager._read_with_jina_reader("https://example.com")
        assert result["urls"] == []
        assert "错误" in result["text"] or "error" in result["text"].lower()
```

### 7. 安全测试 (Security Tests)

**目标**: 测试系统安全性

**测试场景**:
- URL 注入攻击
- 恶意文件下载
- 敏感信息泄露
- 资源耗尽攻击

**测试文件**: `tests/test_webagent_security.py`

**示例测试用例**:
```python
def test_url_injection_prevention():
    """测试URL注入防护"""
    malicious_urls = [
        "javascript:alert('xss')",
        "data:text/html,<script>alert('xss')</script>",
        "file:///etc/passwd"
    ]
    for url in malicious_urls:
        result = check_if_download_link(url)
        assert result["is_download"] == False

def test_sensitive_info_leak():
    """测试敏感信息泄露"""
    # 验证日志中不包含敏感信息
    log_output = capture_logs()
    assert "api_key" not in log_output.lower()
    assert "password" not in log_output.lower()
```

## 📝 测试步骤

### 阶段1: 测试准备

1. **环境配置**
   ```bash
   # 安装测试依赖
   pip install -e ".[dev]"
   
   # 配置环境变量
   export DF_API_KEY=your_api_key
   export TAVILY_API_KEY=your_tavily_key
   export OPENAI_API_KEY=your_openai_key
   ```

2. **测试数据准备**
   - 创建测试URL列表
   - 准备模拟响应数据
   - 设置测试目录结构

3. **Mock服务设置**
   - 设置HTTP mock服务器
   - 配置测试数据库
   - 准备测试文件

### 阶段2: 单元测试执行

```bash
# 运行所有单元测试
pytest tests/test_webagent_unit.py -v

# 运行特定测试
pytest tests/test_webagent_unit.py::test_jina_reader_timeout -v

# 生成覆盖率报告
pytest tests/test_webagent_unit.py --cov=dataflow_agent.agentroles.webresearch --cov-report=html
```

### 阶段3: 集成测试执行

```bash
# 运行集成测试（需要真实API）
pytest tests/test_webagent_integration.py -v -s

# 标记为集成测试
pytest -m integration tests/ -v
```

### 阶段4: 端到端测试执行

```bash
# 运行E2E测试
pytest tests/test_webagent_e2e.py -v -s

# 使用标记
pytest -m e2e tests/ -v
```

### 阶段5: 性能测试执行

```bash
# 运行性能测试
pytest tests/test_webagent_performance.py -v --benchmark-only

# 生成性能报告
pytest tests/test_webagent_performance.py --benchmark-json=benchmark.json
```

### 阶段6: 压力测试执行

```bash
# 运行压力测试（需要较长时间）
pytest tests/test_webagent_stress.py -v -s --timeout=3600

# 监控资源使用
pytest tests/test_webagent_stress.py -v --profile
```

## 🛠️ 测试工具和框架

### 推荐工具

1. **pytest**: 主要测试框架
2. **pytest-asyncio**: 异步测试支持
3. **pytest-cov**: 代码覆盖率
4. **pytest-mock**: Mock支持
5. **pytest-benchmark**: 性能测试
6. **httpx**: HTTP客户端（用于测试）
7. **pytest-timeout**: 超时控制
8. **pytest-xdist**: 并行测试

### 配置文件

在 `pyproject.toml` 中添加：

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
asyncio_mode = "auto"
markers = [
    "unit: 单元测试",
    "integration: 集成测试",
    "e2e: 端到端测试",
    "performance: 性能测试",
    "stress: 压力测试",
    "security: 安全测试",
]
timeout = 300
```

## 📊 测试覆盖率目标

- **单元测试覆盖率**: ≥ 80%
- **集成测试覆盖率**: ≥ 60%
- **关键路径覆盖率**: 100%

## 🔄 CI/CD 集成

### GitHub Actions 示例

```yaml
name: WebAgent Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - run: pip install -e ".[dev]"
      - run: pytest tests/test_webagent_unit.py --cov --cov-report=xml
      - run: pytest tests/test_webagent_integration.py -m integration
      - uses: codecov/codecov-action@v3
```

## 📈 测试报告

### 生成HTML报告

```bash
pytest tests/ --html=report.html --self-contained-html
```

### 生成覆盖率报告

```bash
pytest tests/ --cov=dataflow_agent --cov-report=html
# 打开 htmlcov/index.html 查看
```

## ✅ 测试检查清单

### 功能测试
- [ ] Jina Reader 超时机制正常工作
- [ ] 并发控制信号量正常工作
- [ ] URL 过滤逻辑正确
- [ ] 下载功能正常
- [ ] 页面解析正确
- [ ] 错误处理完善

### 性能测试
- [ ] 响应时间满足要求
- [ ] 并发处理能力满足要求
- [ ] 资源使用在合理范围内
- [ ] 没有内存泄漏

### 稳定性测试
- [ ] 长时间运行稳定
- [ ] 异常情况能正确恢复
- [ ] 并发场景不卡死
- [ ] 超时机制有效

### 安全测试
- [ ] URL注入防护有效
- [ ] 敏感信息不泄露
- [ ] 资源耗尽攻击防护有效

## 🚀 快速开始

1. **创建测试文件结构**
   ```bash
   mkdir -p tests/webagent
   touch tests/test_webagent_unit.py
   touch tests/test_webagent_integration.py
   touch tests/test_webagent_e2e.py
   ```

2. **编写第一个测试**
   ```python
   import pytest
   from dataflow_agent.agentroles.webresearch import ToolManager
   
   @pytest.mark.asyncio
   async def test_jina_reader_basic():
       result = await ToolManager._read_with_jina_reader("https://example.com")
       assert "urls" in result
       assert "text" in result
   ```

3. **运行测试**
   ```bash
   pytest tests/test_webagent_unit.py -v
   ```

## 📚 参考资源

- [pytest 文档](https://docs.pytest.org/)
- [pytest-asyncio 文档](https://pytest-asyncio.readthedocs.io/)
- [企业级测试最佳实践](https://martinfowler.com/articles/practical-test-pyramid.html)

## 🔗 相关文档

- [集成测试说明](./README_INTEGRATION_TEST.md)
- [项目README](../README.md)


