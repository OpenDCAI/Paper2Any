# 多模态供应商与 API 开发指南

DataFlow-Agent 采用灵活的策略模式来支持多种多模态 AI 供应商（如 OpenAI DALL-E, Google Gemini, APIYI 等）。本指南将介绍如何扩展系统以支持新的多模态 API。

## 核心架构

多模态功能的实现主要依赖于以下几个核心组件：

1.  **`VisionLLMCaller`** (`dataflow_agent/llm_callers/image.py`):
    *   这是多模态调用的统一入口。
    *   支持 `generation` (生图), `edit` (修图), `understanding` (图像理解), `ocr` (文字识别), `video_understanding` (视频理解) 等模式。
    *   根据配置的 `mode` 调用相应的底层逻辑。

2.  **`AIProviderStrategy`** (`dataflow_agent/toolkits/multimodaltool/providers.py`):
    *   这是一个抽象基类，定义了所有多模态供应商必须实现的接口。
    *   包括请求构建 (`build_request`) 和响应解析 (`parse_response`)。

3.  **`VLMStrategy`** (`dataflow_agent/agentroles/cores/strategies.py`):
    *   Agent 执行策略的一种，负责配置和调度 `VisionLLMCaller`。

## 接口定义

所有的供应商策略都继承自 `AIProviderStrategy`。主要接口如下：

```python
class AIProviderStrategy(ABC):
    
    @abstractmethod
    def match(self, api_url: str, model: str) -> bool:
        """判断当前策略是否适用于给定的 API URL 和模型名称"""
        pass
        
    @abstractmethod
    def build_generation_request(self, api_url, model, prompt, **kwargs) -> Tuple[str, Dict, bool]:
        """构造文生图请求。返回: (url, payload, is_stream)"""
        pass

    @abstractmethod
    def parse_generation_response(self, response_data: Dict) -> str:
        """解析生图响应，返回 Base64 图片字符串"""
        pass

    # 可选实现：图生图/编辑
    def build_edit_request(self, ...): ...
    
    # 可选实现：多图编辑
    def build_multi_image_edit_request(self, ...): ...

    # 可选实现：TTS (语音合成)
    def build_tts_request(self, ...): ...
    
    # 可选实现：对话/多模态理解 (默认实现了 OpenAI 兼容格式)
    def build_chat_request(self, ...): ...
    def parse_chat_response(self, ...): ...
```

## 如何添加新供应商

要添加一个新的多模态供应商（例如 "MyNewAI"），请按照以下步骤操作：

### 步骤 1: 创建策略类

在 `dataflow_agent/toolkits/multimodaltool/providers.py` 中定义一个新的类，继承自 `AIProviderStrategy`。

```python
class MyNewAIProvider(AIProviderStrategy):
    """
    MyNewAI 服务商支持
    """
    def match(self, api_url: str, model: str) -> bool:
        # 定义匹配规则，例如检查 URL 或模型前缀
        return "mynewai.com" in api_url or model.startswith("mynewai-")

    def build_generation_request(self, api_url: str, model: str, prompt: str, **kwargs) -> Tuple[str, Dict[str, Any], bool]:
        # 构造请求
        url = f"{api_url.rstrip('/')}/v1/images/generate"
        payload = {
            "model": model,
            "text": prompt,
            "width": 1024,
            "height": 1024
        }
        is_stream = False
        return url, payload, is_stream

    def parse_generation_response(self, data: Dict[str, Any]) -> str:
        # 解析响应，提取图片 Base64
        if "data" in data and "image_base64" in data["data"]:
            return data["data"]["image_base64"]
        raise RuntimeError("Failed to parse MyNewAI response")
```

### 步骤 2: 实现高级功能 (可选)

如果该供应商支持图生图 (Image Editing) 或多模态理解，请重写相应的方法：

*   `build_edit_request`: 用于处理图像编辑任务。
*   `build_chat_request` / `parse_chat_response`: 用于处理 Vision/Chat 任务（如果 API 格式不兼容 OpenAI 标准）。

### 步骤 3: 注册策略

在 `dataflow_agent/toolkits/multimodaltool/providers.py` 文件底部的 `STRATEGIES` 列表中注册你的新策略类。**注意顺序很重要**，系统会按顺序尝试匹配。

```python
STRATEGIES = [
    ApiYiGeminiProvider(),
    MyNewAIProvider(),  # <--- 添加在这里
    ApiYiSeeDreamProvider(),
    # ...
    OpenAICompatGeminiProvider(), # 默认回退
]
```

## 调试与测试

1.  配置环境变量 `DF_API_URL` 和 `DF_API_KEY` 指向你的新服务商。
2.  使用 `VisionLLMCaller` 进行测试，或者直接运行 `dataflow_agent/llm_callers/image.py` 中的 `_quick_test` 函数（需要稍作修改以支持你的模型参数）。
3.  检查 `dataflow_agent.logger` 输出的日志以排查请求构建或响应解析的问题。

## 常见问题

*   **Multipart Upload**: 如果编辑接口需要上传文件（`multipart/form-data`），在 `build_edit_request` 中返回的 payload 应包含 `{"__is_multipart__": True, "files": ..., "data": ...}`。参考 `ApiYiGPTImageProvider` 的实现。
*   **流式响应**: 如果 API 返回流式数据 (SSE)，`build_*_request` 返回的第三个参数 `is_stream` 应设为 `True`。
