# Paper2Graph / Paper2PPT 前后端交互说明

## 1. 整体架构

- 前端项目：`frontend-workflow/`
  - 入口：`src/main.tsx`
  - 根组件：`src/App.tsx`
  - 页面组件：
    - `src/components/Paper2GraphPage.tsx`
    - `src/components/Paper2PptPage.tsx`
- 后端项目：`fastapi_app/`（假定）
  - 对应路由：
    - `POST /api/paper2graph/generate`
    - `POST /api/paper2ppt/generate`

前端通过 `fetch` 以 `multipart/form-data` 方式 POST 到上述接口，由后端生成 PPTX（二进制流），返回给前端下载。

---

## 2. App 路由和页面切换

文件：`frontend-workflow/src/App.tsx`

```tsx
import { useState } from 'react';
import ParticleBackground from './components/ParticleBackground';
import Paper2GraphPage from './components/Paper2GraphPage';
import Paper2PptPage from './components/Paper2PptPage';
import { Workflow, Zap } from 'lucide-react';

function App() {
  const [activePage, setActivePage] = useState<'paper2graph' | 'paper2ppt'>('paper2graph');

  return (
    <div className="w-screen h-screen bg-[#0a0a1a] overflow-hidden relative">
      {/* 粒子背景 */}
      <ParticleBackground />

      {/* 顶部导航栏 */}
      <header className="absolute top-0 left-0 right-0 h-16 glass-dark border-b border-white/10 z-10">
        <div className="h-full px-6 flex items-center justify-between">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-primary-500/20">
              <Workflow className="text-primary-400" size={24} />
            </div>
            <div>
              <h1 className="text-lg font-bold text-white glow-text">
                DataFlow Agent
              </h1>
              <p className="text-xs text-gray-400">Workflow Editor</p>
            </div>
          </div>

          {/* 工具栏：只保留 Paper2Graph / Paper2PPT Tab */}
          <div className="flex items-center gap-4">
            {/* 页面切换 Tab */}
            <div className="flex items-center gap-2">
              <button
                onClick={() => setActivePage('paper2graph')}
                className={`px-3 py-1.5 rounded-full text-sm ${
                  activePage === 'paper2graph'
                    ? 'bg-primary-500 text-white shadow'
                    : 'glass text-gray-300 hover:bg-white/10'
                }`}
              >
                Paper2Graph 生成科研绘图
              </button>
              <button
                onClick={() => setActivePage('paper2ppt')}
                className={`px-3 py-1.5 rounded-full text-sm ${
                  activePage === 'paper2ppt'
                    ? 'bg-primary-500 text-white shadow'
                    : 'glass text-gray-300 hover:bg-white/10'
                }`}
              >
                Paper2PPT 生成
              </button>
            </div>

            {/* 右侧操作按钮（占位，实际生成由各页面自身按钮触发） */}
            <div className="flex items-center gap-2">
              <button className="flex items-center gap-2 px-4 py-2 rounded-lg bg-primary-500 hover:bg-primary-600 transition-colors glow">
                <Zap size={18} className="text-white" />
                <span className="text-sm text-white font-medium">
                  {activePage === 'paper2graph' ? 'Paper2Graph' : 'Paper2PPT'}
                </span>
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* 主内容区：根据 activePage 渲染不同页面 */}
      <main className="absolute top-16 bottom-0 left-0 right-0 flex">
        <div className="flex-1">
          {activePage === 'paper2graph' ? <Paper2GraphPage /> : <Paper2PptPage />}
        </div>
      </main>

      {/* 底部状态栏（简化版） */}
      <footer className="absolute bottom-0 left-0 right-0 h-8 glass-dark border-t border-white/10 z-10">
        <div className="h-full px-4 flex items-center justify-between text-xs text-gray-500">
          <div className="flex items-center gap-4">
            <span>DataFlow Agent v1.0.0</span>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
              <span>就绪</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
```

App 本身只做页面切换，具体与后端交互都在子页面组件内部完成。

---

## 3. Paper2GraphPage 前端参数和调用说明

文件：`frontend-workflow/src/components/Paper2GraphPage.tsx`

### 3.1 后端接口

```ts
const BACKEND_API = '/api/paper2graph/generate';
```

开发环境下，`/api/...` 建议在 `vite.config.ts` 中配置 dev 代理，将请求转发到 FastAPI 后端（例如 `http://127.0.0.1:8000`）。

### 3.2 前端内部状态

```ts
type UploadMode = 'file' | 'url' | 'text';
type FileKind = 'pdf' | 'image' | null;

const [uploadMode, setUploadMode] = useState<UploadMode>('file');
const [selectedFile, setSelectedFile] = useState<File | null>(null);
const [fileKind, setFileKind] = useState<FileKind>(null);
const [sourceUrl, setSourceUrl] = useState('');
const [textContent, setTextContent] = useState('');

// LLM 调用配置
const [llmApiUrl, setLlmApiUrl] = useState('https://api.openai.com/v1/chat/completions');
const [apiKey, setApiKey] = useState('');
const [model, setModel] = useState('NanoBanana');

// 请求状态 / 下载信息
const [isLoading, setIsLoading] = useState(false);
const [error, setError] = useState<string | null>(null);
const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
const [lastFilename, setLastFilename] = useState('paper2graph.pptx');
const [successMessage, setSuccessMessage] = useState<string | null>(null);
```

前端通过这些 state 收集用户配置，然后在点击“生成 Paper2Graph PPTX”时统一打包进 `FormData` 请求里。

### 3.3 表单参数构造（核心逻辑）

```ts
const handleSubmit = async () => {
  if (isLoading) return;
  setError(null);
  setSuccessMessage(null);
  setDownloadUrl(null);

  if (!llmApiUrl.trim() || !apiKey.trim()) {
    setError('请先配置模型 API URL 和 API Key');
    return;
  }

  const formData = new FormData();
  formData.append('model_name', model);
  formData.append('chat_api_url', llmApiUrl.trim());
  formData.append('api_key', apiKey.trim());
  formData.append('input_type', uploadMode);

  if (uploadMode === 'file') {
    if (!selectedFile) {
      setError('请先选择要上传的文件');
      return;
    }
    const kind = fileKind ?? detectFileKind(selectedFile);
    if (!kind) {
      setError('仅支持 PDF 和常见图片格式，请检查文件类型');
      return;
    }
    formData.append('file', selectedFile);
    formData.append('file_kind', kind); // 'pdf' | 'image'
  } else if (uploadMode === 'url') {
    if (!sourceUrl.trim()) {
      setError('请输入文档 URL');
      return;
    }
    formData.append('source_url', sourceUrl.trim());
  } else if (uploadMode === 'text') {
    if (!textContent.trim()) {
      setError('请输入要解析的文本内容');
      return;
    }
    formData.append('text', textContent.trim());
  }

  const res = await fetch(BACKEND_API, {
    method: 'POST',
    body: formData,
  });

  // 后续处理见 3.5
};
```

### 3.4 前端向后端传的字段明细

统一字段（所有 input_type 都会带）：

- `model_name`：字符串
  - 例：`"NanoBanana"`（Graph 页默认值）
  - 后端可根据这个模型名称决定调用哪个模型。
- `chat_api_url`：字符串
  - 例：`"https://api.openai.com/v1/chat/completions"`
  - 用于自定义 OpenAI / 兼容 API 地址。
- `api_key`：字符串
  - 用于后端调用 LLM。
- `input_type`：`"file" | "url" | "text"`
  - 前端根据当前模式传递，后端用来区分解析方式。

仅在 `input_type === 'file'` 时携带：

- `file`：上传的二进制文件
- `file_kind`：`"pdf"` 或 `"image"`
  - 由前端根据扩展名自动判断。
  - 后端可以根据此字段决定解析逻辑（PDF 解析 / 图片 OCR 等）。

仅在 `input_type === 'url'` 时携带：

- `source_url`：文档 URL 字符串

仅在 `input_type === 'text'` 时携带：

- `text`：用户粘贴的原始文本内容

### 3.5 后端返回要求

- HTTP 状态码：
  - 200：生成成功
  - 4xx / 5xx：失败（前端会尝试读取错误文本显示）
- Header：
  - `Content-Disposition: attachment; filename="xxx.pptx"`（推荐）
    - 前端通过正则解析 filename 作为下载名称。
- Body：
  - 二进制流：
    - 内容类型建议为 `application/vnd.openxmlformats-officedocument.presentationml.presentation`（PPTX）

前端处理下载的部分：

```ts
if (!res.ok) {
  let msg = '生成 PPTX 失败';
  try {
    const text = await res.text();
    if (text) msg = text;
  } catch {}
  throw new Error(msg);
}

// 从 header 解析文件名
const disposition = res.headers.get('content-disposition') || '';
let filename = 'paper2graph.pptx';
const match = disposition.match(/filename="?([^";]+)"?/i);
if (match?.[1]) {
  filename = decodeURIComponent(match[1]);
}

const blob = await res.blob();
const url = URL.createObjectURL(blob);
setDownloadUrl(url);
setLastFilename(filename);
setSuccessMessage('PPTX 已生成，正在下载...');

// 触发浏览器下载
const a = document.createElement('a');
a.href = url;
a.download = filename;
document.body.appendChild(a);
a.click();
a.remove();
```

---

## 4. Paper2PptPage 前端参数和调用说明

文件：`frontend-workflow/src/components/Paper2PptPage.tsx`

### 4.1 后端接口

```ts
const BACKEND_API = '/api/paper2ppt/generate';
```

Paper2PPT 使用单独的路由，后端可以实现不同的 PPT 模板 / 布局逻辑。

### 4.2 前端内部状态（与 Graph 页高度一致）

```ts
type UploadMode = 'file' | 'url' | 'text';
type FileKind = 'pdf' | 'image' | null;

const [uploadMode, setUploadMode] = useState<UploadMode>('file');
const [selectedFile, setSelectedFile] = useState<File | null>(null);
const [fileKind, setFileKind] = useState<FileKind>(null);
const [sourceUrl, setSourceUrl] = useState('');
const [textContent, setTextContent] = useState('');

// LLM 调用配置
const [llmApiUrl, setLlmApiUrl] = useState('https://api.openai.com/v1/chat/completions');
const [apiKey, setApiKey] = useState('');
// 默认模型为 gpt-4o，允许自由输入
const [model, setModel] = useState('gpt-4o');

// 请求状态 / 下载信息
const [isLoading, setIsLoading] = useState(false);
const [error, setError] = useState<string | null>(null);
const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
const [lastFilename, setLastFilename] = useState('paper2ppt.pptx');
const [successMessage, setSuccessMessage] = useState<string | null>(null);
```

与 `Paper2GraphPage` 的主要区别：

- `model` 默认值改为 `"gpt-4o"`。
- 模型名称是 `<input type="text">` 自由填写，不再限制为固定枚举。

### 4.3 构造 `FormData` 的逻辑

基本与 `Paper2GraphPage` 一致，仅默认文件名不同：

```ts
const formData = new FormData();
formData.append('model_name', model);
formData.append('chat_api_url', llmApiUrl.trim());
formData.append('api_key', apiKey.trim());
formData.append('input_type', uploadMode);

// 后面 file/url/text 分支完全相同
// ...

let filename = 'paper2ppt.pptx';
```

### 4.4 Paper2PPT 前端字段一览

与 Paper2Graph 完全一致：

- `model_name`：字符串（默认 gpt-4o，可自由填）
- `chat_api_url`：字符串
- `api_key`：字符串
- `input_type`：`"file" | "url" | "text"`
- `file` / `file_kind`：仅 file 模式
- `source_url`：仅 url 模式
- `text`：仅 text 模式

因此，后端可以共用一套解析输入的部分，只是在“生成 PPT 内容”的业务逻辑上区分 Paper2Graph / Paper2PPT。

---

## 5. 后端如何对接这些参数（FastAPI 示例）

下面是一个简化的 FastAPI 伪代码示例，说明如何接收前端传参。真实实现请参照你的业务逻辑和 DataFlow Agent 调用方式。

```python
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/api/paper2graph/generate")
async def generate_paper2graph(
    model_name: str = Form(...),
    chat_api_url: str = Form(...),
    api_key: str = Form(...),
    input_type: str = Form(...),  # 'file' | 'url' | 'text'
    file: UploadFile | None = File(None),
    file_kind: str | None = Form(None),   # 'pdf' | 'image'
    source_url: str | None = Form(None),
    text: str | None = Form(None),
):
    # 1. 根据 input_type 读取内容
    if input_type == "file":
        if not file:
            raise HTTPException(status_code=400, detail="file is required when input_type=file")
        content_bytes = await file.read()
        # 根据 file_kind 决定解析方式
    elif input_type == "url":
        if not source_url:
            raise HTTPException(status_code=400, detail="source_url is required when input_type=url")
        # 从 source_url 抓取 PDF/HTML
    elif input_type == "text":
        if not text:
            raise HTTPException(status_code=400, detail="text is required when input_type=text")
        content_text = text
    else:
        raise HTTPException(status_code=400, detail="invalid input_type")

    # 2. 使用 model_name / chat_api_url / api_key 调用 LLM
    #    这里可以封装为 DataFlow Agent 的一个 workflow 或 agent
    #    例如调用 dataflow_agent.llm_callers.text 中的封装

    # 3. 生成 PPTX 文件（bytes）
    pptx_bytes = b"...your generated pptx..."

    # 4. 返回 StreamingResponse
    filename = "paper2graph.pptx"
    return StreamingResponse(
        iter([pptx_bytes]),
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        headers={"Content-Disposition": f'attachment; filename=\"{filename}\"'},
    )
```

`/api/paper2ppt/generate` 的签名与上面几乎一样，只是业务部分（PPT 模板、结构）不同。

---

## 6. 如何修改前端参数 / 行为

### 6.1 改前端传参（字段名 / 增删字段）

操作步骤：

1. 找到对应页面组件：
   - Graph：`src/components/Paper2GraphPage.tsx`
   - PPT：`src/components/Paper2PptPage.tsx`
2. 修改 `handleSubmit` 函数中的 `formData.append(...)` 部分：
   - 新增字段：增加一行 `formData.append('new_field', someState)`。
   - 删除字段：删除对应的 `append` 代码，同时在后端去掉相应参数。
   - 改字段名：前端 `append('xxx', ...)` 和后端 `Form(...)` 参数名要同时改。

示例：增加一个“语言 language”字段：

前端：

```ts
const [language, setLanguage] = useState<'zh' | 'en'>('zh');
// 表单中 append：
formData.append('language', language);
```

后端：

```python
@app.post("/api/paper2ppt/generate")
async def generate_paper2ppt(
    # ...
    language: str = Form("zh"),
):
    # 根据 language = 'zh' / 'en' 选择生成中文或英文 PPT
```

### 6.2 改默认模型 / 模型列表

- `Paper2GraphPage`：
  - 当前：

    ```ts
    const [model, setModel] = useState('NanoBanana');
    ```

  - 改成默认 `gpt-4o`：

    ```ts
    const [model, setModel] = useState('gpt-4o');
    ```

  - 如需改为自由输入模型名，可以参考 `Paper2PptPage`，将 `<select>` 换成 `<input type="text">`。

- `Paper2PptPage`：
  - 当前默认已经是 `gpt-4o`。如需更换，直接改默认值即可。

### 6.3 改后端 URL / 接口路径

- Graph：
  - 在 `Paper2GraphPage.tsx` 中：

    ```ts
    const BACKEND_API = '/api/paper2graph/generate';
    ```

    如需改为完整外网地址：

    ```ts
    const BACKEND_API = 'https://your-backend-domain.com/api/paper2graph/generate';
    ```

- PPT：
  - 在 `Paper2PptPage.tsx` 中：

    ```ts
    const BACKEND_API = '/api/paper2ppt/generate';
    ```

如果使用完整外网 URL，Vite 开发时将直接跨域访问该地址，需要在后端启用 CORS；此时不再依赖 Vite 的 `/api` 代理。

---

## 7. 前后端联调 / 调试流程

### 7.1 启动后端 FastAPI

示例（根据实际项目调整）：

```bash
uvicorn fastapi_app.main:app --host 0.0.0.0 --port 8000
```

确保实现了：

- `POST /api/paper2graph/generate`
- `POST /api/paper2ppt/generate`

### 7.2 Vite dev 代理配置（推荐）

在 `frontend-workflow/vite.config.ts` 中增加：

```ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    open: true,
    allowedHosts: true,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  }
})
```

这样前端用相对路径 `/api/...` 即可，无需关心后端 Host/端口。

### 7.3 启动前端

```bash
cd frontend-workflow
npm install
npm run dev
```

访问 `http://localhost:3000`，切换到：

- `Paper2Graph 生成科研绘图`
- `Paper2PPT 生成`

填写：

- 模型 API URL（如 `https://api.openai.com/v1/chat/completions`）
- API Key
- 模型名称（Graph 页：NanoBanana；PPT 页默认 gpt-4o，也可自定义）
- 上传文件 / URL / 文本

点击“生成 PPTX”，查看：

- 浏览器 Network 面板请求参数是否正确
- 后端日志 / 返回状态码
- 是否自动触发 PPTX 下载

---

## 8. 小结

- 与后端交互的核心在于两个组件：
  - `Paper2GraphPage.tsx` → `/api/paper2graph/generate`
  - `Paper2PptPage.tsx` → `/api/paper2ppt/generate`
- 前端通过 `FormData` 传递 LLM 配置和输入源（文件 / URL / 文本），字段名在本说明文档中已经列清。
- 后端 FastAPI 接口通过 `Form(...)` 与 `File(...)` 即可直接接收。
- 如需修改参数（增加、删除、改名），只要同步修改：
  - 前端组件中的 `formData.append(...)`
  - 后端接口函数参数列表
即可完成前后端协同修改。
