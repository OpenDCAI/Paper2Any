# paper2ppt 接口说明（前端视角）

本页面基于 `tests/backend_test/backend_test.py` 中的调用方式，以及你实际跑出来的结果进行整理。

- 所有接口基准路径：FastAPI app 根路径，例如 `http://localhost:8000`
- 调用方式：HTTP POST
- 有的接口返回 JSON，有的直接返回文件（二进制）

---

## 一、`/api/pdf2ppt/generate`

用于将上传的 PDF 直接转换为 PPTX 文件，内部会调用 `pdf2ppt_with_sam` workflow。

### 1.1 请求

- 方法：`POST`
- 路径：`/api/pdf2ppt/generate`
- Content-Type：`multipart/form-data`

字段说明：

| 字段名       | 类型   | 是否必填 | 说明                          |
|-------------|--------|----------|-------------------------------|
| pdf_file    | file   | 是       | 论文 PDF 文件，字段名固定     |
| invite_code | string | 建议必填 | 邀请码，例如 `ABC123`        |

请求示例（伪 HTTP）：

```http
POST /api/pdf2ppt/generate
Content-Type: multipart/form-data

invite_code=ABC123
pdf_file=@/path/to/your_paper.pdf;type=application/pdf
```

### 1.2 响应

- 成功时：HTTP 200，响应体为一个 PPTX 二进制文件

关键响应头：

```http
Content-Type: application/vnd.openxmlformats-officedocument.presentationml.presentation
Content-Disposition: attachment; filename="xxx.pptx"
```

前端可以直接把该接口当做「文件下载接口」来用：

#### 1.2.1 普通表单提交

- 使用 `<form method="POST" enctype="multipart/form-data">` 包含一个 `<input type="file" name="pdf_file">`
- 浏览器会自动弹出下载保存对话框

#### 1.2.2 使用 fetch/axios 方式

以 fetch 为例：

```ts
const formData = new FormData();
formData.append("invite_code", "ABC123");
formData.append("pdf_file", fileInput.files[0]); // file 来自 <input type="file">

const resp = await fetch("/api/pdf2ppt/generate", {
  method: "POST",
  body: formData,
});

if (!resp.ok) {
  // 这里可以根据后端返回的错误信息做提示
  const text = await resp.text();
  console.error("pdf2ppt error:", resp.status, text);
  return;
}

const blob = await resp.blob();
const url = URL.createObjectURL(blob);
const a = document.createElement("a");
a.href = url;
a.download = "paper2ppt_result.pptx"; // 可以根据业务自定义文件名
a.click();
URL.revokeObjectURL(url);
```

这样前端完全不需要了解内部 `pdf2ppt_with_sam` workflow 的细节，只需要知道：

- 上传 PDF (`pdf_file`)
- 带上 `invite_code`
- 收到一个 PPTX 文件（二进制）

---

## 二、`/paper2ppt/pagecontent_json`

用于从「文本 / PDF / PPTX」中抽取/生成 PPT 的结构化 `pagecontent` 信息（标题、布局描述、要点等），供后续生成 PPT 使用。

### 2.1 通用表单字段（3 种 input_type 共用）

```text
chat_api_url   string  必填  LLM 服务地址，例如 https://api.apiyi.com/v1
api_key        string  必填  LLM API Key
model          string  必填  LLM 模型名，例如 gpt-5.1
language       string  必填  生成语言，例如 zh / en
style          string  必填  风格描述，例如 "多啦A梦风格；英文；"
gen_fig_model  string  必填  图像生成模型，例如 gemini-2.5-flash-image-preview
page_count     int     必填  期望生成的页数
invite_code    string  必填  邀请码，例如 ABC123
input_type     string  必填  输入类型：text / pdf / pptx
```

#### 2.1.1 作为 `form-data` 传递示例

```http
POST /paper2ppt/pagecontent_json
Content-Type: multipart/form-data

chat_api_url=https://api.apiyi.com/v1
api_key=...
model=gpt-5.1
language=zh
style=多啦A梦风格；英文；
gen_fig_model=gemini-2.5-flash-image-preview
page_count=2
invite_code=ABC123
input_type=text|pdf|pptx
[text/pdf/pptx 专用字段见下]
```

---

### 2.2 `input_type = text`

#### 2.2.1 请求字段（在通用字段基础上增加）

```text
input_type  = "text"
text        string  必填  原始文本内容
```

#### 2.2.2 典型响应示例（已缩减非关键字段）

```json
{
  "success": true,
  "ppt_pdf_path": "",
  "ppt_pptx_path": "",
  "pagecontent": [
    {
      "title": "this is a test text for paper2ppt",
      "layout_description": "整页居中布局，上方为报告标题，中间位置显示报告人姓名和单位，下方预留汇报时间；不放任何图表。",
      "key_points": [
        "报告题目：this is a test text for paper2ppt",
        "报告人：XXX",
        "单位：XXX",
        "汇报时间：XXXX年XX月XX日"
      ],
      "asset_ref": null
    },
    {
      "title": "致谢",
      "layout_description": "标题置于页面上方居中，正文采用居中段落形式，分行列出需要感谢的对象；页面不使用图片或表格。",
      "key_points": [
        "感谢各位老师和同事在本工作的支持与帮助。",
        "感谢课题组成员在讨论与实验中的合作与付出。",
        "感谢资金支持和相关机构提供的资源保障。",
        "感谢各位专家和同学的聆听与宝贵意见。"
      ],
      "asset_ref": null
    }
  ],
  "result_path": "/home/ubuntu/liuzhou/myproj/dev_2/DataFlow-Agent/outputs/ABC123/paper2ppt/1766077323",
  "all_output_files": []
}
```

#### 2.2.3 前端关心字段

- `pagecontent`: 数组，每一项是一页的说明
  - `title`: 页标题
  - `layout_description`: 布局描述（中文）
  - `key_points`: 要点列表（字符串数组）
  - `asset_ref`: 关联资源（如图片路径），目前示例中为 `null`
- `result_path`: 服务端该次任务的数据目录，可透传给后续 `/paper2ppt/ppt_json` 使用

---

### 2.3 `input_type = pdf`

#### 2.3.1 请求字段（额外）

```text
input_type  = "pdf"
file        file  必填  PDF 文件（multipart/form-data 里的文件字段名为 "file"）
```

- 其他字段同 2.1

#### 2.3.2 典型响应示例（缩减版）

```json
{
  "success": true,
  "ppt_pdf_path": "",
  "ppt_pptx_path": "",
  "pagecontent": [
    {
      "title": "Multimodal DeepResearcher：从零生成文本-图表交错报告的智能体框架",
      "layout_description": "页面居中放置论文标题，标题下方居中显示作者及单位信息，页面底部右下角以较小字号标注汇报人姓名，其余区域留白以突出主题。",
      "key_points": [
        "论文题目：Multimodal DeepResearcher: Generating Text-Chart Interleaved Reports From Scratch with Agentic Framework",
        "作者：Zhaorui Yang, Bo Pan, Han Wang, Yiyao Wang, Xingyu Liu, Minfeng Zhu, Bo Zhang, Wei Chen",
        "单位：State Key Lab of CAD&CG, Zhejiang University；Zhejiang University",
        "汇报人：XXX"
      ],
      "asset_ref": null
    },
    {
      "title": "致谢",
      "layout_description": "标题置顶居中，下方空白区域中间以较大字号显示“感谢聆听”，底部可预留位置简要致谢导师、合作团队或资助机构。",
      "key_points": [
        "感谢各位老师和同学的聆听与指导。",
        "感谢论文作者及相关开源社区的工作。",
        "欢迎交流与讨论。"
      ],
      "asset_ref": null
    }
  ],
  "result_path": "/home/ubuntu/liuzhou/myproj/dev_2/DataFlow-Agent/outputs/ABC123/paper2ppt/1766077329",
  "all_output_files": [
    "http://testserver/outputs/ABC123/paper2ppt/1766077329/input/auto/input_span.pdf",
    "http://testserver/outputs/ABC123/paper2ppt/1766077329/input/auto/input_origin.pdf",
    "http://testserver/outputs/ABC123/paper2ppt/1766077329/input/auto/input_layout.pdf"
  ]
}
```

---

### 2.4 `input_type = pptx`

#### 2.4.1 请求字段（额外）

```text
input_type  = "pptx"
file        file  必填  PPTX 文件（multipart/form-data 字段名 "file"）
```

#### 2.4.2 典型响应示例（缩减）

```json
{
  "success": true,
  "ppt_pdf_path": "",
  "ppt_pptx_path": "",
  "pagecontent": [
    {
      "ppt_img_path": "/home/ubuntu/liuzhou/myproj/dev_2/DataFlow-Agent/outputs/ABC123/paper2ppt/1766077408/ppt_images/slide_000.png"
    },
    {
      "ppt_img_path": "/home/ubuntu/liuzhou/myproj/dev_2/DataFlow-Agent/outputs/ABC123/paper2ppt/1766077408/ppt_images/slide_001.png"
    }
  ],
  "result_path": "/home/ubuntu/liuzhou/myproj/dev_2/DataFlow-Agent/outputs/ABC123/paper2ppt/1766077408",
  "all_output_files": [
    "http://testserver/outputs/ABC123/paper2ppt/1766077408/ppt_images/slide_000.png",
    "http://testserver/outputs/ABC123/paper2ppt/1766077408/ppt_images/input.pdf",
    "http://testserver/outputs/ABC123/paper2ppt/1766077408/ppt_images/slide_001.png"
  ]
}
```

---

## 三、`/paper2ppt/ppt_json`

`ppt_json` 是更「后期处理/编辑」的接口，分几种典型使用场景：

1. 直接传 `pagecontent` 里的图片路径（首次生成场景）
2. 传结构化 `pagecontent`（title/layout_description/key_points/...），由后端去生成整套 PPT 与图片
3. 编辑模式：`get_down = true`，指定 `page_id` 和 `edit_prompt` 对某一页做图像编辑/重绘

### 3.1 通用表单字段

```text
img_gen_model_name  string  必填  图像生成模型，例如 gemini-2.5-flash-image-preview
chat_api_url        string  必填  LLM / 图像模型服务地址，例如 https://api.apiyi.com/v1
api_key             string  必填  API Key
model               string  必填  LLM 模型名，例如 gpt-5.1
language            string  必填  语言
style               string  必填  风格
aspect_ratio        string  必填  比例，例如 16:9
invite_code         string  必填  邀请码，例如 ABC123

result_path         string  必填  后端输出根目录，一般直接用上一步返回的 result_path
pagecontent         string  必填  JSON 字符串（注意：是字符串形式传输）
get_down            string  必填  "false" 或 "true"（字符串）
```

当 `get_down = "false"`：表示「首次生成」或「再生成」  
当 `get_down = "true"`：表示「编辑模式」，需要额外字段（见 3.4）

### 3.2 场景一：pagecontent 为「直接图片路径」（首次生成）

对应测试：`test_ppt_json_with_direct_image_pagecontent`

- 传入的 `pagecontent` 形如：

```json
[
  {"ppt_img_path": "/.../ppt_images/slide_000.png"},
  {"ppt_img_path": "/.../ppt_images/slide_001.png"}
]
```

后端会生成：

- 可编辑 PPTX：`paper2ppt_editable.pptx`
- PDF：`paper2ppt.pdf`
- 每一页的展示 PNG：`ppt_pages/page_*.png`

所有这些路径都在 `all_output_files` 里通过 `http://testserver/outputs/...` 暴露，可直接在前端展示或下载。

### 3.3 场景二：pagecontent 为结构化内容（首次生成）

对应测试：`test_ppt_json_with_structured_pagecontent`

- 传入的 `pagecontent` 形如：

```json
[
  {
    "title": "xxx",
    "layout_description": "xxx",
    "key_points": ["...", "..."],
    "asset_ref": "images/a.jpg,images/b.jpg"
  },
  ...
]
```

后端会根据这些结构化描述自动生成整套 PPT（包含图），输出同样的 PPTX / PDF / PNG 集合，并通过 `all_output_files` 返回 HTTP 链接。

### 3.4 场景三：编辑模式（get_down = true）

对应测试：`test_ppt_json_edit_mode`

- 额外字段：

```text
get_down    = "true"
page_id     int     必填  要编辑的页索引（从 0 开始）
edit_prompt string  必填  对这一页的编辑指令（自然语言）
```

后端会：

- 根据 `result_path` 找到已有的 PPT / PNG
- 根据 `page_id` 和 `edit_prompt` 重绘/编辑对应页
- 更新 PPTX / PNG，再通过 `all_output_files` 返回最新资源路径

前端流程建议：

1. 首次生成后，拿到 `result_path` + `all_output_files` 展示各页。
2. 用户选中一页，填写编辑说明。
3. 前端调用 `/paper2ppt/ppt_json`（`get_down = "true"`），传 `page_id` 和 `edit_prompt`。
4. 收到新的 `all_output_files` 后刷新对应页的预览和下载链接。

---

## 四、快速对照表

| 接口                        | 用途                             | 典型返回                     |
|-----------------------------|----------------------------------|------------------------------|
| `/api/pdf2ppt/generate`     | 上传 PDF，直接返回 PPTX 文件     | PPTX 二进制（下载接口）      |
| `/paper2ppt/pagecontent_json` | 从 Text/PDF/PPTX 生成结构化页信息 | JSON：`pagecontent[]` 等    |
| `/paper2ppt/ppt_json`（首次/再生成） | 根据 pagecontent 生成整套 PPT    | JSON：PPTX/PDF/PNG 链接列表 |
| `/paper2ppt/ppt_json`（编辑） | 编辑某一页图像/样式              | JSON：更新后的资源链接      |

前端根据需要选择：

- 只要一个「从 PDF 到 PPTX」的一步式接口时，用 `/api/pdf2ppt/generate`。
- 要更细粒度控制 PPT 内容、样式和逐页编辑时，用 `/paper2ppt/pagecontent_json` + `/paper2ppt/ppt_json` 组合。
