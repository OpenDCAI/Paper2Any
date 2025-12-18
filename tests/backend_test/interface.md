
# paper2ppt 接口说明（前端视角）

本页面基于 `tests/backend_test/backend_test.py` 中的调用方式，以及你刚才实际跑出来的结果进行整理。

## 通用说明

- 所有接口基准路径：FastAPI app 根路径，例如 `http://localhost:8000`
- 调用方式：HTTP POST，`Content-Type: multipart/form-data` 或 `application/x-www-form-urlencoded`（下面会标明）
- 所有接口都返回 JSON

## 一、`/paper2ppt/pagecontent_json`

用于从「文本 / PDF / PPTX」中抽取/生成 PPT 的结构化 `pagecontent` 信息（标题、布局描述、要点等），供后续生成 PPT 使用。

### 1.1 通用表单字段（3 种 input_type 共用）

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

#### 1.1.1 作为 `form-data` 传递示例

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

### 1.2 `input_type = text`

#### 1.2.1 请求字段（在通用字段基础上增加）

```text
input_type  = "text"
text        string  必填  原始文本内容
```

#### 1.2.2 典型响应示例（已缩减非关键字段）

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

#### 1.2.3 前端关心字段

- `pagecontent`: 数组，每一项是一页的说明
  - `title`: 页标题
  - `layout_description`: 布局描述（中文）
  - `key_points`: 要点列表（字符串数组）
  - `asset_ref`: 关联资源（如图片路径），目前示例中为 `null`
- `result_path`: 服务端该次任务的数据目录，可透传给后续 `/paper2ppt/ppt_json` 使用

---

### 1.3 `input_type = pdf`

#### 1.3.1 请求字段（额外）

```text
input_type  = "pdf"
file        file  必填  PDF 文件（multipart/form-data 里的文件字段名为 "file"）
```

- 其他字段同 1.1

#### 1.3.2 典型响应示例（缩减版）

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

#### 1.3.3 额外说明

- `all_output_files` 会包含一些辅助 PDF（如 `input_span`、`input_origin`、`input_layout`），前端一般只需在调试时查看。

---

### 1.4 `input_type = pptx`

#### 1.4.1 请求字段（额外）

```text
input_type  = "pptx"
file        file  必填  PPTX 文件（multipart/form-data 字段名 "file"）
```

#### 1.4.2 典型响应示例（缩减）

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

#### 1.4.3 前端关心字段

- `pagecontent[i].ppt_img_path`: 这一页 PPT 转成的 PNG 截图的本地绝对路径（服务内部用）
- `all_output_files`: 对外访问的 HTTP 地址前缀（`http://testserver/...`），前端可以直接用来展示缩略图

---

## 二、`/paper2ppt/ppt_json`

`ppt_json` 是更「后期处理/编辑」的接口，分几种典型使用场景：

1. 直接传 `pagecontent` 里的图片路径（首次生成场景）
2. 传结构化 `pagecontent`（title/layout_description/key_points/...），由后端去生成整套 PPT 与图片
3. 编辑模式：`get_down = true`，指定 `page_id` 和 `edit_prompt` 对某一页做图像编辑/重绘

### 2.1 通用表单字段

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

- 当 `get_down = "false"`：表示「首次生成」或「再生成」
- 当 `get_down = "true"`：表示「编辑模式」，需要额外字段（见 2.4）

---

### 2.2 场景一：pagecontent 为「直接图片路径」（首次生成）

对应测试：`test_ppt_json_with_direct_image_pagecontent`

#### 2.2.1 请求示例

```python
result_path = "/home/.../outputs/ABC123/paper2ppt/1766070298"

pagecontent = [
    {"ppt_img_path": "/home/.../ppt_images/slide_000.png"},
    {"ppt_img_path": "/home/.../ppt_images/slide_001.png"},
]

data = {
  "img_gen_model_name": "gemini-2.5-flash-image-preview",
  "chat_api_url": "https://api.apiyi.com/v1",
  "api_key": "...",
  "model": "gpt-5.1",
  "language": "zh",
  "style": "多啦A梦风格；英文；",
  "aspect_ratio": "16:9",
  "invite_code": "ABC123",
  "result_path": result_path,
  "pagecontent": json.dumps(pagecontent, ensure_ascii=False),
  "get_down": "false"
}
```

#### 2.2.2 典型响应示例

```json
{
  "success": true,
  "ppt_pdf_path": "",
  "ppt_pptx_path": "",
  "pagecontent": [],
  "result_path": "/home/ubuntu/liuzhou/myproj/dev_2/DataFlow-Agent/outputs/ABC123/paper2ppt/1766070298",
  "all_output_files": [
    "http://testserver/outputs/ABC123/paper2ppt/1766070298/paper2ppt_editable.pptx",
    "http://testserver/outputs/ABC123/paper2ppt/1766070298/paper2ppt.pdf",
    "http://testserver/outputs/ABC123/paper2ppt/1766070298/ppt_images/slide_000.png",
    "http://testserver/outputs/ABC123/paper2ppt/1766070298/ppt_images/input.pdf",
    "http://testserver/outputs/ABC123/paper2ppt/1766070298/ppt_images/slide_001.png",
    "http://testserver/outputs/ABC123/paper2ppt/1766070298/ppt_pages/page_000.png",
    "http://testserver/outputs/ABC123/paper2ppt/1766070298/ppt_pages/page_002.png",
    "http://testserver/outputs/ABC123/paper2ppt/1766070298/ppt_pages/page_001.png"
  ]
}
```

#### 2.2.3 前端用法

- 传入已有的 slide PNG 路径（通常是后端内部的）；后端会：
  - 生成可编辑 PPTX：`paper2ppt_editable.pptx`
  - 生成 PDF：`paper2ppt.pdf`
  - 生成每一页的展示 PNG：`ppt_pages/page_*.png`
- 前端可以直接用 `all_output_files` 里的 HTTP 链接展示：

  - PPT 下载地址：`.../paper2ppt_editable.pptx`
  - PDF 下载地址：`.../paper2ppt.pdf`
  - 各页预览：`.../ppt_pages/page_*.png`

---

### 2.3 场景二：pagecontent 为结构化内容（首次生成）

对应测试：`test_ppt_json_with_structured_pagecontent`

#### 2.3.1 请求示例（关键结构）

```python
result_path = "/home/.../outputs/ABC123/paper2ppt/1766067301"

pagecontent = [
  {
    "title": "Multimodal DeepResearcher：从零生成文本-图表交织报告的智能框架",
    "layout_description": "全版居中排版，上方为大标题...",
    "key_points": [
      "论文题目：...",
      "作者与单位：...",
      "汇报人：XXX"
    ],
    "asset_ref": null
  },
  {
    "title": "方法概览：Formal Description of Visualization 与 Multimodal DeepResearcher",
    "layout_description": "左侧用要点概述...右侧两幅示意图...",
    "key_points": [
      "研究任务：...",
      "核心挑战：...",
      "...",
      "实验与效果：..."
    ],
    "asset_ref": "images/xxx.jpg,images/yyy.jpg"
  },
  {
    "title": "致谢",
    "layout_description": "标题置于顶部居中，主体区域采用居中单栏布局...",
    "key_points": [
      "感谢论文作者及其团队...",
      "感谢所在课题组/实验室...",
      "感谢各位老师和同学..."
    ],
    "asset_ref": null
  }
]

data = {
  ...同 2.2 通用字段...,
  "result_path": result_path,
  "pagecontent": json.dumps(pagecontent, ensure_ascii=False),
  "get_down": "false"
}
```

#### 2.3.2 典型响应示例

```json
{
  "success": true,
  "ppt_pdf_path": "",
  "ppt_pptx_path": "",
  "pagecontent": [],
  "result_path": "/home/ubuntu/liuzhou/myproj/dev_2/DataFlow-Agent/outputs/ABC123/paper2ppt/1766067301",
  "all_output_files": [
    "http://testserver/outputs/ABC123/paper2ppt/1766067301/paper2ppt_editable.pptx",
    "http://testserver/outputs/ABC123/paper2ppt/1766067301/paper2ppt.pdf",
    "http://testserver/outputs/ABC123/paper2ppt/1766067301/ppt_pages/page_000.png",
    "http://testserver/outputs/ABC123/paper2ppt/1766067301/ppt_pages/page_002.png",
    "http://testserver/outputs/ABC123/paper2ppt/1766067301/ppt_pages/page_001.png",
    "http://testserver/outputs/ABC123/paper2ppt/1766067301/input/auto/input_span.pdf",
    "http://testserver/outputs/ABC123/paper2ppt/1766067301/input/auto/input_origin.pdf",
    "http://testserver/outputs/ABC123/paper2ppt/1766067301/input/auto/input_layout.pdf"
  ]
}
```

#### 2.3.3 前端用法

- 前端只需要构造 `pagecontent` 的结构化描述，后端会：
  - 调用 LLM/图像模型，生成整套 PPT（含图）
  - 输出可编辑 PPTX、PDF、各页 PNG 等
- 和 2.2 一样，前端主要关注 `all_output_files`。

---

### 2.4 场景三：编辑模式（get_down = true）

对应测试：`test_ppt_json_edit_mode`

#### 2.4.1 请求字段（在 2.1 基础上增加）

```text
get_down    = "true"
page_id     int     必填  要编辑的页索引（从 0 开始）
edit_prompt string  必填  对这一页的编辑指令（自然语言）
```

示例：

```python
result_path = "/home/.../outputs/ABC123/paper2ppt/1766070298"

data = _base_form_ppt_json()
data.update({
  "result_path": result_path,
  "get_down": "true",
  "page_id": 0,
  "edit_prompt": "请把这一页的配色改成赛博朋克主题风格"
})

resp = POST /paper2ppt/ppt_json
```

#### 2.4.2 典型响应（你刚刚跑出来的实际）

```json
{
  "success": true,
  "ppt_pdf_path": "",
  "ppt_pptx_path": "",
  "pagecontent": [],
  "result_path": "/home/ubuntu/liuzhou/myproj/dev_2/DataFlow-Agent/outputs/ABC123/paper2ppt/1766070298",
  "all_output_files": [
    "http://testserver/outputs/ABC123/paper2ppt/1766070298/paper2ppt_editable.pptx",
    "http://testserver/outputs/ABC123/paper2ppt/1766070298/paper2ppt.pdf",
    "http://testserver/outputs/ABC123/paper2ppt/1766070298/ppt_images/slide_000.png",
    "http://testserver/outputs/ABC123/paper2ppt/1766070298/ppt_images/input.pdf",
    "http://testserver/outputs/ABC123/paper2ppt/1766070298/ppt_images/slide_001.png",
    "http://testserver/outputs/ABC123/paper2ppt/1766070298/ppt_pages/page_000.png",
    "http://testserver/outputs/ABC123/paper2ppt/1766070298/ppt_pages/page_002.png",
    "http://testserver/outputs/ABC123/paper2ppt/1766070298/ppt_pages/page_001.png"
  ]
}
```

#### 2.4.3 前端交互建议

1. 首次生成完成后，拿到 `result_path` + `all_output_files`（以及各页 PNG）
2. 用户在前端选中某一页（有页号 `page_id`），输入「编辑指令」
3. 前端调用：

   ```http
   POST /paper2ppt/ppt_json
   Content-Type: application/x-www-form-urlencoded

   img_gen_model_name=...
   chat_api_url=...
   api_key=...
   model=...
   language=...
   style=...
   aspect_ratio=16:9
   invite_code=ABC123
   result_path=[同之前的 result_path]
   pagecontent=[可以不变 / 后端根据 result_path 读取]
   get_down=true
   page_id=选中的页索引
   edit_prompt=用户自然语言编辑描述
   ```

4. 后端会：
   - 找到对应页的旧图 / 旧配置
   - 调用 Gemini 等进行图像编辑或重生成
   - 更新 PPTX / PNG，并返回最新 `all_output_files`

前端只需要在收到新 `all_output_files` 后，刷新对应页的预览即可。

---

## 三、快速对照表

| 接口                     | 用途                         | 关键入参                      | 关键出参                          |
|--------------------------|------------------------------|-------------------------------|-----------------------------------|
| `/paper2ppt/pagecontent_json` | 从 Text/PDF/PPTX 抽取/生成页结构 | `input_type` + `text/file`    | `pagecontent[]`, `result_path`   |
| `/paper2ppt/ppt_json`（首次） | 根据 pagecontent 生成整套 PPT   | `pagecontent`, `get_down=false` | `paper2ppt_editable.pptx`, `paper2ppt.pdf`, `ppt_pages/*.png` |
| `/paper2ppt/ppt_json`（编辑） | 编辑某一页图像/样式            | `result_path`, `page_id`, `edit_prompt`, `get_down=true` | 更新后的同一批文件（paths 在 `all_output_files`） |

---

以上内容可以直接保存为例如 `docs/paper2ppt_api_frontend.md` 提供给前端同学使用，无需了解内部 workflow 细节即可完成集成。