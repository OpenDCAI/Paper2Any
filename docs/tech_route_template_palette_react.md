# 技术路线图：模板参考 + 可选配色 + SVG 渲染校验（ReAct重试）

本文档记录本次开发任务的：需求、具体设计、开发计划、当前开发进度与后续待办。

---

## 1. 背景与现状

现有 `paper2technical` 工作流会生成一份技术路线图 SVG，并同时渲染 PNG，前端提供：
- SVG 源文件下载按钮
- SVG 图片（PNG 预览）URL

当前问题/目标：
- 生成 SVG 过程中偶发语法错误或渲染失败（导致显示乱码/无法预览）。
- 需要支持“可选配色”：不选配色只输出黑白；选配色同时输出黑白+彩色两套，并在前端提供两套下载/链接。
- 生成 SVG 的整体结构/样式需参考一张固定模板图（`temp.png`），但允许根据用户内容微调排版。

> 备注：本次 ReAct 的目的仅用于 **SVG 语法正确性/可渲染性**，不用于语义完整性检查。

---

## 2. 需求（确认版）

### 2.1 输出规则

1) **未选择配色**（`tech_route_palette == ""`）：
- 仅生成黑白 SVG（以及对应 PNG）。
- 仍使用原有字段：`svg_filename/svg_image_filename` 指向黑白版本。

2) **选择配色**（`tech_route_palette != ""`）：
- 同时生成黑白 SVG + 彩色 SVG（以及各自 PNG）。
- `svg_filename/svg_image_filename` 继续指向黑白（兼容旧逻辑）。
- 新增两套字段用于彩色：
  - `svg_color_filename/svg_color_image_filename`
  - 同时也新增黑白明确字段：`svg_bw_filename/svg_bw_image_filename`
- 前端展示两套下载按钮与 URL。

### 2.2 ReAct 诉求（仅渲染正确性）

黑白 SVG 生成后需要验证：
- SVG XML 结构合法（可解析）
- 必须有 `viewBox`
- 必须能通过 CairoSVG 渲染为 PNG（`local_tool_for_svg_render` 成功）

若验证失败，则应自动重试（最多 N 次），并把失败原因反馈给模型以修复。

> 注意：由于不能修改 `BaseAgent`，这里的“ReAct”以 **workflow 节点内的循环重试 + validator** 形式实现。

### 2.3 模板与上色要求

- 模板：使用仓库内 `temp.png` 作为技术路线图模板，生成时需参考该图的格式/层级/布局风格，可微调排版以适配实际内容。
- 上色：同类型或同层级内容最好使用同一颜色（例如同一个阶段的节点一致、箭头一致、文字一致）。
- 色卡：提供 3–4 套预设，每套约 3–4 种颜色，并在前端可预览色卡颜色。

---

## 3. 具体设计

### 3.1 模板文件与默认路径

- 模板文件落位：
  - `static/paper2any_imgs/p2t/temp.png`
- Workflow 默认模板路径：
  - 优先 `request.tech_route_template`（支持传入 `temp.png` 或绝对路径）
  - 为空时使用默认 `static/paper2any_imgs/p2t/temp.png`

### 3.2 色卡设计（预设 4 套，每套 4 色，可视化）

色卡字段含义（用于上色 agent）：
- `colors`: 提供给模型可用颜色集合（4个 hex）
- `level_colors`: “同层级/同阶段颜色”的推荐序列（4个 hex）
- `arrow_color`: 箭头/连线强调色
- `text_color`: 文本颜色（确保可读性）

目前内置色卡（与前端保持一致）：
- `academic_blue`: `#1F6FEB #60A5FA #A7C7FF #0B3D91`
- `teal_orange`: `#0F766E #14B8A6 #F59E0B #FB923C`
- `slate_rose`: `#334155 #64748B #F43F5E #FCA5A5`
- `indigo_amber`: `#4338CA #6366F1 #F59E0B #FCD34D`

### 3.3 Workflow 设计（不改 BaseAgent）

工作流：`dataflow_agent/workflow/wf_paper2technical.py`

整体流程：
1) `_start_`：初始化 `result_path`
2) `paper_idea_extractor`（PDF 模式才走）
3) `technical_route_bw_svg_generator`：黑白 SVG（VLM，参考模板 PNG）
4) **条件分支**：
   - 未选配色：直接 `technical_ppt_generator`
   - 选配色：`technical_route_colorize_svg` → `technical_ppt_generator`
5) `technical_ppt_generator`：若选配色则 PPT 插彩色，否则插黑白

关键点：黑白/上色两个节点都实现“手写 ReAct 重试”：
- 每次调用 agent 得到 SVG
- 做校验（XML + viewBox + CairoSVG 渲染）
- 不通过则把错误写入 `state.temp_data["validation_feedback"]`，进入下一次尝试

黑白产物写入：
- `state.figure_tec_svg_bw_content`
- `state.svg_bw_file_path / state.svg_bw_img_path`
- 同时为了兼容旧字段：`state.svg_file_path / state.svg_img_path` 指向黑白

彩色产物写入：
- `state.figure_tec_svg_color_content`
- `state.svg_color_file_path / state.svg_color_img_path`

### 3.4 Agent 与 Prompt 设计（尽量简化）

#### 3.4.1 黑白 SVG 生成（参考模板 PNG）

- Agent：`technical_route_bw_svg_generator`
  - 文件：`dataflow_agent/agentroles/paper2any_agents/technical_route_bw_svg_generator.py`
  - 使用 VLM：`use_vlm=True`，`vlm_config.mode="understanding"`，`vlm_config.input_image=模板路径`
  - 固定模型：`gpt-5.2`
  - 输出：严格 JSON `{"svg_code":"..."}`，写入 `state.figure_tec_svg_bw_content`

- Prompt（仅保留关键约束）：
  - 参考模板图结构/排版
  - 根据 `paper_idea` 生成内容
  - 黑白/灰度（限制 fill/stroke）
  - `viewBox` 必须存在
  - 如有 `validation_feedback` 则修复

#### 3.4.2 彩色 SVG 上色（仅输入黑白SVG + 色卡）

- Agent：`technical_route_colorize_svg`
  - 文件：`dataflow_agent/agentroles/paper2any_agents/technical_route_colorize_svg_agent.py`
  - 文本模式（非 VLM）
  - 固定模型：`gpt-5.2`
  - 输入：`bw_svg_code` + `palette_json` + `validation_feedback`
  - 输出：严格 JSON `{"svg_code":"..."}`，写入 `state.figure_tec_svg_color_content`

- Prompt（仅保留关键约束）：
  - 不改几何结构/坐标/path d/文字内容，只改 fill/stroke/style/class
  - 同层级同色、同类型同色（引导使用 `level_colors`）
  - 箭头统一 `arrow_color`，文字统一 `text_color`
  - 如有 `validation_feedback` 则修复

### 3.5 前后端返回与兼容策略

后端响应模型：`fastapi_app/schemas.py`
- 兼容字段（始终存在/可能为空）：
  - `svg_filename/svg_image_filename`：**黑白**（即使选配色也指向黑白）
- 新增字段（选配色时返回）：
  - `svg_bw_filename/svg_bw_image_filename`：黑白（更明确）
  - `svg_color_filename/svg_color_image_filename`：彩色

前端展示策略：
- 未选配色：只显示黑白下载与链接
- 选配色：显示黑白 + 彩色两组下载与链接
- 色卡选择：`SettingsCard` 中新增下拉与颜色圆点预览

---

## 4. 开发计划（执行拆解）

1) **模板与配色参数接入**
   - request：`tech_route_template`、`tech_route_palette`
   - 默认模板：`static/paper2any_imgs/p2t/temp.png`

2) **黑白 SVG 节点**
   - 新增 VLM agent + prompt
   - workflow 内实现渲染校验重试（XML+viewBox+CairoSVG）
   - 输出黑白 svg/png 并写回 state

3) **可选上色节点**
   - 新增上色 agent + prompt
   - workflow 内实现渲染校验重试
   - 输出彩色 svg/png 并写回 state

4) **后端双套字段返回**
   - adapter 取出 state 中的 bw/color 路径并返回
   - service 层将路径转换为 outputs URL

5) **前端：色卡下拉 + 两套下载/URL**
   - 常量定义色卡列表 + 颜色预览
   - formData 透传 `tech_route_palette` + `tech_route_template`
   - 页面展示黑白/彩色两套链接

---

## 5. 当前开发进度（已完成项）

### 5.1 模板落位
- 已将仓库根目录 `temp.png` 复制到：`static/paper2any_imgs/p2t/temp.png`

### 5.2 后端接口与返回字段
- `fastapi_app/schemas.py`
  - `Paper2FigureRequest` 增加：`tech_route_template`, `tech_route_palette`
  - `Paper2FigureResponse` 增加：`svg_bw_*`, `svg_color_*`
- `fastapi_app/routers/paper2any.py`
  - `generate_paper2figure_json` 增加 Form 参数并传入 service
- `fastapi_app/services/paper2any_service.py`
  - `generate_paper2figure_json` 透传 palette/template 到 request
  - 响应中新增 4 个 URL 字段（bw/color）
- `fastapi_app/workflow_adapters/wa_paper2figure.py`
  - 从 state 读取 `svg_bw_*`、`svg_color_*` 并填充到 response

### 5.3 State 扩展
- `dataflow_agent/state.py`
  - `Paper2FigureRequest`：新增 `tech_route_template`/`tech_route_palette`
  - `Paper2FigureState`：新增黑白/彩色 SVG/PNG 路径字段

### 5.4 Workflow 拆分与校验重试
- `dataflow_agent/workflow/wf_paper2technical.py`
  - 原单节点生成 SVG 改为：
    - `technical_route_bw_svg_generator`（黑白、模板参考、循环重试渲染校验）
    - `technical_route_colorize_svg`（可选、仅输入黑白SVG+色卡、循环重试渲染校验）
  - PPT 选择插入：选配色→彩色，否则→黑白

### 5.5 Agent 与 Prompt
- 新增 agents：
  - `dataflow_agent/agentroles/paper2any_agents/technical_route_bw_svg_generator.py`
  - `dataflow_agent/agentroles/paper2any_agents/technical_route_colorize_svg_agent.py`
- 新增 prompts（在现有模板文件中追加）：
  - `dataflow_agent/promptstemplates/resources/pt_technical_route_desc_generator_repo.py`

### 5.6 前端交互与色卡预览
- `frontend-workflow/src/components/paper2graph/constants.ts`
  - 增加 `TECH_ROUTE_PALETTES`（4套色卡 + “不配色”）
- `frontend-workflow/src/components/paper2graph/index.tsx`
  - localStorage 持久化 palette
  - tech_route 请求时附带 `tech_route_palette` 与 `tech_route_template=temp.png`
  - 解析并保存 `svg_bw_*`、`svg_color_*`
- `frontend-workflow/src/components/paper2graph/SettingsCard.tsx`
  - 技术路线图时显示色卡下拉与颜色圆点预览
  - 选配色后显示黑白+彩色两套下载与链接
- i18n：
  - `frontend-workflow/src/locales/zh/paper2graph.json` 增加 `techRoute.paletteLabel`
  - `frontend-workflow/src/locales/en/paper2graph.json` 增加 `techRoute.paletteLabel`

---

## 6. 待办与风险点

### 6.1 待办
- 跑一次最小联调/自测（建议）：
  - `TEXT` 模式不选配色：应只返回黑白 SVG/PNG
  - `TEXT` 模式选配色：应返回黑白+彩色两套 SVG/PNG，且 PPT 插入彩色
  - `PDF` 模式同上
- 前端可选：若你需要展示 PNG 预览链接（彩色/黑白），可在 SettingsCard 增加对应展示（当前只做了 SVG 下载/链接）。

### 6.2 风险点
- VLM “understanding” 输出 SVG 的能力取决于模型服务对多模态的支持：若 `gpt-5.2` 在你的后端服务不支持 image-understanding，需要调整为支持多模态的 gpt-5.2 端点/代理。
- 渲染校验依赖 CairoSVG 与字体环境：若部署环境缺字体，已在 workflow 里做了中文字体注入兜底，但仍可能存在缺字风险。

---

## 7. 关键文件清单（便于代码审查）

- 模板：
  - `static/paper2any_imgs/p2t/temp.png`
- Workflow：
  - `dataflow_agent/workflow/wf_paper2technical.py`
- Agents：
  - `dataflow_agent/agentroles/paper2any_agents/technical_route_bw_svg_generator.py`
  - `dataflow_agent/agentroles/paper2any_agents/technical_route_colorize_svg_agent.py`
- Prompts：
  - `dataflow_agent/promptstemplates/resources/pt_technical_route_desc_generator_repo.py`
- Backend：
  - `fastapi_app/schemas.py`
  - `fastapi_app/routers/paper2any.py`
  - `fastapi_app/services/paper2any_service.py`
  - `fastapi_app/workflow_adapters/wa_paper2figure.py`
- Frontend：
  - `frontend-workflow/src/components/paper2graph/constants.ts`
  - `frontend-workflow/src/components/paper2graph/index.tsx`
  - `frontend-workflow/src/components/paper2graph/SettingsCard.tsx`
  - `frontend-workflow/src/locales/zh/paper2graph.json`
  - `frontend-workflow/src/locales/en/paper2graph.json`

