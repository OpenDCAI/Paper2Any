前端契约回归测试放这里。

这批脚本的目标不是直接调某个裸 workflow 名，而是模拟前端真实发给后端的 HTTP 请求。
这样可以更早暴露：
- 后端接口字段不兼容
- 部署环境缺依赖
- 模型服务没起来
- 两步流程只测了第一步、第二步没测到

当前覆盖的主要页面 / 接口：
- `image2ppt` -> `/api/v1/image2ppt/generate`
- `pdf2ppt` -> `/api/v1/pdf2ppt/generate`
- `image2drawio` -> `/api/v1/image2drawio/generate`
- `paper2drawio_pdf` -> `/api/v1/paper2drawio/generate`
- `paper2figure_model_arch` -> `/api/v1/paper2figure/generate-json`
- `paper2video_subtitle` -> `/api/v1/paper2video/generate-subtitle`
- `paper2video_full` -> `generate-subtitle + generate-video`

默认测试夹具：
- `image2ppt` / `image2drawio`: `tests/test_02.png`
- `pdf2ppt` / `paper2drawio_pdf` / `paper2figure_model_arch`: `tests/test_03.pdf`
- `paper2video_*`: 默认 `tests/paper2ppt_editable.pptx`（2 页，回归更快）
- 若要覆盖其他 PPTX，可设：
  - `FRONTEND_WORKFLOW_TEST_PAPER2VIDEO_INPUT=tests/test.pptx`

运行前要求：
- 后端已经启动
- `fastapi_app/.env` 里至少有：
  - `BACKEND_API_KEY`
  - `DF_API_URL` 或 `DEFAULT_LLM_API_URL`
  - `DF_API_KEY`
- 若跑 `paper2video_full`，还需要当前 Python 环境已安装 `dashscope`

常用命令：
- 跑全部手工回归：
  - `python tests/manual/frontend_workflows/test_frontend_workflows.py --case all`
- 只跑 paper2video 两步：
  - `python tests/manual/frontend_workflows/test_frontend_workflows.py --case paper2video_full`
- 用 pytest 显式跑某一个 case：
  - `pytest tests/manual/frontend_workflows/test_frontend_workflows.py -s -k paper2video_full`

结果会保存到：
- `outputs/manual/frontend_workflows/<timestamp>/`

如果你改了前端页面请求字段，应该同步更新这里的 case，保证“前端页面 -> 测试脚本”一一对应。
