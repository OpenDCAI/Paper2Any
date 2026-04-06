# 功能指南总览

这组文档只回答一类问题：各个 workflow 是做什么的、适合什么输入、典型输出是什么。

如果你当前卡在部署、环境、`.env`、Supabase、模型服务或端口配置，请先返回：

- [开源部署与配置总指南](open_source_deployment.md)
- [配置文件参考](configuration.md)

## 主要 workflow

### Paper2Figure

把论文内容整理成学术示意图、结构图、流程图和图文页面。

- 文档入口：[Paper2Figure](paper2figure.md)
- 典型输入：论文 PDF、摘要、段落、技术路线描述
- 常见输出：示意图、图表、PPT 页面素材

### Paper2PPT

把论文或主题转成结构化演示文稿。

- 文档入口：[Paper2PPT](paper2ppt.md)
- 典型输入：论文 PDF、主题文本、长文档内容
- 常见输出：可编辑 PPT 页面与导出文件

### Paper2Video

把论文内容转成讲解视频链路，包括脚本、配音、页面和视频子 worker。

- 文档入口：[Paper2Video](paper2video.md)
- 典型输入：论文 PDF、主题、PPT 页面
- 常见输出：脚本、字幕、视频片段、成片

### Paper2Technical

提取论文方法细节，生成技术路线说明和结构化技术报告。

- 文档入口：[Paper2Technical](paper2technical.md)
- 典型输入：论文 PDF、方法章节、实验说明
- 常见输出：技术解读、流程说明、结构化报告

## 开发接口相关

如果你要把本项目当作多模态 API 或 workflow 后端来接入，请看：

- [多模态 API 开发](multimodal_api.md)

## 阅读建议

推荐顺序：

1. 先看 [开源部署与配置总指南](open_source_deployment.md)
2. 跑通 [快速开始](../quickstart.md)
3. 再按实际目标阅读对应 workflow 文档
