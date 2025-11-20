#!/bin/bash
# 运行 WebStructuredDataExtractionNode 脚本
# 直接运行即可，如需修改参数请编辑下方配置部分

# ==================== 配置区域 ====================
# 如需修改参数，请编辑以下变量（留空则使用环境变量或默认值）

DOWNLOAD_DIR="/mnt/DataFlow/lz/proj/agentgroup/binrui/commit/1113/DataFlow-Agent/script/downloaded_data_finally2"          # 下载目录路径，留空则使用环境变量 DF_DOWNLOAD_DIR 或默认值
CATEGORY="SFT"           # 输出数据类别: PT 或 SFT
MAX_RECORDS=""           # 最大输出记录数量，留空则不限制
OUTPUT_SUBDIR=""         # 输出子目录名称，留空则使用环境变量或默认值 "web_get_extracted"
CONCURRENCY="100"        # 并发处理数量
OBJECTIVE="收集代码数据用于大模型微调"             # 用户需求/提炼目标，留空则使用环境变量 DF_USER_OBJECTIVE
MAX_MARKDOWN_CHARS=""    # 网页内容最大字符数（超出部分会被截断），留空则使用默认值 9000，可通过 DF_WEB_GET_MAX_MARKDOWN_CHARS 环境变量设置
VERBOSE="1"               # 是否启用调试日志，设置为 "1" 启用，留空则禁用
# ==================================================

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/run_web_structured_extraction.py"

# 检查 Python 脚本是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "错误: 找不到 Python 脚本: $PYTHON_SCRIPT"
    exit 1
fi

# 构建参数列表
ARGS=()

if [ -n "$DOWNLOAD_DIR" ]; then
    ARGS+=("--download-dir" "$DOWNLOAD_DIR")
fi

if [ -n "$CATEGORY" ]; then
    ARGS+=("--category" "$CATEGORY")
fi

if [ -n "$MAX_RECORDS" ]; then
    ARGS+=("--max-records" "$MAX_RECORDS")
fi

if [ -n "$OUTPUT_SUBDIR" ]; then
    ARGS+=("--output-subdir" "$OUTPUT_SUBDIR")
fi

if [ -n "$CONCURRENCY" ]; then
    ARGS+=("--concurrency" "$CONCURRENCY")
fi

if [ -n "$OBJECTIVE" ]; then
    ARGS+=("--objective" "$OBJECTIVE")
fi

if [ -n "$MAX_MARKDOWN_CHARS" ]; then
    ARGS+=("--max-markdown-chars" "$MAX_MARKDOWN_CHARS")
fi

if [ "$VERBOSE" = "1" ]; then
    ARGS+=("--verbose")
fi

# 运行 Python 脚本
python3 "$PYTHON_SCRIPT" "${ARGS[@]}"

