# 算子编写功能说明

算子编写功能是 DataFlow-Agent 的核心功能之一，能够根据用户的需求描述，自动生成符合 DataFlow 规范的算子代码。该功能通过多阶段的 AI Agent 协作，实现了从需求理解、算子匹配、代码生成到自动验证的完整流程。

## 功能概述

算子编写功能的主要目标是：

- **自动化生成**：根据用户的自然语言描述，自动生成完整的算子代码
- **智能匹配**：基于现有算子库，找到相似的算子作为参考模板
- **代码验证**：自动生成可运行的测试代码，并执行验证确保算子可用
- **错误修复**：当生成的代码出现错误时，自动进行调试和修复

## 工作流程

算子编写功能采用基于状态图（StateGraph）的工作流编排，主要包含以下阶段：

```
match_operator → write_the_operator → llm_append_serving → llm_instantiate
                                                              ↓ (失败时)
                                         code_debugger → rewriter → after_rewrite
                                                              ↓ (循环修复)
                                         llm_append_serving → llm_instantiate
```

### 1. 算子匹配阶段（match_operator）

**功能**：在现有算子库中查找与用户需求最相似的算子，作为代码生成的参考。

**实现细节**：
- 从算子库中获取指定类别（category）的所有算子列表
- 使用 LLM 分析用户需求（purpose）和算子库内容
- 返回最多 4 个最相似的算子名称
- 匹配结果会保存在 `state.matched_ops` 中，供后续阶段使用

**前置工具**：
- `get_operator_content`：获取算子库中指定类别的算子列表和描述
- `purpose`：从用户请求中提取的需求描述

### 2. 算子编写阶段（write_the_operator）

**功能**：基于匹配到的相似算子作为示例，生成新的算子代码。

**实现细节**：
- 获取匹配到的算子完整源码（包括 import 和类定义）
- 将多个匹配算子拼接为示例代码（分批处理，每次最多 3 个，避免提示过长）
- 使用 LLM 根据用户目标和示例代码生成新算子
- 生成代码需要满足最小运行要求，不包含多余代码或注释

**前置工具**：
- `example`：匹配到的相似算子源码示例
- `target`：用户的目标描述

**输出格式**：
```json
{
  "code": "完整的算子源代码",
  "desc": "算子的功能描述和输入输出说明"
}
```

### 3. LLM 服务注入阶段（llm_append_serving）

**功能**：检测算子是否需要使用 LLM，如果需要则自动注入 LLM 服务初始化代码。

**实现细节**：
- 检查算子代码中是否已包含 `llm_serving` 或 `APILLMServing_request`
- 如果需要使用 LLM 但未包含，则通过 LLM 智能注入相关代码
- 如果 LLM 注入失败，会回退到硬编码注入方式作为保底

**前置工具**：
- `pipeline_code`：当前算子代码
- `llm_serving_snippet`：LLM 服务的标准代码片段
- `example_data`：示例数据（仅作提示，不用于运行逻辑）
- `available_keys`：数据中的可用键（仅作提示）

### 4. 实例化与验证阶段（llm_instantiate）

**功能**：生成可运行的测试代码，执行算子并验证其功能正确性。

**实现细节**：
- LLM 生成包含实例化代码和执行逻辑的完整脚本
- 执行生成的代码，捕获 stdout 和 stderr
- 从输出中解析 `[selected_input_key]` 标记，确定算子自动选择的输入键
- 验证输出文件是否成功生成（默认路径：`./cache_local/dataflow_cache_step_step1.jsonl`）
- 验证 `selected_input_key` 是否在真实的 `available_keys` 中

**验证标准**：
- 代码执行无异常
- 成功解析到 `selected_input_key`
- 输出文件非空
- `selected_input_key` 在数据可用键列表中

**前置工具**：
- `pipeline_code`：算子代码
- `target`：目标描述
- `example_data`：示例数据
- `available_keys`：可用键列表
- `preselected_input_key`：预选的输入键（基于数据平均长度计算）
- `test_data_path`：测试数据文件路径

### 5. 调试循环阶段（可选）

当 `need_debug=True` 且实例化验证失败时，会进入调试循环：

#### 5.1 代码调试（code_debugger）

**功能**：分析执行错误，提供调试建议。

**实现细节**：
- 读取算子代码和错误堆栈
- 使用 LLM 分析错误原因
- 输出调试分析结果（`reason`）

#### 5.2 代码重写（rewriter）

**功能**：根据调试分析结果，修复算子代码。

**实现细节**：
- 最小化修改原则，只修复必要的问题
- 考虑错误信息、调试分析、目标描述和数据样本
- 特别关注输入键的选择和修复

**前置工具**：
- `pipeline_code`：需要修复的代码
- `error_trace`：错误堆栈
- `debug_reason`：调试分析结果
- `data_sample`：数据样本
- `available_keys`：可用键
- `target`：目标描述
- `preselected_input_key`：预选输入键

#### 5.3 重写后处理（after_rewrite）

**功能**：重写后的状态更新和处理。

#### 5.4 循环返回

重写后重新执行 `llm_append_serving` 和 `llm_instantiate`，最多循环 `max_debug_rounds` 次。

## 使用方式

### CLI 命令行方式

使用 `script/run_dfa_operator_write.py` 脚本：

```bash
python script/run_dfa_operator_write.py \
    --target "生成一个数据质量分析的算子" \
    --language zh \
    --category "Default" \
    --json-file "tests/test.jsonl" \
    --model "gpt-4o" \
    --chat-api-url "http://123.129.219.111:3000/v1/" \
    --need-debug \
    --max-debug-rounds 3 \
    --output "tests/output1.py"
```

**参数说明**：

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--target` | 是 | - | 用户需求/新算子的目标描述 |
| `--category` | 否 | `Default` | 算子类别，用于匹配阶段筛选算子 |
| `--json-file` | 否 | `tests/test.jsonl` | 测试数据文件路径 |
| `--model` | 否 | `gpt-4o` | 使用的 LLM 模型名称 |
| `--chat-api-url` | 否 | `http://123.129.219.111:3000/v1/` | LLM API 基础地址 |
| `--language` | 否 | `en` | 提示输出语言（en/zh） |
| `--need-debug` | 否 | `False` | 是否启用调试循环 |
| `--max-debug-rounds` | 否 | `3` | 最大调试轮次 |
| `--output` | 否 | `tests/output1.py` | 生成的算子代码保存路径 |

**环境变量**：

- `DF_API_KEY`：LLM API 密钥（如果通过 `--chat-api-url` 调用外部 API）

### Gradio Web 界面

启动 Gradio 应用后，访问算子编写页面：

```bash
python gradio_app/app.py
```

在 Web 界面中：
1. 输入目标描述
2. 选择算子类别（可选）
3. 配置测试数据路径
4. 设置 LLM 模型和 API 参数
5. 选择是否启用调试
6. 点击生成按钮

界面会实时显示：
- 匹配到的算子列表
- 生成的代码预览
- 执行结果
- 调试信息（如果启用）

## 代码结构

### 工作流定义

工作流的定义位于 `dataflow_agent/workflow/wf_pipeline_write.py`：

```python
def create_operator_write_graph() -> GenericGraphBuilder:
    """构建算子编写工作流图"""
    builder = GenericGraphBuilder(state_model=DFState, entry_point="match_operator")
    # ... 节点和边的定义
    return builder
```

### 核心 Agent

- **MatchOperator** (`agentroles/match.py`)：算子匹配 Agent
- **Writer** (`agentroles/writer.py`)：算子编写 Agent
- **AppendLLMServingAgent** (`agentroles/append_llm_serving.py`)：LLM 服务注入 Agent
- **InstantiateAgent** (`agentroles/instantiator.py`)：实例化 Agent
- **CodeDebugger** (`agentroles/debugger.py`)：代码调试 Agent
- **Rewriter** (`agentroles/oprewriter.py`)：代码重写 Agent

### 状态管理

工作流使用 `DFState` 对象管理状态，主要字段包括：

- `request`：用户请求信息（DFRequest）
- `matched_ops`：匹配到的算子名称列表
- `temp_data`：临时数据
  - `pipeline_code`：生成的算子代码
  - `pipeline_file_path`：代码保存路径
  - `category`：算子类别
  - `debug_runtime`：调试运行时的信息（stdout、stderr、input_key、available_keys 等）
  - `round`：当前调试轮次
- `agent_results`：各 Agent 的执行结果
- `execution_result`：执行结果（success、stdout、stderr、file_path）
- `code_debug_result`：代码调试结果（reason）

## 工作原理详解

### 算子匹配机制

匹配阶段会从算子库（`ops.json`）中获取指定类别的所有算子，包括：
- 算子名称
- 算子描述
- 算子类别

LLM 基于用户需求（purpose）和算子内容，选择最相似的 4 个算子。匹配结果会保存到 `state.matched_ops`，供后续阶段使用。

### 代码生成策略

编写阶段会：
1. 从匹配结果中获取算子的完整源码（通过 `local_tool_for_get_match_operator_code`）
2. 将多个算子源码拼接为示例（分批处理，避免提示过长）
3. 要求 LLM 参考示例代码的风格和结构，生成新算子
4. 生成的代码需要满足最小运行要求，遵循项目代码风格

### LLM 服务自动注入

如果算子需要调用 LLM API，`llm_append_serving` 阶段会：
1. 检查代码中是否已有 LLM 服务相关代码
2. 如果需要，通过 LLM 智能分析并注入标准的 LLM 服务初始化代码
3. 注入的代码包括：
   ```python
   from dataflow.serving import APILLMServing_request
   
   self.llm_serving = APILLMServing_request(
       api_url="http://123.129.219.111:3000/v1/chat/completions",
       key_name_of_api_key="DF_API_KEY",
       model_name="gpt-4o",
       max_workers=100,
   )
   ```

### 实例化验证机制

`llm_instantiate` 阶段会：
1. LLM 生成包含算子实例化和执行的完整代码
2. 代码执行时会：
   - 读取测试数据文件
   - 实例化算子对象
   - 处理数据并输出结果
   - 打印 `[selected_input_key]` 标记，指示自动选择的输入键
3. 验证执行结果：
   - 检查输出文件是否生成且非空
   - 验证 `selected_input_key` 是否解析成功
   - 确认 `selected_input_key` 在数据可用键中

### 调试循环机制

当验证失败且 `need_debug=True` 时：
1. `code_debugger` 分析错误原因
2. `rewriter` 根据错误信息和调试分析修复代码
3. 修复后的代码重新进入 `llm_append_serving` 和 `llm_instantiate`
4. 循环最多执行 `max_debug_rounds` 次，或直到成功

## 注意事项

1. **算子类别**：正确设置 `category` 可以提高匹配的准确性
2. **测试数据**：确保 `json_file` 指向有效的测试数据文件
3. **API 密钥**：使用外部 LLM API 时需要设置 `DF_API_KEY` 环境变量
4. **调试轮次**：合理设置 `max_debug_rounds`，避免无限循环
5. **递归限制**：工作流会自动计算合适的 `recursion_limit`，公式为：`4 + 5 * max_debug_rounds + 5`

## 示例

### 示例 1：生成数据质量分析算子

```bash
python script/run_dfa_operator_write.py \
    --target "生成一个数据质量分析的算子，能够检测数据中的缺失值、异常值和重复值" \
    --language zh \
    --category "DataQuality" \
    --need-debug
```

### 示例 2：生成文本分类算子

```bash
python script/run_dfa_operator_write.py \
    --target "Generate a text classification operator using LLM" \
    --category "TextProcessing" \
    --model "gpt-4o" \
    --output "operators/text_classifier.py"
```

## 相关文件

- **工作流定义**：`dataflow_agent/workflow/wf_pipeline_write.py`
- **CLI 入口**：`script/run_dfa_operator_write.py`
- **Gradio 界面**：`gradio_app/pages/operator_write.py`
- **核心 Agent**：
  - `dataflow_agent/agentroles/match.py`
  - `dataflow_agent/agentroles/writer.py`
  - `dataflow_agent/agentroles/append_llm_serving.py`
  - `dataflow_agent/agentroles/instantiator.py`
  - `dataflow_agent/agentroles/debugger.py`
  - `dataflow_agent/agentroles/oprewriter.py`
- **提示词模板**：`dataflow_agent/promptstemplates/prompts_repo.py`

## 扩展与定制

如需定制算子编写功能：

1. **修改匹配逻辑**：编辑 `MatchOperator` Agent 或调整匹配工具
2. **调整代码风格**：修改 `WriteOperator` 的提示词模板
3. **扩展验证机制**：在 `instantiate_operator_main_node` 中添加更多验证逻辑
4. **自定义调试策略**：修改 `CodeDebugger` 和 `Rewriter` 的实现

## 常见问题

**Q: 生成的算子代码质量不高怎么办？**

A: 可以尝试：
- 提供更详细的目标描述
- 选择合适的算子类别
- 启用调试循环，让系统自动修复
- 检查匹配到的参考算子是否合适

**Q: 调试循环一直失败怎么办？**

A: 可能的原因：
- 目标描述过于复杂或模糊
- 测试数据格式不符合算子要求
- LLM API 响应不稳定

建议：
- 简化目标描述
- 检查测试数据格式
- 增加调试轮次
- 手动检查生成的代码

**Q: 如何为特定场景定制算子生成？**

A: 可以：
- 修改提示词模板（`prompts_repo.py`）
- 调整前置工具的数据准备逻辑
- 在验证阶段添加自定义检查
- 扩展 Agent 的实现逻辑

