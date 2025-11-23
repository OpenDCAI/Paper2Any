# DataConvertor 集成测试说明

## 概述

`test_dataconvertor_integration.py` 是使用真实 API LLM 的集成测试文件，用于测试数据转换功能的完整流程。

## 环境配置

### 必需的环境变量

```bash
export DF_API_KEY=your_api_key_here
```

### 可选的环境变量

```bash
# API地址（默认: http://123.129.219.111:3000/v1）
export DF_API_URL=http://your-api-url/v1

# 模型名称（默认: gpt-4o）
export DF_MODEL=gpt-4o
```

## 运行测试

### 方式1: 使用pytest

```bash
# 运行所有集成测试
pytest tests/test_dataconvertor_integration.py -v -s

# 运行特定测试
pytest tests/test_dataconvertor_integration.py::test_pt_mode_real_llm_mapping -v -s

# 运行并显示详细输出
pytest tests/test_dataconvertor_integration.py -v -s --tb=short
```

### 方式2: 直接运行Python文件

```bash
python tests/test_dataconvertor_integration.py
```

## 测试用例

### 1. test_pt_mode_real_llm_mapping
- **目的**: 测试PT模式下的真实LLM映射
- **验证**: 
  - LLM返回的映射结果符合PT模式Schema
  - 中间格式构建正确
  - 包含id, dataset_type, text, meta字段

### 2. test_sft_mode_real_llm_mapping
- **目的**: 测试SFT模式下的真实LLM映射
- **验证**:
  - LLM返回的映射结果符合SFT模式Schema
  - messages结构正确
  - 中间格式构建正确

### 3. test_sft_mode_nested_structure
- **目的**: 测试嵌套对话结构的处理
- **验证**:
  - LLM能正确识别嵌套结构
  - 能正确展开嵌套对话为messages列表

### 4. test_pt_mode_multi_field_concatenation
- **目的**: 测试多字段拼接功能
- **验证**:
  - LLM能识别需要拼接的字段
  - 拼接后的text包含所有相关字段内容

### 5. test_meta_fields_mapping
- **目的**: 测试Meta字段的完整映射
- **验证**:
  - LLM能正确映射所有meta字段
  - source, language, timestamp等字段正确提取

## 测试输出

每个测试会输出：
1. 原始数据行
2. LLM返回的映射结果
3. 构建的中间格式

示例输出：
```
=== 测试PT模式LLM映射 ===
数据行: {
  "content": "人工智能（AI）...",
  "dataset_name": "wikipedia",
  ...
}

LLM返回的映射结果:
{
  "text": "content",
  "meta": {
    "source": "dataset_name",
    "language": "lang"
  }
}

构建的中间格式:
{
  "id": "abc123...",
  "dataset_type": "pretrain",
  "text": "人工智能（AI）...",
  "meta": {
    "source": "wikipedia",
    "language": "zh"
  }
}

✓ PT模式测试通过
```

## 注意事项

1. **API费用**: 这些测试会调用真实的LLM API，会产生API调用费用
2. **网络要求**: 需要能够访问API服务器
3. **测试时间**: 由于需要等待LLM响应，测试可能需要较长时间
4. **环境变量**: 如果未设置 `DF_API_KEY` 或值为 `"test"`，所有测试会被跳过

## 故障排查

### 问题1: 测试被跳过
```
SKIPPED [1] tests/test_dataconvertor_integration.py:需要设置 DF_API_KEY 环境变量才能运行集成测试
```
**解决**: 设置 `DF_API_KEY` 环境变量

### 问题2: API调用失败
```
Error: Failed to call LLM API
```
**解决**: 
- 检查API URL是否正确
- 检查API Key是否有效
- 检查网络连接

### 问题3: 映射结果验证失败
```
AssertionError: PT映射结果验证失败
```
**解决**: 
- 检查LLM返回的映射结果是否符合Schema
- 查看详细输出，确认LLM是否正确理解了提示词

## 与单元测试的区别

- **单元测试** (`test_dataconvertor.py`): 
  - 不调用真实LLM
  - 测试各个方法的逻辑
  - 运行速度快
  - 不需要API配置

- **集成测试** (`test_dataconvertor_integration.py`):
  - 调用真实LLM API
  - 测试完整的数据转换流程
  - 验证LLM映射结果的正确性
  - 需要API配置和网络连接

## 建议

1. 在开发阶段使用单元测试进行快速验证
2. 在提交代码前运行集成测试确保完整流程正常
3. 定期运行集成测试验证LLM映射质量

