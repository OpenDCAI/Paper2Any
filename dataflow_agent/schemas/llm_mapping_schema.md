# LLM 映射输出 JSON Schema 规范

本文档定义了数据转换Agent中LLM应返回的字段映射结构。LLM负责识别原始数据中的字段路径，系统会根据这些路径提取数据并构建统一的中间格式。

## 通用说明

### 字段路径格式
- **字符串路径**：单个字段路径，使用点号和方括号表示嵌套结构
  - 示例：`"content"`, `"articles[*].body"`, `"metadata.description"`
- **数组路径**：多个字段路径的数组，系统会按顺序拼接
  - 示例：`["stem", "options[*]"]` → 拼接为 `stem + "\n" + options[0] + "\n" + options[1] + ...`
- **直接值**：某些meta字段可以直接返回字符串值（如source可以从文件路径推断）
  - 示例：`"wikipedia"` 或 `"field_path"`

### 字段路径语法
- 顶层字段：`"field_name"`
- 嵌套字段：`"parent.child"`
- 数组元素：`"items[0]"` 或 `"items[*]"`（表示所有元素）
- 混合路径：`"dialogues[*].turns[0].text"`

---

## 1. PT (Pretrain) 模式映射 Schema

### 完整结构
```json
{
  "text": "field_path | [field_path, ...] | null",
  "meta": {
    "source": "field_path | string_value | null",
    "language": "field_path | string_value | null",
    "timestamp": "field_path | null",
    "token_count": "field_path | null",
    "quality_score": "field_path | null",
    "original_id": "field_path | null"
  }
}
```

### 字段说明

#### text (必须)
- **类型**：`string | array<string> | null`
- **必填性**：必须（如果数据集相关）
- **说明**：预训练文本内容的字段路径
  - 可以是单个字段路径：`"content"`
  - 可以是多字段拼接：`["title", "body"]`
  - 如果数据集不相关，返回 `null`

#### meta (推荐)
- **类型**：`object`
- **必填性**：推荐（至少包含source）
- **说明**：元数据字段映射

##### meta.source (必须)
- **类型**：`string | null`
- **必填性**：必须（meta存在时）
- **说明**：数据来源标识
  - 可以是字段路径：`"dataset_name"` 或 `"meta.source"`
  - 可以是直接字符串值：`"wikipedia"`（LLM从文件名或内容推断）
  - 如果无法确定，返回 `null`（系统会从文件路径推断）

##### meta.language (推荐)
- **类型**：`string | null`
- **必填性**：推荐
- **说明**：语言代码（ISO 639-1）
  - 可以是字段路径：`"lang"` 或 `"metadata.language"`
  - 可以是直接字符串值：`"zh"`, `"en"`, `"mix"`
  - 如果无法确定，返回 `null`（系统会从state.request.language获取）

##### meta.timestamp (可选)
- **类型**：`string | null`
- **必填性**：可选
- **说明**：时间戳字段路径
  - 字段路径：`"created_at"` 或 `"meta.timestamp"`
  - 如果不存在，返回 `null`

##### meta.token_count (可选)
- **类型**：`string | null`
- **必填性**：可选
- **说明**：预计算的token数量字段路径
  - 字段路径：`"token_count"` 或 `"stats.tokens"`
  - 如果不存在，返回 `null`

##### meta.quality_score (可选)
- **类型**：`string | null`
- **必填性**：可选
- **说明**：质量评分字段路径（0.0-1.0）
  - 字段路径：`"quality"` 或 `"scores.quality"`
  - 如果不存在，返回 `null`

##### meta.original_id (可选)
- **类型**：`string | null`
- **必填性**：可选
- **说明**：原始数据集中的ID字段路径
  - 字段路径：`"id"` 或 `"meta.id"`
  - 如果不存在，返回 `null`

### PT模式示例

#### 示例1：简单文本字段
```json
{
  "text": "content",
  "meta": {
    "source": "dataset_name",
    "language": "lang",
    "timestamp": null,
    "token_count": null,
    "quality_score": null,
    "original_id": "id"
  }
}
```

#### 示例2：多字段拼接
```json
{
  "text": ["title", "body", "tags[*]"],
  "meta": {
    "source": "wikipedia",
    "language": "en",
    "timestamp": "created_at",
    "token_count": null,
    "quality_score": "quality_score",
    "original_id": "article_id"
  }
}
```

#### 示例3：嵌套结构
```json
{
  "text": "articles[*].body",
  "meta": {
    "source": "meta.dataset",
    "language": "meta.lang",
    "timestamp": null,
    "token_count": null,
    "quality_score": null,
    "original_id": "meta.article_id"
  }
}
```

#### 示例4：不相关数据集
```json
{
  "text": null,
  "meta": null
}
```

---

## 2. SFT (Supervised Fine-Tuning) 模式映射 Schema

### 完整结构
```json
{
  "messages": [
    {
      "role": "user | assistant | system | tool",
      "content": "field_path | [field_path, ...] | null",
      "loss_mask": true | false | null
    }
  ],
  "system": "field_path | string_value | null",
  "meta": {
    "source": "field_path | string_value | null",
    "language": "field_path | string_value | null",
    "timestamp": "field_path | null",
    "token_count": "field_path | null",
    "quality_score": "field_path | null",
    "original_id": "field_path | null"
  }
}
```

### 字段说明

#### messages (必须)
- **类型**：`array<object>`
- **必填性**：必须（如果数据集相关）
- **说明**：对话消息列表，按时间顺序排列
- **最小长度**：1（至少需要一条消息）

##### messages[].role (必须)
- **类型**：`"user" | "assistant" | "system" | "tool"`
- **必填性**：必须
- **说明**：消息角色
  - `"user"`: 用户输入/指令
  - `"assistant"`: 模型回答
  - `"system"`: 系统提示词（可选，通常放在messages开头或使用顶层system字段）
  - `"tool"`: 工具调用结果（用于Agent场景）
- **识别策略**：
  - LLM应尽可能明确指定role
  - 如果字段名包含"question"/"input"/"prompt"等 → `"user"`
  - 如果字段名包含"answer"/"response"/"output"等 → `"assistant"`
  - 如果字段名包含"system"/"instruction"等 → `"system"`

##### messages[].content (必须)
- **类型**：`string | array<string> | null`
- **必填性**：必须（role存在时）
- **说明**：消息内容的字段路径
  - 可以是单个字段路径：`"question"`
  - 可以是多字段拼接：`["stem", "options[*]"]`
  - 如果该role的消息不存在，返回 `null`

##### messages[].loss_mask (可选)
- **类型**：`boolean | null`
- **必填性**：可选
- **说明**：是否计算loss的标记
  - `true`: 需要计算loss（通常assistant消息）
  - `false`: 不计算loss（通常user/system消息）
  - `null`: 系统会根据role自动推断（assistant→true，其他→false）

#### system (可选)
- **类型**：`string | null`
- **必填性**：可选
- **说明**：全局系统提示词
  - 可以是字段路径：`"system_prompt"` 或 `"meta.system"`
  - 可以是直接字符串值：`"You are a helpful assistant."`
  - 如果不存在，返回 `null`
  - **注意**：如果messages中已有system角色的消息，优先使用messages中的

#### meta (推荐)
- **类型**：`object`
- **必填性**：推荐（至少包含source）
- **说明**：与PT模式相同的元数据结构
- **字段说明**：参考PT模式的meta字段说明

### SFT模式示例

#### 示例1：经典QA对
```json
{
  "messages": [
    {
      "role": "user",
      "content": "question",
      "loss_mask": false
    },
    {
      "role": "assistant",
      "content": "answer",
      "loss_mask": true
    }
  ],
  "system": null,
  "meta": {
    "source": "alpaca",
    "language": "en",
    "timestamp": null,
    "token_count": null,
    "quality_score": null,
    "original_id": "id"
  }
}
```

#### 示例2：多轮对话
```json
{
  "messages": [
    {
      "role": "user",
      "content": "dialogues[*].user",
      "loss_mask": false
    },
    {
      "role": "assistant",
      "content": "dialogues[*].assistant",
      "loss_mask": true
    }
  ],
  "system": "system_prompt",
  "meta": {
    "source": "sharegpt",
    "language": "mix",
    "timestamp": "created_at",
    "token_count": null,
    "quality_score": "quality",
    "original_id": "conversation_id"
  }
}
```

#### 示例3：多字段拼接的问题
```json
{
  "messages": [
    {
      "role": "user",
      "content": ["stem", "options[*]"],
      "loss_mask": false
    },
    {
      "role": "assistant",
      "content": "analysis",
      "loss_mask": true
    }
  ],
  "system": null,
  "meta": {
    "source": "exam_dataset",
    "language": "zh",
    "timestamp": null,
    "token_count": null,
    "quality_score": null,
    "original_id": "question_id"
  }
}
```

#### 示例4：嵌套对话结构
```json
{
  "messages": [
    {
      "role": "user",
      "content": "conversation.turns[*].user_input",
      "loss_mask": false
    },
    {
      "role": "assistant",
      "content": "conversation.turns[*].assistant_response",
      "loss_mask": true
    }
  ],
  "system": "conversation.system_instruction",
  "meta": {
    "source": "meta.dataset_name",
    "language": "meta.lang",
    "timestamp": "meta.timestamp",
    "token_count": null,
    "quality_score": null,
    "original_id": "meta.id"
  }
}
```

#### 示例5：包含system角色的消息
```json
{
  "messages": [
    {
      "role": "system",
      "content": "instruction",
      "loss_mask": false
    },
    {
      "role": "user",
      "content": "input",
      "loss_mask": false
    },
    {
      "role": "assistant",
      "content": "output",
      "loss_mask": true
    }
  ],
  "system": null,
  "meta": {
    "source": "custom_dataset",
    "language": "en",
    "timestamp": null,
    "token_count": null,
    "quality_score": null,
    "original_id": "id"
  }
}
```

#### 示例6：不相关数据集
```json
{
  "messages": null,
  "system": null,
  "meta": null
}
```

---

## 3. 验证规则

### PT模式验证
1. `text` 必须存在且不为null（除非数据集不相关）
2. 如果 `meta` 存在，`meta.source` 必须存在且不为null
3. 字段路径必须存在于样本数据的列名中（或可通过嵌套路径访问）

### SFT模式验证
1. `messages` 必须存在且为数组，长度至少为1（除非数据集不相关）
2. `messages` 中每个元素必须包含 `role` 和 `content`
3. `role` 必须是有效的枚举值
4. 如果 `meta` 存在，`meta.source` 必须存在且不为null
5. 字段路径必须存在于样本数据的列名中（或可通过嵌套路径访问）

### 通用验证
1. 所有字段路径必须使用有效的语法
2. 数组路径中的每个元素都必须是有效的字段路径
3. 如果字段不存在，应返回 `null` 而不是空字符串

---

## 4. 错误处理

### LLM返回格式错误
- 如果JSON解析失败，系统应记录错误并尝试使用robust_parse_json
- 如果必填字段缺失，系统应记录警告并使用默认值或跳过该数据集

### 字段路径无效
- 如果字段路径在数据中不存在，系统应记录警告
- 对于可选字段，使用 `null` 或默认值
- 对于必填字段，跳过该记录或使用fallback策略

### 部分字段缺失
- PT模式：如果 `text` 为null，跳过该数据集
- SFT模式：如果 `messages` 为空或null，跳过该数据集
- meta字段：如果部分meta字段缺失，使用默认值填充

---

## 5. 实现注意事项

1. **字段路径解析**：系统需要支持点号、方括号、通配符等语法
2. **多字段拼接**：当content为数组时，按顺序拼接，使用换行符分隔
3. **role推断**：当LLM未明确指定role时，系统应根据字段名和位置推断
4. **loss_mask默认值**：assistant角色默认为true，其他角色默认为false
5. **meta默认值**：source从文件路径推断，language从state获取，其他字段为null

---

## 6. 版本历史

- **v1.0** (2024): 初始版本，支持PT和SFT模式的完整映射结构

