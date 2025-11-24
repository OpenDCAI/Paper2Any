# DataFlow Agent 工作流编辑器

一个基于 React + ReactFlow 的可视化工作流编辑器，用于配置和管理 DataFlow Agent 工作流。

## 功能特性

### ✨ 核心功能

1. **可视化工作流编辑**
   - 拖拽式节点添加
   - 节点连线创建工作流
   - 实时预览工作流结构

2. **Agent 节点类型**
   - **自定义 Agent**（5种模式）
     - Simple Agent - 简单模式，单次LLM调用
     - ReAct Agent - 推理+行动模式，支持重试验证
     - Graph Agent - 图模式，复杂任务流程
     - VLM Agent - 视觉语言模型，图像理解/生成
     - Parallel Agent - 并行模式，批量处理
   
   - **预设 Agent**（20+种）
     - 数据处理：DataCollector, DataConvertor, DataExporter, Classifier
     - 代码生成：Writer, Rewriter, Debugger, OperatorExecutor
     - Pipeline：PipelineBuilder, Recommender, Refiner
     - 图像处理：IconGenerator, IconEditor, IconPromptGenerator
     - 辅助工具：Intenter, InfoRequester, GrammarChecker, Matcher, PromptWriter

3. **节点配置面板**
   - 基础参数配置
     - 模型名称（model_name）
     - API URL（chat_api_url）
     - 温度参数（temperature）
     - 最大Token数（max_tokens）
     - 工具模式（tool_mode）
     - 解析器类型（parser_type）
   
   - 模式特定参数
     - ReAct模式：最大重试次数
     - Parallel模式：并发限制
     - VLM模式：视觉模式、图像细节
   
   - 提示词配置
     - System Prompt - 定义角色和行为
     - Task Prompt - 描述具体任务
   
   - JSON Schema配置
     - Response Schema - 定义返回结构
     - Schema Description - 格式描述
     - Response Example - 返回示例
     - Required Fields - 必填字段

4. **工作流管理**
   - 保存/加载工作流（待实现）
   - 清空工作流
   - 运行工作流（待实现）
   - 实时统计（节点数、连接数）

## 使用指南

### 启动应用

```bash
cd frontend-workflow
npm install
npm run dev
```

访问 http://localhost:3000

### 创建工作流

1. **添加节点**
   - 从左侧面板选择 Agent 类型
   - 拖拽到中央画布
   - 节点自动显示在画布上

2. **配置节点**
   - 点击画布上的节点
   - 右侧面板显示配置选项
   - 修改参数后点击"保存配置"

3. **连接节点**
   - 鼠标悬停在节点边缘的圆点（端口）
   - 拖拽到另一个节点的端口
   - 创建连接线

4. **删除节点**
   - 选中节点后在右侧面板点击"删除节点"
   - 或选中节点后按 Delete 键

### Agent 配置示例

#### Simple Agent 配置
```json
{
  "model_name": "gpt-4",
  "temperature": 0.7,
  "max_tokens": 4096,
  "system_prompt": "你是一个代码生成助手",
  "task_prompt": "生成Python函数",
  "response_schema": {
    "code": "完整代码",
    "explanation": "代码说明"
  },
  "required_fields": ["code"]
}
```

#### ReAct Agent 配置
```json
{
  "model_name": "gpt-4",
  "temperature": 0.3,
  "max_retries": 5,
  "system_prompt": "你是一个具有推理能力的AI助手",
  "task_prompt": "分析并解决问题"
}
```

#### VLM Agent 配置
```json
{
  "model_name": "gpt-4-vision",
  "vlm_mode": "understanding",
  "image_detail": "high",
  "system_prompt": "你是一个图像分析专家",
  "task_prompt": "分析图像内容"
}
```

## 技术栈

- **React 18** - UI框架
- **TypeScript** - 类型安全
- **ReactFlow** - 工作流可视化
- **Tailwind CSS** - 样式框架
- **Vite** - 构建工具
- **Zustand** - 状态管理（可选）
- **Lucide React** - 图标库

## 项目结构

```
frontend-workflow/
├── src/
│   ├── components/          # React组件
│   │   ├── AgentNodePanel.tsx      # 左侧Agent列表
│   │   ├── WorkflowCanvas.tsx      # 中央画布
│   │   ├── NodeConfigPanel.tsx     # 右侧配置面板
│   │   ├── CustomNode.tsx          # 自定义节点组件
│   │   └── ParticleBackground.tsx  # 粒子背景
│   ├── data/                # 数据定义
│   │   └── agentTypes.ts           # Agent类型定义
│   ├── types/               # TypeScript类型
│   │   └── index.ts                # 类型定义
│   ├── styles/              # 样式文件
│   │   └── globals.css             # 全局样式
│   ├── App.tsx              # 主应用组件
│   └── main.tsx             # 应用入口
├── public/                  # 静态资源
├── index.html              # HTML模板
├── package.json            # 依赖配置
├── vite.config.ts          # Vite配置
├── tailwind.config.js      # Tailwind配置
└── tsconfig.json           # TypeScript配置
```

## 与后端集成

### Agent 参数映射

前端配置的参数会映射到后端的 Agent 创建函数：

```python
from dataflow_agent.agentroles import (
    create_simple_agent,
    create_react_agent,
    create_graph_agent,
    create_vlm_agent,
    create_parallel_agent
)

# Simple Agent
agent = create_simple_agent(
    "writer",
    model_name="gpt-4",
    temperature=0.7,
    max_tokens=4096,
    response_schema={"code": "string", "files": "list"},
    required_fields=["code"]
)

# ReAct Agent
agent = create_react_agent(
    "planner",
    max_retries=5,
    temperature=0.3
)

# VLM Agent
agent = create_vlm_agent(
    "image_analyzer",
    vlm_mode="understanding",
    image_detail="high"
)
```

## 待实现功能

- [ ] 工作流保存/加载（JSON格式）
- [ ] 工作流执行引擎集成
- [ ] 节点执行状态可视化
- [ ] 撤销/重做功能
- [ ] 节点复制/粘贴
- [ ] 工作流模板库
- [ ] 导出为Python代码
- [ ] 实时协作编辑

## 开发指南

### 添加新的 Agent 类型

1. 在 `src/data/agentTypes.ts` 中添加定义：

```typescript
{
  id: 'my-agent',
  name: 'MyAgent',
  displayName: '我的Agent',
  category: 'custom',
  description: 'Agent描述',
  icon: 'Zap',
  color: '#3b82f6',
  inputs: 1,
  outputs: 1,
  mode: 'simple',
  defaultConfig: {
    model_name: 'gpt-4',
    temperature: 0.0,
    // ... 其他默认配置
  },
}
```

2. Agent 会自动出现在左侧面板中

### 自定义节点样式

在 `src/styles/globals.css` 中添加样式：

```css
.node-custom {
  --node-color: #your-color;
  --node-glow: rgba(r, g, b, 0.5);
}
```

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
