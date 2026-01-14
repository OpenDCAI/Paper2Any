# 计费系统配置指南

## 概述

Paper2Any 集成了基于 Cookie 的计费系统，支持对各个工作流接口进行灵活的扣费配置。

## 功能特性

- ✅ **Cookie 认证**：从用户 Cookie 中获取 `appAccessKey` 和 `clientName`
- ✅ **成功后计费**：只有业务逻辑执行成功后才扣费
- ✅ **幂等性保护**：防止 5 分钟内的重复提交
- ✅ **灵活配置**：通过环境变量配置各接口的扣费金额
- ✅ **开发模式**：可通过 `BILLING_ENABLED=false` 禁用计费

## 快速开始

### 1. 配置环境变量

复制 `.env.example` 为 `.env`：

```bash
cp .env.example .env
```

编辑 `.env` 文件，填写以下必填项：

```bash
# 启用计费
BILLING_ENABLED=true

# 填写你的 App SKU ID（联系管理员获取）
BILLING_SKU_ID=your_app_sku_id_here

# 扣费 API 地址（默认无需修改）
BILLING_API_URL=https://openapi.dp.tech/openapi/v1/api/integral/consume
```

### 2. 配置各接口扣费金额

根据实际需求调整各接口的 `eventValue`：

```bash
# Paper2Figure 相关
EVENT_VALUE_PAPER2FIGURE_GENERATE=10
EVENT_VALUE_PAPER2FIGURE_JSON=10
EVENT_VALUE_PAPER2BEAMER=15

# Paper2PPT 相关
EVENT_VALUE_PAPER2PPT_PAGECONTENT=5
EVENT_VALUE_PAPER2PPT_PPT=15
EVENT_VALUE_PAPER2PPT_FULL=20

# PDF2PPT
EVENT_VALUE_PDF2PPT=25

# Image2PPT
EVENT_VALUE_IMAGE2PPT=15
```

### 3. 启动服务

```bash
# 启动后端
cd fastapi_app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 启动前端
cd frontend-workflow
npm run dev
```

## 计费接口列表

以下接口已集成计费功能：

| 接口路径 | 功能 | 环境变量 |
|---------|------|---------|
| `/api/paper2figure/generate` | Paper2Figure 生成 | `EVENT_VALUE_PAPER2FIGURE_GENERATE` |
| `/api/paper2figure/generate_json` | Paper2Figure JSON | `EVENT_VALUE_PAPER2FIGURE_JSON` |
| `/api/paper2beamer/generate` | Paper2Beamer 生成 | `EVENT_VALUE_PAPER2BEAMER` |
| `/api/paper2ppt/pagecontent_json` | Paper2PPT 页面内容 | `EVENT_VALUE_PAPER2PPT_PAGECONTENT` |
| `/api/paper2ppt/ppt_json` | Paper2PPT 生成 | `EVENT_VALUE_PAPER2PPT_PPT` |
| `/api/paper2ppt/full_json` | Paper2PPT 完整流程 | `EVENT_VALUE_PAPER2PPT_FULL` |
| `/api/pdf2ppt/generate` | PDF2PPT 生成 | `EVENT_VALUE_PDF2PPT` |
| `/api/image2ppt/generate` | Image2PPT 生成 | `EVENT_VALUE_IMAGE2PPT` |

## 计费流程

```
用户请求 → 验证 Cookie → 幂等性检查 → 执行业务逻辑 → 业务成功 → 扣费 → 返回结果
                ↓                                    ↓
            401 未登录                          不扣费（业务失败）
```

### 关键点

1. **计费时机**：业务逻辑执行成功后才扣费
2. **失败不扣费**：业务失败或异常时不扣费
3. **幂等性**：5 分钟内相同的 `bizNo` 只能提交一次
4. **用户刷新**：用户刷新页面会生成新的 `bizNo`，视为新请求

## 错误码说明

| 状态码 | 说明 | 处理建议 |
|-------|------|---------|
| 401 | 未找到用户凭证 | 提示用户登录 |
| 402 | 扣费失败 | 检查余额或联系管理员 |
| 409 | 重复请求 | 提示用户不要重复提交 |

## 开发模式

在开发环境中，可以禁用计费：

```bash
BILLING_ENABLED=false
```

此时所有接口将跳过扣费逻辑，直接执行业务。

## 业务流水号（bizNo）生成规则

```
bizNo = timestamp(10位) + clientName前4位hash(4位) + random(4位)
```

示例：
- timestamp: `1736841234`
- clientName: `user123` → hash: `0456`
- random: `7890`
- bizNo: `17368412340456789`

## 日志查看

计费相关日志以 `[Billing]` 前缀标识：

```bash
# 查看计费日志
tail -f dataflow_agent.log | grep "\[Billing\]"
```

## 常见问题

### Q: 如何测试计费功能？

A: 
1. 设置 `BILLING_ENABLED=false` 进行功能测试
2. 设置 `BILLING_ENABLED=true` 并配置测试 SKU ID 进行集成测试

### Q: 用户刷新页面会重复扣费吗？

A: 不会。每次请求都会生成新的 `bizNo`，但幂等性保护会防止 5 分钟内的重复提交。

### Q: 如何修改扣费金额？

A: 修改 `.env` 文件中对应的 `EVENT_VALUE_*` 环境变量，然后重启服务。

### Q: 扣费失败后业务会继续执行吗？

A: 不会。扣费失败会返回 402 错误，业务不会继续执行。

## 技术架构

```
┌─────────────────┐
│  前端请求       │
│  (带 Cookie)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  路由装饰器     │
│  @with_billing  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  BillingService │
│  - 读取 Cookie  │
│  - 生成 bizNo   │
│  - 调用扣费 API │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  扣费 API       │
│  (外部服务)     │
└─────────────────┘
```

## 联系支持

如有问题，请联系技术支持团队。
