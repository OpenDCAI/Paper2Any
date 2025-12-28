# Frontend-Workflow 部署指南

本指南介绍如何部署 Paper2Any 系列工作流的 Web 前端（`frontend-workflow/`）。

## 功能特性

| 功能 | 描述 |
|------|------|
| **Paper2Figure** | 上传论文 PDF，自动生成技术路线图、模型架构图等 |
| **Paper2PPT** | 上传论文，AI 自动生成演示文稿 |
| **Pdf2PPT** | PDF 文档转换为可编辑 PPT |
| **PptPolish** | PPT 美化和润色 |
| **My Files** | 查看和管理生成的文件 |

---

## 快速开始

### 1. 安装依赖

```bash
cd frontend-workflow
npm install
```

### 2. 配置环境变量

环境变量在 `frontend-workflow/.env` 中配置：

```bash
cd frontend-workflow
cp .env.example .env
```

编辑 `.env` 文件：

```env
# Frontend (Vite) - 可选，留空则禁用用户管理功能
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-key

# Backend
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
SUPABASE_JWT_SECRET=your-jwt-secret
```

### 3. 启动开发服务器

```bash
npm run dev
```

默认访问 http://localhost:3000

---

## 有无 Supabase 的区别

Supabase 配置是**可选**的。不配置时，应用仍可正常使用核心功能：

| 功能 | 无 Supabase | 有 Supabase |
|------|-------------|-------------|
| 核心工作流 | ✓ 正常使用 | ✓ 正常使用 |
| 用户登录/注册 | ✗ 跳过 | ✓ 支持 |
| 配额限制 | ✗ 无限制 | ✓ 5次/天（游客）或 10次/天（登录） |
| My Files 云存储 | ✗ 仅本地下载 | ✓ 自动上传到 Supabase Storage |
| 用户菜单 | ✗ 隐藏 | ✓ 显示 |

---

## Supabase 配置指南（可选）

如果你需要用户管理、配额限制和文件云存储功能，按以下步骤配置 Supabase。

### 1. 创建 Supabase 项目

访问 [supabase.com](https://supabase.com) 创建新项目。

### 2. 创建数据库表

在 Supabase SQL Editor 中执行：

```sql
-- 使用记录表（配额追踪）
CREATE TABLE IF NOT EXISTS usage_records (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id TEXT NOT NULL,
  workflow_type TEXT NOT NULL,
  called_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_usage_records_user_date ON usage_records (user_id, called_at);

-- 用户文件表
CREATE TABLE IF NOT EXISTS user_files (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID NOT NULL,
  file_name TEXT NOT NULL,
  file_size BIGINT,
  workflow_type TEXT,
  file_path TEXT,
  download_url TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_user_files_user_id ON user_files (user_id);
```

### 3. 配置 RLS 策略

```sql
-- usage_records RLS
ALTER TABLE usage_records ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can insert own records" ON usage_records
  FOR INSERT WITH CHECK (true);

CREATE POLICY "Users can read own records" ON usage_records
  FOR SELECT USING (user_id = auth.uid()::text OR user_id LIKE 'anon_%');

-- user_files RLS
ALTER TABLE user_files ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can insert own files" ON user_files
  FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can read own files" ON user_files
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own files" ON user_files
  FOR DELETE USING (auth.uid() = user_id);
```

### 4. 创建 Storage Bucket

```sql
-- 创建文件存储桶（公开访问）
INSERT INTO storage.buckets (id, name, public, file_size_limit)
VALUES ('user-files', 'user-files', true, 52428800);

-- Storage RLS 策略
CREATE POLICY "Users upload own files" ON storage.objects
  FOR INSERT WITH CHECK (
    bucket_id = 'user-files' AND
    (auth.uid())::text = (storage.foldername(name))[1]
  );

CREATE POLICY "Users read own files" ON storage.objects
  FOR SELECT USING (
    bucket_id = 'user-files' AND
    (auth.uid())::text = (storage.foldername(name))[1]
  );

CREATE POLICY "Users delete own files" ON storage.objects
  FOR DELETE USING (
    bucket_id = 'user-files' AND
    (auth.uid())::text = (storage.foldername(name))[1]
  );
```

### 5. 启用匿名登录（可选）

在 Supabase Dashboard → Authentication → Providers → 启用 "Anonymous Sign-ins"。

### 6. 获取配置

在 Supabase Dashboard → Settings → API 中获取：
- **Project URL** → `VITE_SUPABASE_URL`
- **anon public** key → `VITE_SUPABASE_ANON_KEY`

---

## 连接自定义后端

修改 `vite.config.ts` 中的代理配置：

```typescript
proxy: {
  '/api': {
    target: 'http://localhost:8001',  // 修改为你的后端地址
    changeOrigin: true,
  },
},
```

---

## 技术栈

- **React 18** + **TypeScript**
- **Vite** - 构建工具
- **Tailwind CSS** - 样式
- **Zustand** - 状态管理
- **Supabase** - 认证 + 数据库 + 存储（可选）
- **Lucide React** - 图标

---

## 开发命令

```bash
npm run dev      # 启动开发服务器
npm run build    # 构建生产版本
npm run preview  # 预览生产构建
```
