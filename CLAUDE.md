# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DataFlow-Agent is a LangGraph-based multi-agent workflow platform focused on:
- **Paper2Any**: Research paper processing workflows (Paper2Figure, Paper2PPT, Paper2Video, etc.)
- **Easy-DataFlow**: Data governance and pipeline workflows

## Build & Development Commands

```bash
# Install in development mode
pip install -e .
pip install -r requirements-dev.txt

# Optional: Paper processing dependencies
pip install -r requirements-paper.txt
conda install -c conda-forge tectonic  # LaTeX engine

# Run tests
pytest

# Code formatting and linting
black .
ruff check .

# CLI for scaffolding new components
dfa create --wf_name my_workflow      # Create new workflow
dfa create --agent_name my_agent      # Create new agent
dfa create --gradio_name my_page      # Create new Gradio page
```

## Running Services

```bash
# FastAPI backend (port 8000)
uvicorn fastapi_app.main:app --host 0.0.0.0 --port 8000

# Gradio UI (port 7860)
python gradio_app/app.py --page_set {all|data|paper}

# React frontend (port 3000)
cd frontend-workflow && npm install && npm run dev
```

## Environment Variables

```bash
# Core API Configuration
DF_API_KEY=<your_api_key>            # OpenAI-compatible API key
DF_API_URL=<optional_api_url>        # Custom API endpoint
MINERU_DEVICES="0,1,2,3"             # GPU pool for PDF parsing

# Supabase Configuration (Required for Auth & Rate Limiting)
SUPABASE_URL=https://xciveaaildyzbreltihu.supabase.co
SUPABASE_ANON_KEY=<anon_key>         # Public key (respects RLS)
SUPABASE_SERVICE_ROLE_KEY=<key>      # Server-only (bypasses RLS)
SUPABASE_JWT_SECRET=<jwt_secret>     # For token verification
DAILY_WORKFLOW_LIMIT=10              # Rate limit per user/day
```

## Supabase Setup

**Project**: `dataflow-agent` (Region: ap-southeast-1)
**Dashboard**: https://supabase.com/dashboard/project/xciveaaildyzbreltihu

### Database Schema

| Table | Purpose |
|-------|---------|
| `usage_records` | Track daily API calls per user for rate limiting |
| `user_files` | Track generated files stored in Supabase Storage |

Both tables have **Row Level Security (RLS)** enabled - users can only access their own data.

### Storage

- **Bucket**: `user-files` (private, 50MB limit)
- **Path convention**: `user-files/{user_id}/{timestamp}_{filename}`
- RLS policies ensure users can only access their own folder

### Migrations

SQL migration files are in `fastapi_app/migrations/`:
- `001_initial_schema.sql` - Tables and indexes
- `002_rls_policies.sql` - Row Level Security policies
- `003_storage_policies.sql` - Storage bucket policies

### Getting Credentials

1. Go to [Project Settings > API](https://supabase.com/dashboard/project/xciveaaildyzbreltihu/settings/api)
2. Copy: Project URL, anon key, service_role key
3. Go to API > JWT Settings for JWT Secret
4. Create `.env` from `.env.example` and fill in values

## Architecture

### Core Components

```
dataflow_agent/
├── agentroles/              # Agent definitions (auto-registered via @register)
│   ├── common_agents/       # Reusable utility agents
│   ├── data_agents/         # Data processing agents
│   ├── paper2any_agents/    # Paper workflow agents
│   └── cores/               # Agent execution strategies (Simple, ReAct, Graph, VLM)
├── workflow/                # Workflow definitions (wf_*.py)
│   └── registry.py          # Workflow registration system
├── states/                  # State class definitions
├── promptstemplates/        # Prompt template library
└── toolkits/                # Tool implementations (LLM callers, Docker, Image)
```

### Key Patterns

1. **Agent Registration**: Use `@register("agent_name")` decorator for auto-discovery
2. **Workflow Files**: Named `wf_<workflow_name>.py`, use LangGraph StateGraph
3. **State Management**: Dataclass-based with inheritance (MainState → DFState, DataCollectionState)
4. **Gradio Pages**: Named `<page_name>.py` with `create_<page_name>()` function for auto-loading

### Service Architecture

- **FastAPI** (`fastapi_app/`): REST API with workflow adapters
- **Gradio** (`gradio_app/`): Web UI with dynamic page discovery
- **React** (`frontend-workflow/`): Paper2Any visual interface with ReactFlow

## Dependency Tiers

The project has tiered dependencies to support different development scenarios:

| File | Purpose | Size |
|------|---------|------|
| `requirements-lite.txt` | API dev, auth testing, no ML models | ~200MB |
| `requirements-base.txt` | Full backend with torch/transformers | ~4GB |
| `requirements-paper.txt` | Paper workflows (MinerU, SAM, PaddleOCR) | ~8GB |

### Lightweight Development (Recommended for API work)

```bash
# Install lite dependencies (no heavy ML models)
pip install -r requirements-lite.txt

# Run tests with mock API server
./scripts/run_lite_tests.sh
```

### Mock API Server

For testing without real LLM/image API calls:

```bash
# Start mock server (simulates OpenAI-compatible API)
python -m tests.mocks.mock_api_server

# Or use in tests with fixtures from tests/mocks/conftest_mock.py
```

The mock server provides:
- `POST /v1/chat/completions` - Chat completions (streaming supported)
- `POST /v1/images/generations` - Image generation
- `GET /v1/models` - Model listing

### ML Model Dependencies

| Model | Package | VRAM | Purpose |
|-------|---------|------|---------|
| MinerU | `mineru-vl-utils` | 4GB | PDF parsing |
| SAM | `ultralytics` | 2GB | Image segmentation |
| YOLO | `ultralytics` | 1GB | Object detection |
| PaddleOCR | `paddleocr` | 512MB | Text recognition |
| RMBG | ONNX model | 1GB | Background removal |

## System Dependencies

Linux: `inkscape`, `libreoffice`, `poppler-utils`, `wkhtmltopdf`, `tectonic`
Windows: Inkscape (in PATH), tectonic (via conda)

## Git Workflow

Branch naming: `{username}/dev` (e.g., `lz/dev`, `tch/dev`)
Feature branches: `{username}/{feature-name}`
