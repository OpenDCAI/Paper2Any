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
DF_API_KEY=<your_api_key>            # OpenAI-compatible API key
DF_API_URL=<optional_api_url>        # Custom API endpoint
MINERU_DEVICES="0,1,2,3"             # GPU pool for PDF parsing
```

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

## System Dependencies

Linux: `inkscape`, `libreoffice`, `poppler-utils`, `wkhtmltopdf`, `tectonic`
Windows: Inkscape (in PATH), tectonic (via conda)

## Git Workflow

Branch naming: `{username}/dev` (e.g., `lz/dev`, `tch/dev`)
Feature branches: `{username}/{feature-name}`
