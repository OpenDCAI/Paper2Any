Manual and environment-dependent test scripts live here.

These files are intentionally excluded from default `pytest` discovery via
`pyproject.toml` because they require one or more of:
- live LLM/API credentials
- running local services or GPU models
- large fixture files or manually prepared outputs

Run them explicitly when needed, for example:
- `python tests/_archive/manual/sam3/test_paper2drawio_sam3_back.py`  # visual drawio workflow
- `python tests/_archive/manual/frontend_workflows/test_frontend_workflows.py --case all`  # live frontend-contract smoke tests

Keep fast, deterministic unit tests in `tests/`.
