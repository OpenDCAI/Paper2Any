Manual and environment-dependent test scripts live here.

These files are intentionally excluded from default `pytest` discovery via
`pyproject.toml` because they require one or more of:
- live LLM/API credentials
- running local services or GPU models
- large fixture files or manually prepared outputs

Run them explicitly when needed, for example:
- `pytest tests/manual/workflows/test_pdf2ppt.py -s`
- `python tests/manual/sam3/test_paper2drawio_sam3_back.py`  # visual drawio workflow

Keep fast, deterministic unit tests in `tests/`.
