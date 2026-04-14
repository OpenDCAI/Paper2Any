from __future__ import annotations

import os
import sys
import types


fake_mineru_vl_utils = types.ModuleType("mineru_vl_utils")


class _FakeMinerUClient:
    async def extract_from_file(self, *args, **kwargs):  # pragma: no cover - test stub only
        return {}


fake_mineru_vl_utils.MinerUClient = _FakeMinerUClient
sys.modules.setdefault("mineru_vl_utils", fake_mineru_vl_utils)

os.environ.pop("ALL_PROXY", None)
os.environ.pop("all_proxy", None)
