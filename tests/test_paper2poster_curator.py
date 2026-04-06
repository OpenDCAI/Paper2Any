from __future__ import annotations

import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POSTERTOOL_ROOT = PROJECT_ROOT / "dataflow_agent" / "toolkits" / "postertool"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(POSTERTOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(POSTERTOOL_ROOT))


def _install_import_stubs() -> None:
    langgraph_utils_module = types.ModuleType("utils.langgraph_utils")

    class _DummyAgent:
        def __init__(self, *args, **kwargs):
            pass

        def reset(self) -> None:
            pass

        def step(self, _message: str):
            raise RuntimeError("LangGraphAgent should not be used in this unit test")

    langgraph_utils_module.LangGraphAgent = _DummyAgent
    langgraph_utils_module.extract_json = lambda content: content
    langgraph_utils_module.load_prompt = lambda _path: ""
    sys.modules["utils.langgraph_utils"] = langgraph_utils_module


_install_import_stubs()

from src.agents.curator import StoryBoardCurator


def test_curator_removes_nonexistent_visual_assets() -> None:
    curator = StoryBoardCurator()
    story_board = {
        "spatial_content_plan": {
            "sections": [
                {
                    "section_id": "method",
                    "section_title": "Method",
                    "column_assignment": "middle",
                    "vertical_priority": "top",
                    "text_content": ["Point A", "Point B"],
                    "visual_assets": [
                        {"visual_id": "figure_1"},
                        {"visual_id": "key_visual_workflow"},
                    ],
                }
            ]
        }
    }

    sanitized = curator._sanitize_story_board_visuals(story_board, {"figure_1"})
    visuals = sanitized["spatial_content_plan"]["sections"][0]["visual_assets"]

    assert visuals == [{"visual_id": "figure_1"}]
