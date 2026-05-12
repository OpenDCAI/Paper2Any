from .contracts import EditablePPTInputRunRequest, EditablePPTRunArtifacts, EditablePPTRunRequest

__all__ = [
    "EditablePPTInputRunRequest",
    "EditablePPTRunArtifacts",
    "EditablePPTRunRequest",
    "generate_slide_code_from_ir",
    "parse_pagecontent_from_input",
    "plan_deck_ir_and_materials",
    "run_from_input",
    "run_from_pagecontent",
]


def __getattr__(name: str):
    if name in {
        "generate_slide_code_from_ir",
        "parse_pagecontent_from_input",
        "plan_deck_ir_and_materials",
        "run_from_input",
        "run_from_pagecontent",
    }:
        from . import runner

        return getattr(runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
