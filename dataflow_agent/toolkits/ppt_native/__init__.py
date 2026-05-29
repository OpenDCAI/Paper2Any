"""Native editable PPTX helpers for paper2ppt.

This package vendors the PPT Master SVG-to-DrawingML converter under
``svg_to_pptx`` and exposes a small Paper2Any-facing wrapper in ``render``.
"""

from .render import export_svg_deck_to_pptx, render_pagecontent_to_svg_deck

__all__ = ["export_svg_deck_to_pptx", "render_pagecontent_to_svg_deck"]
