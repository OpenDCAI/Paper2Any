"""LayoutPlanner — deterministic layout sanity + content-aware adjustments.

The planner sits between the LLM-picked `layout_name` and the downstream PPT
coder. Its job is narrow: take the raw slot ratios (either LLM-produced or the
pagecontent template defaults) and make sure they do not overlap, overflow the
slide, or cram more content than their h_ratio can hold.

Everything here is deterministic. The LLM only picks a name; geometry lives in
`PAGECONTENT_LAYOUT_SLOTS` and the helpers below.
"""

from __future__ import annotations

from typing import Any

from .pagecontent_adapter import PAGECONTENT_LAYOUT_SLOTS

# Tolerances for float comparisons so validators do not trip on rounding.
_EPS = 1e-3
# One visible bullet line costs roughly this fraction of slide height at the
# default body font size. Used to check whether a body slot can hold N lines.
_LINE_H_RATIO = 0.062
_MIN_SLOT_H = 0.05
_MIN_SLOT_W = 0.05


def _slot_bounds(slot: dict[str, Any]) -> tuple[float, float, float, float]:
    x = float(slot.get("x_ratio", 0.0))
    y = float(slot.get("y_ratio", 0.0))
    w = float(slot.get("w_ratio", 0.0))
    h = float(slot.get("h_ratio", 0.0))
    return x, y, w, h


def _slots_overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    ax, ay, aw, ah = _slot_bounds(a)
    bx, by, bw, bh = _slot_bounds(b)
    return not (ax + aw <= bx + _EPS or bx + bw <= ax + _EPS or ay + ah <= by + _EPS or by + bh <= ay + _EPS)


def validate_layout(slots: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    """Check slot ratios stay inside [0,1] and do not mutually overlap.

    Returns (is_valid, list_of_reasons). Callers typically use the boolean and
    fall back to the template when it is False; reasons are kept for logging.
    """
    reasons: list[str] = []
    for slot in slots:
        slot_id = str(slot.get("slot_id", "?"))
        x, y, w, h = _slot_bounds(slot)
        if w < _MIN_SLOT_W or h < _MIN_SLOT_H:
            reasons.append(f"{slot_id}: slot too small ({w:.3f}x{h:.3f})")
        if x < -_EPS or y < -_EPS:
            reasons.append(f"{slot_id}: negative origin ({x:.3f},{y:.3f})")
        if x + w > 1.0 + _EPS or y + h > 1.0 + _EPS:
            reasons.append(f"{slot_id}: overflows slide ({x + w:.3f},{y + h:.3f})")

    for i, slot_a in enumerate(slots):
        for slot_b in slots[i + 1 :]:
            if _slots_overlap(slot_a, slot_b):
                reasons.append(
                    f"{slot_a.get('slot_id', '?')} overlaps {slot_b.get('slot_id', '?')}"
                )
    return (not reasons), reasons


def template_slots_for(layout_name: str) -> list[dict[str, Any]]:
    """Deep-copy the template slots for a layout, falling back to section_divider."""

    base = PAGECONTENT_LAYOUT_SLOTS.get(layout_name) or PAGECONTENT_LAYOUT_SLOTS["section_divider"]
    return [dict(slot) for slot in base]


def sanitize_slots(
    slots: list[dict[str, Any]] | None,
    *,
    layout_name: str,
) -> list[dict[str, Any]]:
    """Validate slots; on any problem, fall back to the template for layout_name."""

    candidate = list(slots or [])
    if not candidate:
        return template_slots_for(layout_name)
    ok, _reasons = validate_layout(candidate)
    if ok:
        return [dict(slot) for slot in candidate]
    return template_slots_for(layout_name)


def densify_body(
    slots: list[dict[str, Any]],
    *,
    points_count: int,
    body_slot_ids: tuple[str, ...] = ("body", "supporting_body", "callout"),
) -> list[dict[str, Any]]:
    """Check that the combined body slot height can hold points_count lines.

    If not, try to steal vertical room from an empty tail strip below the body.
    If still not enough, leave the slots alone — the caller should have picked a
    different layout_name via `auto_repick_layout` before reaching here.
    """

    if points_count <= 0:
        return slots
    required = points_count * _LINE_H_RATIO
    text_slots = [slot for slot in slots if str(slot.get("slot_id", "")) in body_slot_ids]
    if not text_slots:
        return slots
    # Grow the first body slot downward into unused space, up to 0.9 of slide.
    primary = text_slots[0]
    x, y, w, h = _slot_bounds(primary)
    if h >= required - _EPS:
        return slots
    # Find the largest y below primary that stays clear of other slots at this x-range.
    ceiling = 0.92
    for other in slots:
        if other is primary:
            continue
        ox, oy, ow, oh = _slot_bounds(other)
        overlaps_x = not (ox + ow <= x + _EPS or x + w <= ox + _EPS)
        if overlaps_x and oy > y + _EPS:
            ceiling = min(ceiling, oy)
    new_h = max(h, min(required, ceiling - y))
    if new_h > h + _EPS:
        primary = dict(primary)
        primary["h_ratio"] = round(new_h, 4)
        return [primary if slot is text_slots[0] else slot for slot in slots]
    return slots


def grid_image_slots(
    base_slot: dict[str, Any],
    image_count: int,
) -> list[dict[str, Any]]:
    """Split a single image slot into a grid of image_count sub-slots.

    Produces a grid whose cells all sit inside `base_slot`. Does not touch any
    non-image slots, so it cannot cause overlap with body/callout. Superior to
    the legacy vertical-only split when image_count >= 3.
    """

    if image_count <= 1:
        return [dict(base_slot, slot_id="image")]
    x, y, w, h = _slot_bounds(base_slot)
    gap = 0.008
    # Pick a grid shape: wide box gets more columns, tall box gets more rows.
    if image_count == 2:
        cols, rows = (2, 1) if w >= h else (1, 2)
    elif image_count <= 4:
        cols, rows = 2, 2
    elif image_count <= 6:
        cols, rows = 3, 2
    else:
        cols, rows = 3, (image_count + 2) // 3
    cell_w = (w - gap * (cols - 1)) / cols
    cell_h = (h - gap * (rows - 1)) / rows
    out: list[dict[str, Any]] = []
    for index in range(image_count):
        col = index % cols
        row = index // cols
        if row >= rows:
            break
        slot_id = "image" if index == 0 else f"image_{index + 1:02d}"
        out.append(
            {
                "slot_id": slot_id,
                "x_ratio": round(x + col * (cell_w + gap), 4),
                "y_ratio": round(y + row * (cell_h + gap), 4),
                "w_ratio": round(cell_w, 4),
                "h_ratio": round(cell_h, 4),
            }
        )
    return out


def plan_layout(
    layout: dict[str, Any],
    *,
    layout_name: str,
    points_count: int,
    image_count: int,
) -> dict[str, Any]:
    """End-to-end layout sanitisation: validate, densify body, grid images.

    Call site: `llm_planner._merge_slide` right after the LLM slide patch has
    been merged. The returned dict has `name` set to layout_name and `slots`
    sanitised against the geometry of that template.
    """

    slots = sanitize_slots(layout.get("slots"), layout_name=layout_name)
    if image_count >= 1:
        primary_index = next(
            (
                index
                for index, slot in enumerate(slots)
                if str(slot.get("slot_id", "")).startswith("image")
            ),
            None,
        )
        if primary_index is not None:
            base_slot = slots[primary_index]
            image_slots = grid_image_slots(base_slot, max(image_count, 1))
            slots = [
                slot
                for slot in slots
                if not str(slot.get("slot_id", "")).startswith("image")
            ]
            slots[primary_index:primary_index] = image_slots
    slots = densify_body(slots, points_count=points_count)
    # Last-resort validation; fall back to template if densify_body re-introduced
    # an overlap (rare, but cheap insurance).
    ok, _ = validate_layout(slots)
    if not ok:
        slots = template_slots_for(layout_name)
    planned = dict(layout)
    planned["name"] = layout_name
    planned["slots"] = slots
    return planned


__all__ = [
    "grid_image_slots",
    "densify_body",
    "plan_layout",
    "sanitize_slots",
    "template_slots_for",
    "validate_layout",
]
