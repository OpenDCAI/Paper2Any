from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml

from fastapi_app.config.settings import settings


DEFAULT_PRICING: Dict[str, Any] = {
    "billing": {
        "signup_bonus_points": 0,
        "daily_grant_points": 5,
        "daily_grant_balance_cap": 15,
        "referral_inviter_points": 5,
        "referral_invitee_points": 0,
        "guest_daily_limit": 0,
        "points_purchase_url": "",
        "redeem_code_files": {
            "10": "data/redeem_codes/points_10.txt",
            "50": "data/redeem_codes/points_50.txt",
            "100": "data/redeem_codes/points_100.txt",
        },
        "mindmap": {
            "node_tiers": [
                {"max_nodes": 8, "points": 2, "label": "1-8 节点"},
                {"max_nodes": 16, "points": 3, "label": "9-16 节点"},
                {"max_nodes": 32, "points": 4, "label": "17-32 节点"},
                {"max_nodes": 64, "points": 6, "label": "33-64 节点"},
                {"max_nodes": None, "points": 8, "label": "65+ 节点"},
            ],
            "depth_bonus": {"threshold": 4, "per_level": 1, "max_bonus": 2},
        },
    },
    "workflows": {
        "paper2figure": 1,
        "paper2ppt": 1,
        "pdf2ppt": 1,
        "image2ppt": 1,
        "image2drawio": 1,
        "paper2drawio": 1,
        "paper2poster": 1,
        "paper2video": 5,
        "paper2citation": 1,
        "paper2rebuttal": 1,
        "image_playground": 2,
        "ppt2polish": 1,
        "kb_report": 1,
        "kb_deepresearch": 2,
        "kb_podcast": 2,
        "kb_mindmap": 2,
        "kb_ppt": 1,
        "kb_chat": 1,
        "kb_search": 1,
    },
}

_pricing_cache: Dict[str, Any] | None = None


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_pricing_config() -> Dict[str, Any]:
    global _pricing_cache

    if _pricing_cache is not None:
        return deepcopy(_pricing_cache)

    config_path = Path(settings.BILLING_PRICING_CONFIG_PATH).expanduser().resolve()
    merged = deepcopy(DEFAULT_PRICING)

    if config_path.is_file():
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if isinstance(raw, dict):
            merged = _deep_merge(merged, raw)

    _pricing_cache = merged
    return deepcopy(_pricing_cache)


def get_workflow_cost(workflow_type: str, default: int = 1) -> int:
    pricing = get_pricing_config()
    workflows = pricing.get("workflows", {})
    raw_cost = workflows.get(workflow_type, default)
    try:
        return max(0, int(raw_cost))
    except (TypeError, ValueError):
        return default


def get_mindmap_pricing() -> Dict[str, Any]:
    billing = get_billing_config()
    config = billing.get("mindmap", {})
    return config if isinstance(config, dict) else {}


def estimate_mindmap_points(node_count: int, depth: int) -> Dict[str, Any]:
    pricing = get_mindmap_pricing()
    tiers = pricing.get("node_tiers", [])
    selected_tier: Dict[str, Any] = {}
    base_points = max(2, get_workflow_cost("kb_mindmap", default=2))

    if isinstance(tiers, list):
        for tier in tiers:
            if not isinstance(tier, dict):
                continue
            max_nodes = tier.get("max_nodes")
            try:
                tier_points = max(0, int(tier.get("points", base_points)))
            except (TypeError, ValueError):
                tier_points = base_points

            if max_nodes is None:
                selected_tier = dict(tier)
                base_points = tier_points
                break

            try:
                limit = int(max_nodes)
            except (TypeError, ValueError):
                continue

            if node_count <= limit:
                selected_tier = dict(tier)
                base_points = tier_points
                break

    depth_bonus_cfg = pricing.get("depth_bonus", {})
    threshold = 4
    per_level = 1
    max_bonus = 2
    if isinstance(depth_bonus_cfg, dict):
        try:
            threshold = max(0, int(depth_bonus_cfg.get("threshold", threshold)))
        except (TypeError, ValueError):
            threshold = 4
        try:
            per_level = max(0, int(depth_bonus_cfg.get("per_level", per_level)))
        except (TypeError, ValueError):
            per_level = 1
        try:
            max_bonus = max(0, int(depth_bonus_cfg.get("max_bonus", max_bonus)))
        except (TypeError, ValueError):
            max_bonus = 2

    depth_bonus = max(0, int(depth) - threshold) * per_level
    if max_bonus > 0:
        depth_bonus = min(depth_bonus, max_bonus)

    total_points = max(2, base_points + depth_bonus)

    tier_label = str(selected_tier.get("label") or "").strip()
    if not tier_label:
        if selected_tier.get("max_nodes") is None:
            tier_label = "65+ 节点"
        elif selected_tier.get("max_nodes") is not None:
            tier_label = f"≤ {selected_tier.get('max_nodes')} 节点"
        else:
            tier_label = "默认档位"

    return {
        "node_count": max(1, int(node_count)),
        "depth": max(1, int(depth)),
        "base_points": base_points,
        "depth_bonus": depth_bonus,
        "points": total_points,
        "tier_label": tier_label,
        "rule": {
            "threshold": threshold,
            "per_level": per_level,
            "max_bonus": max_bonus,
        },
    }


def get_billing_config() -> Dict[str, Any]:
    pricing = get_pricing_config()
    billing = pricing.get("billing", {})
    return billing if isinstance(billing, dict) else {}


def get_points_purchase_url() -> str:
    billing = get_billing_config()
    raw_value = billing.get("points_purchase_url", "")
    if isinstance(raw_value, str) and raw_value.strip():
        return raw_value.strip()
    return (settings.POINTS_PURCHASE_URL or "").strip()


def get_redeem_code_files() -> Dict[int, str]:
    billing = get_billing_config()
    configured = billing.get("redeem_code_files", {})
    result: Dict[int, str] = {}

    if isinstance(configured, dict):
        for raw_key, raw_value in configured.items():
            try:
                points = int(raw_key)
            except (TypeError, ValueError):
                continue
            if isinstance(raw_value, str) and raw_value.strip():
                result[points] = raw_value.strip()

    if result:
        return result

    fallback = {
        10: settings.POINTS_REDEEM_CODE_FILE_10,
        50: settings.POINTS_REDEEM_CODE_FILE_50,
        100: settings.POINTS_REDEEM_CODE_FILE_100,
    }
    return {points: value for points, value in fallback.items() if isinstance(value, str) and value.strip()}
