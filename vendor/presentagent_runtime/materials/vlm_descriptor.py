from __future__ import annotations

from typing import Any


class VLMDescriptor:
    def __init__(self, client) -> None:
        self.client = client

    def score_candidates(
        self,
        request: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        scored: list[dict[str, Any]] = []
        for candidate in candidates:
            prompt = self._build_score_prompt(request, candidate)
            raw = self.client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format="json",
            )
            payload = self._parse_json(raw)
            score = float(payload.get("score", 0.0) or 0.0)
            scored.append(
                {
                    **candidate,
                    "vlm_score": score,
                    "score_reason": str(payload.get("reason") or ""),
                }
            )
        return sorted(scored, key=lambda item: item.get("vlm_score", 0.0), reverse=True)

    def _build_score_prompt(self, request: dict[str, Any], candidate: dict[str, Any]) -> str:
        return (
            "Score how well this asset matches the slide request.\n"
            "Return JSON only with keys: score, reason.\n"
            f"Request: {request}\n"
            f"Candidate: {candidate}"
        )

    def _parse_json(self, raw: str) -> dict[str, Any]:
        import json

        text = (raw or "").strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return json.loads(text or "{}")
