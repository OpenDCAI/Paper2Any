#!/usr/bin/env python3

from __future__ import annotations

import argparse
import secrets
import string
from pathlib import Path


ALPHABET = string.ascii_uppercase + string.digits
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "redeem_codes"
TIERS = (
    (10, "points_10.txt"),
    (50, "points_50.txt"),
    (100, "points_100.txt"),
)


def make_token(prefix: str, seen: set[str]) -> str:
    while True:
        parts = [
            "".join(secrets.choice(ALPHABET) for _ in range(4)),
            "".join(secrets.choice(ALPHABET) for _ in range(4)),
            "".join(secrets.choice(ALPHABET) for _ in range(4)),
        ]
        token = f"P2A-{prefix}-{'-'.join(parts)}"
        if token not in seen:
            seen.add(token)
            return token


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Paper2Any points redeem codes.")
    parser.add_argument("--count", type=int, default=200, help="Number of tokens to generate per tier.")
    args = parser.parse_args()

    if args.count <= 0:
        raise SystemExit("--count must be greater than 0")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()

    for points, filename in TIERS:
        tokens = [make_token(str(points), seen) for _ in range(args.count)]
        output_path = OUTPUT_DIR / filename
        output_path.write_text("\n".join(tokens) + "\n", encoding="utf-8")
        print(f"generated {len(tokens)} tokens -> {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
