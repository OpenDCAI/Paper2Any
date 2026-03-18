from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Iterable, List


def add_site_packages_for_packages(
    package_names: Iterable[str],
    extra_paths_env: str = "SAM3_EXTRA_SITE_PACKAGES",
) -> List[str]:
    """Best-effort fallback for optional SAM3 dependencies from sibling conda envs."""
    missing = [name for name in package_names if importlib.util.find_spec(name) is None]
    if not missing:
        return []

    added_paths: List[str] = []
    for site_packages in _candidate_site_packages(extra_paths_env, missing):
        site_str = str(site_packages.resolve())
        if site_str in sys.path:
            continue
        sys.path.insert(0, site_str)
        added_paths.append(site_str)
        missing = [name for name in missing if importlib.util.find_spec(name) is None]
        if not missing:
            break
    return added_paths


def _candidate_site_packages(extra_paths_env: str, package_names: Iterable[str]) -> List[Path]:
    candidates: List[Path] = []

    for raw_path in os.environ.get(extra_paths_env, "").split(os.pathsep):
        raw_path = raw_path.strip()
        if not raw_path:
            continue
        site_packages = Path(raw_path)
        if _has_any_package(site_packages, package_names):
            candidates.append(site_packages)

    version_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    env_roots = _conda_env_roots()
    for env_root in env_roots:
        for env_dir in sorted(env_root.iterdir()):
            site_packages = env_dir / "lib" / version_dir / "site-packages"
            if _has_any_package(site_packages, package_names):
                candidates.append(site_packages)

    deduped: List[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = str(candidate.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(candidate)
    return deduped


def _conda_env_roots() -> List[Path]:
    roots: List[Path] = []

    conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    if conda_prefix:
        env_root = Path(conda_prefix).expanduser().resolve().parent
        if env_root.exists():
            roots.append(env_root)

    default_root = Path("/root/miniconda3/envs")
    if default_root.exists():
        roots.append(default_root)

    deduped: List[Path] = []
    seen: set[str] = set()
    for root in roots:
        resolved = str(root.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(root)
    return deduped


def _has_any_package(site_packages: Path, package_names: Iterable[str]) -> bool:
    if not site_packages.exists() or not site_packages.is_dir():
        return False
    return any(_package_exists(site_packages, name) for name in package_names)


def _package_exists(site_packages: Path, package_name: str) -> bool:
    direct_dir = site_packages / package_name
    if direct_dir.exists():
        return True

    normalized = package_name.replace("_", "-")
    patterns = [
        f"{package_name}-*.dist-info",
        f"{package_name}-*.egg-info",
        f"{normalized}-*.dist-info",
        f"{normalized}-*.egg-info",
    ]
    for pattern in patterns:
        if next(site_packages.glob(pattern), None) is not None:
            return True
    return False
