"""Record the environment a run used. Results depend on installed libraries (a solution that
imports sympy passes on Colab and fails on a clean machine), so save it next to every log."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib import metadata
from pathlib import Path


def snapshot() -> dict:
    pkgs = sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()
                  if d.metadata["Name"])
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages_sha256": hashlib.sha256("\n".join(pkgs).encode()).hexdigest(),
        "packages": pkgs,
    }


def write_once(log_path) -> Path:
    p = Path(str(log_path) + ".env.json")
    if not p.exists():
        p.write_text(json.dumps(snapshot(), indent=1), encoding="utf-8")
    return p
