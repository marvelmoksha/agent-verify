"""Fixed 50-problem dev subset (25 HumanEval + 25 MBPP).

Pick once, commit the JSON file, never edit it. The file stores a checksum of the id
list so accidental edits are caught on load.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Iterable, Union

DEFAULT_PATH = Path(__file__).parent / "dev_ids.json"


def _num(task_id: str) -> int:
    return int(task_id.split("/")[1])


def select_dev_ids(
    humaneval_ids: Iterable[str],
    mbpp_ids: Iterable[str],
    n_each: int = 25,
    seed: int = 42,
) -> list[str]:
    """Deterministic: same inputs and seed always give the same ids."""
    rng = random.Random(seed)
    he = sorted(rng.sample(sorted(humaneval_ids, key=_num), n_each), key=_num)
    mb = sorted(rng.sample(sorted(mbpp_ids, key=_num), n_each), key=_num)
    return he + mb


def _checksum(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def save_dev_ids(
    ids: list[str],
    seed: int,
    n_each: int,
    path: Union[str, Path] = DEFAULT_PATH,
    force: bool = False,
) -> Path:
    p = Path(path)
    if p.exists() and not force:
        raise FileExistsError(
            f"{p} already exists. The dev subset must never change. "
            f"Use force=True only if you are certain."
        )
    payload = {
        "note": "DO NOT EDIT. Fixed dev subset for prompt development.",
        "seed": seed,
        "n_each": n_each,
        "sha256": _checksum(ids),
        "ids": ids,
    }
    p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return p


def load_dev_ids(path: Union[str, Path] = DEFAULT_PATH) -> list[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = data["ids"]
    if _checksum(ids) != data["sha256"]:
        raise ValueError("dev_ids.json was edited: checksum mismatch.")
    return ids
