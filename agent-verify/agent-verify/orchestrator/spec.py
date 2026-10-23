"""A configuration (C1..C5) is just a small YAML file; the loop reads these switches."""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Union

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


@dataclass
class ConfigSpec:
    name: str
    plan: bool = False          # planner agent before coding
    reviewer: bool = False      # reviewer agent diagnoses each failure
    test_writer: bool = False   # extra checks written from the task text alone
    max_rounds: int = 0         # repair rounds after the first attempt (0 = single call)
    token_budget: int = 12000   # stop starting new calls once a task has used this many tokens
    temperature: float = 0.2
    max_tokens: int = 1024


def load_spec(name_or_path: Union[str, Path]) -> ConfigSpec:
    p = Path(name_or_path)
    if not p.exists():
        p = CONFIG_DIR / f"{name_or_path}.yaml"
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    allowed = {f.name for f in fields(ConfigSpec)}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown keys in {p}: {sorted(unknown)}")
    data.setdefault("name", p.stem)
    return ConfigSpec(**data)
