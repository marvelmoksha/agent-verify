"""Paired comparison of two configurations on the SAME tasks, with a bootstrap interval.

Unit of resampling is the task (not the run), because the task sample is the main source of
uncertainty. Seeds are averaged per task first.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable, Optional


def per_task_pass(records: Iterable[dict], seeds: Optional[set[int]] = None) -> dict[str, float]:
    """task_id -> fraction of seeds passed (API-error results are ignored)."""
    seen = defaultdict(dict)   # task -> {seed: 0/1}; a later record for the same seed wins
    for r in records:
        if r.get("kind") != "task_result" or r.get("result") == "error":
            continue
        if seeds is not None and int(r["seed"]) not in seeds:
            continue
        seen[r["task_id"]][int(r["seed"])] = 1.0 if r["result"] == "pass" else 0.0
    return {t: sum(v.values()) / len(v) for t, v in seen.items()}


def tokens_per_task(records: Iterable[dict], seeds: Optional[set[int]] = None) -> dict[str, float]:
    tot, runs = defaultdict(float), defaultdict(set)
    for r in records:
        if r.get("kind") != "llm_call":
            continue
        if seeds is not None and int(r["seed"]) not in seeds:
            continue
        tot[r["task_id"]] += int(r.get("prompt_tokens", 0)) + int(r.get("completion_tokens", 0))
        runs[r["task_id"]].add(int(r["seed"]))
    return {t: tot[t] / len(runs[t]) for t in tot}


def paired_bootstrap(a: dict[str, float], b: dict[str, float], n_boot: int = 10000,
                     seed: int = 0) -> dict:
    """Difference b - a over tasks both configs finished, with a 95% bootstrap interval."""
    common = sorted(set(a) & set(b))
    n = len(common)
    if n == 0:
        raise ValueError("The two logs share no tasks.")
    diffs = [b[t] - a[t] for t in common]
    mean = sum(diffs) / n
    rng = random.Random(seed)
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return {
        "n": n,
        "a_rate": sum(a[t] for t in common) / n,
        "b_rate": sum(b[t] for t in common) / n,
        "diff": mean,
        "ci_low": boots[int(0.025 * n_boot)],
        "ci_high": boots[int(0.975 * n_boot) - 1],
        "b_wins": sum(d > 0 for d in diffs),
        "a_wins": sum(d < 0 for d in diffs),
        "ties": sum(d == 0 for d in diffs),
    }
