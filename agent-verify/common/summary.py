"""Turn a JSONL log into per-seed and per-config results. Pure Python, easy to test."""
from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Iterable


def summarize(records: Iterable[dict]) -> dict:
    results = defaultdict(dict)          # (config, seed) -> {task_id: result record}
    tokens = defaultdict(int)            # (config, seed, task_id) -> total tokens
    models = defaultdict(set)            # config -> served models
    for r in records:
        key = (r["config"], int(r["seed"]))
        if r.get("kind") == "task_result":
            results[key][r["task_id"]] = r   # a re-run of the same task keeps the last record
        elif r.get("kind") == "llm_call":
            tokens[(r["config"], int(r["seed"]), r["task_id"])] += (
                int(r.get("prompt_tokens", 0)) + int(r.get("completion_tokens", 0))
            )
            if r.get("model_served"):
                models[r["config"]].add(r["model_served"])

    per_seed = {}
    for (config, seed), all_tasks in sorted(results.items()):
        # Infrastructure errors (API down) are reported separately, never counted as model failures.
        errors = sum(1 for t in all_tasks.values() if t["result"] == "error")
        tasks = {k: t for k, t in all_tasks.items() if t["result"] != "error"}
        n = len(tasks)
        passed = sum(1 for t in tasks.values() if t["result"] == "pass")
        fails = Counter(t.get("error_type") or "unknown" for t in tasks.values() if t["result"] != "pass")
        tok = [tokens[(config, seed, tid)] for tid in tasks]
        per_seed[(config, seed)] = {
            "errors": errors,
            "n": n,
            "passed": passed,
            "pass_rate": passed / n if n else 0.0,
            "failures": dict(fails),
            "mean_tokens_per_task": statistics.mean(tok) if tok else 0.0,
            "tokens_per_solved": (sum(tok) / passed) if passed else float("inf"),
        }

    per_config = {}
    for config in sorted({c for c, _ in per_seed}):
        rates = [v["pass_rate"] for (c, _), v in per_seed.items() if c == config]
        per_config[config] = {
            "seeds": len(rates),
            "mean_pass_rate": statistics.mean(rates),
            "std_pass_rate": statistics.stdev(rates) if len(rates) > 1 else 0.0,
            "models_served": sorted(models[config]),
            "valid": len(models[config]) <= 1,   # >1 served model = silent swap = invalid
        }
    return {"per_seed": per_seed, "per_config": per_config}
