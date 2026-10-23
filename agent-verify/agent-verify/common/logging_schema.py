"""JSONL logging for every LLM call and every finished task.

One line per event. Append-only, flushed and fsynced on every write, so a Colab
disconnect loses at most the call in flight. The same file is used to RESUME a run:
any (config, seed, task_id) that already has a `task_result` record is skipped.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Union

KIND_LLM_CALL = "llm_call"
KIND_TASK_RESULT = "task_result"
KIND_ROUND = "round"          # one verifier check inside the repair loop


def log_dir() -> Path:
    """Directory for logs. Set AGENT_VERIFY_LOG_DIR to point at Google Drive on Colab."""
    p = Path(os.environ.get("AGENT_VERIFY_LOG_DIR", "logs"))
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class LogRecord:
    kind: str                 # KIND_LLM_CALL or KIND_TASK_RESULT
    run_id: str               # one id per experiment launch
    config: str               # e.g. "c1_single", "c5_full"
    seed: int
    task_id: str              # e.g. "humaneval/0", "mbpp/11"
    source: str = ""          # "humaneval" or "mbpp"
    round: int = 0            # repair round (0 = first attempt)
    role: str = ""            # planner / coder / testwriter / reviewer / baseline
    backend: str = ""         # local / openrouter / freellmapi
    model_requested: str = ""
    model_served: str = ""    # what the server reports; used to detect silent model swaps
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    result: str = ""          # task_result only: "pass" | "fail" | "error"
    error_type: str = ""      # e.g. "timeout", "syntax", "wrong_answer", "extraction"
    code: str = ""            # task_result only: the code that was verified (for failure analysis)
    detail: str = ""          # task_result only: stderr tail / extra diagnosis
    stop_reason: str = ""     # task_result only: checks_passed | max_rounds | budget | stalled | tests_wrong
    timestamp: float = field(default_factory=time.time)


class JsonlLogger:
    _lock = threading.Lock()  # safe to share between worker threads

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: LogRecord) -> None:
        line = json.dumps(asdict(record), ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())


def read_records(path: Union[str, Path]) -> Iterator[dict]:
    """Yield records; silently skip a truncated last line left by a crash."""
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def completed_task_keys(path: Union[str, Path]) -> set[tuple[str, int, str]]:
    """(config, seed, task_id) triples that already finished, for resuming a run.

    Tasks that ended in an infrastructure error (result == "error", e.g. the API was down)
    are NOT finished: running the command again retries them.
    """
    return {
        (r["config"], int(r["seed"]), r["task_id"])
        for r in read_records(path)
        if r.get("kind") == KIND_TASK_RESULT and r.get("result") != "error"
    }


def models_served(path: Union[str, Path], config: Optional[str] = None) -> set[str]:
    """Distinct served-model names in a log. More than one means the run is INVALID."""
    return {
        r["model_served"]
        for r in read_records(path)
        if r.get("kind") == KIND_LLM_CALL
        and r.get("model_served")
        and (config is None or r.get("config") == config)
    }
