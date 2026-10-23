"""Glue: task + extracted code -> sandboxed run against the HIDDEN tests."""
from __future__ import annotations

import re
from dataclasses import dataclass

from benchmarks.loaders import Task
from verifier.sandbox import run_python


@dataclass
class VerifyResult:
    passed: bool
    error_type: str
    stderr_tail: str
    stdout_tail: str
    duration_s: float
    limits_enforced: bool


def assemble_program(task: Task, code: str, tests: str | None = None) -> str:
    """Build the file that gets executed: [prompt context] + candidate + hidden tests."""
    body = code
    if task.source == "humaneval" and task.entry_point:
        defines_entry = re.search(
            rf"^\s*(?:async\s+)?def\s+{re.escape(task.entry_point)}\b", code, re.MULTILINE
        )
        # The HumanEval prompt carries imports and helper functions the model may omit.
        # A model's own redefinition of the entry function simply overrides the stub.
        if defines_entry:
            body = task.prompt.rstrip() + "\n\n" + code
        else:  # model returned only a function body
            body = task.prompt + code
    return body.rstrip() + "\n\n" + (task.hidden_tests if tests is None else tests)


def visible_check_source(task: Task) -> str:
    """Checks agents may see: a smoke check plus the examples already in the prompt."""
    smoke = f"assert callable({task.entry_point})\n" if task.entry_point else ""
    return smoke + task.visible_tests


def verify(task: Task, code: str, timeout_s: float = 10.0, mem_mb: int = 1024,
           tests: str | None = None) -> VerifyResult:
    """Run `code` against `tests`. tests=None means the HIDDEN tests (final scoring only)."""
    r = run_python(assemble_program(task, code, tests), timeout_s=timeout_s, mem_mb=mem_mb)
    return VerifyResult(
        passed=r.passed,
        error_type=r.error_type,
        stderr_tail=r.stderr_tail,
        stdout_tail=r.stdout_tail,
        duration_s=r.duration_s,
        limits_enforced=r.limits_enforced,
    )
