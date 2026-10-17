"""Load HumanEval and MBPP into one common `Task` shape.

IMPORTANT (no leakage): `hidden_tests` is for the verifier ONLY. Agent prompts may
use `task.prompt` and nothing else.

MBPP convention: the standard MBPP prompt shows the FIRST test case to the model so
it can see the expected function name and signature. We keep that convention
(show_first_test=True) but the verifier still runs ALL test cases. Say so in your
README so readers know exactly what the model saw.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Task:
    task_id: str              # namespaced: "humaneval/0", "mbpp/11"
    source: str               # "humaneval" | "mbpp"
    prompt: str               # the ONLY text agents may see
    hidden_tests: str         # python source that raises on failure; verifier only
    entry_point: Optional[str] = None


def humaneval_row_to_task(row: dict) -> Task:
    entry = row["entry_point"]
    tests = row["test"].rstrip() + f"\n\ncheck({entry})\n"
    return Task(
        task_id=row["task_id"].lower(),   # "HumanEval/0" -> "humaneval/0"
        source="humaneval",
        prompt=row["prompt"],
        hidden_tests=tests,
        entry_point=entry,
    )


def mbpp_row_to_task(row: dict, show_first_test: bool = True) -> Task:
    tests = list(row["test_list"])
    prompt = row["text"].strip()
    if show_first_test and tests:
        prompt += "\nYour code should pass this test:\n" + tests[0]
    setup = (row.get("test_setup_code") or "").strip()
    hidden = (setup + "\n" if setup else "") + "\n".join(tests) + "\n"
    return Task(
        task_id=f"mbpp/{row['task_id']}",
        source="mbpp",
        prompt=prompt,
        hidden_tests=hidden,
        entry_point=None,
    )


def load_humaneval() -> list[Task]:
    from datasets import load_dataset  # lazy: keeps unit tests light

    ds = load_dataset("openai/openai_humaneval", split="test")  # 164 problems
    return [humaneval_row_to_task(r) for r in ds]


def load_mbpp(show_first_test: bool = True) -> list[Task]:
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/mbpp", "full", split="test")  # 500 problems
    return [mbpp_row_to_task(r, show_first_test) for r in ds]


def load_all_tasks() -> dict[str, Task]:
    tasks = load_humaneval() + load_mbpp()
    return {t.task_id: t for t in tasks}
