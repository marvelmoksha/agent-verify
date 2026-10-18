"""Prompt templates. Version these in git; freeze them before final experiments."""
from benchmarks.loaders import Task

BASELINE_SYSTEM = (
    "You are an expert Python programmer. Solve the task. Reply with ONE complete Python "
    "code block that contains the full solution, including any imports it needs. "
    "No explanation, no tests, no example usage."
)


def baseline_user(task: Task) -> str:
    if task.source == "humaneval":
        lead = (
            "Complete the following function. Return the full function, including its "
            "signature, in a single ```python block."
        )
    else:
        lead = (
            "Write a Python function that solves the task below. Use the exact function "
            "name and arguments implied by the test. Reply in a single ```python block."
        )
    return f"{lead}\n\n{task.prompt}"
