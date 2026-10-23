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


# ----------------------------------------------------------------------------
# Phase 2 roles. Tune these on the 50-task dev subset ONLY, then freeze them.
# ----------------------------------------------------------------------------
PLANNER_SYSTEM = (
    "You are a careful software engineer. Do NOT write code. Read the task and reply with: "
    "(1) one sentence restating what the function must do, (2) the key edge cases, "
    "(3) the approach in 2-4 short steps. Under 150 words. Your plan must be consistent "
    "with EVERY example in the task."
)

TESTWRITER_SYSTEM = (
    "You write unit tests. Reply with ONE ```python block of 3 to 5 single-line `assert` "
    "statements that check the function described in the task, using the exact function name "
    "given. Cover normal and edge cases. Only use expected values you are SURE are correct "
    "from the task description and its examples. No explanations."
)

REVIEWER_SYSTEM = (
    "You are a code reviewer. You get a task, a Python solution, and the output of a failing "
    "check. Diagnose the root cause in 1-3 sentences and state the specific change needed. "
    "Do NOT write the full solution. If the check is marked 'AI-written test' and the code "
    "satisfies the task description and its examples, start your reply with TESTS_WRONG and "
    "explain why the test is wrong."
)


def planner_user(task: Task) -> str:
    return f"Task:\n\n{task.prompt}"


def testwriter_user(task: Task) -> str:
    name = task.func_name or "the function"
    return f"Function name: {name}\n\nTask:\n\n{task.prompt}"


def coder_user(task: Task, plan: str = "") -> str:
    text = baseline_user(task)
    return text + (f"\n\nPlan to follow:\n{plan.strip()}" if plan.strip() else "")


def reviewer_user(task: Task, code: str, kind: str, trace: str) -> str:
    label = "AI-written test" if kind == "selftest" else "check"
    return (
        f"Task:\n\n{task.prompt}\n\nSolution:\n```python\n{code}\n```\n\n"
        f"Failing {label} output:\n{trace}"
    )


def repair_user(task: Task, code: str, trace: str, diagnosis: str = "", plan: str = "") -> str:
    parts = [baseline_user(task)]
    if plan.strip():
        parts.append(f"Plan to follow:\n{plan.strip()}")
    parts.append(f"Your previous solution:\n```python\n{code}\n```")
    parts.append(f"It failed this check:\n{trace}")
    if diagnosis.strip():
        parts.append(f"A reviewer's diagnosis:\n{diagnosis.strip()}")
    parts.append(
        "Fix the problem. Reply with the complete corrected solution in one ```python block. "
        "No explanation."
    )
    return "\n\n".join(parts)
