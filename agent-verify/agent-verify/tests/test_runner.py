import sys

import pytest

from benchmarks.loaders import humaneval_row_to_task, mbpp_row_to_task
from verifier.runner import assemble_program, verify

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="needs Linux/macOS")

HE_ROW = {
    "task_id": "HumanEval/9",
    "prompt": "from typing import List\n\n\ndef helper(x: int) -> int:\n    return x + 1\n\n\n"
              "def f(xs: List[int]) -> List[int]:\n    \"\"\"increment every element\"\"\"\n",
    "canonical_solution": "    return [helper(x) for x in xs]\n",
    "test": "def check(candidate):\n    assert candidate([1, 2]) == [2, 3]\n",
    "entry_point": "f",
}
MB_ROW = {
    "task_id": 5, "text": "Write a function to double a number.", "code": "def dbl(x): return 2*x",
    "test_list": ["assert dbl(2) == 4", "assert dbl(5) == 10"], "test_setup_code": "",
}


@posix_only
def test_humaneval_full_function_with_prompt_helpers():
    task = humaneval_row_to_task(HE_ROW)
    code = "def f(xs):\n    return [helper(x) for x in xs]"   # no imports, uses the prompt's helper
    assert verify(task, code, timeout_s=5).passed


@posix_only
def test_humaneval_body_only_completion():
    task = humaneval_row_to_task(HE_ROW)
    assert verify(task, "    return [x + 1 for x in xs]\n", timeout_s=5).passed


@posix_only
def test_humaneval_wrong_solution_fails():
    task = humaneval_row_to_task(HE_ROW)
    v = verify(task, "def f(xs):\n    return xs", timeout_s=5)
    assert not v.passed and v.error_type == "wrong_answer"


@posix_only
def test_mbpp_pass_and_fail():
    task = mbpp_row_to_task(MB_ROW)
    assert verify(task, "def dbl(x):\n    return 2 * x", timeout_s=5).passed
    bad = verify(task, "def dbl(x):\n    return x + x + 1", timeout_s=5)
    assert not bad.passed and bad.error_type == "wrong_answer"


@posix_only
def test_main_guard_example_usage_does_not_run():
    task = mbpp_row_to_task(MB_ROW)
    code = "def dbl(x):\n    return 2 * x\n\nif __name__ == '__main__':\n    input()\n"
    assert verify(task, code, timeout_s=5).passed


def test_assemble_puts_hidden_tests_last():
    prog = assemble_program(mbpp_row_to_task(MB_ROW), "def dbl(x):\n    return 2 * x")
    assert prog.rstrip().endswith("assert dbl(5) == 10")
