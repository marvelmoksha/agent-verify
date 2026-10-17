from benchmarks.loaders import humaneval_row_to_task, mbpp_row_to_task

HE_ROW = {
    "task_id": "HumanEval/0",
    "prompt": "def f(x):\n    \"\"\"double x\"\"\"\n",
    "canonical_solution": "    return 2 * x\n",
    "test": "def check(candidate):\n    assert candidate(2) == 4\n",
    "entry_point": "f",
}
MBPP_ROW = {
    "task_id": 11,
    "text": "Write a function to double a number.",
    "code": "def dbl(x): return 2*x",
    "test_list": ["assert dbl(2) == 4", "assert dbl(3) == 6", "assert dbl(0) == 0"],
    "test_setup_code": "",
}


def test_humaneval_task():
    t = humaneval_row_to_task(HE_ROW)
    assert t.task_id == "humaneval/0"
    assert t.entry_point == "f"
    assert t.hidden_tests.rstrip().endswith("check(f)")
    assert "return 2 * x" not in t.prompt  # canonical solution must never reach the prompt


def test_mbpp_shows_only_first_test_in_prompt_but_runs_all():
    t = mbpp_row_to_task(MBPP_ROW)
    assert t.task_id == "mbpp/11"
    assert "assert dbl(2) == 4" in t.prompt
    assert "assert dbl(3) == 6" not in t.prompt
    assert "assert dbl(3) == 6" in t.hidden_tests
    assert "def dbl" not in t.prompt  # reference code never leaks


def test_mbpp_can_hide_first_test():
    t = mbpp_row_to_task(MBPP_ROW, show_first_test=False)
    assert "assert" not in t.prompt


def test_mbpp_setup_code_goes_into_hidden_tests():
    row = dict(MBPP_ROW, test_setup_code="import math")
    assert mbpp_row_to_task(row).hidden_tests.startswith("import math")
