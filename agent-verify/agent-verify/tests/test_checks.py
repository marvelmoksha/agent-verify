import sys

import pytest

from benchmarks.loaders import humaneval_row_to_task, mbpp_row_to_task
from verifier.checks import (
    feedback_text, humaneval_visible_checks, informative_asserts, mbpp_func_name,
    sanitize_asserts, short_trace,
)
from verifier.runner import verify, visible_check_source

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="needs Linux/macOS")

HE_PROMPT = '''from typing import List


def remove_duplicates(numbers: List[int]) -> List[int]:
    """ From a list of integers, remove all elements that occur more than once.
    >>> remove_duplicates([1, 2, 3, 2, 4])
    [1, 3, 4]
    >>> remove_duplicates([])
    []
    """
'''


def test_humaneval_examples_become_asserts():
    src = humaneval_visible_checks(HE_PROMPT)
    assert "remove_duplicates([1, 2, 3, 2, 4])" in src and "[1, 3, 4]" in src
    assert src.count("assert _got == _want") == 2


def test_prompt_without_examples_gives_no_checks():
    assert humaneval_visible_checks('def f(x):\n    """double x"""\n') == ""


def test_informative_assert_reports_both_values():
    src = informative_asserts("assert f(2) == 4")
    ns = {"f": lambda x: x}
    with pytest.raises(AssertionError) as e:
        exec(src, ns)
    assert "f(2)" in str(e.value) and "returned 2" in str(e.value) and "expected 4" in str(e.value)


def test_non_equality_asserts_are_left_alone():
    assert "_got" not in informative_asserts("assert isinstance(x, int)")


def test_mbpp_func_name_skips_builtins():
    assert mbpp_func_name("assert set(find_dup([1,1])) == {1}") == "find_dup"
    assert mbpp_func_name("assert max_difference([(3, 5)]) == 2") == "max_difference"


def test_sanitize_keeps_only_plain_asserts():
    reply = "```python\nimport os\nos.remove('x')\nassert f(1) == 2\nprint('hi')\nassert f(2) == 3\n```"
    out = sanitize_asserts(reply)
    assert out.count("assert _got") == 2 and "remove" not in out and "print" not in out


def test_sanitize_caps_count_and_handles_garbage():
    many = "```python\n" + "\n".join(f"assert f({i}) == {i}" for i in range(20)) + "\n```"
    assert sanitize_asserts(many, max_n=5).count("assert _got") == 5
    assert sanitize_asserts("no code here") == ""
    assert sanitize_asserts("```python\nassert (\n```") == ""


def test_short_trace_drops_launcher_frames():
    stderr = ('Traceback (most recent call last):\n  File "<string>", line 41, in <module>\n'
              '    exec(code)\n  File "candidate.py", line 3, in <module>\n    assert f(2) == 4\n'
              "AssertionError: f(2) returned 2 but expected 4\n")
    t = short_trace(stderr)
    assert "<string>" not in t and "candidate.py" in t and "returned 2 but expected 4" in t


def test_feedback_text_for_timeouts():
    assert "5 seconds" in feedback_text("timeout", "", 5)


MB_ROW = {"task_id": 9, "text": "Double a number.", "code": "", "test_setup_code": "",
          "test_list": ["assert dbl(2) == 4", "assert dbl(3) == 6", "assert dbl(10) == 20"]}


def test_mbpp_visible_check_contains_only_the_shown_test():
    t = mbpp_row_to_task(MB_ROW)
    assert "dbl(2)" in t.visible_tests and "dbl(3)" not in t.visible_tests
    assert t.func_name == "dbl"
    assert mbpp_row_to_task(MB_ROW, show_first_test=False).visible_tests == ""


@posix_only
def test_visible_checks_catch_a_bug_with_helpful_feedback():
    t = mbpp_row_to_task(MB_ROW)
    v = verify(t, "def dbl(x):\n    return x", timeout_s=5, tests=visible_check_source(t))
    assert not v.passed and v.error_type == "wrong_answer"
    assert "returned 2 but expected 4" in short_trace(v.stderr_tail)


@posix_only
def test_visible_checks_can_pass_while_hidden_tests_fail():
    """The loop would stop here, and only the hidden tests expose the bug. That is intended."""
    t = mbpp_row_to_task(MB_ROW)
    code = "def dbl(x):\n    return 4 if x == 2 else 0"
    assert verify(t, code, timeout_s=5, tests=visible_check_source(t)).passed
    assert not verify(t, code, timeout_s=5).passed


def test_humaneval_smoke_check_requires_entry_point():
    row = {"task_id": "HumanEval/1", "prompt": HE_PROMPT, "canonical_solution": "",
           "test": "def check(c):\n    pass\n", "entry_point": "remove_duplicates"}
    assert visible_check_source(humaneval_row_to_task(row)).startswith("assert callable(remove_duplicates)")
