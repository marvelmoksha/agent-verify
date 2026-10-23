"""Checks that agents ARE allowed to see, plus helpers that turn failures into feedback text.

Leak rule: the benchmark's hidden tests never feed the repair loop. The loop may only use
  (1) the example(s) already shown in the prompt,
  (2) crashes / syntax errors / timeouts,
  (3) tests an AI wrote from the task text alone.
Hidden tests are run once, on the final code, to score it.
"""
from __future__ import annotations

import ast
import builtins
import doctest
import re
from typing import Optional

from verifier.extract import extract_code


def informative_asserts(source: str) -> str:
    """Rewrite `assert f(x) == y` so a failure says what was returned and what was expected."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    out = []
    for node in tree.body:
        t = getattr(node, "test", None)
        if (isinstance(node, ast.Assert) and isinstance(t, ast.Compare) and len(t.ops) == 1
                and isinstance(t.ops[0], ast.Eq) and node.msg is None):
            left, right = ast.unparse(t.left), ast.unparse(t.comparators[0])
            out.append(
                f"_got = ({left})\n_want = ({right})\n"
                f"assert _got == _want, {left!r} + ' returned ' + repr(_got) + ' but expected ' + repr(_want)"
            )
        else:
            out.append(ast.unparse(node))
    return "\n".join(out) + "\n"


def humaneval_visible_checks(prompt: str, max_n: int = 5) -> str:
    """Turn the `>>> f(x)` / expected-value examples in a HumanEval docstring into asserts."""
    try:
        examples = doctest.DocTestParser().get_examples(prompt)
    except ValueError:
        return ""
    lines = []
    for ex in examples:
        src = ex.source.strip()
        want = re.sub(r"\s*(\"\"\"|''')\s*$", "", ex.want.strip())  # drop closing docstring quotes
        if not want or ex.exc_msg:
            continue
        try:
            ast.parse(src, mode="eval")
            ast.parse(want, mode="eval")
        except SyntaxError:
            continue
        lines.append(f"assert ({src}) == ({want})")
        if len(lines) >= max_n:
            break
    return informative_asserts("\n".join(lines)) if lines else ""


def mbpp_func_name(test_source: str) -> Optional[str]:
    """First non-builtin function called in an MBPP test, e.g. 'max_difference'."""
    try:
        tree = ast.parse(test_source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in dir(builtins):
                return node.func.id
    return None


def sanitize_asserts(model_reply: str, max_n: int = 5, max_len: int = 300) -> str:
    """Keep only plain top-level `assert` statements from a test-writer reply."""
    ex = extract_code(model_reply)
    if not ex.ok:
        return ""
    try:
        tree = ast.parse(ex.code)
    except SyntaxError:
        return ""
    kept = [ast.unparse(n) for n in tree.body if isinstance(n, ast.Assert)]
    kept = [k for k in kept if len(k) <= max_len][:max_n]
    return informative_asserts("\n".join(kept)) if kept else ""


def short_trace(stderr: str, max_chars: int = 1200) -> str:
    """Drop sandbox-launcher frames; keep the candidate's frames and the final error line."""
    idx = stderr.find('File "candidate.py"')
    if idx == -1:
        return stderr.strip()[-max_chars:]
    start = stderr.rfind("\n", 0, idx) + 1
    body = stderr[start:].strip()
    return ("Traceback (most recent call last):\n" + body)[-max_chars:]


def feedback_text(error_type: str, stderr_tail: str, timeout_s: float) -> str:
    if error_type == "timeout":
        return f"The program ran longer than {timeout_s:g} seconds (infinite loop or far too slow)."
    if error_type == "memory":
        return "The program ran out of memory."
    if error_type == "forbidden":
        return "The program tried a blocked operation (network, launching processes, or deleting files)."
    if error_type == "early_exit":
        return "The program exited before the checks finished running."
    return short_trace(stderr_tail) or f"The check failed ({error_type})."
