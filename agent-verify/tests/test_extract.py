from verifier.extract import extract_code


def test_simple_fence():
    e = extract_code("Here:\n```python\ndef f():\n    return 1\n```\nDone.", "f")
    assert e.ok and e.code.startswith("def f") and e.method == "fenced_entry"


def test_prefers_final_block_with_entry_point():
    text = "```python\ndef f():\n    return 1\n```\nbetter:\n```python\ndef f():\n    return 2\n```"
    assert "return 2" in extract_code(text, "f").code


def test_skips_usage_example_block():
    text = "```python\ndef f():\n    return 1\n```\n```python\nprint(f())\n```"
    assert "def f" in extract_code(text, "f").code


def test_no_entry_point_takes_longest_block():
    text = "```python\nx=1\n```\n```python\ndef g(a):\n    return a\n```"
    assert "def g" in extract_code(text).code


def test_py_and_uppercase_fence_tags():
    assert extract_code("```py\ndef f():\n    pass\n```", "f").ok
    assert extract_code("```Python\ndef f():\n    pass\n```", "f").ok


def test_fence_without_language():
    assert extract_code("```\ndef f():\n    pass\n```", "f").ok


def test_unclosed_fence_from_truncated_reply():
    e = extract_code("```python\ndef f():\n    return 1\n", "f")
    assert e.ok and e.method == "unclosed_fence" and "return 1" in e.code


def test_raw_code_without_fences():
    e = extract_code("def f():\n    return 1\n", "f")
    assert e.ok and e.method == "raw"


def test_prose_then_code_without_fences():
    e = extract_code("Sure! Here you go:\n\ndef f():\n    return 1\n", "f")
    assert e.ok and e.method == "heuristic"


def test_pure_prose_fails_cleanly():
    e = extract_code("I cannot help with that.", "f")
    assert not e.ok and e.method == "none"


def test_empty_reply():
    assert not extract_code("", "f").ok
    assert not extract_code(None, "f").ok
