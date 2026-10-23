"""End-to-end wiring of run_one() with a fake model: LLM reply -> extract -> sandbox -> JSONL."""
import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from benchmarks.loaders import mbpp_row_to_task
from common.logging_schema import JsonlLogger, completed_task_keys, read_records
from common.summary import summarize
from llm.client import LLMError, LLMResponse

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs Linux/macOS")

spec = importlib.util.spec_from_file_location(
    "run_baseline", Path(__file__).parent.parent / "scripts" / "run_baseline.py")
rb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rb)

TASK = mbpp_row_to_task({
    "task_id": 7, "text": "Write a function to double a number.", "code": "",
    "test_list": ["assert dbl(2) == 4", "assert dbl(5) == 10"], "test_setup_code": ""})
ARGS = Namespace(backend="local", model="fake-model", temperature=0.2, max_tokens=256,
                 timeout=5.0, mem_mb=512)


def fake_chat(reply):
    def _chat(client, model, messages, **kw):
        if isinstance(reply, Exception):
            raise reply
        return LLMResponse(text=reply, prompt_tokens=10, completion_tokens=5, total_tokens=15,
                           served_model="fake-model", latency_s=0.1)
    return _chat


def run(monkeypatch, tmp_path, reply):
    monkeypatch.setattr(rb, "chat", fake_chat(reply))
    path = tmp_path / "log.jsonl"
    result = rb.run_one(TASK, 0, ARGS, None, JsonlLogger(path), "run1")
    return result, path


def test_correct_reply_is_logged_as_pass(monkeypatch, tmp_path):
    res, path = run(monkeypatch, tmp_path, "```python\ndef dbl(x):\n    return 2 * x\n```")
    assert res == "pass"
    assert completed_task_keys(path) == {("c1_single", 0, "mbpp/7")}
    assert summarize(read_records(path))["per_seed"][("c1_single", 0)]["pass_rate"] == 1.0


def test_wrong_reply_is_logged_with_code_and_error(monkeypatch, tmp_path):
    res, path = run(monkeypatch, tmp_path, "```python\ndef dbl(x):\n    return x\n```")
    assert res == "fail"
    rec = [r for r in read_records(path) if r["kind"] == "task_result"][0]
    assert rec["error_type"] == "wrong_answer" and "return x" in rec["code"]


def test_no_code_is_an_extraction_failure(monkeypatch, tmp_path):
    res, path = run(monkeypatch, tmp_path, "Sorry, I cannot do that.")
    rec = [r for r in read_records(path) if r["kind"] == "task_result"][0]
    assert res == "fail" and rec["error_type"] == "extraction"


def test_llm_outage_is_logged_as_error_not_fail(monkeypatch, tmp_path):
    res, path = run(monkeypatch, tmp_path, LLMError("boom"))
    rec = [r for r in read_records(path) if r["kind"] == "task_result"][0]
    assert res == "error" and rec["error_type"] == "llm_error"
