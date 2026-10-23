"""The CLI scripts end to end, with a fake model and a tiny fake benchmark."""
import importlib.util
import sys
from pathlib import Path

import pytest

from benchmarks.loaders import mbpp_row_to_task
from common.logging_schema import read_records
from llm.client import LLMResponse

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs Linux/macOS")
SCRIPTS = Path(__file__).parent.parent / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_task(i, doubled=True):
    row = {"task_id": i, "text": "Double a number.", "code": "", "test_setup_code": "",
           "test_list": [f"assert dbl{i}(2) == 4", f"assert dbl{i}(3) == 6"]}
    return mbpp_row_to_task(row)


@pytest.fixture
def env(tmp_path, monkeypatch):
    rc = load_script("run_config")
    tasks = {t.task_id: t for t in (make_task(1), make_task(2), make_task(3))}
    monkeypatch.setattr(rc, "load_all_tasks", lambda: tasks)
    monkeypatch.setattr(rc, "load_dev_ids", lambda: list(tasks))
    monkeypatch.setattr(rc, "get_client", lambda backend: None)
    monkeypatch.setenv("AGENT_VERIFY_LOG_DIR", str(tmp_path))
    calls = {"n": 0}

    def fake_chat(client, model, messages, **kw):
        calls["n"] += 1
        name = next(n for n in ("dbl1", "dbl2", "dbl3") if n in messages[1]["content"])
        # task 2 is wrong the first time, then right after seeing the feedback
        wrong = name == "dbl2" and "failed this check" not in messages[1]["content"]
        body = "x" if wrong else "2 * x"
        return LLMResponse(f"```python\ndef {name}(x):\n    return {body}\n```", 80, 20, 100, "fake", 0.0)

    monkeypatch.setattr("orchestrator.loop.chat", fake_chat)
    return rc, tmp_path, calls


def run_cli(rc, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["run_config.py", *argv])
    return rc.main()


def test_run_config_end_to_end_and_resume(env, monkeypatch, capsys):
    rc, tmp, calls = env
    args = ("--config", "c2_retry", "--backend", "local", "--model", "fake-model", "--subset", "dev")
    assert run_cli(rc, monkeypatch, *args) == 0
    out = tmp / "c2_retry__fake-model__dev.jsonl"
    finals = [r for r in read_records(out) if r["kind"] == "task_result"]
    assert len(finals) == 3 and all(r["result"] == "pass" for r in finals)
    assert {r["task_id"]: r["round"] for r in finals} == {"mbpp/1": 0, "mbpp/2": 1, "mbpp/3": 0}
    assert (tmp / "c2_retry__fake-model__dev.jsonl.env.json").exists()
    first_calls = calls["n"]
    assert "3/3 = 100.0%" in capsys.readouterr().out

    run_cli(rc, monkeypatch, *args)            # same command again: everything is skipped
    assert calls["n"] == first_calls
    assert "0 to run" in capsys.readouterr().out


def test_compare_script_end_to_end(env, monkeypatch, capsys):
    rc, tmp, _ = env
    for cfg in ("c1_single", "c2_retry"):
        run_cli(rc, monkeypatch, "--config", cfg, "--backend", "local", "--model", "fake-model",
                "--subset", "dev")
    capsys.readouterr()
    cmp_ = load_script("compare")
    monkeypatch.setattr(sys, "argv", ["compare.py", "--a", str(tmp / "c1_single__fake-model__dev.jsonl"),
                                      "--b", str(tmp / "c2_retry__fake-model__dev.jsonl"),
                                      "--names", "C1", "C2", "--boot", "200"])
    assert cmp_.main() == 0
    out = capsys.readouterr().out
    assert "tasks compared (both finished): 3" in out
    assert "C1: 66.7%" in out and "C2: 100.0%" in out
    assert "tasks where C2 won: 1" in out
