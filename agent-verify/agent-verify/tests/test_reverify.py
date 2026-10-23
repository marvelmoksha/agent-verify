import importlib.util
import sys
from pathlib import Path

import pytest

from benchmarks.loaders import mbpp_row_to_task
from common.compare import per_task_pass
from common.logging_schema import KIND_TASK_RESULT, JsonlLogger, LogRecord, read_records

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs Linux/macOS")
spec = importlib.util.spec_from_file_location("reverify", Path(__file__).parent.parent / "scripts" / "reverify.py")
rv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rv)

TASK = mbpp_row_to_task({"task_id": 4, "text": "Double.", "code": "", "test_setup_code": "",
                         "test_list": ["assert dbl(2) == 4", "assert dbl(3) == 6"]})
# valid code that the OLD sandbox rejected: it imports subprocess at import time
CODE = "import subprocess\n\ndef dbl(x):\n    return 2 * x\n"


def rec(result, error, task="mbpp/4", seed=0):
    return LogRecord(kind=KIND_TASK_RESULT, run_id="r", config="c1", seed=seed, task_id=task,
                     result=result, error_type=error, code=CODE)


def test_forbidden_false_alarm_is_corrected_and_original_kept(tmp_path):
    p = tmp_path / "log.jsonl"
    lg = JsonlLogger(p)
    lg.log(rec("fail", "forbidden"))
    lg.log(rec("fail", "wrong_answer", task="mbpp/5"))       # not selected, must be untouched
    fixed = list(rv.reverify_records(read_records(p), {"mbpp/4": TASK}, {"forbidden"}))
    assert len(fixed) == 1 and fixed[0].result == "pass"
    for f in fixed:
        lg.log(f)
    assert len(list(read_records(p))) == 3                    # original line still there
    assert per_task_pass(read_records(p)) == {"mbpp/4": 1.0, "mbpp/5": 0.0}   # last record wins


def test_rows_without_code_are_skipped(tmp_path):
    r = rec("fail", "forbidden")
    r.code = ""
    assert list(rv.reverify_records([r.__dict__], {"mbpp/4": TASK}, {"forbidden"})) == []
