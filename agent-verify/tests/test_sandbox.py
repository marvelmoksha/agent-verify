"""Hostile-case tests: the verifier must be trusted completely before anything is built on it."""
import sys

import pytest

from verifier.sandbox import run_python

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs Linux/macOS (use Colab/WSL)")

T = dict(timeout_s=3.0, mem_mb=512)


def test_correct_code_passes():
    r = run_python("def f(x):\n    return 2 * x\nassert f(2) == 4", **T)
    assert r.passed and r.error_type == ""


def test_wrong_answer_is_detected():
    r = run_python("def f(x):\n    return x\nassert f(2) == 4", **T)
    assert not r.passed and r.error_type == "wrong_answer"


def test_syntax_error():
    r = run_python("def f(:\n  pass", **T)
    assert r.error_type == "syntax"


def test_runtime_error():
    r = run_python("x = 1 / 0", **T)
    assert r.error_type == "runtime"


def test_infinite_loop_times_out_quickly():
    r = run_python("while True:\n    pass", timeout_s=1.5, mem_mb=512)
    assert r.error_type == "timeout"
    assert r.duration_s < 6


def test_sleep_times_out():
    assert run_python("import time\ntime.sleep(30)", timeout_s=1.0, mem_mb=512).error_type == "timeout"


@pytest.mark.skipif(sys.platform == "darwin", reason="RLIMIT_AS is not enforced on macOS")
def test_memory_bomb_is_stopped():
    r = run_python("x = bytearray(4 * 1024**3)", timeout_s=3.0, mem_mb=256)
    assert not r.passed and r.error_type in {"memory", "killed"}


def test_output_flood_is_capped():
    r = run_python("while True:\n    print('x' * 1000)", timeout_s=3.0, mem_mb=512)
    assert not r.passed
    assert len(r.stdout_tail) <= 4000


def test_subprocess_import_blocked():
    r = run_python("import subprocess\nsubprocess.run(['echo', 'hi'])", **T)
    assert r.error_type == "forbidden"


def test_os_system_blocked():
    assert run_python("import os\nos.system('echo hi')", **T).error_type == "forbidden"


def test_fork_blocked():
    assert run_python("import os\nos.fork()", **T).error_type == "forbidden"


def test_network_blocked():
    r = run_python("import socket\ns = socket.socket()\ns.connect(('example.com', 80))", **T)
    assert r.error_type == "forbidden"


def test_file_deletion_blocked():
    assert run_python("import os\nopen('a.txt','w').close()\nos.remove('a.txt')", **T).error_type == "forbidden"


def test_early_sys_exit_is_not_a_pass():
    """A program that exits 0 before the tests run must NOT count as passing."""
    r = run_python("import sys\nsys.exit(0)\nassert False", **T)
    assert not r.passed and r.error_type == "early_exit"


def test_forged_sentinel_does_not_pass():
    r = run_python("print('__VERIFY_OK_deadbeef__')\nimport sys\nsys.exit(0)", **T)
    assert not r.passed


def test_stdin_is_closed():
    r = run_python("input()", **T)
    assert not r.passed


def test_runs_in_a_throwaway_directory():
    run_python("open('leftover.txt', 'w').write('x')", **T)
    import os
    assert not os.path.exists("leftover.txt")
