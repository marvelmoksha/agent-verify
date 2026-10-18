"""Run untrusted, LLM-written Python in a separate process with hard limits.

What it enforces (Linux/macOS):
  * wall-clock timeout (whole process group is killed)
  * CPU-time, memory (address space) and output-file-size limits via `resource`
  * a throw-away working directory, stdin closed, minimal environment
  * guards that block network use, process spawning, and file deletion helpers
  * a random sentinel printed AFTER the tests, so a program that calls sys.exit(0)
    early can never be counted as a pass

Honest limits: the guards stop ACCIDENTAL damage from model-written code. They are not
a security boundary against deliberately hostile code. For stronger isolation run
the whole experiment inside Docker or a throw-away Colab VM.
On Windows the resource limits are unavailable; only the timeout applies
(`limits_enforced=False`). Use Colab, WSL, or Linux for real runs.
"""
from __future__ import annotations

import os
import secrets
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

IS_POSIX = os.name == "posix"
TAIL_CHARS = 4000

# Runs inside the child process: set limits, install guards, then exec the program.
_LAUNCHER = r'''
import sys, os, builtins
path, mem, cpu, fsize = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
try:
    import resource
    for name, val in (("RLIMIT_AS", mem), ("RLIMIT_CPU", cpu), ("RLIMIT_FSIZE", fsize), ("RLIMIT_CORE", 0)):
        try:
            resource.setrlimit(getattr(resource, name), (val, val))
        except (ValueError, OSError):
            pass  # e.g. RLIMIT_AS is not enforced on macOS
except ImportError:
    pass

src = open(path, encoding="utf-8").read()
os.remove(path)            # the program text (and its sentinel) is not readable later
sys.argv = ["candidate"]
code = compile(src, "candidate.py", "exec")
del src

import socket, shutil

class ForbiddenOperation(Exception):
    pass

def _deny(*a, **k):
    raise ForbiddenOperation("blocked by sandbox")

for _n in ("socket", "create_connection", "getaddrinfo", "gethostbyname", "socketpair"):
    if hasattr(socket, _n):
        setattr(socket, _n, _deny)
for _n in ("system", "popen", "fork", "forkpty", "kill", "killpg", "remove", "unlink",
           "rmdir", "removedirs", "rename", "renames", "truncate", "chmod", "chown",
           "execv", "execve", "execvp", "execvpe", "execl", "execle", "execlp", "execlpe",
           "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe"):
    if hasattr(os, _n):
        setattr(os, _n, _deny)
shutil.rmtree = _deny
shutil.move = _deny
sys.modules["subprocess"] = None

exec(code, {"__name__": "candidate", "__builtins__": builtins})
'''


@dataclass
class SandboxResult:
    passed: bool
    error_type: str          # "" when passed
    stdout_tail: str
    stderr_tail: str
    duration_s: float
    returncode: Optional[int]
    limits_enforced: bool


def _tail(path: Path) -> tuple[str, str]:
    """Return (full_text, tail_text) of a file, tolerant of bad bytes."""
    try:
        text = path.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        text = ""
    return text, text[-TAIL_CHARS:]


def classify(returncode: Optional[int], stderr: str, timed_out: bool, sentinel_seen: bool) -> str:
    """Map a finished run to ONE error type ('' means passed)."""
    if timed_out:
        return "timeout"
    if returncode is not None and returncode < 0:
        sig = -returncode
        if sig == getattr(signal, "SIGXCPU", 24):
            return "timeout"
        if sig == getattr(signal, "SIGXFSZ", 25):
            return "output_limit"
        if sig == signal.SIGKILL:
            return "killed"
        return "crash"
    if "ForbiddenOperation" in stderr or "import of subprocess halted" in stderr:
        return "forbidden"
    if "SyntaxError" in stderr or "IndentationError" in stderr:
        return "syntax"
    if "MemoryError" in stderr:
        return "memory"
    if returncode == 0 and sentinel_seen:
        return ""
    if returncode == 0:
        return "early_exit"      # exited 0 but never reached the end of the tests
    if "AssertionError" in stderr:
        return "wrong_answer"
    return "runtime"


def run_python(
    source: str,
    *,
    timeout_s: float = 10.0,
    mem_mb: int = 1024,
    fsize_mb: int = 10,
) -> SandboxResult:
    """Execute `source` (candidate code + hidden tests). Pass = clean exit AND sentinel seen."""
    sentinel = f"__VERIFY_OK_{secrets.token_hex(8)}__"
    program = (
        source.rstrip()
        + f"\n\n__import__('sys').stdout.write({sentinel!r} + '\\n')\n"
        + "__import__('sys').stdout.flush()\n"
    )

    start = time.time()
    with tempfile.TemporaryDirectory(prefix="verify_") as td:
        td_path = Path(td)
        prog = td_path / "candidate.py"
        prog.write_text(program, encoding="utf-8")
        out_p, err_p = td_path / "stdout.txt", td_path / "stderr.txt"

        cmd = [
            sys.executable, "-I", "-B", "-c", _LAUNCHER, str(prog),
            str(mem_mb * 1024 * 1024), str(int(timeout_s) + 2), str(fsize_mb * 1024 * 1024),
        ]
        env = {"PYTHONHASHSEED": "0", "PATH": "/usr/bin:/bin"}
        timed_out = False
        with out_p.open("wb") as outf, err_p.open("wb") as errf:
            proc = subprocess.Popen(
                cmd, cwd=td, env=env, stdin=subprocess.DEVNULL, stdout=outf, stderr=errf,
                start_new_session=IS_POSIX,
            )
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    if IS_POSIX:
                        os.killpg(proc.pid, signal.SIGKILL)
                    else:
                        proc.kill()
                except (ProcessLookupError, PermissionError):
                    pass
                proc.wait()

        out_full, out_tail = _tail(out_p)
        _, err_tail = _tail(err_p)

    sentinel_seen = sentinel in out_full
    error_type = classify(proc.returncode, err_tail, timed_out, sentinel_seen)
    return SandboxResult(
        passed=(error_type == ""),
        error_type=error_type,
        stdout_tail=out_tail.replace(sentinel, "").strip(),
        stderr_tail=err_tail,
        duration_s=time.time() - start,
        returncode=proc.returncode,
        limits_enforced=IS_POSIX,
    )
