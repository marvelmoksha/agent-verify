"""The agent loop. Plain Python on purpose: all the interesting behaviour is in the prompts,
the checks and the budgets, not in a framework.

Per task:
  [planner] -> [test writer] -> coder -> check -> (reviewer) -> coder -> check -> ...
Checks use ONLY agent-visible information. The hidden benchmark tests score the final code once.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Optional

from agents.prompts import (
    BASELINE_SYSTEM, PLANNER_SYSTEM, REVIEWER_SYSTEM, TESTWRITER_SYSTEM,
    coder_user, planner_user, repair_user, reviewer_user, testwriter_user,
)
from benchmarks.loaders import Task
from common.logging_schema import (
    KIND_LLM_CALL, KIND_ROUND, KIND_TASK_RESULT, JsonlLogger, LogRecord,
)
from llm.client import LLMError, chat
from orchestrator.spec import ConfigSpec
from verifier.checks import feedback_text, sanitize_asserts
from verifier.extract import extract_code
from verifier.runner import verify, visible_check_source


@dataclass
class Feedback:
    passed: bool
    kind: str          # "ok" | "extraction" | "visible" | "selftest"
    text: str
    error_type: str = ""


@dataclass
class Outcome:
    result: str        # "pass" | "fail" | "error"
    error_type: str
    code: str
    rounds: int
    stop_reason: str
    tokens: int


def _digest(code: str) -> str:
    return hashlib.sha256(code.strip().encode()).hexdigest()


class Solver:
    def __init__(self, spec: ConfigSpec, client, model: str, backend: str, logger: JsonlLogger,
                 run_id: str, timeout_s: float = 10.0, mem_mb: int = 1024,
                 chat_fn: Optional[Callable] = None):
        self.spec, self.client, self.model, self.backend = spec, client, model, backend
        self.logger, self.run_id = logger, run_id
        self.timeout_s, self.mem_mb = timeout_s, mem_mb
        self.chat = chat_fn or chat

    # -- logging helpers -------------------------------------------------------------------
    def _base(self, task: Task, seed: int) -> dict:
        return dict(run_id=self.run_id, config=self.spec.name, seed=seed, task_id=task.task_id,
                    source=task.source, backend=self.backend, model_requested=self.model)

    def _call(self, task, seed, state, role, system, user, rnd):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        r = self.chat(self.client, self.model, messages, temperature=self.spec.temperature,
                      max_tokens=self.spec.max_tokens, seed=seed)
        state["tokens"] += r.prompt_tokens + r.completion_tokens
        self.logger.log(LogRecord(
            kind=KIND_LLM_CALL, role=role, round=rnd, model_served=r.served_model,
            prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens,
            latency_s=r.latency_s, **self._base(task, seed)))
        return r.text

    # -- checks ----------------------------------------------------------------------------
    def _check(self, task: Task, code: str, self_tests: str) -> Feedback:
        if not code:
            return Feedback(False, "extraction", "No Python code block was found in your reply.",
                            "extraction")
        v = verify(task, code, self.timeout_s, self.mem_mb, tests=visible_check_source(task))
        if not v.passed:
            return Feedback(False, "visible",
                            feedback_text(v.error_type, v.stderr_tail, self.timeout_s), v.error_type)
        if self_tests:
            s = verify(task, code, self.timeout_s, self.mem_mb, tests=self_tests)
            if not s.passed:
                return Feedback(False, "selftest",
                                feedback_text(s.error_type, s.stderr_tail, self.timeout_s),
                                s.error_type)
        return Feedback(True, "ok", "")

    # -- main entry ------------------------------------------------------------------------
    def solve(self, task: Task, seed: int) -> Outcome:
        state = {"tokens": 0}
        try:
            return self._solve(task, seed, state)
        except LLMError as e:
            self.logger.log(LogRecord(kind=KIND_TASK_RESULT, result="error", error_type="llm_error",
                                      detail=str(e)[:500], **self._base(task, seed)))
            return Outcome("error", "llm_error", "", 0, "llm_error", state["tokens"])

    def _solve(self, task: Task, seed: int, state: dict) -> Outcome:
        spec = self.spec
        plan = ""
        if spec.plan:
            plan = self._call(task, seed, state, "planner", PLANNER_SYSTEM, planner_user(task), 0)
        self_tests = ""
        if spec.test_writer:
            reply = self._call(task, seed, state, "testwriter", TESTWRITER_SYSTEM,
                               testwriter_user(task), 0)
            self_tests = sanitize_asserts(reply)

        reply = self._call(task, seed, state, "coder", BASELINE_SYSTEM, coder_user(task, plan), 0)
        ex = extract_code(reply, task.entry_point)
        code = ex.code if ex.ok else ""

        stop, rnd, seen = "max_rounds", 0, {_digest(code)}
        if spec.max_rounds == 0:
            stop = "single_call"
        else:
            while True:
                fb = self._check(task, code, self_tests)
                self.logger.log(LogRecord(
                    kind=KIND_ROUND, role="verifier", round=rnd, result="pass" if fb.passed else "fail",
                    error_type=fb.error_type or fb.kind, detail=fb.text[:800], code=code[:4000],
                    **self._base(task, seed)))
                if fb.passed:
                    stop = "checks_passed"
                    break
                if rnd >= spec.max_rounds:
                    stop = "max_rounds"
                    break
                if state["tokens"] >= spec.token_budget:
                    stop = "budget"
                    break
                rnd += 1
                diagnosis = ""
                if spec.reviewer:
                    diagnosis = self._call(task, seed, state, "reviewer", REVIEWER_SYSTEM,
                                           reviewer_user(task, code, fb.kind, fb.text), rnd)
                    if fb.kind == "selftest" and diagnosis.strip().upper().startswith("TESTS_WRONG"):
                        stop = "tests_wrong"
                        break
                    if state["tokens"] >= spec.token_budget:
                        stop = "budget"
                        break
                reply = self._call(task, seed, state, "coder", BASELINE_SYSTEM,
                                   repair_user(task, code, fb.text, diagnosis, plan), rnd)
                ex = extract_code(reply, task.entry_point)
                new_code = ex.code if ex.ok else code
                if _digest(new_code) in seen:      # same code again: more rounds will not help
                    stop = "stalled"
                    break
                seen.add(_digest(new_code))
                code = new_code

        return self._score(task, seed, state, code, rnd, stop)

    def _score(self, task, seed, state, code, rounds, stop) -> Outcome:
        """Hidden tests, run once on the final code. Never used for feedback."""
        if not code:
            result, err, detail = "fail", "extraction", "no code found"
        else:
            vr = verify(task, code, self.timeout_s, self.mem_mb)
            result, err, detail = ("pass" if vr.passed else "fail"), vr.error_type, vr.stderr_tail[-1500:]
        self.logger.log(LogRecord(
            kind=KIND_TASK_RESULT, result=result, error_type=err, round=rounds, stop_reason=stop,
            code=code[:4000], detail=detail, role="final", **self._base(task, seed)))
        return Outcome(result, err, code, rounds, stop, state["tokens"])
