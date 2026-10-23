"""The agent loop, driven by a scripted fake model. No network, no real LLM."""
import sys

import pytest

from benchmarks.loaders import mbpp_row_to_task
from common.logging_schema import JsonlLogger, read_records
from llm.client import LLMError, LLMResponse
from orchestrator.loop import Solver
from orchestrator.spec import ConfigSpec, load_spec

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs Linux/macOS")

ROW = {"task_id": 9, "text": "Double a number.", "code": "", "test_setup_code": "",
       "test_list": ["assert dbl(2) == 4", "assert dbl(3) == 6", "assert dbl(10) == 20"]}
TASK = mbpp_row_to_task(ROW)

GOOD = "```python\ndef dbl(x):\n    return 2 * x\n```"
BAD = "```python\ndef dbl(x):\n    return x\n```"
CHEAT = "```python\ndef dbl(x):\n    return 4 if x == 2 else 0\n```"   # passes the visible check only


class Script:
    """Replies in order; records every message list the model was shown."""
    def __init__(self, replies, tokens=(100, 50)):
        self.replies, self.seen, self.tokens = list(replies), [], tokens

    def __call__(self, client, model, messages, **kw):
        self.seen.append(messages)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return LLMResponse(text=r, prompt_tokens=self.tokens[0], completion_tokens=self.tokens[1],
                           total_tokens=sum(self.tokens), served_model="fake", latency_s=0.0)


def run(tmp_path, spec, replies, **kw):
    script = Script(replies, **kw)
    path = tmp_path / "log.jsonl"
    solver = Solver(spec, None, "fake", "local", JsonlLogger(path), "r1", timeout_s=5, chat_fn=script)
    return solver.solve(TASK, 0), script, list(read_records(path))


def results(records, kind="task_result"):
    return [r for r in records if r["kind"] == kind]


def test_single_call_config_makes_one_call_and_scores_hidden(tmp_path):
    o, script, recs = run(tmp_path, ConfigSpec("c1", max_rounds=0), [GOOD])
    assert o.result == "pass" and o.stop_reason == "single_call" and len(script.seen) == 1


def test_retry_repairs_a_failure_using_visible_feedback(tmp_path):
    o, script, recs = run(tmp_path, ConfigSpec("c2", max_rounds=3), [BAD, GOOD])
    assert o.result == "pass" and o.rounds == 1 and o.stop_reason == "checks_passed"
    repair_prompt = script.seen[1][1]["content"]
    assert "returned 2 but expected 4" in repair_prompt and "return x" in repair_prompt
    assert [r["result"] for r in results(recs, "round")] == ["fail", "pass"]


def test_hidden_tests_never_reach_any_prompt(tmp_path):
    """Leak check: the 2nd and 3rd MBPP tests exist only in the hidden tests."""
    spec = ConfigSpec("c5", plan=True, reviewer=True, test_writer=True, max_rounds=3)
    plan, tests = "restate. edge cases. steps.", "```python\nassert dbl(1) == 2\n```"
    o, script, _ = run(tmp_path, spec, [plan, tests, BAD, "diagnosis: returns x", GOOD])
    blob = " ".join(m["content"] for msgs in script.seen for m in msgs)
    assert "dbl(3)" not in blob and "dbl(10)" not in blob
    assert o.result == "pass"


def test_visible_check_can_pass_while_hidden_tests_fail(tmp_path):
    o, script, _ = run(tmp_path, ConfigSpec("c2", max_rounds=3), [CHEAT])
    assert o.stop_reason == "checks_passed" and o.result == "fail" and len(script.seen) == 1


def test_max_rounds_stops_and_scores_last_attempt(tmp_path):
    other_bad = "```python\ndef dbl(x):\n    return x + 0.5\n```"
    o, script, _ = run(tmp_path, ConfigSpec("c2", max_rounds=1), [BAD, other_bad])
    assert o.stop_reason == "max_rounds" and o.rounds == 1 and o.result == "fail"
    assert len(script.seen) == 2


def test_stalled_when_model_repeats_itself(tmp_path):
    o, script, _ = run(tmp_path, ConfigSpec("c2", max_rounds=5), [BAD, BAD])
    assert o.stop_reason == "stalled" and len(script.seen) == 2


def test_token_budget_stops_the_loop(tmp_path):
    other_bad = "```python\ndef dbl(x):\n    return x - 1\n```"
    spec = ConfigSpec("c2", max_rounds=5, token_budget=200)
    o, script, _ = run(tmp_path, spec, [BAD, other_bad, "```python\ndef dbl(x):\n    return 0\n```"],
                       tokens=(100, 50))
    assert o.stop_reason == "budget" and len(script.seen) == 2


def test_reviewer_runs_each_failed_round_and_its_diagnosis_reaches_the_coder(tmp_path):
    o, script, recs = run(tmp_path, ConfigSpec("c4", reviewer=True, max_rounds=3),
                          [BAD, "You return x; you must double it.", GOOD])
    assert o.result == "pass"
    assert [r["role"] for r in results(recs, "llm_call")] == ["coder", "reviewer", "coder"]
    assert "you must double it" in script.seen[2][1]["content"]


def test_wrong_ai_test_is_overruled_by_the_reviewer(tmp_path):
    spec = ConfigSpec("c5", test_writer=True, reviewer=True, max_rounds=3)
    wrong_test = "```python\nassert dbl(2) == 5\n```"          # the test itself is wrong
    o, script, _ = run(tmp_path, spec, [wrong_test, GOOD, "TESTS_WRONG: doubling 2 gives 4."])
    assert o.stop_reason == "tests_wrong" and o.result == "pass" and len(script.seen) == 3


def test_ai_test_failure_drives_a_repair_when_it_is_right(tmp_path):
    spec = ConfigSpec("c5", test_writer=True, max_rounds=3)
    # visible example (dbl(2)==4) passes with CHEAT, the AI-written test dbl(10)==20 catches it
    o, script, _ = run(tmp_path, spec, ["```python\nassert dbl(10) == 20\n```", CHEAT, GOOD])
    assert o.result == "pass" and o.rounds == 1


def test_missing_code_in_first_reply_gets_a_repair_round(tmp_path):
    o, script, _ = run(tmp_path, ConfigSpec("c2", max_rounds=2), ["Sorry, I cannot.", GOOD])
    assert o.result == "pass" and "No Python code block" in script.seen[1][1]["content"]


def test_llm_outage_is_logged_as_error(tmp_path):
    o, _, recs = run(tmp_path, ConfigSpec("c2", max_rounds=2), [LLMError("down")])
    assert o.result == "error" and results(recs)[0]["error_type"] == "llm_error"


def test_every_call_and_the_final_result_are_logged(tmp_path):
    o, _, recs = run(tmp_path, ConfigSpec("c2", max_rounds=3), [BAD, GOOD])
    assert len(results(recs, "llm_call")) == 2 and len(results(recs)) == 1
    final = results(recs)[0]
    assert final["stop_reason"] == "checks_passed" and final["round"] == 1 and "return 2 * x" in final["code"]


@pytest.mark.parametrize("name", ["c1_single", "c2_retry", "c3_plan_retry", "c4_review", "c5_full"])
def test_shipped_configs_load(name):
    s = load_spec(name)
    assert s.name == name
    assert (s.max_rounds == 0) == (name == "c1_single")


def test_unknown_config_key_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("name: x\nplaner: true\n")
    with pytest.raises(ValueError):
        load_spec(p)
