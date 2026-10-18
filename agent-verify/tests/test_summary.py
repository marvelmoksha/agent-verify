from common.summary import summarize


def tr(task, result, seed=0, config="c1", error=""):
    return {"kind": "task_result", "config": config, "seed": seed, "task_id": task,
            "result": result, "error_type": error}


def call(task, seed=0, config="c1", model="m", p=10, c=5):
    return {"kind": "llm_call", "config": config, "seed": seed, "task_id": task,
            "model_served": model, "prompt_tokens": p, "completion_tokens": c}


def test_pass_rate_and_failure_breakdown():
    recs = [call("a"), tr("a", "pass"), call("b"), tr("b", "fail", error="wrong_answer"),
            call("c"), tr("c", "fail", error="timeout")]
    v = summarize(recs)["per_seed"][("c1", 0)]
    assert v["n"] == 3 and v["passed"] == 1
    assert v["failures"] == {"wrong_answer": 1, "timeout": 1}
    assert v["mean_tokens_per_task"] == 15
    assert v["tokens_per_solved"] == 45


def test_mean_and_std_across_seeds():
    recs = [tr("a", "pass", 0), tr("b", "fail", 0), tr("a", "pass", 1), tr("b", "pass", 1)]
    c = summarize(recs)["per_config"]["c1"]
    assert c["seeds"] == 2 and abs(c["mean_pass_rate"] - 0.75) < 1e-9 and c["std_pass_rate"] > 0


def test_mixed_models_flagged_invalid():
    recs = [call("a", model="A"), tr("a", "pass"), call("b", model="B"), tr("b", "pass")]
    assert summarize(recs)["per_config"]["c1"]["valid"] is False


def test_rerun_keeps_last_record():
    recs = [tr("a", "fail", error="timeout"), tr("a", "pass")]
    assert summarize(recs)["per_seed"][("c1", 0)]["passed"] == 1


def test_api_errors_are_not_model_failures():
    recs = [tr("a", "pass"), tr("b", "error", error="llm_error")]
    v = summarize(recs)["per_seed"][("c1", 0)]
    assert v["n"] == 1 and v["errors"] == 1 and v["pass_rate"] == 1.0
