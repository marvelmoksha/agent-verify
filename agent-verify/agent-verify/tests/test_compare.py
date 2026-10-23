from common.compare import paired_bootstrap, per_task_pass, tokens_per_task


def tr(task, result, seed=0):
    return {"kind": "task_result", "task_id": task, "seed": seed, "result": result}


def test_per_task_pass_averages_seeds_and_skips_errors():
    recs = [tr("a", "pass", 0), tr("a", "fail", 1), tr("b", "error", 0), tr("c", "fail", 0)]
    assert per_task_pass(recs) == {"a": 0.5, "c": 0.0}
    assert per_task_pass(recs, seeds={0}) == {"a": 1.0, "c": 0.0}


def test_identical_configs_have_zero_difference():
    a = {f"t{i}": float(i % 2) for i in range(100)}
    r = paired_bootstrap(a, dict(a), n_boot=500)
    assert r["diff"] == 0 and r["ci_low"] <= 0 <= r["ci_high"] and r["ties"] == 100


def test_clear_improvement_has_ci_above_zero():
    a = {f"t{i}": 0.0 for i in range(100)}
    b = {f"t{i}": 1.0 if i < 30 else 0.0 for i in range(100)}
    r = paired_bootstrap(a, b, n_boot=500)
    assert abs(r["diff"] - 0.30) < 1e-9 and r["ci_low"] > 0
    assert r["b_wins"] == 30 and r["a_wins"] == 0


def test_only_common_tasks_are_compared():
    r = paired_bootstrap({"a": 1.0, "b": 0.0, "x": 1.0}, {"a": 1.0, "b": 1.0, "y": 0.0}, n_boot=100)
    assert r["n"] == 2 and r["b_wins"] == 1


def test_tokens_per_task_sums_calls():
    recs = [{"kind": "llm_call", "task_id": "a", "seed": 0, "prompt_tokens": 10, "completion_tokens": 5},
            {"kind": "llm_call", "task_id": "a", "seed": 0, "prompt_tokens": 20, "completion_tokens": 5}]
    assert tokens_per_task(recs) == {"a": 40.0}
