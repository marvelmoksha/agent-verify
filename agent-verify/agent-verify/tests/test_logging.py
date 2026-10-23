from common.logging_schema import (
    KIND_LLM_CALL, KIND_TASK_RESULT, JsonlLogger, LogRecord,
    completed_task_keys, models_served, read_records,
)


def rec(kind, task, config="c1", seed=0, model="m1", result=""):
    return LogRecord(kind=kind, run_id="r", config=config, seed=seed, task_id=task,
                     model_served=model, result=result)


def test_roundtrip_and_resume_keys(tmp_path):
    p = tmp_path / "run.jsonl"
    lg = JsonlLogger(p)
    lg.log(rec(KIND_LLM_CALL, "humaneval/0"))
    lg.log(rec(KIND_TASK_RESULT, "humaneval/0", result="pass"))
    lg.log(rec(KIND_LLM_CALL, "humaneval/1"))  # started but not finished
    assert len(list(read_records(p))) == 3
    assert completed_task_keys(p) == {("c1", 0, "humaneval/0")}


def test_truncated_last_line_is_ignored(tmp_path):
    p = tmp_path / "run.jsonl"
    JsonlLogger(p).log(rec(KIND_TASK_RESULT, "mbpp/11", result="fail"))
    with p.open("a") as f:
        f.write('{"kind": "task_res')  # simulated crash mid-write
    assert completed_task_keys(p) == {("c1", 0, "mbpp/11")}


def test_missing_file_is_empty(tmp_path):
    assert completed_task_keys(tmp_path / "nope.jsonl") == set()


def test_detects_silent_model_swap(tmp_path):
    p = tmp_path / "run.jsonl"
    lg = JsonlLogger(p)
    lg.log(rec(KIND_LLM_CALL, "a", model="model-A"))
    lg.log(rec(KIND_LLM_CALL, "b", model="model-B"))
    assert models_served(p) == {"model-A", "model-B"}  # >1 means invalid run


def test_api_errors_are_retried_on_resume(tmp_path):
    p = tmp_path / "run.jsonl"
    lg = JsonlLogger(p)
    lg.log(rec(KIND_TASK_RESULT, "a", result="error"))
    lg.log(rec(KIND_TASK_RESULT, "b", result="fail"))
    assert completed_task_keys(p) == {("c1", 0, "b")}
