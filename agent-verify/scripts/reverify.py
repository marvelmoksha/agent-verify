"""Re-score logged results whose failure was caused by a sandbox bug, WITHOUT calling the model.

Example (the sympy false alarm in the C1 baseline):
  python scripts/reverify.py --log <C1 full log> --error-types forbidden

For every task_result with one of the given error types, the logged code is run again through the
(fixed) verifier. A corrected record is APPENDED (the original line stays in the file); summaries
and comparisons use the last record for a task, so the corrected result wins.
"""
import argparse
import time

from benchmarks.loaders import load_all_tasks
from common.logging_schema import KIND_TASK_RESULT, JsonlLogger, LogRecord, read_records
from verifier.runner import verify


def reverify_records(records, tasks, error_types, timeout_s=10.0, mem_mb=1024):
    """Yield corrected LogRecords for matching task_result rows."""
    for r in records:
        if r.get("kind") != KIND_TASK_RESULT or r.get("error_type") not in error_types:
            continue
        task, code = tasks.get(r["task_id"]), r.get("code", "")
        if task is None or not code:
            continue
        vr = verify(task, code, timeout_s, mem_mb)
        yield LogRecord(
            kind=KIND_TASK_RESULT, run_id=r["run_id"], config=r["config"], seed=int(r["seed"]),
            task_id=r["task_id"], source=r.get("source", ""), round=int(r.get("round", 0)),
            role=r.get("role", ""), backend=r.get("backend", ""),
            model_requested=r.get("model_requested", ""), model_served=r.get("model_served", ""),
            result="pass" if vr.passed else "fail", error_type=vr.error_type, code=code,
            detail=f"[reverified after sandbox fix; was {r['error_type']}] {vr.stderr_tail[-800:]}",
            stop_reason=r.get("stop_reason", ""), timestamp=time.time(),
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--error-types", nargs="+", default=["forbidden"])
    args = ap.parse_args()

    records = list(read_records(args.log))
    logger, n = JsonlLogger(args.log), 0
    for rec in reverify_records(records, load_all_tasks(), set(args.error_types)):
        logger.log(rec)
        n += 1
        print(f"{rec.task_id} seed={rec.seed}: now {rec.result} ({rec.error_type or 'ok'})")
    print(f"{n} record(s) re-scored and appended to {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
