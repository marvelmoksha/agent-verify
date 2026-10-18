"""Config C1: single LLM call per task, verified in the sandbox. Resumable.

Examples
  python scripts/run_baseline.py --backend openrouter --model "<id>" --subset dev --limit 5
  python scripts/run_baseline.py --backend local --model qwen2.5-coder:7b --subset dev --seeds 0 1 2
  python scripts/run_baseline.py --backend local --model qwen2.5-coder:7b --subset full --seeds 0 1 2

Stop it any time (Ctrl-C or a Colab disconnect) and run the same command again:
finished (config, seed, task) triples are skipped.
"""
import argparse
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from agents.prompts import BASELINE_SYSTEM, baseline_user
from benchmarks.dev_subset import load_dev_ids
from benchmarks.loaders import load_all_tasks
from common.logging_schema import (
    KIND_LLM_CALL, KIND_TASK_RESULT, JsonlLogger, LogRecord,
    completed_task_keys, log_dir, models_served, read_records,
)
from common.summary import summarize
from llm.client import BACKENDS, LLMError, chat, get_client
from verifier.extract import extract_code
from verifier.runner import verify

CONFIG = "c1_single"


def run_one(task, seed, args, client, logger, run_id):
    base = dict(run_id=run_id, config=CONFIG, seed=seed, task_id=task.task_id,
                source=task.source, backend=args.backend, model_requested=args.model)
    messages = [
        {"role": "system", "content": BASELINE_SYSTEM},
        {"role": "user", "content": baseline_user(task)},
    ]
    try:
        resp = chat(client, args.model, messages, temperature=args.temperature,
                    max_tokens=args.max_tokens, seed=seed)
    except LLMError as e:
        logger.log(LogRecord(kind=KIND_TASK_RESULT, result="error", error_type="llm_error",
                             detail=str(e)[:500], role="baseline", **base))
        return "error"

    logger.log(LogRecord(kind=KIND_LLM_CALL, role="baseline", model_served=resp.served_model,
                         prompt_tokens=resp.prompt_tokens, completion_tokens=resp.completion_tokens,
                         latency_s=resp.latency_s, **base))

    ex = extract_code(resp.text, task.entry_point)
    if not ex.ok:
        logger.log(LogRecord(kind=KIND_TASK_RESULT, result="fail", error_type="extraction",
                             code=resp.text[:4000], detail="no code found", role="baseline",
                             model_served=resp.served_model, **base))
        return "fail"

    vr = verify(task, ex.code, timeout_s=args.timeout, mem_mb=args.mem_mb)
    result = "pass" if vr.passed else "fail"
    logger.log(LogRecord(kind=KIND_TASK_RESULT, result=result, error_type=vr.error_type,
                         code=ex.code[:4000], detail=f"[{ex.method}] {vr.stderr_tail[-1500:]}",
                         role="baseline", model_served=resp.served_model, **base))
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=BACKENDS)
    ap.add_argument("--model", required=True, help="pinned model id (not 'auto')")
    ap.add_argument("--subset", choices=["dev", "full"], default="dev")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--limit", type=int, default=None, help="only the first N tasks (for tests)")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--timeout", type=float, default=10.0, help="seconds per sandbox run")
    ap.add_argument("--mem-mb", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--sleep", type=float, default=0.0, help="pause after each task (rate limits)")
    ap.add_argument("--out", default=None, help="log file (default: logs dir, named by model)")
    args = ap.parse_args()

    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model)
    out = args.out or str(log_dir() / f"{CONFIG}__{slug}__{args.subset}.jsonl")

    all_tasks = load_all_tasks()
    if args.subset == "dev":
        tasks = [all_tasks[i] for i in load_dev_ids()]
    else:
        tasks = list(all_tasks.values())
    if args.limit:
        tasks = tasks[: args.limit]

    done = completed_task_keys(out)
    todo = [(t, s) for s in args.seeds for t in tasks if (CONFIG, s, t.task_id) not in done]
    print(f"{len(tasks)} tasks x {len(args.seeds)} seed(s); {len(done)} already done; "
          f"{len(todo)} to run -> {out}")

    client, logger, run_id = get_client(args.backend), JsonlLogger(out), uuid.uuid4().hex[:8]
    finished = 0

    def work(item):
        task, seed = item
        res = run_one(task, seed, args, client, logger, run_id)
        if args.sleep:
            time.sleep(args.sleep)
        return task.task_id, seed, res

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(work, it) for it in todo]):
            tid, seed, res = fut.result()
            finished += 1
            print(f"[{finished}/{len(todo)}] seed={seed} {tid}: {res}", flush=True)

    s = summarize(read_records(out))
    for (config, seed), v in s["per_seed"].items():
        extra = f"  ({v['errors']} API errors: rerun the same command to retry them)" if v["errors"] else ""
        print(f"seed {seed}: {v['passed']}/{v['n']} = {v['pass_rate']:.1%}  failures={v['failures']}{extra}")
    served = models_served(out)
    if len(served) > 1:
        print(f"WARNING: more than one served model in this log: {served}. Run is INVALID.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
