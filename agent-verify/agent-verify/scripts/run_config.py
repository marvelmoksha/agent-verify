"""Run any configuration (C1..C5) on the dev subset or the full benchmarks. Resumable.

  python scripts/run_config.py --config c2_retry --backend local --model qwen2.5-coder:7b --subset dev
  python scripts/run_config.py --config c5_full  --backend local --model qwen2.5-coder:7b --subset full --seeds 0

Prompts are tuned on --subset dev ONLY. Freeze them (commit) before any --subset full run.
Stop any time and run the same command again; finished tasks are skipped, API errors retried.
"""
import argparse
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from benchmarks.dev_subset import load_dev_ids
from benchmarks.loaders import load_all_tasks
from common import envinfo
from common.logging_schema import JsonlLogger, completed_task_keys, log_dir, models_served, read_records
from common.summary import summarize
from llm.client import BACKENDS, get_client
from orchestrator.loop import Solver
from orchestrator.spec import load_spec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="name in configs/ (e.g. c2_retry) or a yaml path")
    ap.add_argument("--backend", required=True, choices=BACKENDS)
    ap.add_argument("--model", required=True, help="pinned model id (not 'auto')")
    ap.add_argument("--subset", choices=["dev", "full"], default="dev")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--mem-mb", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    spec = load_spec(args.config)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model)
    out = args.out or str(log_dir() / f"{spec.name}__{slug}__{args.subset}.jsonl")

    all_tasks = load_all_tasks()
    tasks = [all_tasks[i] for i in load_dev_ids()] if args.subset == "dev" else list(all_tasks.values())
    if args.limit:
        tasks = tasks[: args.limit]

    done = completed_task_keys(out)
    todo = [(t, s) for s in args.seeds for t in tasks if (spec.name, s, t.task_id) not in done]
    print(f"{spec.name}: {len(tasks)} tasks x {len(args.seeds)} seed(s); {len(done)} done; "
          f"{len(todo)} to run -> {out}\n  settings: {spec}")
    envinfo.write_once(out)

    solver = Solver(spec, get_client(args.backend), args.model, args.backend, JsonlLogger(out),
                    uuid.uuid4().hex[:8], timeout_s=args.timeout, mem_mb=args.mem_mb)

    def work(item):
        task, seed = item
        o = solver.solve(task, seed)
        if args.sleep:
            time.sleep(args.sleep)
        return task.task_id, seed, o

    n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(work, it) for it in todo]):
            tid, seed, o = fut.result()
            n += 1
            print(f"[{n}/{len(todo)}] seed={seed} {tid}: {o.result} (rounds={o.rounds}, "
                  f"stop={o.stop_reason}, tokens={o.tokens})", flush=True)

    s = summarize(read_records(out))
    for (config, seed), v in s["per_seed"].items():
        extra = f"  ({v['errors']} API errors: rerun to retry)" if v["errors"] else ""
        print(f"seed {seed}: {v['passed']}/{v['n']} = {v['pass_rate']:.1%}  "
              f"rounds={v['mean_rounds']:.2f}  stops={v['stop_reasons']}  failures={v['failures']}{extra}")
    served = models_served(out)
    if len(served) > 1:
        print(f"WARNING: more than one served model: {served}. Run is INVALID.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
