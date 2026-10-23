"""Paired comparison of two logs on the same tasks.

  python scripts/compare.py --a <C1 log> --b <C2 log> --seeds 0 [--names C1 C2]
"""
import argparse

from common.compare import paired_bootstrap, per_task_pass, tokens_per_task
from common.logging_schema import read_records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline log (e.g. C1)")
    ap.add_argument("--b", required=True, help="log to compare against it")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="default: all seeds in each log")
    ap.add_argument("--names", nargs=2, default=["A", "B"])
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()
    seeds = set(args.seeds) if args.seeds else None

    ra, rb = list(read_records(args.a)), list(read_records(args.b))
    pa, pb = per_task_pass(ra, seeds), per_task_pass(rb, seeds)
    res = paired_bootstrap(pa, pb, n_boot=args.boot)
    ta, tb = tokens_per_task(ra, seeds), tokens_per_task(rb, seeds)
    common = sorted(set(pa) & set(pb))
    na, nb = args.names

    def per_solved(pass_map, tok_map):
        solved = sum(pass_map[t] for t in common)
        return sum(tok_map.get(t, 0) for t in common) / solved if solved else float("inf")

    print(f"tasks compared (both finished): {res['n']}")
    print(f"{na}: {res['a_rate']:.1%}   {nb}: {res['b_rate']:.1%}")
    print(f"difference ({nb} - {na}): {res['diff']*100:+.1f} points   "
          f"95% CI [{res['ci_low']*100:+.1f}, {res['ci_high']*100:+.1f}]")
    print(f"tasks where {nb} won: {res['b_wins']}   {na} won: {res['a_wins']}   tied: {res['ties']}")
    print(f"tokens per task: {na} {sum(ta.get(t,0) for t in common)/res['n']:.0f}   "
          f"{nb} {sum(tb.get(t,0) for t in common)/res['n']:.0f}")
    print(f"tokens per solved task: {na} {per_solved(pa, ta):.0f}   {nb} {per_solved(pb, tb):.0f}")
    verdict = ("CI excludes 0: the difference is unlikely to be noise."
               if res["ci_low"] > 0 or res["ci_high"] < 0 else "CI includes 0: not a clear difference.")
    print(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
