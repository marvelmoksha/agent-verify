"""Print pass rates from one or more JSONL logs:  python scripts/summarize.py logs/*.jsonl"""
import sys

from common.logging_schema import read_records
from common.summary import summarize


def main(paths: list[str]) -> int:
    if not paths:
        print("usage: python scripts/summarize.py <log.jsonl> [more.jsonl ...]")
        return 1
    records = [r for p in paths for r in read_records(p)]
    s = summarize(records)
    print(f"{'config':<14}{'seed':>5}{'n':>6}{'pass':>6}{'rate':>8}{'tok/task':>10}{'errors':>8}  failures")
    for (config, seed), v in s["per_seed"].items():
        print(f"{config:<14}{seed:>5}{v['n']:>6}{v['passed']:>6}{v['pass_rate']:>8.1%}"
              f"{v['mean_tokens_per_task']:>10.0f}{v['errors']:>8}  {v['failures']}")
    print()
    for config, v in s["per_config"].items():
        flag = "" if v["valid"] else "   *** INVALID: more than one served model ***"
        print(f"{config}: {v['mean_pass_rate']:.1%} +/- {v['std_pass_rate']:.1%} "
              f"over {v['seeds']} seed(s); models: {v['models_served']}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
