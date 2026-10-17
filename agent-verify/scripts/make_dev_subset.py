"""Create benchmarks/dev_ids.json (run once; needs internet to download datasets)."""
import argparse

from benchmarks.dev_subset import DEFAULT_PATH, save_dev_ids, select_dev_ids
from benchmarks.loaders import load_humaneval, load_mbpp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-each", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="overwrite existing file (avoid!)")
    args = ap.parse_args()

    he = load_humaneval()
    mb = load_mbpp()
    print(f"Loaded {len(he)} HumanEval and {len(mb)} MBPP tasks.")
    ids = select_dev_ids([t.task_id for t in he], [t.task_id for t in mb], args.n_each, args.seed)
    path = save_dev_ids(ids, args.seed, args.n_each, force=args.force)
    print(f"Saved {len(ids)} ids to {path}")
    print("Commit this file to git now and never change it.")


if __name__ == "__main__":
    main()
