"""Check that your machine is ready. Never prints key values."""
import argparse
import importlib
import os
import sys

from llm.client import BACKENDS, _KEY_ENV, get_client


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="*", default=list(BACKENDS), choices=BACKENDS)
    args = ap.parse_args()

    ok = True
    print(f"Python {sys.version.split()[0]}", "OK" if sys.version_info >= (3, 10) else "NEED >= 3.10")
    for pkg in ("openai", "datasets", "pandas", "yaml", "dotenv"):
        try:
            importlib.import_module(pkg)
            print(f"  [ok]   package {pkg}")
        except ImportError:
            print(f"  [FAIL] package {pkg} missing -> pip install -e .")
            ok = False

    for b in args.backends:
        print(f"\nBackend: {b}")
        if b != "local":
            has_key = bool(os.environ.get(_KEY_ENV[b]))
            print(f"  key {_KEY_ENV[b]}: {'set' if has_key else 'NOT SET'}")
            if not has_key:
                ok = False
                continue
        try:
            models = get_client(b).with_options(timeout=10).models.list()
            names = [m.id for m in models.data][:5]
            print(f"  [ok]   reachable; first models: {names}")
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] not reachable: {type(e).__name__}: {str(e)[:120]}")
            print("         (fine if you are not using this backend right now)")
    print("\nDone.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
