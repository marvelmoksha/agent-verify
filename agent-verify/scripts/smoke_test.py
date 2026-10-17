"""Phase 0 finish line: load one task, call the model, print the reply, write one log line."""
import argparse
import sys

from common.logging_schema import KIND_LLM_CALL, JsonlLogger, LogRecord, log_dir
from llm.client import BACKENDS, LLMError, chat, get_client

SYSTEM = "You are an expert Python programmer. Reply with one Python code block and nothing else."
FALLBACK_PROMPT = (
    "def add(a: int, b: int) -> int:\n"
    '    """Return the sum of a and b."""\n'
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=BACKENDS)
    ap.add_argument("--model", required=True, help="pinned model id (not 'auto')")
    ap.add_argument("--task-index", type=int, default=0)
    ap.add_argument("--no-dataset", action="store_true", help="skip HF download; use a tiny prompt")
    args = ap.parse_args()

    if args.no_dataset:
        task_id, source, prompt = "smoke/0", "smoke", FALLBACK_PROMPT
    else:
        from benchmarks.loaders import load_humaneval

        task = load_humaneval()[args.task_index]
        task_id, source, prompt = task.task_id, task.source, task.prompt
        print(f"Loaded task {task_id} ({len(prompt)} chars of prompt)")

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Complete the following function.\n\n{prompt}"},
    ]
    try:
        resp = chat(get_client(args.backend), args.model, messages, max_tokens=512)
    except (LLMError, RuntimeError, ValueError) as e:
        print(f"FAILED: {e}")
        return 1

    print("\n--- model reply ---")
    print(resp.text)
    print("--- end ---")
    print(f"requested model: {args.model}")
    print(f"served model:    {resp.served_model}")
    print(f"tokens: prompt={resp.prompt_tokens} completion={resp.completion_tokens}")
    print(f"latency: {resp.latency_s:.1f}s")

    path = log_dir() / "smoke.jsonl"
    JsonlLogger(path).log(
        LogRecord(
            kind=KIND_LLM_CALL, run_id="smoke", config="smoke", seed=0, task_id=task_id,
            source=source, role="baseline", backend=args.backend, model_requested=args.model,
            model_served=resp.served_model, prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens, latency_s=resp.latency_s,
        )
    )
    print(f"Logged to {path}\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
