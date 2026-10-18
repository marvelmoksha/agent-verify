# agent-verify

Multi-agent code solver with verifiable guarantees. A planner, coder, test writer and
reviewer solve programming tasks; a deterministic, non-LLM verifier decides pass or fail.

**Status: Phase 0 (skeleton) and Phase 1 (verifier + single-call baseline) are complete.**

## Layout

```
llm/client.py            one chat() for local (Ollama/vLLM), OpenRouter, FreeLLMAPI
benchmarks/loaders.py    HumanEval + MBPP -> common Task shape (hidden tests kept separate)
benchmarks/dev_subset.py fixed 50-task dev subset with checksum
verifier/sandbox.py      subprocess sandbox: timeout, memory/CPU/output limits, guards, sentinel
verifier/extract.py      pulls code out of model replies (fences, truncated, raw, heuristic)
verifier/runner.py       task + code -> sandboxed run against the HIDDEN tests
agents/prompts.py        prompt templates (baseline for now)
common/logging_schema.py JSONL logger, resume keys, silent-model-swap detector
common/summary.py        pass rates per seed/config, failure breakdown, token cost
scripts/                 check_env, make_dev_subset, smoke_test, run_baseline, summarize
colab/start_ollama.sh    one command: GPU check, Ollama, pull the pinned model
colab/start_vllm.sh      optional vLLM server (fails fast now); Ollama is the default
tests/                   69 offline unit tests (sandbox, extractor, runner, wiring, logging)
```

## Phase 0 steps

### Step 1. Install tools (laptop)
Python 3.10+, git, and a GitHub account.

### Step 2. Create the repo and environment
```bash
cd agent-verify
git init
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### Step 3. Add your keys
```bash
cp .env.example .env               # Windows: copy .env.example .env
```
Edit `.env` and fill in `OPENROUTER_API_KEY` and `FREELLMAPI_KEY`. Never commit `.env`
(it is already in `.gitignore`). Start your FreeLLMAPI proxy so it listens on port 3001.

### Step 4. Check the environment
```bash
python scripts/check_env.py --backends openrouter freellmapi
```
Every backend you plan to use should say `[ok] reachable`.

### Step 5. Run the unit tests
```bash
pytest -q
```
Expect all tests to pass (69 at Phase 1). The sandbox tests need Linux/macOS (use Colab or WSL on Windows).

### Step 6. Create the dev subset (once)
```bash
python scripts/make_dev_subset.py
git add benchmarks/dev_ids.json
```
This downloads HumanEval (164) and MBPP (500), picks 25 of each with seed 42, and saves
`benchmarks/dev_ids.json`. It refuses to overwrite. Never edit or regenerate it.

### Step 7. Smoke test on your laptop (API backends)
Use a **specific** model id, never `auto`. Look the id up in the OpenRouter model list or
the FreeLLMAPI dashboard.
```bash
python scripts/smoke_test.py --backend openrouter --model "<pinned-model-id>"
python scripts/smoke_test.py --backend freellmapi --model "<pinned-model-id>"
```
If the Hugging Face download is blocked, add `--no-dataset`.

### Step 8. Smoke test on Colab (local model, used for all final runs)
(Phase 0 originally used vLLM here. Ollama is now the default because it is more reliable on Colab.)
1. Push the repo to GitHub (private is fine).
2. In Colab: Runtime > Change runtime type > T4 GPU.
3. Run these cells:
```bash
!git clone https://github.com/<you>/agent-verify.git
%cd agent-verify
!pip install -q -e .
!bash colab/start_ollama.sh                    # GPU check, install, pull qwen2.5-coder:7b
```
```python
from google.colab import drive
drive.mount('/content/drive')
import os
os.environ["AGENT_VERIFY_LOG_DIR"] = "/content/drive/MyDrive/agent-verify-logs"
```
```bash
!python scripts/smoke_test.py --backend local --model qwen2.5-coder:7b
!ollama ps          # must say 100% GPU; if it says CPU, something else is holding GPU memory
```
If the model runs out of memory, use a smaller one (for example `qwen2.5-coder:3b`).

Note: the FreeLLMAPI proxy runs on your laptop, so Colab cannot reach `localhost:3001`.
Use FreeLLMAPI and OpenRouter from the laptop, and the local model on Colab.

### Step 9. Commit
```bash
git add -A && git commit -m "Phase 0: skeleton, client, loaders, dev subset, logging"
```

## Phase 0 is done when
- [ ] `pytest -q` passes
- [ ] `benchmarks/dev_ids.json` exists and is committed
- [ ] smoke test passes on at least one API backend (laptop)
- [ ] smoke test passes on the local model (Colab)
- [ ] `logs/smoke.jsonl` has a line showing `model_served`

## Rules to keep for the whole project
1. **Pinned models only.** The client rejects router aliases like `auto` on purpose.
2. **Check `model_served`.** If one run's log shows more than one model, the run is invalid
   (`common.logging_schema.models_served`).
3. **Hidden tests are for the verifier only.** Agent prompts use `task.prompt` and nothing else.
4. **Never edit `dev_ids.json`.** Its checksum is verified on load.
5. **All runs must be resumable.** Use `completed_task_keys()` to skip finished work.
6. **Final runs use the local Colab model.** API tiers are for development and the
   stronger-reviewer / second-model replication experiments.

## Phase 1: verifier and single-call baseline

### What was built
- **Sandbox** (`verifier/sandbox.py`): each candidate runs in its own process with a wall-clock
  timeout, CPU/memory/output-size limits, a throw-away directory, closed stdin, and guards
  against network, process spawning and file deletion. A random sentinel printed *after* the
  tests means `sys.exit(0)` tricks cannot fake a pass.
  *These guards stop accidental damage from model-written code; they are not a security
  boundary against deliberately hostile code. For stronger isolation, run in Docker or a
  disposable Colab VM.*
- **Extractor** (`verifier/extract.py`): fenced blocks (prefers the last block that defines the
  entry function), truncated replies, raw code, and a heuristic fallback. Every result records how
  the code was found.
- **Baseline C1** (`scripts/run_baseline.py`): one model call per task, then verification.
  Resumable, multi-seed, logs the verified code and stderr for later failure analysis.
- **Summary** (`scripts/summarize.py`): pass rate per seed, mean and std per config, failure types,
  tokens per task and per solved task, and an INVALID flag if more than one model served a run.

### Run it (Colab, local model)
```bash
# 1. five tasks, to check the whole pipeline
!python scripts/run_baseline.py --backend local --model qwen2.5-coder:7b --subset dev --limit 5

# 2. the 50-task dev subset, 3 seeds
!python scripts/run_baseline.py --backend local --model qwen2.5-coder:7b --subset dev --seeds 0 1 2

# 3. read the results
!python scripts/summarize.py $AGENT_VERIFY_LOG_DIR/*.jsonl      # or logs/*.jsonl

# 4. later: the full benchmarks (164 + 500 tasks)
!python scripts/run_baseline.py --backend local --model qwen2.5-coder:7b --subset full --seeds 0 1 2
```
If Colab disconnects, run the same command again; finished tasks are skipped, and tasks that
failed only because the API or server was down are retried.

For OpenRouter or FreeLLMAPI from your laptop, add `--sleep 3` to stay under free-tier rate limits.

### Phase 1 is done when
- [ ] `pytest -q` passes (69 tests)
- [ ] the 5-task run completes and `summarize.py` prints a table
- [ ] the 50-task dev run with 3 seeds completes; pass rate is in a believable range for the model
- [ ] you have opened 20 failures in the log (`code` and `detail` fields) and confirmed they are
      real model errors, not extraction or sandbox bugs
- [ ] `models_served` shows exactly one model for the run

### Known decisions to mention in the write-up
- MBPP prompts show the first test case (standard convention); all tests are run.
- Default temperature 0.2 with seeds 0, 1, 2. The seed is sent to the server; some servers
  ignore it, so report the std across seeds rather than assuming exact reproducibility.

## Next: Phase 2
Planner, coder, test writer and reviewer agents; the orchestrator loop; configs C2 to C5.
