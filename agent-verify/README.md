# agent-verify

Multi-agent code solver with verifiable guarantees. A planner, coder, test writer and
reviewer solve programming tasks; a deterministic, non-LLM verifier decides pass or fail.

**This is the Phase 0 skeleton:** project layout, one LLM client for three backends,
benchmark loaders, a fixed dev subset, JSONL logging with resume, and a smoke test.

## Layout

```
llm/client.py            one chat() for local vLLM, OpenRouter, FreeLLMAPI
benchmarks/loaders.py    HumanEval + MBPP -> common Task shape (hidden tests kept separate)
benchmarks/dev_subset.py fixed 50-task dev subset with checksum
common/logging_schema.py JSONL logger, resume keys, silent-model-swap detector
scripts/                 check_env.py, make_dev_subset.py, smoke_test.py
colab/start_vllm.sh      starts a local model server on a Colab GPU
tests/                   25 unit tests (run offline)
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
Expect `25 passed`.

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
1. Push the repo to GitHub (private is fine).
2. In Colab: Runtime > Change runtime type > T4 GPU.
3. Run these cells:
```bash
!git clone https://github.com/<you>/agent-verify.git
%cd agent-verify
!pip install -q -e .
!bash colab/start_vllm.sh                      # first start downloads weights
```
```python
from google.colab import drive
drive.mount('/content/drive')
import os
os.environ["AGENT_VERIFY_LOG_DIR"] = "/content/drive/MyDrive/agent-verify-logs"
```
```bash
!python scripts/smoke_test.py --backend local --model Qwen/Qwen2.5-Coder-7B-Instruct-AWQ
```
If the model runs out of memory, use a smaller one (for example a 3B coder model) and pass
its id to `start_vllm.sh` and `--model`.

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

## Next: Phase 1
Sandboxed verifier (timeout, memory cap, no network), code extraction, and the single-call baseline.
