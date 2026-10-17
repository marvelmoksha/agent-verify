#!/usr/bin/env bash
# Start a local OpenAI-compatible model server on a Colab GPU (T4 works).
# Usage:  bash colab/start_vllm.sh [model-id]
# Flags and model names change between vLLM versions; if it fails, read vllm.log.
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-Coder-7B-Instruct-AWQ}"
PORT="${PORT:-8000}"

pip install -q vllm

nohup vllm serve "$MODEL" \
  --dtype half \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --port "$PORT" > vllm.log 2>&1 &

echo "Starting $MODEL on port $PORT (first start downloads weights; can take several minutes)..."
for i in $(seq 1 120); do
  if curl -s "http://localhost:${PORT}/v1/models" > /dev/null; then
    echo "Server ready."
    exit 0
  fi
  sleep 5
done
echo "Server did not start in time. Last log lines:"
tail -n 40 vllm.log
exit 1
