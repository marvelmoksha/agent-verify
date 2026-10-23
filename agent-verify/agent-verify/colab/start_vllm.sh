#!/usr/bin/env bash
# Start a local OpenAI-compatible model server on a Colab GPU (T4 works).
# Usage:  bash colab/start_vllm.sh [model-id]
# Flags and model names change between vLLM versions; if it fails, read vllm.log.
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-Coder-7B-Instruct-AWQ}"
PORT="${PORT:-8000}"

if ! nvidia-smi > /dev/null 2>&1; then
  echo "No GPU visible. Switch Colab to a T4 GPU runtime (or use colab/start_ollama.sh)."
  exit 1
fi

pip install -q vllm

nohup vllm serve "$MODEL" \
  --dtype half \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --port "$PORT" > vllm.log 2>&1 &
PID=$!

echo "Starting $MODEL on port $PORT (first start downloads weights; can take several minutes)..."
for i in $(seq 1 120); do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "vLLM exited early. Last log lines:"; tail -n 40 vllm.log; exit 1
  fi
  if curl -s "http://localhost:${PORT}/v1/models" > /dev/null; then
    echo "Server ready."
    exit 0
  fi
  sleep 5
done
echo "Server did not start in time. Last log lines:"
tail -n 40 vllm.log
exit 1
