#!/usr/bin/env bash
# One command to get a local model on Colab:  bash colab/start_ollama.sh [model]
# Needs a GPU runtime (Runtime > Change runtime type > T4 GPU).
set -euo pipefail
MODEL="${1:-qwen2.5-coder:7b}"

if ! nvidia-smi > /dev/null 2>&1; then
  echo "No GPU visible. Runtime > Change runtime type > T4 GPU, reconnect, run again."
  exit 1
fi

# Free GPU memory held by any earlier vLLM attempt (it caused 2 tokens/sec before).
pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f EngineCore 2>/dev/null || true

if ! command -v ollama > /dev/null; then
  apt-get -qq install -y zstd > /dev/null
  curl -fsSL https://ollama.com/install.sh | sh
fi

if ! curl -s http://localhost:11434/api/tags > /dev/null; then
  nohup ollama serve > ollama.log 2>&1 &
  for i in $(seq 1 30); do
    curl -s http://localhost:11434/api/tags > /dev/null && break
    sleep 2
  done
fi

ollama pull "$MODEL"
echo
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
echo "Ollama ready with $MODEL (default LOCAL_BASE_URL already points to port 11434)."
echo "After your first request, run 'ollama ps' and check it says 100% GPU."
