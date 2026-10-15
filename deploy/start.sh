#!/usr/bin/env bash
# Start Ollama (if not already running) and the app, in the background. Logs go to logs/.
# Usage: bash deploy/start.sh [all|ollama|app]     (default: all)
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
unset LD_LIBRARY_PATH
mkdir -p logs run

start_ollama() {
  if curl -sf http://127.0.0.1:11434/api/version >/dev/null; then echo "Ollama already running"; return; fi
  echo "Starting Ollama"
  # Gemma loads onto the GPU when a request comes in and unloads after 5 idle minutes.
  # 16K context: Ollama's 4K default is too small for summarizing longer documents.
  OLLAMA_HOST=127.0.0.1:11434 OLLAMA_CONTEXT_LENGTH=16384 \
    setsid nohup ollama serve > logs/ollama.log 2>&1 &
  echo $! > run/ollama.pid
  for _ in $(seq 1 30); do
    curl -sf http://127.0.0.1:11434/api/version >/dev/null && { echo "Ollama ready"; return; }
    sleep 1
  done
  echo "Ollama did not start; see logs/ollama.log"; exit 1
}

start_app() {
  if [ -f run/app.pid ] && kill -0 "$(cat run/app.pid)" 2>/dev/null; then echo "App already running"; return; fi
  echo "Starting app (first start loads the speech models, ~1 min)"
  setsid nohup uv run python -m app.server > logs/app.log 2>&1 &
  echo $! > run/app.pid
  for _ in $(seq 1 60); do
    if curl -sf http://127.0.0.1:7860/api/config >/dev/null; then
      echo "App ready. On your laptop run:  ssh -N -L 7860:localhost:7860 l40-aws"
      echo "then open http://localhost:7860"
      return
    fi
    kill -0 "$(cat run/app.pid)" 2>/dev/null || { echo "App exited; last log lines:"; tail -20 logs/app.log; exit 1; }
    sleep 3
  done
  echo "App not ready after 3 min; see logs/app.log"; exit 1
}

case "${1:-all}" in
  all) start_ollama; start_app ;;
  ollama) start_ollama ;;
  app) start_app ;;
  *) echo "usage: $0 [all|ollama|app]"; exit 2 ;;
esac
