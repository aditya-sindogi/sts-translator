#!/usr/bin/env bash
# Stop the app and/or vLLM started by deploy/start.sh.
# Usage: bash deploy/stop.sh [all|vllm|app]     (default: all)
set -uo pipefail
cd "$(dirname "$0")/.."
WHAT=${1:-all}

stop() {
  local name=$1 pidfile="run/$1.pid"
  if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    local pid; pid=$(cat "$pidfile")
    # Kill the whole process group (uv/vllm spawn children).
    kill -TERM -- "-$(ps -o pgid= "$pid" | tr -d ' ')" 2>/dev/null || kill -TERM "$pid"
    for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -0 "$pid" 2>/dev/null && kill -KILL "$pid"
    echo "stopped $name"
  else
    echo "$name not running"
  fi
  rm -f "$pidfile"
}

case "$WHAT" in
  all) stop app; stop vllm ;;
  vllm|app) stop "$WHAT" ;;
  *) echo "usage: $0 [all|vllm|app]"; exit 2 ;;
esac
