#!/usr/bin/env bash
# Stop what deploy/start.sh started. Usage: bash deploy/stop.sh [all|ollama|app]
cd "$(dirname "$0")/.."

stop() {
  local pidfile="run/$1.pid"
  if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    kill -TERM -- "-$(cat "$pidfile")" 2>/dev/null || kill -TERM "$(cat "$pidfile")"
    echo "stopped $1"
  else
    echo "$1 not running"
  fi
  rm -f "$pidfile"
}

case "${1:-all}" in
  all) stop app; stop ollama ;;
  ollama|app) stop "$1" ;;
  *) echo "usage: $0 [all|ollama|app]"; exit 2 ;;
esac
