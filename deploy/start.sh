#!/usr/bin/env bash
# Start vLLM (Gemma 4 12B) and the app in the background; logs in logs/, PIDs in run/.
# Usage: bash deploy/start.sh [all|vllm|app]     (default: all)
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
mkdir -p logs run
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"
# PyTorch and vLLM wheels bundle their own CUDA/cuDNN. A system LD_LIBRARY_PATH (e.g. from
# another project's CUDA 12.1 activate.sh) makes them load mismatched libraries and fail
# with CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED, so it is cleared for everything we start.
unset LD_LIBRARY_PATH
set -a; source .env; set +a
VLLM_VENV=${VLLM_VENV/#\$HOME/$HOME}
WHAT=${1:-all}

running() { [ -f "run/$1.pid" ] && kill -0 "$(cat "run/$1.pid")" 2>/dev/null; }

start_vllm() {
  if running vllm; then echo "vLLM already running (pid $(cat run/vllm.pid))"; return; fi
  if pgrep -u "$USER" -f "vllm serve" >/dev/null; then
    echo "A 'vllm serve' process not started by this script is running (it holds GPU memory):"
    pgrep -u "$USER" -af "vllm serve"; echo "Stop it first: pkill -u $USER -f 'vllm serve'"; exit 1
  fi
  echo "starting vLLM: $VLLM_MODEL on 127.0.0.1:$VLLM_PORT"
  # setsid: own process group, so stop.sh can stop vLLM and the app independently.
  setsid nohup "$VLLM_VENV/bin/vllm" serve "$VLLM_MODEL" \
    --host 127.0.0.1 --port "$VLLM_PORT" \
    --api-key "$VLLM_API_KEY" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --enable-prefix-caching \
    > logs/vllm.log 2>&1 &
  echo $! > run/vllm.pid
  echo -n "waiting for vLLM to load (first start ~2-5 min)"
  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$VLLM_PORT/health" >/dev/null; then echo " ready"; return; fi
    running vllm || {
      echo; echo "vLLM exited. Root cause from logs/vllm.log:"
      grep -i -E "error|out of memory|kv cache|max_model_len" logs/vllm.log | grep -v -i warning | head -15
      exit 1
    }
    echo -n "."; sleep 5
  done
  echo; echo "vLLM not ready after 10 min; see logs/vllm.log"; exit 1
}

start_app() {
  if running app; then echo "app already running (pid $(cat run/app.pid))"; return; fi
  echo "starting app on $STS_HOST:$STS_PORT"
  setsid nohup uv run python -m app.server > logs/app.log 2>&1 &
  echo $! > run/app.pid
  for _ in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:$STS_PORT/api/config" >/dev/null; then
      echo "app ready. From your laptop:"
      echo "  ssh -N -L $STS_PORT:localhost:$STS_PORT <this-ec2>   then open http://localhost:$STS_PORT"
      return
    fi
    running app || { echo "app exited; see logs/app.log"; tail -20 logs/app.log; exit 1; }
    sleep 3
  done
  echo "app not ready after 3 min; see logs/app.log"; exit 1
}

case "$WHAT" in
  all) start_vllm; start_app ;;
  vllm) start_vllm ;;
  app) start_app ;;
  *) echo "usage: $0 [all|vllm|app]"; exit 2 ;;
esac
