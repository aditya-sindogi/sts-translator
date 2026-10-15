#!/usr/bin/env bash
# One-time setup on the EC2 GPU instance (no sudo or Docker needed).
#   1. app environment (uv, with the NeMo extra for Parakeet on GPU)
#   2. a separate vLLM nightly environment (Gemma 4 12B isn't in a stable vLLM yet)
#   3. model downloads (Gemma ~24 GB, Parakeet, LID, Moonshine, Kokoro)
# Usage: bash deploy/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "Checks"
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"
# PyTorch and vLLM wheels bundle their own CUDA/cuDNN. A system LD_LIBRARY_PATH (e.g. from
# another project's CUDA 12.1 activate.sh) makes them load mismatched libraries and fail
# with CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED, so it is cleared for everything we start.
unset LD_LIBRARY_PATH
if ! command -v uv >/dev/null; then
  echo "uv not found; installing it for this user (no sudo needed)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  command -v uv >/dev/null || { echo "uv install failed; see https://docs.astral.sh/uv/"; exit 1; }
fi
uv --version
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
FREE_GB=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
echo "free disk: ${FREE_GB} GB"
[ "$FREE_GB" -ge 45 ] || { echo "Need ~45 GB free (Gemma weights + two Python envs)."; exit 1; }

say "Config (.env)"
if [ ! -f .env ]; then
  KEY=$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')
  sed "s/__GENERATED__/${KEY}/g" deploy/ec2.env.example > .env
  echo "wrote .env (vLLM API key generated)"
else
  echo ".env exists, leaving it as is"
fi
set -a; source .env; set +a
VLLM_VENV=${VLLM_VENV/#\$HOME/$HOME}

say "App environment"
uv sync --extra nemo

say "vLLM nightly environment at $VLLM_VENV"
uv venv --python 3.12 "$VLLM_VENV"
VIRTUAL_ENV="$VLLM_VENV" uv pip install -U vllm --pre \
  --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match
"$VLLM_VENV/bin/python" -c "import vllm; print('vllm', vllm.__version__)"

say "Download Gemma ($VLLM_MODEL)"
"$VLLM_VENV/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$VLLM_MODEL'))"

say "Download app models (Parakeet, LID, Moonshine, Kokoro)"
STS_LLM_REQUIRED=false uv run python -m scripts.check_stack --models-only

mkdir -p "$REPO/logs" "$REPO/run"
say "Done. Next: bash deploy/start.sh   then   uv run python -m scripts.check_stack"
