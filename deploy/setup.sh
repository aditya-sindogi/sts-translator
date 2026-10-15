#!/usr/bin/env bash
# One-time setup on the EC2 GPU box. Same stack as the laptop: Ollama runs Gemma on the GPU.
# No sudo or Docker needed. Usage: bash deploy/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
unset LD_LIBRARY_PATH  # stray CUDA paths from other projects break PyTorch's bundled libraries

# 1. uv (Python package manager)
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Ollama, installed for this user into ~/.local (it bundles its own CUDA runtime)
if ! command -v ollama >/dev/null; then
  echo "Installing Ollama into ~/.local"
  mkdir -p "$HOME/.local"
  curl -fL https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst -o /tmp/ollama.tar.zst
  if command -v zstd >/dev/null; then
    tar --zstd -xf /tmp/ollama.tar.zst -C "$HOME/.local"
  else  # Python 3.14 can read .tar.zst without the zstd tool
    uv run --no-project --python 3.14 python -c \
      "import os, tarfile; tarfile.open('/tmp/ollama.tar.zst', 'r:zst').extractall(os.path.expanduser('~/.local'), filter='tar')"
  fi
  rm -f /tmp/ollama.tar.zst
fi
ollama --version

# 3. App
[ -f .env ] || cp deploy/ec2.env.example .env
uv sync --extra nemo

# 4. Models: Gemma (via Ollama) and the app's speech models
bash deploy/start.sh ollama
ollama pull gemma4:12b
uv run python -m scripts.check_stack --models-only

echo "Setup done. Next: bash deploy/start.sh"
