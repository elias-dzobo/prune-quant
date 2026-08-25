#!/usr/bin/env bash
# Run prune-quant commands inside the uv virtual environment.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  echo "Creating virtual environment..."
  uv venv --python 3.11 .venv
  uv pip install -e .
fi

exec "$ROOT/.venv/bin/python" "$@"
