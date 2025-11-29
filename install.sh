#!/usr/bin/env bash
set -euo pipefail

pip install -r requirements.txt
pip install "torch==2.10.0.dev20251113+cu128" --pre \
  --index-url https://download.pytorch.org/whl/nightly/cu128 --upgrade

# flash-linear-attention (KimiLinear) is CUDA-only; on non-CUDA platforms
# skip it with a warning instead of failing the whole script.
if python - << 'PY'
import torch
exit(0 if torch.cuda.is_available() else 1)
PY
then
  pip install flash-linear-attention  # KimiLinear
else
  echo "WARNING: CUDA not available; skipping 'flash-linear-attention' install (KimiLinear disabled)." >&2
fi