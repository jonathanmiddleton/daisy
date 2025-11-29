#!/usr/bin/env bash
set -euo pipefail

pip install "torch==2.10.0.dev20251113+cu128" --pre \
  --index-url https://download.pytorch.org/whl/nightly/cu128 --upgrade

pip install -r requirements.txt