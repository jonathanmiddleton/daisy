#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper that delegates to the Python runner.
# The Python runner handles logging, environment, overrides (including comma-separated lists),
# and executes Cartesian product runs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure Python uses repository root for imports
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH-}"
export TORCH_LOGS=recompiles,graph_breaks

python -m training.runner "$@"