#!/usr/bin/env bash
# Simple distributed ping wrapper using the same distributed args as run.sh/runner.py
# Supports optional args via CLI or environment variables:
#   -n NPROC                  Number of processes per node
#   --nnodes NNODES          Number of nodes (omit or 1 for single-node)
#   --node-rank R | --node_rank R
#   --master-addr HOST | --master_addr HOST
#   --master-port PORT | --master_port PORT

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure Python uses repository root for imports
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH-}"

NPROC=1
NNODES=""
NODE_RANK="${NODE_RANK-}"
MASTER_ADDR="${MASTER_ADDR-}"
MASTER_PORT="${MASTER_PORT-}"

# Parse minimal set of distributed args
i=1
while [[ $i -le $# ]]; do
  arg="${!i}"
  case "$arg" in
    -n)
      i=$((i+1)); NPROC="${!i:-$NPROC}" ;;
    --nnodes)
      i=$((i+1)); NNODES="${!i:-}" ;;
    --node-rank|--node_rank)
      i=$((i+1)); NODE_RANK="${!i:-${NODE_RANK-}}" ;;
    --master-addr|--master_addr)
      i=$((i+1)); MASTER_ADDR="${!i:-${MASTER_ADDR-}}" ;;
    --master-port|--master_port)
      i=$((i+1)); MASTER_PORT="${!i:-${MASTER_PORT-}}" ;;
    *)
      # ignore other args
      ;;
  esac
  i=$((i+1))
done

# Build command
if [[ "${NPROC}" -le 1 ]]; then
  CMD=( python -m training.ping )
else
  if [[ -z "${NNODES}" || "${NNODES}" -eq 1 ]]; then
    CMD=( torchrun --standalone --nproc_per_node="${NPROC}" -m training.ping )
  else
    CMD=( torchrun --nproc_per_node="${NPROC}" --nnodes="${NNODES}" )
    # rendezvous if provided
    if [[ -n "${MASTER_ADDR}" && -n "${MASTER_PORT}" ]]; then
      CMD+=( --rdzv_backend=c10d --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" )
    fi
    if [[ -n "${NODE_RANK}" ]]; then
      CMD+=( --node_rank="${NODE_RANK}" )
    fi
    CMD+=( -m training.ping )
  fi
fi

echo "[run_ping.sh] Executing: ${CMD[*]}"
exec "${CMD[@]}"
