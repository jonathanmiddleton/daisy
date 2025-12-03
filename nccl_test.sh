#!/usr/bin/bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"

usage() {
  cat <<EOF
Usage:
  Master:
    $SCRIPT_NAME master [MASTER_ADDR]

    Environment overrides:
      MASTER_ADDR  - master IP/hostname (default: first 10.x.x.x interface)
      MASTER_PORT  - rendezvous port (default: 29500)
      NNODES       - total number of nodes (default: 2)
      NPROC_PER_NODE - processes per node (default: 1)

  Worker:
    $SCRIPT_NAME worker <MASTER_ADDR> <MASTER_PORT> <MASTER_HOSTNAME> [NODE_RANK]

    Environment overrides (if args omitted):
      MASTER_ADDR
      MASTER_PORT
      MASTER_HOSTNAME
      NODE_RANK      - this worker's node_rank (default: 1)

On workers, this script will ensure /etc/hosts contains a line:
  <MASTER_ADDR> <MASTER_HOSTNAME>
EOF
}

ROLE="${1:-}"

if [ -z "$ROLE" ]; then
  usage
  exit 1
fi

NNODES="${NNODES:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

if [ "$ROLE" = "master" ]; then
  # MASTER_ADDR: env, then arg, then auto-detect first 10.x.x.x
  MASTER_ADDR="${MASTER_ADDR:-${2:-}}"
  if [ -z "$MASTER_ADDR" ]; then
    MASTER_ADDR=$(ip -4 -o addr show scope global \
      | awk '$4 ~ /^10\./ {print $4; exit}' \
      | cut -d/ -f1 || true)
    if [ -z "$MASTER_ADDR" ]; then
      echo "Failed to auto-detect a 10.x.x.x address."
      echo "Please rerun with: MASTER_ADDR=<ip> $SCRIPT_NAME master"
      exit 1
    fi
  fi

  MASTER_PORT="${MASTER_PORT:-29500}"
  MASTER_HOSTNAME="${MASTER_HOSTNAME:-$HOSTNAME}"

  export MASTER_ADDR
  export MASTER_PORT
  export MASTER_HOSTNAME

  master_cmd=(
    torchrun
    --nproc_per_node="$NPROC_PER_NODE"
    --nnodes="$NNODES"
    --rdzv_backend=c10d
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT"
    --node_rank=0
    tools/simple_nccl_test.py
  )

  echo
  echo "Execute on each worker:"
  echo "  ./$SCRIPT_NAME worker $MASTER_ADDR $MASTER_PORT $MASTER_HOSTNAME 1"
  echo
  echo "-----------------------------------------------------------"
  echo "Assuming all hosts are on the same 10.0.0.0/24 network..."
  echo "Open ports 1 - 65535 for hostmask 10.0.0.0/24 on all hosts"
  echo
  echo "Master hostname: $MASTER_HOSTNAME"
  echo "Master address : $MASTER_ADDR"
  echo "Master port    : $MASTER_PORT"
  echo
  printf 'Executing (master):'
  printf ' %q' "${master_cmd[@]}"
  echo
  echo "-----------------------------------------------------------"
  echo
  echo

  "${master_cmd[@]}"

elif [ "$ROLE" = "worker" ]; then
  # Args or env
  MASTER_ADDR="${2:-${MASTER_ADDR:-}}"
  MASTER_PORT="${3:-${MASTER_PORT:-}}"
  MASTER_HOSTNAME="${4:-${MASTER_HOSTNAME:-}}"
  NODE_RANK="${5:-${NODE_RANK:-1}}"

  if [ -z "${MASTER_ADDR:-}" ] || [ -z "${MASTER_PORT:-}" ] || [ -z "${MASTER_HOSTNAME:-}" ]; then
    echo "Missing required parameters for worker."
    echo
    usage
    exit 1
  fi

  export MASTER_ADDR
  export MASTER_PORT
  export MASTER_HOSTNAME
  export NODE_RANK

  echo
  echo "-----------------------------------------------------------"
  echo "Worker configuration:"
  echo "  MASTER_ADDR    = $MASTER_ADDR"
  echo "  MASTER_PORT    = $MASTER_PORT"
  echo "  MASTER_HOSTNAME= $MASTER_HOSTNAME"
  echo "  NODE_RANK      = $NODE_RANK"
  echo

  # Ensure /etc/hosts entry exists
  HOSTS_LINE="$MASTER_ADDR $MASTER_HOSTNAME"
  if grep -Eq "^[[:space:]]*${MASTER_ADDR}[[:space:]]+.*\b${MASTER_HOSTNAME}\b" /etc/hosts; then
    echo "/etc/hosts already contains an entry for $MASTER_HOSTNAME ($MASTER_ADDR)"
  else
    echo "Adding to /etc/hosts: $HOSTS_LINE"
    if [ "$(id -u)" -ne 0 ]; then
      echo "Not running as root; attempting to use sudo to modify /etc/hosts..."
      echo "$HOSTS_LINE" | sudo tee -a /etc/hosts >/dev/null
    else
      echo "$HOSTS_LINE" >> /etc/hosts
    fi
  fi
  echo
  worker_cmd=(
    torchrun
    --nproc_per_node="$NPROC_PER_NODE"
    --nnodes="$NNODES"
    --rdzv_backend=c10d
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT"
    --node_rank="$NODE_RANK"
    tools/simple_nccl_test.py
  )

  echo "Starting torchrun on worker (node_rank=$NODE_RANK)..."
  printf 'Executing (worker):'
  printf ' %q' "${worker_cmd[@]}"
  echo
  echo "-----------------------------------------------------------"
  echo

  "${worker_cmd[@]}"

else
  echo "Unknown role: $ROLE"
  echo
  usage
  exit 1
fi