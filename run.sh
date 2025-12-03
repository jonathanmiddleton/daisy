#!/usr/bin/env bash
#
# Enhanced runner wrapper
# - Accepts one or more config files or glob patterns (e.g., config/*.yml)
# - Iterates over each resolved config and invokes the Python runner sequentially
# - Fault tolerant: if a run fails, it logs the error and proceeds to the next config
# - Preserves existing override behavior: all non-config args are forwarded unchanged

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure Python uses repository root for imports
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH-}"
export TORCH_LOGS=recompiles,graph_breaks

# Usage/help text
print_help() {
  cat <<'EOF'
Usage:
  ./run.sh [CONFIG ...] [OPTIONS] [OVERRIDES...]

CONFIG:
  - A YAML file path (e.g., config/pretrain/nano-dclm.yml)
  - A quoted glob pattern matching YAML files (e.g., "config/**/*.yml")

OPTIONS (forwarded to training.runner; it invokes torchrun when needed):
  -n NPROC              Processes per node (macOS forces nproc=1)
  --nnodes N            Number of nodes (omit or 1 for single-node)
  --node-rank R         Node rank for multi-node (alias: --node_rank)
  --master-addr HOST    Master address (alias: --master_addr, or env MASTER_ADDR)
  --master-port PORT    Master port (alias: --master_port, or env MASTER_PORT)
  -p CHECKPOINT         Initial checkpoint path (forwards as init_checkpoint=...)
  -s BEGIN_SHARD        BEGIN_SHARD environment value
  -r RUN_ID             Base RUN_ID for the run(s)
  --full_windows        Example boolean flag forwarded as --full_windows=true
  --any-long-opt        Any additional long options are forwarded as overrides

OVERRIDES (Cartesian product; handled by training.runner):
  key=val               Single override (e.g., lr=3e-4)
  key=v1,v2,...         Grid values combine across keys (e.g., lr=1e-3,1e-4 wd=0,0.1)

Examples (single process):
  ./run.sh config/pretrain/nano-dclm.yml lr=1e-3,1e-4 wd=0,0.1
  ./run.sh config/pretrain/nano-dclm.yml config/old/pico-linear-fineweb-edu.yml --full_windows lr=1e-3,1e-4
  ./run.sh "config/**/*.yml" head_params_lr=0.7,0.8

Distributed training:
  Single node, 8 GPUs:
    ./run.sh -n 8 config/pretrain/nano-dclm.yml lr=1e-3,1e-4

  Multi-node (2 nodes, 8 GPUs each):
    # on node 0
    MASTER_ADDR=node0.example.com MASTER_PORT=29500 \
    ./run.sh --nnodes 2 --node-rank 0 -n 8 config/pretrain/nano-dclm.yml run_name=myrun

    # on node 1
    MASTER_ADDR=node0.example.com MASTER_PORT=29500 \
    ./run.sh --nnodes 2 --node-rank 1 -n 8 config/pretrain/nano-dclm.yml run_name=myrun

  SLURM (example):
    srun --nodes=2 --ntasks-per-node=8 bash -lc '
      MASTER_ADDR=${SLURM_JOB_NODELIST%%,*}; MASTER_PORT=29500;
      ./run.sh -n $SLURM_NTASKS_PER_NODE --nnodes $SLURM_JOB_NUM_NODES \
               --node-rank $SLURM_NODEID config/pretrain/nano-dclm.yml
    '

Notes:
  - Multiple CONFIGs or glob patterns will be executed sequentially; failures are logged and the script continues to the next.
  - If no CONFIGs are detected but arguments are provided, run.sh delegates to training.runner unchanged.
  - For distributed training, pass -n/--nnodes/--node-rank/--master-addr/--master-port to run.sh; it will invoke torchrun automatically.
  - Use -h or --help to show this message.
\n+Multi-node ergonomics (modeled after nccl_test.sh):
  - If --nnodes > 1:
      - On master (node_rank=0):
          * Automatically picks MASTER_PORT=29500 if unset
          * Infers MASTER_ADDR from first 10.x.x.x interface when not provided
          * Exports MASTER_ADDR/MASTER_PORT/MASTER_HOSTNAME and prints worker instructions
      - On workers (node_rank>0):
          * Ensures /etc/hosts contains "<MASTER_ADDR> <MASTER_HOSTNAME>" (using sudo when needed)
          * Exports MASTER_* and NODE_RANK
EOF
}

# Show help when no arguments are provided
if [[ $# -eq 0 ]]; then
  print_help
  exit 0
fi

# Show help when -h/--help is present anywhere in the args
for _arg in "$@"; do
  case "$_arg" in
    -h|--help)
      print_help
      exit 0
      ;;
  esac
done

# Enable safe globbing: unmatched globs expand to nothing instead of literal strings
shopt -s nullglob 2>/dev/null || true
# Enable ** if supported (bash >= 4); ignore if unsupported
shopt -s globstar 2>/dev/null || true

# Separate provided args into:
# - configs: files (or matched globs) that end with .yml/.yaml
# - extras: everything else, forwarded verbatim to the Python runner
configs=()
extras=()

is_glob() {
  case "$1" in
    *'*'*|*'?'*|*'['*']'*) return 0;;
    *) return 1;;
  esac
}

for arg in "$@"; do
  # Overrides like key=1,2 or bare tokens should be forwarded, not treated as configs
  if [[ "$arg" == -* ]] || [[ "$arg" == *=* ]]; then
    extras+=("$arg")
    continue
  fi

  # If it's a glob pattern, expand it and collect matching YAML files
  if is_glob "$arg"; then
    # Use an intermediate array to capture expansion results safely
    expanded=( $arg )
    # Filter to readable .yml/.yaml files
    for f in "${expanded[@]:-}"; do
      if [[ -f "$f" && "$f" == *.yml || -f "$f" && "$f" == *.yaml ]]; then
        configs+=("$f")
      fi
    done
    # If nothing matched, warn and continue (fault tolerant)
    if [[ ${#expanded[@]:-0} -eq 0 ]]; then
      echo "[run.sh] Warning: glob pattern matched no files: $arg" >&2
    fi
    continue
  fi

  # Non-glob token: treat as config only if it is an existing YAML file
  if [[ -f "$arg" && ( "$arg" == *.yml || "$arg" == *.yaml ) ]]; then
    configs+=("$arg")
  else
    # Otherwise forward it as an extra token (e.g., bare override flag)
    extras+=("$arg")
  fi
done

# ---------------------------
# Multi-node ergonomics block
# ---------------------------

# Helper: extract option value from an array (supports --opt=val, --opt val, and -n val)
get_opt() {
  local name="$1"; shift
  local -n _arr="$1"; shift
  local i tok next
  for ((i=0; i<${#_arr[@]}; i++)); do
    tok="${_arr[$i]}"
    # --long=value
    if [[ "$tok" == --${name}=* ]]; then
      echo "${tok#*=}"
      return 0
    fi
    # --long value
    if [[ "$tok" == --${name} ]]; then
      next="${_arr[$((i+1))]:-}"
      if [[ -n "$next" && "$next" != "" && ! "$next" =~ ^- ]]; then
        echo "$next"
        return 0
      fi
    fi
  done
  return 1
}

# Short option -n value
get_short_n() {
  local -n _arr="$1"
  local i tok next
  for ((i=0; i<${#_arr[@]}; i++)); do
    tok="${_arr[$i]}"
    if [[ "$tok" == "-n" ]]; then
      next="${_arr[$((i+1))]:-}"
      if [[ -n "$next" && "$next" != "" && ! "$next" =~ ^- ]]; then
        echo "$next"
        return 0
      fi
    fi
  done
  return 1
}

# Derive distributed params from flags/env
NNODES="${NNODES:-}"
if [[ -z "${NNODES}" ]]; then
  NNODES=$(get_opt nnodes extras || true)
fi
NNODES="${NNODES:-1}"

NODE_RANK="${NODE_RANK:-}"
if [[ -z "${NODE_RANK}" ]]; then
  NODE_RANK=$(get_opt node-rank extras || true)
fi
if [[ -z "${NODE_RANK}" ]]; then
  NODE_RANK=$(get_opt node_rank extras || true)
fi
NODE_RANK="${NODE_RANK:-0}"

# nproc per node from -n
NPROC_PER_NODE="${NPROC_PER_NODE:-}"
if [[ -z "${NPROC_PER_NODE}" ]]; then
  NPROC_PER_NODE=$(get_short_n extras || true)
fi
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

MASTER_ADDR="${MASTER_ADDR:-}"
if [[ -z "${MASTER_ADDR}" ]]; then
  MASTER_ADDR=$(get_opt master-addr extras || true)
fi
if [[ -z "${MASTER_ADDR}" ]]; then
  MASTER_ADDR=$(get_opt master_addr extras || true)
fi

MASTER_PORT="${MASTER_PORT:-}"
if [[ -z "${MASTER_PORT}" ]]; then
  MASTER_PORT=$(get_opt master-port extras || true)
fi
if [[ -z "${MASTER_PORT}" ]]; then
  MASTER_PORT=$(get_opt master_port extras || true)
fi

MASTER_HOSTNAME="${MASTER_HOSTNAME:-${HOSTNAME:-}}"

maybe_detect_master_addr() {
  # Try to detect first 10.x.x.x address
  local ip_out
  ip_out=$(ip -4 -o addr show scope global 2>/dev/null \
    | awk '$4 ~ /^10\./ {print $4; exit}' \
    | cut -d/ -f1 || true)
  echo "${ip_out}"
}

if [[ "${NNODES}" =~ ^[0-9]+$ && ${NNODES} -gt 1 ]]; then
  # Multi-node ergonomics
  if [[ "${NODE_RANK}" = "0" ]]; then
    # Master node
    if [[ -z "${MASTER_ADDR}" ]]; then
      MASTER_ADDR=$(maybe_detect_master_addr)
      if [[ -z "${MASTER_ADDR}" ]]; then
        echo "[run.sh] Failed to auto-detect a 10.x.x.x address for MASTER_ADDR." >&2
        echo "[run.sh] Provide it explicitly, e.g.: MASTER_ADDR=<ip> ./run.sh --nnodes ${NNODES} --node-rank 0 ..." >&2
      fi
    fi
    MASTER_PORT="${MASTER_PORT:-29500}"
    MASTER_HOSTNAME="${MASTER_HOSTNAME:-${HOSTNAME:-master}}"

    export MASTER_ADDR MASTER_PORT MASTER_HOSTNAME NNODES NPROC_PER_NODE NODE_RANK

    echo
    echo "[run.sh] -----------------------------------------------------------"
    echo "[run.sh] Multi-node master configuration:"
    echo "[run.sh]   MASTER_HOSTNAME = ${MASTER_HOSTNAME}"
    echo "[run.sh]   MASTER_ADDR     = ${MASTER_ADDR:-<unset>}"
    echo "[run.sh]   MASTER_PORT     = ${MASTER_PORT}"
    echo "[run.sh]   NNODES          = ${NNODES}"
    echo "[run.sh]   NPROC_PER_NODE  = ${NPROC_PER_NODE}"
    echo
    echo "[run.sh] Execute on each worker (adjust NODE_RANK per host):"
    echo "[run.sh]   MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} MASTER_HOSTNAME=${MASTER_HOSTNAME} NODE_RANK=<rank> \\" 
    echo "[run.sh]     ./run.sh --nnodes ${NNODES} --node-rank <rank> -n ${NPROC_PER_NODE} <CONFIG> [OVERRIDES...]"
    echo "[run.sh] -----------------------------------------------------------"
    echo
  else
    # Worker node
    if [[ -z "${MASTER_PORT}" ]]; then
      MASTER_PORT=29500
    fi
    if [[ -z "${MASTER_ADDR}" ]]; then
      echo "[run.sh] ERROR: MASTER_ADDR is required on worker nodes (node_rank=${NODE_RANK})." >&2
      echo "[run.sh] Set env MASTER_ADDR or pass --master-addr to run.sh." >&2
    fi
    MASTER_HOSTNAME="${MASTER_HOSTNAME:-${MASTER_ADDR}}"

    export MASTER_ADDR MASTER_PORT MASTER_HOSTNAME NNODES NPROC_PER_NODE NODE_RANK

    echo
    echo "[run.sh] -----------------------------------------------------------"
    echo "[run.sh] Worker configuration:"
    echo "[run.sh]   MASTER_HOSTNAME = ${MASTER_HOSTNAME}"
    echo "[run.sh]   MASTER_ADDR     = ${MASTER_ADDR:-<unset>}"
    echo "[run.sh]   MASTER_PORT     = ${MASTER_PORT}"
    echo "[run.sh]   NODE_RANK       = ${NODE_RANK}"
    echo "[run.sh]   NNODES          = ${NNODES}"
    echo "[run.sh]   NPROC_PER_NODE  = ${NPROC_PER_NODE}"
    echo
    # Ensure /etc/hosts entry exists for master
    if [[ -n "${MASTER_ADDR:-}" && -n "${MASTER_HOSTNAME:-}" ]]; then
      HOSTS_LINE="${MASTER_ADDR} ${MASTER_HOSTNAME}"
      if grep -Eq "^[[:space:]]*${MASTER_ADDR}[[:space:]]+.*\b${MASTER_HOSTNAME}\b" /etc/hosts; then
        echo "[run.sh] /etc/hosts already has entry for ${MASTER_HOSTNAME} (${MASTER_ADDR})"
      else
        echo "[run.sh] Adding to /etc/hosts: ${HOSTS_LINE}"
        if [ "$(id -u)" -ne 0 ]; then
          echo "${HOSTS_LINE}" | sudo tee -a /etc/hosts >/dev/null || true
        else
          echo "${HOSTS_LINE}" >> /etc/hosts || true
        fi
      fi
    fi
    echo "[run.sh] -----------------------------------------------------------"
    echo
  fi
fi

# If no configs were identified, fall back to original behavior
if [[ ${#configs[@]} -eq 0 ]]; then
  python -m training.runner "$@"
  exit $?
fi

echo "[run.sh] Discovered ${#configs[@]} config(s)."
for c in "${configs[@]}"; do
  echo "[run.sh]  - $c"
done

fail_count=0
failed_configs=()

run_one() {
  local cfg="$1"; shift
  echo "[run.sh] === Running config: $cfg ==="
  # Use if/else to avoid 'set -e' aborting the script on non-zero exit
  if python -m training.runner "$cfg" "$@"; then
    echo "[run.sh] Completed: $cfg"
  else
    rc=$?
    echo "[run.sh] ERROR: run failed for $cfg (exit $rc)" >&2
    fail_count=$((fail_count+1))
    failed_configs+=("$cfg:$rc")
  fi
}

for cfg in "${configs[@]}"; do
  run_one "$cfg" "${extras[@]}"
done

if [[ $fail_count -gt 0 ]]; then
  echo "[run.sh] Completed with $fail_count failure(s)." >&2
  for item in "${failed_configs[@]}"; do
    echo "[run.sh]  - $item" >&2
  done
  exit 1
fi

echo "[run.sh] All runs completed successfully."
exit 0