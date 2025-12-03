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