#!/usr/bin/env python3
"""
Runner for use in conjunction with run.sh. New behavior: a single process executes all
Cartesian product combinations by forwarding --grid arguments to train.py, which will
reset model/optimizer state between runs without recompiling the model.

Examples:
  python -m training.runner config/pretrain.yml head_params_lr=0.7,0.8,0.9
  python -m training.runner config/pretrain.yml head_params_lr=0.7,0.8,0.9 cooldown_frac=0.9,0.8,0.7

This will launch a single torchrun/python process and let train.py handle all combinations.
"""
import argparse
import itertools
import os
import shlex
import subprocess
import sys
from datetime import datetime
from io import TextIOBase
from pathlib import Path
from typing import List, Tuple
from tools.master_logger import MasterLogger

from tools.helpers import is_mac_os
import socket
import ipaddress

logger = MasterLogger

def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

#TODO single logging ownership - see train.py
def _setup_log_file() -> Tuple[Path, "TextIOBase"]:
    logs_dir = Path(__file__).resolve().parent.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{_timestamp()}.log"
    f = open(log_path, "a", buffering=1, encoding="utf-8")
    logger.info(f"Logging to {log_path}")
    return log_path, f


def _split_override(arg: str) -> Tuple[str, List[str]]:
    """
    Parse a single override token like 'key=1,2,3' or '--key=1,2'.
    Returns (key, [values...]) with values split on commas (commas within quotes are not supported).
    If the token has no '=', it's treated as a boolean flag set to 'true'.
    """
    if arg.startswith("--"):
        arg = arg[2:]
    if "=" not in arg:
        # treat bare flag as true
        k = arg.strip().replace("-", "_")
        return k, ["true"]
    k, v = arg.split("=", 1)
    k = k.strip().replace("-", "_")
    # split by commas, ignore empty strings
    parts = [p for p in (v.split(",") if "," in v else [v])]
    parts = [p.strip() for p in parts if p is not None and p != ""]
    return k, parts if parts else [v]


def _has_commas(vals: List[str]) -> bool:
    return any(
        ("," in v) for v in vals
    )


def _infer_master_addr() -> str | None:
    """Best-effort inference of a non-loopback IPv4 address for rendezvous.

    Uses a UDP connect trick to determine the primary egress interface IP without
    sending any packets. Returns None if a suitable non-loopback address cannot be found.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # Doesn't actually send traffic to 8.8.8.8; just forces routing decision
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        finally:
            s.close()
        # Validate non-loopback
        try:
            ip_obj = ipaddress.ip_address(ip)
            if ip_obj.is_loopback:
                return None
        except ValueError:
            return None
        return ip
    except Exception:
        return None


def build_run_cmd(
    *,
    nproc: int,
    config: str,
    checkpoint: str | None,
    extra_long_opts: List[str],
    singleton_overrides: List[Tuple[str, str]],
    grid_overrides: List[Tuple[str, List[str]]],
    nnodes: int | None = None,
    node_rank: int | None = None,
    master_addr: str | None = None,
    master_port: int | None = None,
) -> List[str]:
    """Build the command to launch training.

    Single-node behavior is preserved for compatibility with existing tests:
    - nproc == 1 -> python train.py
    - nproc > 1 and (nnodes is None or nnodes == 1) -> torchrun --standalone --nproc_per_node=n train.py

    Multi-node behavior (when nnodes > 1):
    - torchrun \
      --nnodes=NN \
      --nproc_per_node=n \
      --node_rank=R \
      --master_addr=ADDR \
      --master_port=PORT \
      train.py ...
    """
    # Prioritize multi-node first: even with nproc==1 we must use torchrun
    if nnodes is not None and nnodes > 1:
        # Multi-node torchrun in deterministic master/node rank form
        base_cmd = [
            "torchrun",
            f"--nnodes={nnodes}",
            f"--nproc_per_node={nproc}",
        ]
        if node_rank is not None:
            base_cmd += [f"--node_rank={node_rank}"]
        if master_addr is not None:
            base_cmd += [f"--master_addr={master_addr}"]
        if master_port is not None:
            base_cmd += [f"--master_port={master_port}"]
    elif nproc > 1:
        # Explicit single-node torchrun
        base_cmd = ["torchrun", "--standalone", f"--nproc_per_node={nproc}"]
    else:
        # Pure single-process run
        base_cmd = ["python"]
    cmd = base_cmd + ["train.py", config]

    # Add grid overrides as proper options first so argparse in train.py can parse them
    for k, vals in grid_overrides:
        joined = ",".join(vals)
        cmd.append(f"--grid={k}={joined}")

    # Everything else (including single-value overrides and passthrough long opts)
    # must be passed as positional tokens after "--" so train.py captures them in
    # its "overrides" list (it doesn't declare these as argparse options).
    positional_overrides: List[str] = []

    # Forward checkpoint as a config override token
    if checkpoint:
        positional_overrides.append(f"init_checkpoint={checkpoint}")

    # Forward any extra long opts as override-style tokens; they may start with
    # "--" (train.py strips it) or be plain key=value.
    positional_overrides.extend(extra_long_opts)

    # Forward singleton overrides as plain key=value tokens (no leading dashes)
    for k, v in singleton_overrides:
        positional_overrides.append(f"{k}={v}")

    if positional_overrides:
        cmd.append("--")
        cmd.extend(positional_overrides)

    return cmd


def _stream_subprocess(cmd: List[str], log_fp) -> int:
    """Run subprocess, streaming stdout/stderr to both console and log file."""
    # flush to avoid out-of-order
    sys.stdout.flush()
    sys.stderr.flush()
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
        env=os.environ.copy(),
    ) as p:
        assert p.stdout is not None
        for line in p.stdout:
            # Child process output is already fully formatted (timestamps, levels, etc.),
            # so just forward it instead of wrapping it in another logger.info().
            sys.stdout.write(line)
            # noinspection PyBroadException
            try:
                log_fp.write(line)
            except Exception:
                pass
        returncode = p.wait()
    return returncode

def main(argv: List[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # We accept a mix of short opts and trailing overrides. Use argparse for known ones and leave the rest.
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("-n", dest="nproc", type=int, default=8, help="nproc per node (nproc=1 if MacOS)")
    parser.add_argument("-p", dest="checkpoint", default="", help="init checkpoint path")
    parser.add_argument("-s", dest="begin_shard", default="", help="BEGIN_SHARD env value")
    parser.add_argument("-r", dest="run_id", default="1", help="RUN_ID env value for the run")
    # Optional distributed multi-node arguments
    parser.add_argument("--nnodes", dest="nnodes", type=int, default=None, help="Number of nodes for torchrun (omit or 1 for single-node)")
    parser.add_argument("--node-rank", dest="node_rank", type=int, default=None, help="Node rank for multi-node runs")
    parser.add_argument("--node_rank", dest="node_rank_alt", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--master-addr", dest="master_addr", default=None, help="Master address (rendezvous endpoint host)")
    parser.add_argument("--master_addr", dest="master_addr_alt", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--master-port", dest="master_port", type=int, default=None, help="Master port (rendezvous endpoint port)")
    parser.add_argument("--master_port", dest="master_port_alt", type=int, default=None, help=argparse.SUPPRESS)
    # Accept passthrough flags like --full_windows and arbitrary long options; we'll collect them

    # Parse known args and capture the rest for override processing
    args, extras = parser.parse_known_args(argv)

    config = args.config
    checkpoint = args.checkpoint or ""
    begin_shard = args.begin_shard or ""
    run_id = str(args.run_id)
    # prefer explicit flags; fall back to env vars
    nnodes = args.nnodes
    node_rank = args.node_rank if args.node_rank is not None else args.node_rank_alt
    master_addr = args.master_addr or args.master_addr_alt or os.environ.get("MASTER_ADDR")
    master_port = (
        args.master_port if args.master_port is not None else (args.master_port_alt if args.master_port_alt is not None else None)
    )
    if master_port is None:
        env_mp = os.environ.get("MASTER_PORT")
        master_port = int(env_mp) if env_mp and env_mp.isdigit() else None
    if node_rank is None:
        env_nr = os.environ.get("NODE_RANK")
        node_rank = int(env_nr) if env_nr and env_nr.isdigit() else None

    # Split the leftover overrides/long options
    raw_tail = list(extras or [])

    passthrough_long_opts: List[str] = []
    override_pairs: List[Tuple[str, List[str]]] = []

    i = 0
    while i < len(raw_tail):
        tok = raw_tail[i]
        if tok == "--":
            i += 1
            continue
        # Handle common flag aliases
        if tok in ("--full_windows", "--full-windows"):
            passthrough_long_opts.append("--full_windows=true")
            i += 1
            continue
        # --run_id style should map to env var RUN_ID but maintain compatibility
        if tok.startswith("--run_id="):
            run_id = tok.split("=", 1)[1]
            i += 1
            continue
        if tok.startswith("--run-id="):
            run_id = tok.split("=", 1)[1]
            i += 1
            continue
        # Any other token that starts with -- and either has = or not
        if tok.startswith("--") and ("=" not in tok):
            # preserve as-is long option without value
            passthrough_long_opts.append(tok)
            i += 1
            continue
        # For everything else, treat as an override key=value or --key=value
        k, vals = _split_override(tok)
        # If this looks like a long option not present in train Hyperparameters (can't easily know here),
        # but still in --foo=bar style, we'll forward as override; train.py will ignore unknown keys.
        override_pairs.append((k, vals))
        i += 1

    # Environment setup
    os.environ.setdefault("TORCH_DISABLE_MODEL_COMPILE", "0")
    os.environ.setdefault("TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS", "1")
    os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "8")
    if begin_shard:
        os.environ["BEGIN_SHARD"] = str(begin_shard)
    # RUN_ID may be overridden per combo; we will increment optionally, but here set base
    base_run_id = int(run_id) if run_id.isdigit() else run_id

    log_path, log_fp = _setup_log_file()

    nproc = 1 if is_mac_os() else args.nproc

    try:
        logger.info(f"Config: {config}")
        logger.info(f"nproc: {nproc}")
        if checkpoint:
            logger.info(f"checkpoint: {checkpoint}")
        if begin_shard:
            logger.info(f"BEGIN_SHARD: {begin_shard}")
        logger.info(f"RUN_ID base: {base_run_id}")
        logger.info(f"Extra opts: {' '.join(passthrough_long_opts) if passthrough_long_opts else '(none)'}")

        # Resolve rendezvous for multi-node runs before building the command
        resolved_master_addr = master_addr
        resolved_master_port = master_port
        if nnodes and nnodes > 1:
            # Default port if not provided
            if resolved_master_port is None:
                resolved_master_port = 29500
                logger.info("No --master-port/MASTER_PORT provided; defaulting to 29500")
            # If master addr missing, try to infer on rank 0; otherwise, fail fast on workers
            if not resolved_master_addr:
                if node_rank in (0, None):
                    inferred = _infer_master_addr()
                    if inferred:
                        resolved_master_addr = inferred
                        logger.info(f"No --master-addr/MASTER_ADDR provided; inferred MASTER_ADDR={resolved_master_addr}")
                    else:
                        logger.error("Unable to infer a non-loopback MASTER_ADDR for rank 0. Please provide --master-addr or set MASTER_ADDR.")
                        return 2
                else:
                    logger.error("Missing --master-addr/MASTER_ADDR on non-master node (node_rank>0). Provide the master's address to avoid rendezvous hang.")
                    return 2

            # Export to environment for child process consistency
            os.environ["MASTER_ADDR"] = str(resolved_master_addr)
            os.environ["MASTER_PORT"] = str(resolved_master_port)
            # Log final dist setup
            logger.info(f"Distributed multi-node: nnodes={nnodes}, node_rank={node_rank}, master={resolved_master_addr}:{resolved_master_port}")

        # Separate singleton overrides from grids
        singletons: List[Tuple[str, str]] = []
        grids: List[Tuple[str, List[str]]] = []
        for k, vals in override_pairs:
            if len(vals) == 1:
                singletons.append((k, vals[0]))
            else:
                grids.append((k, vals))

        # Set base RUN_ID; train.py will auto-increment per combo
        os.environ["RUN_ID"] = str(base_run_id)

        cmd = build_run_cmd(
            nproc=nproc,
            config=config,
            checkpoint=checkpoint or None,
            extra_long_opts=passthrough_long_opts,
            singleton_overrides=singletons,
            grid_overrides=grids,
            nnodes=nnodes,
            node_rank=node_rank,
            master_addr=resolved_master_addr if (nnodes and nnodes > 1) else master_addr,
            master_port=resolved_master_port if (nnodes and nnodes > 1) else master_port,
        )
        logger.info("\n=== Running single multi-run process: " + shlex.join(cmd))
        rc = _stream_subprocess(cmd, log_fp)
        return rc
    finally:
        # noinspection PyBroadException
        try:
            log_fp.flush()
            log_fp.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
