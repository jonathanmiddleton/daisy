import argparse
import dataclasses
import os
import sys
from typing import List, Tuple

import torch

from tools.master_logger import MasterLogger
from training.hparams import load_hparams_from_yaml, apply_cli_overrides, Hyperparameters
from training.trainer import (
    CompiledRuntime,
    TrainingSession,
    partition_runs_by_compile_key,
    compute_group_max_seq_len,
)


logger = MasterLogger


def _parse_grid_item(s: str) -> Tuple[str, List[str]]:
    """Parse 'key=v1,v2' into (key, [v1, v2])."""
    if s.startswith("--"):
        s = s[2:]
    if s.startswith("grid="):
        s = s[len("grid="):]
    if "=" not in s:
        raise ValueError("--grid expects 'key=v1,v2' format")
    k, v = s.split("=", 1)
    parts = [p.strip() for p in (v.split(",") if "," in v else [v]) if p.strip() != ""]
    return k.strip().replace("-", "_"), parts


def main(argv: List[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("overrides", nargs="*", help="Overrides like key=value or --key=value")
    parser.add_argument("--grid", dest="grids", action="append", default=[], help="Grid override: key=v1,v2 (repeatable)")
    args_ns = parser.parse_args(argv)

    config_path = args_ns.config
    grid_specs_raw: List[str] = args_ns.grids or []
    tail_overrides: List[str] = [s for s in (args_ns.overrides or []) if not (s.startswith("--grid") or s == "--")]

    base_args = load_hparams_from_yaml(config_path)
    base_args = apply_cli_overrides(base_args, tail_overrides)

    # Materialize combinations
    grid_specs: List[Tuple[str, List[str]]] = []
    for g in grid_specs_raw:
        k, vals = _parse_grid_item(g)
        grid_specs.append((k, vals))

    def _cartesian(overrides: List[Tuple[str, List[str]]]) -> List[List[Tuple[str, str]]]:
        if not overrides:
            return [[]]
        keys = [k for k, _ in overrides]
        values_lists = [vals for _, vals in overrides]
        combos: List[List[Tuple[str, str]]] = []
        import itertools
        for prod in itertools.product(*values_lists):
            combos.append(list(zip(keys, prod)))
        return combos

    combos = _cartesian(grid_specs)

    run_args_list: list[tuple[int, Hyperparameters]] = []
    base_run_id = int(os.environ.get("RUN_ID", "0"))
    if not combos:
        combos = [[]]
    for idx, combo in enumerate(combos):
        a = dataclasses.replace(base_args)
        a = apply_cli_overrides(a, [f"{k}={v}" for k, v in combo])
        run_id = base_run_id + idx
        run_args_list.append((run_id, a))

    # Device/world for grouping
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"

    groups = partition_runs_by_compile_key(run_args_list, device_type=device_type, world_size=world_size)

    logger.info(f"Total run combinations: {len(run_args_list)}; compile groups: {len(groups)}")

    for key, group in groups.items():
        # Elevate group max_seq_len so buffers are sized adequately
        group_args_only = [a for _, a in group]
        group_max = compute_group_max_seq_len(group_args_only)
        for _, a in group:
            a.max_seq_len = max(int(getattr(a, "max_seq_len", a.training_sequence_length)), group_max)

        dynamic = any(a.train_mode == "task" for _, a in group)
        runtime = CompiledRuntime(group[0][1], dynamic=dynamic)
        try:
            for run_id, a in group:
                session = TrainingSession(runtime, a, run_id)
                session.run()
        finally:
            runtime.destroy()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
