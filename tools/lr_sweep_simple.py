import argparse
from typing import Any, Dict, List

import torch
from torch import nn

from models import model_from_spec
from training.hparams import load_hparams_from_yaml, apply_cli_overrides
from data.data_gen_stream import DistributedDataGenerator
from training.optim import build_optimizers_from_cfg, get_num_window_blocks, get_referenced_groups


def _filter_optimizer_cfg(full_cfg: List[Dict[str, Any]], group_name: str) -> List[Dict[str, Any]]:
    """Return a filtered optimizer cfg list containing only param groups matching group_name.
    Drops any optimizer entries left without params.
    """
    filtered: List[Dict[str, Any]] = []
    for oc in full_cfg or []:
        params = oc.get("params") or []
        kept_pgs = [pg for pg in params if isinstance(pg, dict) and pg.get("group") == group_name]
        if kept_pgs:
            oc_new = dict(oc)
            oc_new["params"] = kept_pgs
            filtered.append(oc_new)
    return filtered


def _mean_abs_grad(params: List[nn.Parameter]) -> float:
    total = 0.0
    count = 0
    for p in params:
        if p.grad is not None:
            g = p.grad.detach()
            total += g.abs().mean().item()
            count += 1
    return (total / max(1, count)) if count > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Ultra-simple LR sweep isolating one param group")
    parser.add_argument("--config", "-c", required=True, help="Path to training YAML config")
    parser.add_argument("--group", "-g", required=True, help="Name of the parameter group to isolate (e.g., embed_params)")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps to run (default: 100)")
    parser.add_argument("--device", default="cuda", help="Device to use (default: cuda)")
    parser.add_argument("--begin-shard", type=int, default=None, help="Optional 1-based shard to start from")
    # Allow arbitrary CLI overrides like --training_sequence_length=8192, etc.
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="Optional key=value overrides for Hyperparameters")

    args = parser.parse_args()

    # Load and optionally override hyperparameters
    hp = load_hparams_from_yaml(args.config)
    if args.overrides:
        # Strip a leading "--" separator if provided
        overrides = [s for s in args.overrides if s != "--"]
        hp = apply_cli_overrides(hp, overrides)

    device = torch.device(args.device)
    model: nn.Module = model_from_spec(hp.model_spec, device=str(device))
    model.train()
    # Match train.py: embeddings in bf16
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            m.bfloat16()

    # Single-rank data generator (no DDP)
    ddg = DistributedDataGenerator(
        hp.train_shards,
        sequence_length=hp.training_sequence_length,
        rank=0,
        world_size=1,
        start_shard=args.begin_shard,
        device=str(device),
    )

    # Filter optimizer config down to a single param group
    if not hp.optimizers:
        raise ValueError("Config must include an 'optimizers' list")

    # Build the optimizer for the isolated param group
    frozens = [g for g in get_referenced_groups(hp.optimizers) if g != args.group]
    print(f"frozens: {frozens}")
    print(f"args.group: {args.group}")
    print(f"hp.optimizers: {hp.optimizers}")
    optimizer = build_optimizers_from_cfg(cfg_list=hp.optimizers, model=model, rank=0, world_size=1, frozen_groups=frozens)
    if not optimizer:
        raise RuntimeError("Failed to construct optimizer from filtered config")
    if len(optimizer) != 1:
        raise RuntimeError("Expected exactly one optimizer, got multiple")
    optimizer = optimizer[0]


    # Print the effective config, isolated group, and associated optimizer class
    print("=== lr_sweep_simple: effective settings ===")
    print("Isolated parameter group:", args.group)
    print("Frozen parameter groups:", frozens)
    print("Using optimizer:", optimizer.__class__.__name__)

    # Determine sliding window blocks (use full schedule by default)
    sliding_window_num_blocks = get_num_window_blocks(
        1.0,
        attention_window_len=int(hp.train_attention_window_len),
        window_block_size=int(hp.window_block_size),
    )
    # Ensure it's on the right device
    if sliding_window_num_blocks.device.type != device.type:
        sliding_window_num_blocks = sliding_window_num_blocks.to(device)

    # Cache the isolated parameters list for gradient reporting
    iso_params: List[nn.Parameter] = [p for g in optimizer.param_groups for p in g["params"]]
    model = torch.compile(model, dynamic=False)

    steps = int(args.steps)
    for step in range(1, steps + 1):
        inputs, targets = next(ddg)
        # inputs: (T,), targets: (T,)
        loss = model(inputs, sliding_window_num_blocks, targets)
        loss.backward()

        # Report
        mean_abs = _mean_abs_grad(iso_params)
        print(f"step {step:03d} | loss {loss.item():.6f} | mean|grad| {mean_abs:.6e}")

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    main()
