import math
import time
from argparse import ArgumentParser
from typing import Iterable, List, Dict, Any

import torch
import json
import copy
from datetime import datetime
from pathlib import Path

from data.data_gen_stream import DistributedDataGenerator
from training.optim import get_num_window_blocks


# Utilities for group introspection ---------------------------------------------------------------
#noinspection PyShadowingNames
def _enumerate_param_groups(optimizers: Iterable[torch.optim.Optimizer]):
    """
    Yield tuples (opt_idx, group_idx, group_dict) for every param group across all optimizers.
    If the optimizer param group contains a 'name' key (added by our optimizer builder), we expose it.
    """
    for oi, opt in enumerate(optimizers):
        for gi, g in enumerate(opt.param_groups):
            yield oi, gi, g


def _group_key(oi: int, gi: int, g: Dict[str, Any]) -> str:
    """Stable identifier string for a param group."""
    name = g.get("name")
    return name if isinstance(name, str) else f"opt{oi}/group{gi}"


# LR sweep ----------------------------------------------------------------------------------------
#noinspection PyShadowingNames
def lr_sweep(
    model: torch.nn.Module,
    optimizers: List[torch.optim.Optimizer] | torch.optim.Optimizer,
    data_generator: DistributedDataGenerator,
    *,
    # window schedule for attention
    window_schedule: float = 1.0,
    train_attention_window_len: int = 3456,
    window_block_size: int = 128,
    # sweep setup
    num_scales: int = 200,
    scale_min: float = 0.5,
    scale_max: float = 5.0,
    steps_per_scale: int = 20,
    smooth: float = 0.85,  # EMA on loss (computed within each scale window)
    device: str = "cuda",
    accum_steps: int = 1,
    clip_norm: float | None = None,
    blowup_pct: float = 0.30,  # early stop when EMA > (1+blowup_pct)*best
    # scaling control
    scaled_group_names: List[str] | None = None,
    # optional Weights & Biases logging per-scale
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
    wandb_log: bool = False,
    # additional metadata to include in wandb.init config (e.g., full Hyperparameters dict)
    wandb_extra_config: Dict[str, Any] | None = None,
):
    """
    Sweep a multiplicative LR scale between [scale_min, scale_max] across `num_scales` points.

    Semantics:
    - We apply a scalar s to selected optimizer param groups such that lr_group = base_lr[group] * s.
    - The scalar s increases exponentially from scale_min to scale_max over `num_scales` points.
    - For each scalar value, we run `steps_per_scale` (min 2) optimizer steps and compute an EMA of the delta loss (prev_loss - curr_loss)
      within that window. The final EMA(delta) for the window is recorded as the score for that scalar.

    Group control:
    - Swept groups are exactly those present in the provided optimizer(s). To isolate a single group,
      build the optimizer(s) with training.optim.build_optimizers_from_cfg using the frozen_groups argument to
      exclude all other groups.

    Returns (scales, ema_losses, meta) where:
      - scales: list[float] of the applied multiplicative scalar per point
      - ema_losses: list[float] of EMA losses per point (one per scalar)
      - meta: dict with keys:
          'groups': dict[group_key -> {oi, gi, name, base_lr}],
          'steps_per_scale': int,
          'num_scales': int,
          'scale_min': float,
          'scale_max': float,
    """
    # Normalize optimizers input
    if not isinstance(optimizers, (list, tuple)):
        optimizers = [optimizers]

    # Move model
    model = model.to(device)
    model.train()

    # Build param-group registry and capture base LRs
    group_infos: Dict[str, Dict[str, Any]] = {}

    for oi, gi, g in _enumerate_param_groups(optimizers):
        key = _group_key(oi, gi, g)
        # Ensure base_lr is recorded and stable
        base_lr = g.get("base_lr", g.get("lr", optimizers[oi].defaults.get("lr", 1e-3)))
        g["base_lr"] = float(base_lr)
        group_infos[key] = {
            "oi": oi,
            "gi": gi,
            "name": g.get("name"),
            "base_lr": float(base_lr),
            "frozen": False,  # set below
        }

    # Determine which param groups to scale vs. keep fixed at their base LR
    all_keys = list(group_infos.keys())
    # Build lookup from external name and internal key -> keys
    name_to_keys: Dict[str, List[str]] = {}
    for k, info in group_infos.items():
        nm = info.get("name")
        if isinstance(nm, str):
            name_to_keys.setdefault(nm, []).append(k)
        # also support addressing by internal key
        name_to_keys.setdefault(k, []).append(k)

    if scaled_group_names is None:
        scaled_keys = set(all_keys)  # default: scale all
        requested_names: List[str] = []
    else:
        requested_names = [str(n).strip() for n in scaled_group_names if str(n).strip()]
        missing: List[str] = []
        scaled_keys = set()
        for nm in requested_names:
            keys = name_to_keys.get(nm)
            if not keys:
                missing.append(nm)
            else:
                scaled_keys.update(keys)
        if missing:
            available = sorted([k for k in name_to_keys.keys()])
            raise ValueError(
                "Requested groups not found: " + ", ".join(missing) +
                ". Available groups: " + ", ".join(available)
            )
        if not scaled_keys:
            raise ValueError("No valid parameter groups selected to scale.")

    # Function to set LRs based on a scalar for scaled groups; keep others at base LR
    # noinspection PyShadowingNames
    def set_lrs(scalar: float):
        for oi2, gi2, g in _enumerate_param_groups(optimizers):
            k2 = _group_key(oi2, gi2, g)
            base = float(g.get("base_lr", 1e-3))
            g["lr"] = base * float(scalar) if k2 in scaled_keys else base

    # Initialize sweep (compute geometric positions for scales)
    def _scale_at(i: int) -> float:
        if num_scales <= 1:
            return float(scale_min)
        ratio = max(scale_max, 1e-12) / max(scale_min, 1e-12)
        return float(scale_min) * (ratio ** (i / (num_scales - 1)))

    # Coalesce defaults and validate sweep parameters
    if scale_min is None:
        scale_min = 1e-2
    if scale_max is None:
        scale_max = 1e+2
    # Guardrails
    if num_scales is None or num_scales < 1:
        num_scales = 1
    # Ensure positive scales
    eps = 1e-12
    if not isinstance(scale_min, (int, float)) or not isinstance(scale_max, (int, float)):
        raise ValueError("scale_min and scale_max must be numbers or None")
    if scale_min <= 0:
        scale_min = eps
    if scale_max <= 0:
        scale_max = eps
    # Ensure ordering
    if scale_max < scale_min:
        scale_min, scale_max = scale_max, scale_min

    # Enforce minimum steps per scale for delta-loss EMA
    if steps_per_scale is None or steps_per_scale < 2:
        print("[lr_sweep] steps_per_scale < 2; bumping to 2 for delta-loss EMA", flush=True)
        steps_per_scale = 2

    t0 = time.time()
    print(
        f"[lr_sweep] num_scales={num_scales}, scale_min={scale_min:.3e}, scale_max={scale_max:.3e}, "
        f"steps_per_scale={steps_per_scale}, accum_steps={accum_steps}, smooth={smooth}, "
        f"blowup_pct={blowup_pct*100:.1f}%, metric=EMA(delta_loss, debiased)",
        flush=True,
    )
    print("[lr_sweep] Groups:", flush=True)
    for key in all_keys:
        info = group_infos[key]
        name = info.get("name") or key
        tag = "[scaled]" if key in scaled_keys else "[fixed]"
        print(f"  - {name}: base_lr={info['base_lr']:.3e} {tag}", flush=True)

    print_every = max(1, num_scales // 50) if num_scales else 1

    # Capture baselines to reset between scales
    model_baseline = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    opt_baselines = [copy.deepcopy(opt.state_dict()) for opt in optimizers]

    global_best = float("-inf")
    lrs_scalars: List[float] = []
    losses: List[float] = []
    improvements: List[float] = []

    # Preserve original optimizers to rebuild per scale if needed
    orig_optimizers: List[torch.optim.Optimizer] = optimizers if isinstance(optimizers, list) else [optimizers]

    # Sweep loop over scales; reset model/optimizer/data at each new scale
    for i in range(num_scales):
        scalar = _scale_at(i)
        # Optional: start a fresh wandb run for this scale
        _wb = None
        _wb_run = None
        _wb_enabled = bool(wandb_log) and bool(wandb_project)
        _wb_name_base = (wandb_run_name or "lr-sweep").strip()
        if _wb_enabled:
            try:
                import wandb as _wb  # type: ignore
                # Assemble wandb config with sweep settings, model info, and optional hyperparameters
                try:
                    total_params = int(sum(p.numel() for p in model.parameters()))
                except Exception:
                    total_params = None
                try:
                    trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
                except Exception:
                    trainable_params = None
                model_class = f"{model.__class__.__module__}.{model.__class__.__name__}"

                _cfg: Dict[str, Any] = {
                    # Sweep details
                    "lr_sweep/scale": float(scalar),
                    "lr_sweep/steps_per_scale": int(steps_per_scale),
                    "lr_sweep/num_scales": int(num_scales),
                    "lr_sweep/accum_steps": int(accum_steps),
                    "lr_sweep/clip_norm": float(clip_norm) if clip_norm is not None else None,
                    "lr_sweep/smooth": float(smooth),
                    "lr_sweep/blowup_pct": float(blowup_pct),
                    "lr_sweep/scaled_groups": [group_infos[k].get("name") or k for k in (scaled_keys or [])],
                    # Model info
                    "model/class": model_class,
                    "model/parameters_total": total_params,
                    "model/parameters_trainable": trainable_params,
                }
                # If available, include model spec and full hyperparameters under a namespaced key
                if isinstance(wandb_extra_config, dict) and wandb_extra_config:
                    _cfg["hparams"] = wandb_extra_config
                    if "model_spec" in wandb_extra_config:
                        _cfg["model/spec"] = str(wandb_extra_config.get("model_spec"))

                _wb_run = _wb.init(
                    project=str(wandb_project),
                    name=f"{_wb_name_base}-{float(scalar):.3e}",
                    config=_cfg,
                )
            except Exception:
                _wb = None
                _wb_run = None
                _wb_enabled = False

        # local helper to satisfy API shape requested in issue
        def log_wandb(d: dict):  # noqa: F811 - shadows train.log_wandb intentionally in this scope
            if _wb_enabled and _wb_run is not None:
                try:
                    _wb.log(d)
                except Exception:
                    pass
        # Reset states for fair comparison
        model.load_state_dict(model_baseline, strict=True)
        optimizers = orig_optimizers
        for oi, opt in enumerate(optimizers):
            opt.load_state_dict(copy.deepcopy(opt_baselines[oi]))
        set_lrs(scalar)
        data_generator.reset()
        # Prepare snapshots for tracked groups (all groups)
        swept_keys = list(all_keys)
        prev_snapshots: Dict[str, List[torch.Tensor]] = {}
        for oi2, gi2, g2 in _enumerate_param_groups(optimizers):
            k2 = _group_key(oi2, gi2, g2)
            if k2 in swept_keys:
                prev_snapshots[k2] = [p.detach().clone() for p in g2["params"]]

        ema_delta = None
        t_updates = 0
        prev_val = None
        start_val = None
        end_val = None
        early = False
        reason = ""
        # Track tokens processed within this scale (global tokens across ranks per optimizer step)
        tokens_processed = 0
        tokens_per_optim_step = int(getattr(data_generator, "batch_size", 0)) * int(accum_steps)

        for step in range(steps_per_scale):
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)

            total = 0.0
            for _ in range(accum_steps):
                train, target =  next(data_generator)
                loss = model(
                    input_seq=train,
                    target_seq=target,
                    sliding_window_num_blocks=get_num_window_blocks(
                        window_schedule,
                        attention_window_len=train_attention_window_len,
                        window_block_size=window_block_size,
                    ),
                )
                (loss / accum_steps).backward()
                total += float(loss.detach().item())

            if clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

            # Compute grad stats per swept group (after clipping, before step)
            grad_norms_by_group: Dict[str, float] = {}
            grad_max_by_group: Dict[str, float] = {}
            grad_count_by_group: Dict[str, int] = {}
            total_count_by_group: Dict[str, int] = {}
            eff_lr_by_group: Dict[str, float] = {}
            eff_wd_by_group: Dict[str, float] = {}
            for oi2, gi2, g2 in _enumerate_param_groups(optimizers):
                k2 = _group_key(oi2, gi2, g2)
                params = g2["params"]
                total_count_by_group[k2] = len(params)
                with_grad = [p for p in params if p.grad is not None]
                grad_count_by_group[k2] = len(with_grad)
                norms = [p.grad.detach().float().norm().item() for p in with_grad]
                grad_norms_by_group[k2] = (sum(norms) / len(norms)) if norms else float("nan")
                # Max abs grad across the group
                if with_grad:
                    try:
                        gmax = max(float(p.grad.detach().abs().max().item()) for p in with_grad)
                    except Exception:
                        gmax = float("nan")
                else:
                    gmax = float("nan")
                grad_max_by_group[k2] = gmax
                # Effective lr and weight decay as applied this step
                eff_lr_by_group[k2] = float(g2.get("lr", float("nan")))
                eff_wd_by_group[k2] = float(g2.get("weight_decay", float("nan")))

            # Step the optimizers
            for opt in optimizers:
                opt.step()

            # Compute mean absolute param delta since previous step and update snapshots
            param_delta_by_group: Dict[str, float] = {}
            for oi2, gi2, g2 in _enumerate_param_groups(optimizers):
                k2 = _group_key(oi2, gi2, g2)
                prev_list = prev_snapshots.get(k2)
                if prev_list is None:
                    # Initialize if missing (shouldn't happen, but guard anyway)
                    prev_list = [p.detach().clone() for p in g2["params"]]
                    prev_snapshots[k2] = prev_list
                total_abs = 0.0
                total_elems = 0
                for p_curr, p_prev in zip(g2["params"], prev_list):
                    diff = (p_curr.detach() - p_prev).abs()
                    total_abs += float(diff.sum().item())
                    total_elems += diff.numel()
                param_delta_by_group[k2] = (total_abs / total_elems) if total_elems > 0 else float("nan")
                # Update snapshots to current params for next step
                prev_snapshots[k2] = [p.detach().clone() for p in g2["params"]]

            # Print diagnostics for each swept group with high precision
            for k2 in grad_norms_by_group.keys():
                name2 = group_infos[k2].get("name") or k2
                gnorm = grad_norms_by_group.get(k2, float("nan"))
                gmax = grad_max_by_group.get(k2, float("nan"))
                with_grad = grad_count_by_group.get(k2, 0)
                total_params = total_count_by_group.get(k2, 0)
                eff_lr = eff_lr_by_group.get(k2, float("nan"))
                eff_wd = eff_wd_by_group.get(k2, float("nan"))
                pdelta = param_delta_by_group.get(k2, float("nan"))
                print(
                    f"[diag] step={step+1:03d}/{steps_per_scale:03d} group={name2} "
                    f"with_grad={with_grad}/{total_params} grad_norm_mean={gnorm:.12e} grad_max_abs={gmax:.12e} "
                    f"eff_lr={eff_lr:.12e} eff_wd={eff_wd:.12e} param_delta_mean={pdelta:.12e}",
                    flush=True,
                )

            val = total / accum_steps
            # advance token counter for this optimizer step (after GA)
            if tokens_per_optim_step:
                tokens_processed += tokens_per_optim_step
            # per-iteration logging to wandb (if enabled)
            log_wandb({
                "train/loss": float(val),
                "tokens": int(tokens_processed),
                "lr_scale": float(scalar),
            })
            # capture start and latest values for improvement metric
            if start_val is None:
                start_val = val
            end_val = val

            if prev_val is not None:
                delta = prev_val - val  # positive = improvement
                if ema_delta is None:
                    ema_delta = delta
                    t_updates = 1
                else:
                    ema_delta = smooth * ema_delta + (1 - smooth) * delta
                    t_updates += 1

                # Debias the EMA: ema_hat = ema / (1 - smooth^t)
                denom = 1.0 - (smooth ** t_updates)
                ema_debiased = ema_delta / denom if denom > 1e-12 else float("nan")

                # early stop check within this scale window (maximize EMA(delta))
                if not math.isfinite(ema_debiased):
                    early = True
                    reason = "non-finite EMA(delta)"
                    break
                if math.isfinite(global_best) and step > 4 and ema_debiased < (1.0 - blowup_pct) * global_best:
                    early = True
                    reason = "EMA(delta) dropped below blowup threshold vs global best"
                    break
            prev_val = val

        # record results for this scale (debiased EMA and total improvement)
        denom = (1.0 - (smooth ** t_updates)) if (ema_delta is not None and t_updates > 0) else None
        ema_out = (ema_delta / denom) if (denom and denom > 1e-12) else float("nan")
        improvement = (start_val - end_val) if (start_val is not None and end_val is not None) else float("nan")

        lrs_scalars.append(scalar)
        losses.append(ema_out)
        improvements.append(improvement)

        # update global best (maximize EMA(delta))
        if math.isfinite(ema_out) and ema_out > global_best:
            global_best = ema_out

        # periodic outer-loop progress line
        elapsed = time.time() - t0
        done = i + 1
        pct = 100.0 * done / max(1, num_scales)
        eta_s = (elapsed / max(1e-9, done)) * max(0, num_scales - done)
        eff_lrs_map = {
            (group_infos[k].get("name") or k): ((group_infos[k]["base_lr"] * scalar) if k in scaled_keys else group_infos[k]["base_lr"])
            for k in all_keys
        }
        eff_lrs_str = ", ".join(f"{n}={v:.3e}" for n, v in eff_lrs_map.items()) if eff_lrs_map else "none"
        if early:
            print(
                f"[lr_sweep] {done}/{num_scales} ({pct:.1f}%) ema_delta={ema_out:.6f} improvement={improvement:.6f} early={early} reason={reason} eff_lrs=[{eff_lrs_str}] scale={scalar:.3e} eta={eta_s:.1f}s",
            )
        print(
            f"[lr_sweep] {done}/{num_scales} ({pct:.1f}%) ema_delta={ema_out:.6f} improvement={improvement:.6f} eff_lrs=[{eff_lrs_str}] scale={scalar:.3e} eta={eta_s:.1f}s",
            flush=True,
        )
        # Finish wandb run for this scale
        if _wb_enabled and _wb is not None:
            try:
                _wb.finish()
            except Exception:
                pass

    # summary print
    if losses:
        i_max = max(range(len(losses)), key=lambda i: losses[i])
        best_scalar = lrs_scalars[i_max]
        best_eff_lrs_map = {
            (group_infos[k].get("name") or k): ((group_infos[k]["base_lr"] * best_scalar) if k in scaled_keys else group_infos[k]["base_lr"])
            for k in all_keys
        }
        eff_best_str = ", ".join(f"{n}={v:.3e}" for n, v in best_eff_lrs_map.items()) if best_eff_lrs_map else "none"
        print(
            f"[lr_sweep] collected {len(losses)} points; max EMA(delta) at step {i_max+1} ema_delta={losses[i_max]:.6f} eff_lrs=[{eff_best_str}] scale={best_scalar:.3e}",
            flush=True,
        )
    else:
        print("[lr_sweep] collected 0 points", flush=True)

    meta = {
        "groups": group_infos,
        "steps_per_scale": steps_per_scale,
        "num_scales": num_scales,
        "scale_min": float(scale_min),
        "scale_max": float(scale_max),
        "metric": "ema_delta_loss_debiased",
        "scaled_keys": list(scaled_keys),
        "scaled_names": requested_names,
    }
    return lrs_scalars, losses, improvements, meta


if __name__ == "__main__":
    from models import get_model_class, model_from_spec
    from training.hparams import load_hparams_from_yaml
    from dataclasses import asdict
    import os

    device = "cuda"
    parser = ArgumentParser("Sweep learning rate scales across optimizer param groups")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML training config.")
    parser.add_argument("--num_scales", type=int, default=10)
    parser.add_argument("--steps_per_scale", type=int, default=200, help="Number of steps per scale (total steps = num_scales * steps_per_scale)")
    parser.add_argument("--scale_min", type=float, default=0.5, help="Multiplicative LR scale min")
    parser.add_argument("--scale_max", type=float, default=5.0, help="Multiplicative LR scale max")
    parser.add_argument("--accum_steps", type=int, default=1, help="Grad accumulation steps")
    parser.add_argument("--clip_norm", type=float, default=None)
    parser.add_argument("--smooth", type=float, default=0.85)
    parser.add_argument("--blowup_pct", type=float, default=0.30)
    parser.add_argument(
        "--group",
        "-g",
        required=True,
        help="Comma-separated list of parameter group names to scale (e.g., embed_params,attn_params). Unspecified groups keep their configured LR.",
    )
    cli = parser.parse_args()

    from training.hparams import load_hparams_from_yaml

    params = load_hparams_from_yaml(cli.config)
    model = model_from_spec(params.model_spec, device=device)
    model = torch.compile(model, dynamic=False)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    data_loader = DistributedDataGenerator(
        params.train_shards,
        world_size * params.training_sequence_length,
        rank=rank,
        world_size=world_size,
        device=device,
    )

    from training.optim import build_optimizers_from_cfg

    # Parse comma-separated groups; apply scaling only to these groups. Others remain at config LR.
    selected_groups = [g.strip() for g in str(cli.group).split(",") if g.strip()]
    if not selected_groups:
        raise SystemExit("--group must specify at least one parameter group name")

    # Build optimizers with ALL groups; do not freeze unspecified groups
    optimizers = build_optimizers_from_cfg(
        cfg_list=params.optimizers, model=model, rank=rank, world_size=world_size
    )


    # Resolve deprecated aliases
    n_scales = cli.num_scales
    sc_min = cli.scale_min
    sc_max = cli.scale_max

    lrs, losses, improvements, meta = lr_sweep(
        model,
        optimizers=optimizers,
        data_generator=data_loader,
        train_attention_window_len=params.train_attention_window_len,
        window_block_size=128,
        num_scales=n_scales,
        scale_min=sc_min,
        scale_max=sc_max,
        steps_per_scale=cli.steps_per_scale,
        accum_steps=cli.accum_steps,
        clip_norm=cli.clip_norm,
        smooth=cli.smooth,
        blowup_pct=cli.blowup_pct,
        scaled_group_names=selected_groups,
        wandb_project=getattr(params, "wandb_project", "") or None,
        wandb_run_name=getattr(params, "wandb_run_name", "") or None,
        wandb_log=bool(getattr(params, "wandb_log", False)),
        wandb_extra_config=asdict(params),
    )

    # Log JSON and print a table focused on effective LRs (scale reported secondarily)
    groups = meta.get("groups", {})
    def _name_of(k: str) -> str:
        info = groups[k]
        return info.get("name") or k
    scaled_keys = meta.get("scaled_keys", [])
    swept_names = [_name_of(k) for k in scaled_keys]

    def _eff_lrs_for_scalar(s: float) -> Dict[str, float]:
        # effective LRs for scaled groups only
        return { _name_of(k): (float(groups[k]["base_lr"]) * float(s)) for k in scaled_keys }

    results = [
        {
            "index": i,
            "ema": float(l),  # debiased EMA(delta)
            "improvement": float(imp),  # start_loss - end_loss
            "effective_lrs": _eff_lrs_for_scalar(s),
            "scale": float(s),
        }
        for i, (s, l, imp) in enumerate(zip(lrs, losses, improvements))
    ]
    meta_out = dict(meta)
    meta_out["swept_groups"] = swept_names
    meta_out["base_lrs"] = { _name_of(k): float(groups[k]["base_lr"]) for k in groups.keys() }

    log_obj = {"results": results, "meta": meta_out}
    print(json.dumps(log_obj, indent=2))
    # Write to logs dir
    logs_dir = Path(__file__).resolve().parent.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_path = logs_dir / f"lr_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(log_obj, f, indent=2)
    print(f"[lr_sweep] wrote JSON results to {out_path}")

    # Print ASCII table: Idx, EMA_delta (debiased), Improvement, per-swept-group effective LRs, then Scale
    header_cols = [f"{name}_lr" for name in swept_names]
    header = "\nIdx  EMA_delta  Improvement  " + ("  ".join(f"{h:>12}" for h in header_cols) if header_cols else "(no_swept_groups)") + "  Scale"
    print(header)
    for i, (s, l, imp) in enumerate(zip(lrs, losses, improvements)):
        eff = _eff_lrs_for_scalar(s)
        eff_cols = "  ".join(f"{eff.get(name, float('nan')):12.3e}" for name in swept_names) if swept_names else "(none)"
        print(f"{i:3d}  {l:10.6f}  {imp:12.6f}  {eff_cols}  {s:10.3e}")


