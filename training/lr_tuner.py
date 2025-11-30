import math
import statistics
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Union

import torch
from torch import nn
from torch.optim import Optimizer


OptimizerOrIterable = Union[Optimizer, Iterable[Optimizer]]


def _as_optimizer_list(optimizers: OptimizerOrIterable) -> List[Optimizer]:
    if isinstance(optimizers, Optimizer):
        return [optimizers]
    return list(optimizers)


def lr_from_update_ratio(
    model: nn.Module,
    optimizers: OptimizerOrIterable,
    batch_iter: Iterable[Any],
    compute_loss_and_tokens: Callable[[nn.Module, Any, str], tuple[torch.Tensor, int]],
    device: str = "cuda",
    num_steps: int = 10,
    target_ratio: float = 1e-3,
) -> Dict[str, Any]:
    """
    Measure per-group update-to-weight ratios and suggest LR multipliers.

    Arguments:
        model:
        optimizers: a torch.optim.Optimizer or iterable of them
        batch_iter: an iterator over batches (e.g. iter(dataloader)).
        compute_loss_and_tokens: function(model, batch, device) -> (loss, n_tokens).
        device: device string, e.g. "cuda" or "cpu".
        num_steps: number of batches to sample.
        target_ratio: desired median ||Δw||/||w||.

    Returns:
        dict with:
            - 'global_median_ratio'
            - 'global_scale'
            - 'per_group': list of per-group stats with suggested LRs.
    """
    device_obj = torch.device(device)
    model.to(device_obj)
    model.train()

    opt_list = _as_optimizer_list(optimizers)
    it = iter(batch_iter)

    # For each (optimizer, group) we’ll collect ratios across steps
    per_group_ratios = defaultdict(list)

    for _ in range(num_steps):
        try:
            batch = next(it)
        except StopIteration:
            break

        # Zero grads
        for opt in opt_list:
            opt.zero_grad(set_to_none=True)

        loss, _ = compute_loss_and_tokens(model, batch, device)
        loss.backward()

        for opt_idx, opt in enumerate(opt_list):
            for group_idx, group in enumerate(opt.param_groups):
                lr = float(group.get("lr", 0.0))
                if lr == 0.0:
                    continue

                grad_sq = 0.0
                weight_sq = 0.0
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad.detach()
                    w = p.detach()
                    grad_sq += float(torch.sum(g * g).item())
                    weight_sq += float(torch.sum(w * w).item())

                if grad_sq == 0.0 or weight_sq == 0.0:
                    continue

                grad_norm = math.sqrt(grad_sq)
                weight_norm = math.sqrt(weight_sq)
                ratio = lr * grad_norm / weight_norm

                name = group.get("name", f"opt{opt_idx}_group{group_idx}")
                key = (opt_idx, group_idx, name, lr)
                per_group_ratios[key].append(ratio)

    if not per_group_ratios:
        raise RuntimeError("No ratios collected - check that gradients are non-zero.")

    per_group_stats = []
    all_medians = []

    for (opt_idx, group_idx, name, lr), rs in per_group_ratios.items():
        med = statistics.median(rs)
        all_medians.append(med)
        per_group_stats.append(
            {
                "optimizer_index": opt_idx,
                "group_index": group_idx,
                "group_name": name,
                "current_lr": lr,
                "median_ratio": med,
            }
        )

    global_median = statistics.median(all_medians)
    global_scale = target_ratio / global_median

    # Attach suggested LRs
    for s in per_group_stats:
        local_scale = target_ratio / s["median_ratio"]
        s["suggested_lr_local_scale"] = s["current_lr"] * local_scale
        s["suggested_lr_global_scale"] = s["current_lr"] * global_scale

    # Nicely print summary
    print("=== Update-to-weight ratios (median over batches) ===")
    print(f"Global median ratio: {global_median:.3e}")
    print(f"Global LR scale to hit target={target_ratio:.1e}: {global_scale:.3e}")
    for s in sorted(per_group_stats, key=lambda d: d["median_ratio"], reverse=True):
        print(
            f"{s['group_name']:30s} "
            f"lr={s['current_lr']:.3e}  "
            f"median_ratio={s['median_ratio']:.3e}  "
            f"lr(local)={s['suggested_lr_local_scale']:.3e}  "
            f"lr(global)={s['suggested_lr_global_scale']:.3e}"
        )

    return {
        "global_median_ratio": global_median,
        "global_scale": global_scale,
        "per_group": per_group_stats,
    }
