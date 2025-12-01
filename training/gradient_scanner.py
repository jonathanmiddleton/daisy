import math
from typing import Any, Callable, Dict

import torch
from torch import nn


def inspect_activations_and_gradients(
    model: nn.Module,
    batch: Any,
    compute_loss_and_tokens: Callable[[nn.Module, Any, str], tuple[torch.Tensor, int]],
    device: str = "cuda",
) -> Dict[str, Dict[str, float]]:
    """
    Run a single forward/backward pass with hooks to record activation and
    gradient statistics per module that has parameters.

    Returns:
        dict: module_name -> stats dict with keys:
            'act_rms', 'act_mean', 'act_max_abs',
            'grad_rms', 'grad_max_abs', 'grad_to_act'
    """
    device_obj = torch.device(device)
    model.to(device_obj)
    model.train()

    activations: Dict[str, Dict[str, float]] = {}
    gradients: Dict[str, Dict[str, float]] = {}

    def make_fwd_hook(name: str):
        def hook(module, inputs, output):
            with torch.no_grad():
                if isinstance(output, torch.Tensor):
                    t = output
                elif isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                    t = output[0]
                else:
                    return
                t = t.detach()
                activations[name] = {
                    "rms": float(t.pow(2).mean().sqrt().item()),
                    "mean": float(t.mean().item()),
                    "max_abs": float(t.abs().max().item()),
                }
        return hook

    def make_bwd_hook(name: str):
        def hook(module, grad_inputs, grad_outputs):
            with torch.no_grad():
                if not grad_outputs:
                    return
                go = grad_outputs[0]
                if not isinstance(go, torch.Tensor):
                    return
                g = go.detach()
                gradients[name] = {
                    "rms": float(g.pow(2).mean().sqrt().item()),
                    "max_abs": float(g.abs().max().item()),
                }
        return hook

    handles = []
    for name, module in model.named_modules():
        # Only hook modules that actually have parameters (layers, not containers)
        if len(list(module.parameters(recurse=False))) == 0:
            continue
        handles.append(module.register_forward_hook(make_fwd_hook(name)))
        handles.append(module.register_full_backward_hook(make_bwd_hook(name)))

    # Clear existing grads
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    loss, _ = compute_loss_and_tokens(model, batch, device)
    loss.backward()

    for h in handles:
        h.remove()

    combined: Dict[str, Dict[str, float]] = {}
    for name, act in activations.items():
        grad = gradients.get(name, {})
        grad_rms = grad.get("rms", float("nan"))
        grad_to_act = grad_rms / (act["rms"] + 1e-12) if not math.isnan(grad_rms) else float("nan")
        combined[name] = {
            "act_rms": act["rms"],
            "act_mean": act["mean"],
            "act_max_abs": act["max_abs"],
            "grad_rms": grad_rms,
            "grad_max_abs": grad.get("max_abs", float("nan")),
            "grad_to_act": grad_to_act,
        }

    # Simple textual summary: top 20 modules by grad_to_act
    def sort_key(item):
        name, stats = item
        val = stats["grad_to_act"]
        return (1 if math.isnan(val) else 0, -val)

    print("=== Top 20 modules by grad_to_act ratio ===")
    for name, stats in sorted(combined.items(), key=sort_key)[:20]:
        print(
            f"{name:40s} "
            f"act_rms={stats['act_rms']:.3e} "
            f"grad_rms={stats['grad_rms']:.3e} "
            f"grad_to_act={stats['grad_to_act']:.3e}"
        )

    return combined

"""
batch = next(iter(train_dataloader))
stats = inspect_activations_and_gradients(
    model,
    batch,
    compute_loss_and_tokens,
    device="cuda",
)
"""