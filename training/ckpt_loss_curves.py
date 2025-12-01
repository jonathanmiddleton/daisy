import glob
import os
import re
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch
from torch import nn


def _extract_first_int_from_string(s: str) -> Optional[int]:
    m = re.search(r"(\d+)", os.path.basename(s))
    return int(m.group(1)) if m else None


def evaluate_checkpoints(
    model: nn.Module,
    checkpoint_pattern: str,
    val_loader: Iterable[Any],
    compute_loss_and_tokens: Callable[[nn.Module, Any, str], tuple[torch.Tensor, int]],
    device: str = "cuda",
    max_checkpoints: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Evaluate all checkpoints matching glob pattern on val_loader.

    Arguments:
        model: an instance of the model with the right architecture.
        checkpoint_pattern: e.g. 'checkpoints/step_*.pt'.
        val_loader: iterable of validation batches.
        compute_loss_and_tokens: function(model, batch, device) -> (loss, n_tokens).
        device: device string.
        max_checkpoints: optional cap on number of checkpoints to evaluate.

    Returns:
        List of dicts with keys:
            'path', 'step', 'tokens_seen', 'val_loss'
    """
    device_obj = torch.device(device)
    model.to(device_obj)

    ckpts = sorted(glob.glob(checkpoint_pattern))
    if max_checkpoints is not None:
        ckpts = ckpts[:max_checkpoints]

    if not ckpts:
        raise RuntimeError(f"No checkpoints match pattern: {checkpoint_pattern}")

    results: List[Dict[str, Any]] = []

    for path in ckpts:
        state = torch.load(path, map_location=device_obj)

        # Try to be flexible about checkpoint format
        if isinstance(state, dict) and "model" in state:
            model.load_state_dict(state["model"])
            meta = {k: v for k, v in state.items() if k != "model"}
        elif isinstance(state, dict) and all(isinstance(v, torch.Tensor) for v in state.values()):
            model.load_state_dict(state)
            meta = {}
        else:
            raise ValueError(f"Unrecognized checkpoint format for {path}")

        model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in val_loader:
                loss, n_tokens = compute_loss_and_tokens(model, batch, device)
                total_loss += float(loss.item()) * int(n_tokens)
                total_tokens += int(n_tokens)

        avg_loss = total_loss / total_tokens

        step = meta.get("step") or meta.get("global_step") or _extract_first_int_from_string(path)
        tokens_seen = meta.get("tokens_seen") or meta.get("num_tokens") or None

        print(
            f"{os.path.basename(path):30s} "
            f"step={step}  tokens_seen={tokens_seen}  val_loss={avg_loss:.4f}"
        )

        results.append(
            {
                "path": path,
                "step": step,
                "tokens_seen": tokens_seen,
                "val_loss": avg_loss,
            }
        )

    # Sort by tokens_seen if available, else by step
    def sort_key(r: Dict[str, Any]):
        key = r["tokens_seen"]
        if key is None:
            key = r["step"] if r["step"] is not None else 0
        return key

    results.sort(key=sort_key)
    return results

"""
model = ...  
val_loader = ...   

results = evaluate_checkpoints(
    model,
    "checkpoints/step_*.pt",
    val_loader,
    compute_loss_and_tokens,
    device="cuda",
)"""