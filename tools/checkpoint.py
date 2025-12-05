from ctypes import DEFAULT_MODE
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import Module
from torch.types import FileLike

from models import model_from_spec

STANDARD_KEYS = {
    "model",  # state_dict of the model
    "hparams",  # dict of hyperparameters used to build the model
    "step",  # int: global step when saved
    "best_val",  # float: best validation loss so far
}

DEFAULT_PREFIX = "_orig_mod."


@dataclass
class LoadedCheckpoint:
    model: Dict[str, torch.Tensor]
    hparams: Dict[str, Any]
    step: Optional[int]
    best_val: Optional[float]
    tokens_per_step: Optional[int] = None
    progress_state: Optional[Dict[str, Any]] = None


def remove_prefix(state_dict: Dict[str, Any], prefix: str = DEFAULT_PREFIX) -> Dict[str, Any]:
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_sd[k[len(prefix):]] = v
        else:
            new_sd[k] = v
    return new_sd

def restore_prefix(state_dict: Dict[str, Any], prefix: str = DEFAULT_PREFIX) -> Dict[str, Any]:
    """
        Compiled models expect weights with DEFAULT_PREFIX.
    """
    if not isinstance(state_dict, dict):
        return state_dict
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_sd[k] = v
        else:
            new_sd[prefix+k] = v
    return new_sd


def _normalize(obj: Any) -> LoadedCheckpoint:
    # Accept either a bare state_dict or a dict with our standard layout (or a variant).
    if isinstance(obj, dict) and "model" in obj:
        model_sd = obj.get("model")
        hparams = obj.get("hparams") or {}
        step = obj.get("step")
        best_val = obj.get("best_val")
        tokens_per_step = obj.get("tokens_per_step")
        progress_state = obj.get("progress_state")
        if not isinstance(hparams, dict):
            hparams = {}
        if not isinstance(model_sd, dict):
            # Some checkpoints might accidentally store the whole module
            try:
                model_sd = model_sd.state_dict()  # type: ignore[attr-defined]
            except Exception:
                raise TypeError("Unsupported checkpoint format: 'model' field is not a state_dict or module")
        return LoadedCheckpoint(model=model_sd, hparams=hparams, step=step, best_val=best_val,
                                tokens_per_step=tokens_per_step, progress_state=progress_state)

    # Fallback: treat as a raw state_dict
    if isinstance(obj, dict):
        return LoadedCheckpoint(model=obj, hparams={}, step=None, best_val=None)

    # Last resort: try to call state_dict on it
    try:
        sd = obj.state_dict()  # type: ignore[attr-defined]
        return LoadedCheckpoint(model=sd, hparams={}, step=None, best_val=None)
    except Exception as e:
        raise TypeError(f"Unsupported checkpoint object: {type(obj)}") from e


def load_checkpoint(path: FileLike, map_location: Any | None = None, strip_prefix: bool = True) -> LoadedCheckpoint:
    obj = torch.load(path, map_location=map_location)
    ckpt = _normalize(obj)
    if strip_prefix:
        ckpt.model = remove_prefix(ckpt.model)
    return ckpt


def save_checkpoint(
        path: str,
        model: nn.Module | Dict[str, Any],
        hparams: Dict[str, Any],
        step: Optional[int] = None,
        best_val: Optional[float] = None,
        progress_state: Optional[Dict[str, Any]] = None,
) -> None:
    # Accept either a module or a state_dict for model
    if isinstance(model, nn.Module):
        # If torch.compile was used, original module may be under _orig_mod
        model_to_state = getattr(model, "_orig_mod", model)
        model_sd = model_to_state.state_dict()
    elif isinstance(model, dict):
        model_sd = model
    else:
        raise TypeError("model must be an nn.Module or a state_dict dict")

    payload = dict(
        step=step,
        model=model_sd,
        best_val=best_val,
        hparams=hparams,
        progress_state=progress_state,
    )
    torch.save(payload, path)


def apply_model_state(model: nn.Module, state_dict: Dict[str, Any], strict: bool = True, assign: bool = True) -> Tuple[
    List[str], List[str]]:
    # Convenience to load with common prefix stripping already handled by load_checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=strict, assign=assign)
    return list(missing), list(unexpected)


def peek_hparams(path: str, map_location: Any | None = None) -> Dict[str, Any]:
    obj = torch.load(path, map_location=map_location)
    ckpt = _normalize(obj)
    return ckpt.hparams or {}


def model_from_checkpoint(path: FileLike, device: torch.device | str, dynamic_shapes: bool = False) -> tuple[Module, dict[str, Any]]:
    ckpt = load_checkpoint(path, map_location=device)
    state_dict = ckpt.model
    hparams = ckpt.hparams

    model = model_from_spec(hparams['model_spec'], device=device, dynamic_shapes=dynamic_shapes)
    apply_model_state(model, state_dict, strict=True, assign=True)

    return model, hparams
