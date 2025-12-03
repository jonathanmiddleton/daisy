import dataclasses
from dataclasses import dataclass, fields as dataclass_fields
from pathlib import Path
from typing import Any

import yaml

WINDOW_BLOCK_SIZE = 128

@dataclass
class ModelSpec:
    """
    Strict schema for model_specs/*.yml files.
    Only architecture-relevant fields live here. Training config belongs in the training YAML.
    """
    model_class: str
    vocab_size: int
    num_layers: int
    num_heads: int
    model_dim: int
    head_dim: int
    eos_token_id: int
    attention_window_len: int  # largest sliding attention window supported by the model
    max_seq_len: int  # maximum context size supported by the model
    use_value_embeddings: bool = True
    use_tied_embeddings: bool = False
    attn_all_layers: bool = False
    attn_impl: str = 'standard'

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelSpec":
        """
        Construct a ModelSpec from a dict, enforcing the same rules as YAML loading.
        """
        if not isinstance(data, dict):
            raise ValueError("ModelSpec.from_dict expects a dict")

        validated = build_model_dict(dict(data))
        # Only pass recognized fields (defensive, though build_model_dict already enforces keys)
        allowed = {f.name for f in dataclass_fields(cls)}
        kwargs = {k: v for k, v in validated.items() if k in allowed}
        return cls(**kwargs)

def _strict_keys(obj: dict[str, Any], allowed: set[str], ctx: str) -> None:
    unknown = set(obj) - allowed
    if unknown:
        raise ValueError(f"Unknown key(s) in {ctx}: {sorted(unknown)}")

def _require_keys(obj: dict[str, Any], required: set[str], ctx: str) -> None:
    missing = [k for k in required if k not in obj]
    if missing:
        raise ValueError(f"Missing required key(s) in {ctx}: {sorted(missing)}")

def _as_int(name: str, value: Any, ctx: str, min_value: int | None = 1) -> int:
    try:
        iv = int(value)
    except Exception:
        raise ValueError(f"{ctx}.{name} must be an integer; got {value!r}")
    if min_value is not None and iv < min_value:
        raise ValueError(f"{ctx}.{name} must be >= {min_value}; got {iv}")
    return iv

def _as_bool(name: str, value: Any, ctx: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, str)):
        v = str(value).lower()
        if v in ("1", "true", "yes", "y"):
            return True
        if v in ("0", "false", "no", "n"):
            return False
    raise ValueError(f"{ctx}.{name} must be a boolean; got {value!r}")

def _as_float(name: str, value: Any, ctx: str, min_value: float | None = None, max_value: float | None = None) -> float:
    try:
        fv = float(value)
    except Exception:
        raise ValueError(f"{ctx}.{name} must be a float; got {value!r}")
    if min_value is not None and fv < min_value:
        raise ValueError(f"{ctx}.{name} must be >= {min_value}; got {fv}")
    if max_value is not None and fv > max_value:
        raise ValueError(f"{ctx}.{name} must be <= {max_value}; got {fv}")
    return fv

def build_model_dict(spec_dict: dict, ctx: str | None) -> dict[str, Any]:
    global WINDOW_BLOCK_SIZE
    ctx = ctx or "model_spec"
    allowed = {f.name for f in dataclass_fields(ModelSpec)}
    required = {f.name for f in dataclass_fields(ModelSpec) if f.default is dataclasses.MISSING}
    _strict_keys(spec_dict, allowed, f"model_spec:{ctx}")
    _require_keys(spec_dict, required, f"model_spec:{ctx}")

    # ints
    for k in ("vocab_size", "num_layers", "num_heads", "model_dim", "head_dim", "eos_token_id",  "attention_window_len", "max_seq_len"):
        spec_dict[k] = _as_int(k, spec_dict[k], ctx, min_value=1)
    # strings
    if not isinstance(spec_dict["model_class"], str) or not spec_dict["model_class"]:
        raise ValueError(f"{ctx}.model_class must be a non-empty string")

    # Cross-field constraints
    if spec_dict["attention_window_len"] % WINDOW_BLOCK_SIZE != 0:
        raise ValueError(f"{ctx}: attention_window_len ({spec_dict['attention_window_len']}) must be divisible by window_block_size ({WINDOW_BLOCK_SIZE})")
    if spec_dict["max_seq_len"] < spec_dict["attention_window_len"]:
        raise ValueError(f"{ctx}: max_seq_len ({spec_dict['max_seq_len']}) must be >= attention_window_len ({spec_dict['attention_window_len']})")

    return spec_dict


def _validate_optimizers_schema(cfg_list: Any, ctx: str = "optimizers") -> list[dict[str, Any]]:
    if not isinstance(cfg_list, list) or not cfg_list:
        raise ValueError(f"{ctx} must be a non-empty list")

    out: list[dict[str, Any]] = []
    for i, item in enumerate(cfg_list):
        ictx = f"{ctx}[{i}]"
        if not isinstance(item, dict):
            raise ValueError(f"{ictx} must be a mapping/dict")
        allowed_top = {"type", "betas", "eps", "weight_decay", "fused", "momentum", "params"}
        required_top = {"type", "params"}
        _strict_keys(item, allowed_top, ictx)
        _require_keys(item, required_top, ictx)

        # Basic fields
        if not isinstance(item["type"], str) or not item["type"]:
            raise ValueError(f"{ictx}.type must be a non-empty string")

        if "betas" in item:
            betas = item["betas"]
            if (not isinstance(betas, (list, tuple))) or len(betas) != 2:
                raise ValueError(f"{ictx}.betas must be a 2-length list")
            b0 = _as_float("betas[0]", betas[0], ictx, min_value=0.0, max_value=0.9999999)
            b1 = _as_float("betas[1]", betas[1], ictx, min_value=0.0, max_value=0.9999999)
            item["betas"] = [b0, b1]

        if "eps" in item:
            item["eps"] = _as_float("eps", item["eps"], ictx, min_value=0.0)

        if "weight_decay" in item:
            item["weight_decay"] = _as_float("weight_decay", item["weight_decay"], ictx, min_value=0.0)

        if "momentum" in item:
            item["momentum"] = _as_float("momentum", item["momentum"], ictx, min_value=0.0, max_value=0.9999999)

        if "fused" in item:
            item["fused"] = _as_bool("fused", item["fused"], ictx)

        # params groups
        params = item["params"]
        if not isinstance(params, list) or not params:
            raise ValueError(f"{ictx}.params must be a non-empty list")
        validated_groups: list[dict[str, Any]] = []
        for j, g in enumerate(params):
            gctx = f"{ictx}.params[{j}]"
            if not isinstance(g, dict):
                raise ValueError(f"{gctx} must be a mapping/dict")
            allowed_g = {"group", "lr"}
            required_g = {"group", "lr"}
            _strict_keys(g, allowed_g, gctx)
            _require_keys(g, required_g, gctx)
            if not isinstance(g["group"], str) or not g["group"]:
                raise ValueError(f"{gctx}.group must be a non-empty string")
            g["lr"] = _as_float("lr", g["lr"], gctx, min_value=0.0)
            validated_groups.append({"group": g["group"], "lr": g["lr"]})
        item["params"] = validated_groups
        out.append(item)

    return out

