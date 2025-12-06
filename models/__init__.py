from typing import Any, Type
from importlib import import_module
from dataclasses import is_dataclass, asdict as dc_asdict, fields as dc_fields, asdict
from torch import nn
from model_specs import load_model_spec, ModelSpec, override_model_spec

def is_flex_available():
    import torch
    return torch.cuda.is_available()

def _coerce_module_path(path: str):
    if path == "models.gpt2.gpt_core":
        return "models.daisy.daisy_core"
    return path

def _coerce_model_name(model_name: str):
    if model_name == "GPT2Core":
        return "DaisyCore"
    return model_name

def get_model_class(model_class: str) -> Type[nn.Module]:
    """Import and return a model class given its fully-qualified class name.
    Example: 'models.daisy.daisy_core.DaisyCore'. No defaults or fallbacks.
    """
    if not model_class or not isinstance(model_class, str):
        raise ValueError("model_class must be a non-empty fully-qualified class name string")
    if "." not in model_class:
        raise ValueError("model_class must be a fully-qualified class name like 'models.daisy.daisy_core.DaisyCore'")
    module_path, cls_name = model_class.rsplit(".", 1)
    mod = import_module(_coerce_module_path(module_path))
    cls = getattr(mod, _coerce_model_name(cls_name), None)
    if cls is None:
        raise ImportError(f"Class '{cls_name}' not found in module '{module_path}' for model_class '{model_class}'")
    return cls



def model_from_spec(spec_or_cfg: str | dict | ModelSpec | Any, device: str, overrides: dict = None) -> nn.Module:
    # Normalize to ModelSpec for validation of architecture fields
    spec: ModelSpec

    if isinstance(spec_or_cfg, str):
        # YAML -> ModelSpec (load_model_spec already validates)
        spec = load_model_spec(spec_or_cfg)
    elif isinstance(spec_or_cfg, ModelSpec):
        spec = spec_or_cfg
    elif isinstance(spec_or_cfg, dict):
        cfg = dict(spec_or_cfg)
        # Keep extras for aux, but filter to ModelSpec fields for construction
        allowed = {f.name for f in dc_fields(ModelSpec)}
        spec_data = {k: v for k, v in cfg.items() if k in allowed}
        spec_data.setdefault("attention_window_len", 4096)
        spec = ModelSpec(**spec_data)
    elif is_dataclass(spec_or_cfg):
        cfg = dc_asdict(spec_or_cfg)
        allowed = {f.name for f in dc_fields(ModelSpec)}
        spec_data = {k: v for k, v in cfg.items() if k in allowed}
        spec = ModelSpec(**spec_data)
    else:
        raise ValueError(f"Invalid spec_or_cfg type: {type(spec_or_cfg)}")

    if overrides is not None:
        spec = override_model_spec(spec,overrides)

    # Pull fields from validated ModelSpec
    model_class = str(spec.model_class)
    vocab_size = int(spec.vocab_size)
    eos_token_id = int(spec.eos_token_id)
    num_layers = int(spec.num_layers)
    num_heads = int(spec.num_heads)
    model_dim = int(spec.model_dim)
    head_dim = int(spec.head_dim)
    max_seq_len = int(spec.max_seq_len)
    use_value_embeddings = bool(spec.use_value_embeddings)
    use_tied_embeddings = bool(spec.use_tied_embeddings)
    attn_all_layers = bool(spec.attn_all_layers)
    attn_impl = str(spec.attn_impl)

    ModelClass = get_model_class(model_class)
    model: nn.Module = ModelClass(
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        model_dim=model_dim,
        max_seq_len=max_seq_len,
        head_dim=head_dim,
        eos_token_id=eos_token_id,
        desc=asdict(spec),
        use_value_embeddings=use_value_embeddings,
        use_tied_embeddings=use_tied_embeddings,
        attn_all_layers = attn_all_layers,
        attn_impl=attn_impl
    ).to(device)
    return model


__all__ = [
    "get_model_class",
    "model_from_spec",
    "is_flex_available",
]
