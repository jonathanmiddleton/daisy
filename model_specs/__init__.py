from pathlib import Path

import yaml

from .model_spec import ModelSpec, build_model_dict


def resolve_model_spec_path(name_or_path: str) -> Path:
    """
    Resolve a model spec identifier to a concrete file path.
    Accepts:
      - Absolute/relative file paths to a YAML file.
      - Bare names referencing files inside this package directory (with or without .yml/.yaml),
        including any subdirectories.
    Returns a Path that exists or raises FileNotFoundError.
    """
    # Direct path first
    p = Path(name_or_path)
    if p.exists():
        return p

    # Try relative to this package directory (recursively)
    base = Path(__file__).resolve().parent

    # If name with extension, search for that filename under base
    if "." in name_or_path:
        for path in base.rglob(name_or_path):
            if path.is_file():
                return path
    else:
        # Try adding common YAML extensions under package dir, recursively
        for candidate_name in (name_or_path, f"{name_or_path}.yml", f"{name_or_path}.yaml"):
            for path in base.rglob(candidate_name):
                if path.is_file():
                    return path

    # As a last attempt, allow bare files in current working directory with YAML extensions
    cwd1 = Path(f"{name_or_path}.yml")
    cwd2 = Path(f"{name_or_path}.yaml")
    if cwd1.exists():
        return cwd1
    if cwd2.exists():
        return cwd2

    raise FileNotFoundError(
        f"Model spec '{name_or_path}' not found under {base} or as a file path"
    )


def override_model_spec(spec: ModelSpec, overrides: dict):
    for k, v in overrides.items():
        if hasattr(spec, k):
            setattr(spec, k, v)
    return spec


def load_model_spec(name_or_path: str, overrides: dict | None = None) -> ModelSpec:
    """
    Load and parse a model spec YAML into a ModelSpec dataclass.
    Raises FileNotFoundError if the spec cannot be resolved.
    """
    spec_path = resolve_model_spec_path(name_or_path)
    with open(spec_path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Model spec '{name_or_path}' must be a mapping, got {type(data).__name__}")
    # Construct a ModelSpec (this will raise if required fields are missing or unknown keys are present)
    spec_dict = build_model_dict(data, ctx=str(name_or_path))
    spec = ModelSpec(**spec_dict)
    spec = override_model_spec(spec, overrides) if overrides else spec
    return spec


__all__ = [
    "resolve_model_spec_path",
    "load_model_spec",
    "ModelSpec",
    "override_model_spec"
]