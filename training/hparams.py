import dataclasses
from dataclasses import dataclass, fields as dataclass_fields
from pathlib import Path
from typing import List, Optional

import yaml

from model_specs import load_model_spec
from tools.helpers import _coerce_value

WINDOW_BLOCK_SIZE = 128

@dataclass
class Hyperparameters:
    # Required scenario-specific fields

    target_tokens: int
    cooldown_frac: float
    # Learning rate schedule selection
    learning_rate_schedule: str  # {'linear_decay','linear_warmup_cosine_decay','cosine_decay'}
    train_attention_window_len: int  # training-time sliding attention window (<= model spec max)
    attention_window_len: int  # maximum sliding attention window supported by the model # TODO merge with train_attention_window_len
    # Common fields with defaults
    vocab_size: int
    eos_token_id: int
    val_loss_every_tokens: int  # num tokens between validation passes (0 disables)
    checkpoint_warmup_tokens: int  # tokens to skip before taking checkpoints
    checkpoint_per_n_tokens: int  # interval in tokens between checkpoints (0 = every update after warmup)
    save_checkpoint: bool
    num_layers: int
    num_heads: int
    model_dim: int
    head_dim: int
    optimizers: list[dict]
    # Force full attention windows (useful when resuming after smaller windows)
    full_windows: bool
    # Model context size (from ModelSpec); used to instantiate DaisyCore
    max_seq_len: int
    # Gradient accumulation
    grad_acc_steps: int
    # Model selection
    model_spec: str    # name of model spec under model_specs/, or a path to a spec file
    model_class: str  # fully-qualified class name, e.g., 'models.daisy.daisy_core.DaisyCore'
    # Torch compile/tuning flags
    torch_coordinate_descent_tuning: bool = False
    # Weights & Biases minimal logging config
    wandb_log: bool = False
    wandb_project: str = ""
    wandb_run_name: str = ""
    # Optional Weights & Biases group name (for organizing runs in UI)
    wandb_group: Optional[str] = None
    # Optional minimum LR as a fraction of the initial LR (0..1). When 0.5, LR won't go below 50% of initial.
    learning_rate_floor: float = 0.0
    init_checkpoint: str | None = None
    use_value_embeddings: bool = True
    use_tied_embeddings: bool = False
    # Global RNG seed for training (applied to torch and CUDA). Defaults to 1337.
    seed: int = 1337
    train_mode: str = "pretrain"  # "pretrain" or "task"
    train_shards: str = ""
    # List of evaluation datasets with descriptive type and path glob
    val_shards: list[dict] = dataclasses.field(default_factory=list)

    # Task SFT train data
    task_train_root: Optional[str] = None  # e.g. "data/instruct_tasks"
    task_train_split: str = "train"  # "train" / "val" / "test"
    task_global_batch_size: int = 0  # total examples across all ranks
    # Task SFT val data
    task_val_root: Optional[str] = None  # e.g. "data/instruct_tasks"
    task_val_split: str = "val"
    task_val_global_batch_size: int = 0  # if 0, default to task_global_batch_size
    task_val_debug_log_samples: bool = False

    # Each item: {type: str, path: str, split: str = 'val', target_tokens: int}
    task_val_shards: list[dict] = dataclasses.field(default_factory=list)

    # Interval in tokens for emitting training logs (logger and wandb). Must be > 0.
    log_interval: int = 16384
    # Training Defaults
    training_sequence_length: int = 16384
    # Optional global learning-rate scale applied to every configured param-group lr
    # This can be overridden from the CLI as: lr_scale=0.5 (or comma-separated values via runner)
    lr_scale: float = 1.0


def load_hparams_from_yaml(config_path: str, *, validate: bool = True) -> Hyperparameters:
    """
    Load Hyperparameters from a YAML file. If a 'model_spec' key is present, also load and merge
    the named spec from model_specs/<name>.yml (or a provided file path). Training config values
    take precedence over spec values. Validates against the Hyperparameters dataclass.
    """
    global WINDOW_BLOCK_SIZE
    cfg_dict = {}

    used_path = Path(config_path)
    try:
        with open(used_path, "r") as f:
            cfg_dict = yaml.safe_load(f) or {}
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Config file not found: {used_path}") from e

    # Normalize/alias keys that use dots for namespacing
    if "torch.coordinate_descent_tuning" in cfg_dict and "torch_coordinate_descent_tuning" not in cfg_dict:
        cfg_dict["torch_coordinate_descent_tuning"] = cfg_dict.pop("torch.coordinate_descent_tuning")

    # Backward compatibility: alias old snapshot_* keys to checkpoint_* keys
    if "snapshot_warmup_tokens" in cfg_dict and "checkpoint_warmup_tokens" not in cfg_dict:
        cfg_dict["checkpoint_warmup_tokens"] = cfg_dict.pop("snapshot_warmup_tokens")
    if "snapshot_per_n_tokens" in cfg_dict and "checkpoint_per_n_tokens" not in cfg_dict:
        cfg_dict["checkpoint_per_n_tokens"] = cfg_dict.pop("snapshot_per_n_tokens")

    # If a model_spec name/path is provided, load the spec and merge recognized fields
    model_spec_name = cfg_dict.get("model_spec")
    spec_dict = None
    if model_spec_name:
        from dataclasses import asdict
        spec_dict = asdict(load_model_spec(str(model_spec_name))) # returns a dict
        # Merge ModelSpec fields into cfg.
        valid_names_for_merge = {f.name for f in dataclass_fields(Hyperparameters)}
        for k, v in spec_dict.items():
            if k not in valid_names_for_merge:
                continue
            elif k not in cfg_dict or cfg_dict[k] is None:
                cfg_dict[k] = v

    # Validate keys after potential spec merge
    valid_names = {f.name for f in dataclass_fields(Hyperparameters)}
    unknown = set(cfg_dict) - valid_names
    if unknown:
        raise ValueError(f"Unknown hyperparameter(s) in {used_path}: {sorted(unknown)}")

    required = [f.name for f in dataclass_fields(Hyperparameters)
                if f.default is dataclasses.MISSING and getattr(f, "default_factory",
                                                               dataclasses.MISSING) is dataclasses.MISSING]
    missing = [name for name in required if name not in cfg_dict]

    if missing:
        raise ValueError(f"Missing required hyperparameter(s) in {used_path}: {missing}")

    #TODO properly validate : train/val or task_train/task_val is set along with below

    # Normalize and validate pretraining-style validation shards (used in both modes if provided)
    if "val_shards" in cfg_dict:
        vcfg = cfg_dict.get("val_shards")
        if not isinstance(vcfg, list) or len(vcfg) == 0:
            raise ValueError("val_shards must be a non-empty list of objects with keys: path (str), type (optional str), target_tokens (int), sequence_length (int)")
        norm_list: list[dict] = []
        for i, item in enumerate(vcfg, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"val_shards[{i}] must be a mapping")
            path = item.get("path")
            vtype = item.get("type") or f"val{i}"
            t_tokens = item.get("target_tokens")
            seq_len = item.get("sequence_length")
            if not isinstance(path, str) or not path:
                raise ValueError(f"val_shards[{i}].path must be a non-empty string")
            if not isinstance(vtype, str) or not vtype:
                raise ValueError(f"val_shards[{i}].type must be a non-empty string when provided")
            try:
                t_tokens = int(t_tokens)
                if t_tokens <= 0:
                    raise ValueError
            except Exception:
                raise ValueError(f"val_shards[{i}].target_tokens must be a positive integer")
            try:
                seq_len = int(seq_len)
                if seq_len <= 0:
                    raise ValueError
            except Exception:
                raise ValueError(f"val_shards[{i}].sequence_length must be a positive integer")
            norm_list.append({
                "type": vtype,
                "path": path,
                "target_tokens": t_tokens,
                "sequence_length": seq_len,
            })
        cfg_dict["val_shards"] = norm_list

    # Task SFT validation shards normalization (hierarchical style)
    if cfg_dict.get("train_mode") == "task":
        tvs = cfg_dict.get("task_val_shards", [])
        norm_task_list: list[dict] = []
        if not isinstance(tvs, list):
            raise ValueError("task_val_shards must be a list of objects with keys: path (str), split (str, optional), type (str, optional), target_tokens (int), sequence_length (int)")
        for i, item in enumerate(tvs, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"task_val_shards[{i}] must be a mapping")
            path = item.get("path")
            vtype = item.get("type") or f"task_val{i}"
            split = item.get("split", "val")
            t_tokens = item.get("target_tokens")
            seq_len = item.get("sequence_length")
            if not isinstance(path, str) or not path:
                raise ValueError(f"task_val_shards[{i}].path must be a non-empty string")
            if not isinstance(split, str) or not split:
                raise ValueError(f"task_val_shards[{i}].split must be a non-empty string when provided")
            try:
                t_tokens = int(t_tokens)
                if t_tokens <= 0:
                    raise ValueError
            except Exception:
                raise ValueError(f"task_val_shards[{i}].target_tokens must be a positive integer")
            try:
                seq_len = int(seq_len)
                if seq_len <= 0:
                    raise ValueError
            except Exception:
                raise ValueError(f"task_val_shards[{i}].sequence_length must be a positive integer")
            norm_task_list.append({
                "type": vtype,
                "path": path,
                "split": split,
                "target_tokens": t_tokens,
                "sequence_length": seq_len,
            })
        cfg_dict["task_val_shards"] = norm_task_list

    args = Hyperparameters(**cfg_dict)

    # Optionally validate now; train.py may defer validation until after CLI overrides are applied
    if validate:
        validate_hparams(args)
    return args


def apply_cli_overrides(args: Hyperparameters, override_args: List[str]) -> Hyperparameters:
    """
    Apply CLI overrides of the form --key=value or key=value to an existing Hyperparameters object.
    Unknown keys are ignored to allow passing through unrelated CLI args (e.g., torchrun's).
    Values are coerced to the target type using tools.helpers._coerce_value.
    """
    field_map = {f.name: f for f in dataclass_fields(Hyperparameters)}
    for s in override_args:
        if s.startswith("--"):
            s = s[2:]
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip().replace("-", "_").replace(".", "_")
        if k not in field_map:
            # Ignore unknown keys to allow torchrun args if any; alternatively, raise
            continue
        f = field_map[k]
        coerced = _coerce_value(v.strip(), f.type)
        setattr(args, k, coerced)
    return args


def validate_hparams(args: Hyperparameters) -> Hyperparameters:
    """
    Validate a fully-populated Hyperparameters object. This can be invoked after CLI overrides
    are applied to ensure constraints reflect final values.
    Returns the same args for convenience.
    """
    try:
        tsl = int(args.training_sequence_length)
        tawt = int(args.train_attention_window_len)
        gas = int(args.grad_acc_steps)
        cdf = float(args.cooldown_frac)
        vlet = int(args.val_loss_every_tokens)
        logi = int(args.log_interval)
        spnt = int(args.checkpoint_per_n_tokens)
        swt = int(args.checkpoint_warmup_tokens)
    except Exception as e:
        raise ValueError(f"Invalid types in Hyperparameters: {e}")
    if tsl % WINDOW_BLOCK_SIZE != 0:
        raise ValueError(f"training_sequence_length ({tsl}) must be divisible by window_block_size ({WINDOW_BLOCK_SIZE})")
    if tawt % WINDOW_BLOCK_SIZE != 0:
        raise ValueError(f"train_attention_window_len ({tawt}) must be divisible by window_block_size ({WINDOW_BLOCK_SIZE})")
    if tsl < tawt:
        raise ValueError(f"training_sequence_length ({tsl}) must be >= train_attention_window_len ({tawt})")
    # Enforce training window <= model's supported max if model_spec is available
    try:
        from model_specs import load_model_spec
        from dataclasses import asdict as _asdict
        spec = _asdict(load_model_spec(str(args.model_spec)))
        spec_max = int(spec.get("attention_window_len", tawt))
        if tawt > spec_max:
            raise ValueError(
                f"train_attention_window_len ({tawt}) must be <= model_spec.attention_window_len ({spec_max})"
            )
    except Exception:
        # If spec loading fails here, defer to downstream model construction errors
        pass
    if gas < 1:
        raise ValueError(f"grad_acc_steps must be >= 1, got {gas}")
    if not (0.0 <= cdf <= 1.0):
        raise ValueError(f"cooldown_frac must be in [0,1], got {cdf}")
    # Optional: learning rate floor in [0,1]
    try:
        lrf = float(args.learning_rate_floor)
    except Exception:
        raise ValueError(f"learning_rate_floor must be a number in [0,1], got {args.learning_rate_floor}")
    if not (0.0 <= lrf <= 1.0):
        raise ValueError(f"learning_rate_floor must be in [0,1], got {lrf}")
    if vlet < 0:
        raise ValueError(f"val_loss_every_tokens must be >= 0, got {vlet}")
    if logi <= 0:
        raise ValueError(f"log_interval must be > 0, got {logi}")
    if spnt < 0:
        raise ValueError(f"checkpoint_per_n_tokens must be >= 0, got {spnt}")
    if swt < 0:
        raise ValueError(f"checkpoint_warmup_tokens must be >= 0, got {swt}")
    # Validate learning rate schedule option against optim dispatch table
    try:
        from training.optim import LEARNING_RATE_SCHEDULES
        valid_schedules = set(LEARNING_RATE_SCHEDULES.keys())
    except Exception:
        valid_schedules = {"linear_decay", "linear_warmup_cosine_decay", "cosine_decay"}
    if args.learning_rate_schedule not in valid_schedules:
        raise ValueError(
            f"learning_rate_schedule must be one of {sorted(valid_schedules)}, got '{args.learning_rate_schedule}'"
        )
    return args


__all__ = [
    "Hyperparameters",
    "load_hparams_from_yaml",
    "apply_cli_overrides",
    "validate_hparams",
]
