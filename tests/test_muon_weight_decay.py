import io
from pathlib import Path

import pytest
import torch

from training.hparams import load_hparams_from_yaml
from models import model_from_spec
from training.optim import build_optimizers_from_cfg


@pytest.mark.cpu
@torch.no_grad()
def test_muon_weight_decay_override_from_yaml(tmp_path: Path):
    # Create a minimal training config based on config/test_tiny.yml but override Muon weight_decay
    yml = f"""
# Logging
wandb_log: false
wandb_project: 'daisy'
wandb_run_name: 'test-tiny'
# Logging interval in tokens
log_interval: 65536
# Pretraining configuration
train_shards: "data/fineweb/fineweb_train_*.bin"
target_tokens: 1000000
cooldown_frac: 0.7
learning_rate_schedule: linear_decay
training_sequence_length: 1024
train_attention_window_len: 1024
# Optimizers configuration
optimizers:
  - type: AdamW
    betas: [0.9, 0.95]
    eps: 1.0e-10
    weight_decay: 0.1
    fused: false
    params:
      - group: head_params
        lr: 0.02
      - group: embed_params
        lr: 0.3
      - group: scalar_params
        lr: 0.015

  - type: Muon
    momentum: 0.95
    weight_decay: 0.123
    params:
      - group: hidden_matrix_params
        lr: 0.025
full_windows: false
# Gradient accumulation
grad_acc_steps: 1
# Evaluations
val_shards:
  - type: "fineweb"
    path: "data/fineweb/fineweb_val_*.bin"
    target_tokens: 10485760
    sequence_length: 262144
val_loss_every_tokens: 65536
# Snapshots
snapshot_warmup_tokens: 0
snapshot_per_n_tokens: 0
save_checkpoint: false
# Model Definition
model_spec: test-tiny-model
"""
    cfg_path = tmp_path / "override_muon.yml"
    cfg_path.write_text(yml)

    # Load hparams and build a tiny model from spec
    args = load_hparams_from_yaml(str(cfg_path))
    model = model_from_spec(args.model_spec, device='cpu').to(torch.bfloat16)

    opts = build_optimizers_from_cfg(cfg_list=args.optimizers, model=model, rank=0, world_size=1)

    # Find Muon optimizer
    muon_opts = [opt for opt in opts if opt.__class__.__name__ == 'Muon']
    assert len(muon_opts) == 1
    muon = muon_opts[0]

    # Assert weight_decay applied to its param group equals the override value
    for pg in muon.param_groups:
        assert abs(pg["weight_decay"] - 0.123) < 1e-12


@pytest.mark.cpu
@torch.no_grad()
def test_muon_requires_weight_decay_in_yaml(tmp_path: Path):
    # Same as above but omit weight_decay under Muon to assert ValueError (breaking change)
    yml = f"""
# Logging
wandb_log: false
wandb_project: 'daisy'
wandb_run_name: 'test-tiny'
# Logging interval in tokens
log_interval: 65536
# Pretraining configuration
train_shards: "data/fineweb/fineweb_train_*.bin"
target_tokens: 1000000
cooldown_frac: 0.7
learning_rate_schedule: linear_decay
training_sequence_length: 1024
train_attention_window_len: 1024
# Optimizers configuration
optimizers:
  - type: AdamW
    betas: [0.9, 0.95]
    eps: 1.0e-10
    weight_decay: 0.1
    fused: false
    params:
      - group: head_params
        lr: 0.02
      - group: embed_params
        lr: 0.3
      - group: scalar_params
        lr: 0.015

  - type: Muon
    momentum: 0.95
    params:
      - group: hidden_matrix_params
        lr: 0.025
full_windows: false
# Gradient accumulation
grad_acc_steps: 1
# Evaluations
val_shards:
  - type: "fineweb"
    path: "data/fineweb/fineweb_val_*.bin"
    target_tokens: 10485760
    sequence_length: 262144
val_loss_every_tokens: 65536
# Snapshots
snapshot_warmup_tokens: 0
snapshot_per_n_tokens: 0
save_checkpoint: false
# Model Definition
model_spec: test-tiny-model
"""
    cfg_path = tmp_path / "muon_missing_wd.yml"
    cfg_path.write_text(yml)

    args = load_hparams_from_yaml(str(cfg_path))
    model = model_from_spec(args.model_spec, device='cpu').to(torch.bfloat16)

    with pytest.raises(ValueError, match="requires 'weight_decay'\s*to be set explicitly"):
        _ = build_optimizers_from_cfg(cfg_list=args.optimizers, model=model, rank=0, world_size=1)
