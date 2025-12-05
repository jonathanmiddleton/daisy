import copy
import dataclasses
import os
from pathlib import Path

import pytest
import torch

from training.hparams import load_hparams_from_yaml, Hyperparameters
from training import trainer as trainer_mod


class TinyToyModel(torch.nn.Module):
    """A tiny model with the forward signature expected by trainer (inputs, n_blocks, targets)."""
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(8, 4)

    def forward(self, inputs, n_blocks, targets=None):
        # inputs: [B, T], but tests won't call this; signature only for compatibility
        x = inputs.float()
        if x.dim() == 1:
            x = x.unsqueeze(-1).repeat(1, 8)
        elif x.size(-1) != 8:
            # pad or truncate to 8 features deterministically for safety
            if x.size(-1) < 8:
                pad = torch.zeros(*x.shape[:-1], 8 - x.size(-1), device=x.device, dtype=x.dtype)
                x = torch.cat([x, pad], dim=-1)
            else:
                x = x[..., :8]
        y = self.lin(x)
        if targets is None:
            return y.abs().mean()
        # simple loss
        tgt = targets.float()
        if tgt.dim() < y.dim():
            tgt = tgt.unsqueeze(-1).expand_as(y)
        return torch.nn.functional.mse_loss(y, tgt)


def test_compute_group_max_seq_len_and_partitioning():
    project_root = Path(__file__).resolve().parents[1]
    cfg_path = project_root / "config" / "test" / "test_tiny_model.yml"

    # Load a real config to get a valid Hyperparameters instance
    base = load_hparams_from_yaml(str(cfg_path))

    a1 = dataclasses.replace(base)
    a1.training_sequence_length = 4096
    a1.val_shards = [{
        "type": "val1", "path": "data/dummy/*.bin", "target_tokens": 4096 * 4, "sequence_length": 8192
    }]

    a2 = dataclasses.replace(base)
    a2.training_sequence_length = 16384
    a2.val_shards = []

    # max should consider both training and validation sequence lengths
    max_len = trainer_mod.compute_group_max_seq_len([a1, a2])
    assert max_len == 16384 or max_len == 8192 or max_len == 16384  # sanity (16384 >= 8192)
    assert max_len == 16384

    # Compile key equality for runtime-only changes
    device_type = "cpu"
    k1 = trainer_mod.derive_compile_key(a1, device_type=device_type, world_size=1, dynamic=False)
    a1_lr = dataclasses.replace(a1)
    a1_lr.lr_scale = 0.5
    k1b = trainer_mod.derive_compile_key(a1_lr, device_type=device_type, world_size=1, dynamic=False)
    assert k1 == k1b, "Changing lr_scale must not alter compile key"

    # Changing architecture must change compile key
    a_arch = dataclasses.replace(a1)
    a_arch.num_layers = a1.num_layers + 2
    k2 = trainer_mod.derive_compile_key(a_arch, device_type=device_type, world_size=1, dynamic=False)
    assert k1 != k2

    # Partition groups should put a1 and a1_lr together, and a_arch separate
    groups = trainer_mod.partition_runs_by_compile_key([
        (0, a1), (1, a1_lr), (2, a_arch)
    ], device_type=device_type, world_size=1)
    sizes = sorted(len(v) for v in groups.values())
    assert sizes == [1, 2]


def test_compiled_runtime_reset_restores_initial_state(monkeypatch):
    # Disable torch.compile to keep test fast and CPU-only
    monkeypatch.setenv("TORCH_DISABLE_MODEL_COMPILE", "1")

    # Monkeypatch model_from_spec used inside CompiledRuntime to return our tiny model
    monkeypatch.setattr(trainer_mod, "model_from_spec", lambda *args, **kwargs: TinyToyModel())

    project_root = Path(__file__).resolve().parents[1]
    cfg_path = project_root / "config" / "test" / "test_tiny_model.yml"

    # Load a real config to get a valid Hyperparameters instance
    args = load_hparams_from_yaml(str(cfg_path))
    # Use tiny steps to avoid any large allocations if accidentally invoked
    args.training_sequence_length = 128
    args.max_seq_len = 128

    rt = trainer_mod.CompiledRuntime(args, dynamic=False)
    try:
        init_state = copy.deepcopy(rt.model.state_dict())
        # Mutate weights
        with torch.no_grad():
            for p in rt.model.parameters():
                p.add_(1.2345)
        # Ensure they differ
        diff_before = any(not torch.equal(init_state[k], v) for k, v in rt.model.state_dict().items())
        assert diff_before, "State must differ after mutation"
        # Reset to initial
        rt.reset_model_to_initial()
        # Check equality
        for k, v in rt.model.state_dict().items():
            assert torch.equal(init_state[k], v), f"Param {k} not restored"
    finally:
        rt.destroy()
