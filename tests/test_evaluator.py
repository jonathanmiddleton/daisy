import pytest
import torch
from torch import nn

from training.eval import Evaluator


class FakeDataGenerator:
    """
    Minimal stand-in for DistributedDataGenerator that yields predictable per-rank slices
    and counts how many times it's been iterated. This lets us validate that Evaluator:
      - Computes steps from the GLOBAL batch_size
      - Calls next(generator) exactly `steps` times per rank
      - Forwards generator-produced inputs/targets to the model
    """

    def __init__(self, *, batch_size: int, world_size: int, rank: int):
        assert batch_size % world_size == 0
        self.batch_size = int(batch_size)        # GLOBAL batch size (matches real generator API)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.local_batch_size = self.batch_size // self.world_size
        self.calls = 0

    def __iter__(self):
        return self

    def __next__(self):
        # Produce deterministic inputs/targets unique to this step and rank so the model can verify
        step_idx = self.calls
        # shape: [local_batch_size]
        base = self.rank * 10_000 + step_idx  # encode rank and step in the values
        inputs = torch.full((self.local_batch_size,), base, dtype=torch.int32)
        targets = inputs.to(torch.int64) + 1  # arbitrary "next token" mapping
        self.calls += 1
        return inputs, targets


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # dummy parameter only to define a device for Evaluator
        self.p = nn.Parameter(torch.zeros(()))
        self.reset_state()

    def reset_state(self):
        self.tokens_seen = 0
        self.seen_batches = []  # collect (inputs, targets) tuples for later assertions

    def forward(self, inputs: torch.Tensor, num_window_blocks: int, targets: torch.Tensor):
        # Record exactly what Evaluator/model consumed
        self.tokens_seen += inputs.numel()
        # Store small copies (CPU tensors already) for assertions
        self.seen_batches.append((inputs.clone(), targets.clone()))
        # Return a simple scalar loss dependent on inputs to avoid division-by-zero issues
        return inputs.to(torch.float32).mean()


@pytest.mark.parametrize("world_size", [1, 2, 4])
@pytest.mark.parametrize("global_batch_size", [8])
@pytest.mark.parametrize("steps", [1, 3, 7])
def test_eval_consumes_total_tokens_globally_across_world_sizes(world_size, global_batch_size, steps):
    # total tokens is in GLOBAL token units
    total_tokens = global_batch_size * steps

    # Simulate running Evaluator.eval() on each rank independently (without torch.distributed)
    # and then aggregating what each rank processed. This mirrors what would happen across
    # distributed processes where each rank consumes its local slice.
    per_rank_calls = []
    global_tokens_processed = 0

    for rank in range(world_size):
        ddg = FakeDataGenerator(batch_size=global_batch_size, world_size=world_size, rank=rank)
        evaluator = Evaluator(
            data_generator=ddg,
            distributed_enabled=False,  # not actually using torch.distributed in this unit test
            world_size=world_size,
            rank=rank,
            training_sequence_length=128,
        )
        model = ToyModel().eval()

        result = evaluator.eval(model=model, total_tokens=total_tokens)
        assert "val_loss" in result and isinstance(result["val_loss"], float)

        # Each rank should have been advanced exactly `steps` by the evaluator
        assert ddg.calls == steps
        per_rank_calls.append(ddg.calls)

        # Each rank processes local_batch_size tokens per step
        expected_local_tokens = (global_batch_size // world_size) * steps
        assert model.tokens_seen == expected_local_tokens
        global_tokens_processed += model.tokens_seen

    # Across all ranks, the total tokens processed equals the requested global total_tokens
    assert global_tokens_processed == total_tokens

    # Sanity: per-rank calls are all equal to steps
    assert all(c == steps for c in per_rank_calls)


def test_eval_uses_data_generator_batches_exactly():
    world_size = 2
    global_batch_size = 6
    steps = 5
    total_tokens = global_batch_size * steps

    # Check that the Evaluator forwards the generator's outputs to the model unchanged
    for rank in range(world_size):
        ddg = FakeDataGenerator(batch_size=global_batch_size, world_size=world_size, rank=rank)
        evaluator = Evaluator(
            data_generator=ddg,
            distributed_enabled=False,
            world_size=world_size,
            rank=rank,
            training_sequence_length=128,
        )
        model = ToyModel().eval()

        evaluator.eval(model=model, total_tokens=total_tokens)

        # The model should have seen exactly `steps` batches, each matching the FakeDataGenerator pattern
        assert len(model.seen_batches) == steps
        local_bs = global_batch_size // world_size
        for s, (inputs, targets) in enumerate(model.seen_batches):
            assert inputs.shape == (local_bs,)
            assert targets.shape == (local_bs,)
            expected_base = rank * 10_000 + s
            assert torch.all(inputs == expected_base)
            assert torch.all(targets == expected_base + 1)

        # And generator should have been advanced exactly `steps` times
        assert ddg.calls == steps
