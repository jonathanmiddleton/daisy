import json
from pathlib import Path

import numpy as np
import torch
import pytest

from data.data_gen_task import _Shard, TaskDataGenerator


def _write_shard(dir_path: Path, num_samples: int, sample_len: int, *, eos_id: int = 0, id_offset: int = 0):
    dir_path.mkdir(parents=True, exist_ok=True)
    # Offsets: [0, L, 2L, ..., N*L]
    offsets = np.arange(0, (num_samples + 1) * sample_len, sample_len, dtype=np.int32)
    # Create tokens/labels with a unique per-sample ID repeated across its length
    tokens = np.empty(num_samples * sample_len, dtype=np.int32)
    labels = np.empty(num_samples * sample_len, dtype=np.int64)
    for i in range(num_samples):
        sid = id_offset + i + 1  # avoid 0 which may be pad/eos
        s = i * sample_len
        e = s + sample_len
        tokens[s:e] = sid
        labels[s:e] = sid
    # Save arrays as proper .npy files
    np.save(dir_path / "tokens.npy", tokens)
    np.save(dir_path / "labels.npy", labels)
    np.save(dir_path / "offsets.npy", offsets)

    meta = {
        "magic": 20240520,
        "version": 2,
        "eos_id": int(eos_id),
    }
    (dir_path / "meta.json").write_text(json.dumps(meta))


def _build_dataset(tmp_path: Path, split: str, num_shards: int, samples_per_shard: int, sample_len: int) -> Path:
    root = tmp_path / "dataset"
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    id_offset = 0
    for s in range(num_shards):
        shard_dir = split_dir / f"shard_{s:03d}"
        _write_shard(shard_dir, samples_per_shard, sample_len, eos_id=0, id_offset=id_offset)
        id_offset += samples_per_shard
    return root


def _sample_id_from_batch(x: torch.Tensor) -> int:
    # All tokens in our synthetic sample are the same ID; take the first element
    return int(x.view(-1)[0].item())


def test__shard_basic_next_sample(tmp_path: Path):
    # Build a single shard with variable sample lengths and verify slicing
    shard_dir = tmp_path / "varlen_shard"
    shard_dir.mkdir(parents=True)
    # Create 3 samples with lengths 2, 3, 4
    lens = [2, 3, 4]
    offsets = np.array([0, 2, 5, 9], dtype=np.int32)
    tokens = np.array([11, 11, 22, 22, 22, 33, 33, 33, 33], dtype=np.int32)
    labels = tokens.astype(np.int64)
    np.save(shard_dir / "tokens.npy", tokens)
    np.save(shard_dir / "labels.npy", labels)
    np.save(shard_dir / "offsets.npy", offsets)
    (shard_dir / "meta.json").write_text(json.dumps({"magic": 20240520, "version": 2, "eos_id": 0}))

    sh = _Shard(shard_dir)
    assert len(sh) == 3
    # Check each sample
    for i, L in enumerate(lens):
        x, y, T = sh.next_sample(torch.tensor(i))
        assert T == L
        assert x.dtype == torch.int32 and y.dtype == torch.int64
        assert x.numel() == L and y.numel() == L
        # all tokens within a sample equal its marker id (11, 22, 33)
        expect = (i + 1) * 11
        assert torch.all(x == expect)
        assert torch.all(y.to(torch.int32) == expect)


@pytest.mark.parametrize("world_size", [2, 3])
def test_task_data_generator_distributed_non_overlap(tmp_path: Path, world_size: int):
    # Build dataset with 3 shards, each with 12 samples of length 5
    root = _build_dataset(tmp_path, split="train", num_shards=3, samples_per_shard=12, sample_len=5)
    sequence_length = 5
    seed = 777

    # Create per-rank generators
    gens = [
        TaskDataGenerator(str(root), "train", sequence_length=sequence_length, world_size=world_size, rank=r, seed=seed, device="cpu")
        for r in range(world_size)
    ]

    # Collect a fair number of samples per rank (spanning multiple shards)
    per_rank_take = 20
    produced: list[list[int]] = [[] for _ in range(world_size)]
    for r, g in enumerate(gens):
        it = iter(g)
        for _ in range(per_rank_take):
            x, y = next(it)
            sid = _sample_id_from_batch(x)
            produced[r].append(sid)

    # Check pairwise disjointness of sample IDs across ranks
    sets = [set(p) for p in produced]
    for i in range(world_size):
        for j in range(i + 1, world_size):
            assert sets[i].isdisjoint(sets[j]), f"Ranks {i} and {j} produced intersecting samples: {sets[i] & sets[j]}"


def test_task_data_generator_determinism_same_rank(tmp_path: Path):
    root = _build_dataset(tmp_path, split="train", num_shards=2, samples_per_shard=10, sample_len=7)
    sequence_length = 7
    seed = 12345

    g0a = TaskDataGenerator(str(root), "train", sequence_length, world_size=4, rank=2, seed=seed, device="cpu")
    g0b = TaskDataGenerator(str(root), "train", sequence_length, world_size=4, rank=2, seed=seed, device="cpu")

    a = [
        _sample_id_from_batch(next(iter(g0a))[0])
        for _ in range(15)
    ]
    b = [
        _sample_id_from_batch(next(iter(g0b))[0])
        for _ in range(15)
    ]
    assert a == b, "Same rank+seed should produce identical sequence of samples"
