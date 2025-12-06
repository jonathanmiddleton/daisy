import hashlib
from pathlib import Path
import warnings
import itertools
import json
import torch
from numpy import load as np_load

from data import DataGeneratorProtocol

from tools.master_logger import MasterLogger
logger = MasterLogger


class _Shard:
    def __init__(self, d: Path):
        self.dir = Path(d)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        assert self.meta["magic"] == 20240520
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "The given NumPy array is not writable.*"
                ),
                category=UserWarning,
            )
            self.tokens: torch.Tensor = torch.from_numpy(np_load(self.dir / "tokens.npy", mmap_mode="r")).to(torch.int32)
            self.labels: torch.Tensor = torch.from_numpy(np_load(self.dir / "labels.npy", mmap_mode="r")).to(torch.int64)
            self.offsets: torch.Tensor = torch.from_numpy(np_load(self.dir / "offsets.npy", mmap_mode="r")).to(torch.int32)
        assert self.tokens.shape[0] == self.labels.shape[0] == int(self.offsets[-1])
        assert self.meta["version"] >= 2
        self.pad_id = int(self.meta["eos_id"])

    def __len__(self): return len(self.offsets) - 1

    def next_sample(self, index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        s, e = int(self.offsets[index]), int(self.offsets[index + 1])
        return self.tokens[s:e], self.labels[s:e], e-s


def _batch_to_tensors(batch:list[tuple[torch.Tensor,torch.Tensor]], T: int, pad_id: int, ignore_index: int = -100, debug_logging: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    assert batch is not None and len(batch) > 0
    x0, y0 = batch[0]
    device = x0.device
    dtype_x = x0.dtype
    dtype_y = y0.dtype

    x = torch.full((T,), pad_id, dtype=dtype_x, device=device)
    y = torch.full((T,), ignore_index, dtype=dtype_y, device=device)

    offset = 0
    for frag_x, frag_y in batch:
        assert frag_x.size(0) == frag_y.size(0)
        frag_len = frag_x.size(0)

        remaining = T - offset
        assert remaining > 0
        x[offset: offset + frag_len] = frag_x[:frag_len]
        y[offset: offset + frag_len] = frag_y[:frag_len]
        offset += frag_len

    if offset < T and debug_logging: logger.debug(f"Padded batch with {T - offset} tokens.")

    return x, y

def _stable_int_from_name(name: str) -> int:
    h = hashlib.sha256(name.encode("utf-8")).digest()
    # take first 8 bytes as an unsigned 64-bit int
    return int.from_bytes(h[:8], "big")


class TaskDataGenerator(DataGeneratorProtocol):
    def __init__(self, root: str, split: str, sequence_length: int, world_size: int = 1, rank: int = 0,
                 seed: int = 1337, device: str = "cpu", start_shard: int | None = None):
        p = Path(root) / split
        self.files = sorted([d for d in p.iterdir() if d.is_dir() and (d / "meta.json").exists()])
        if not self.files: raise FileNotFoundError(f"no shards in {p}")
        if sequence_length == 1: logger.error("sequence_length=1 is likely mistaken for batch dimension or example count instead of token length.")
        self.sequence_length = sequence_length
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.seed = int(seed)
        self.device = torch.device(device)
        i0 = (start_shard or 0) % len(self.files)
        # Preserve the chosen file ordering so we can reset back to it.
        self._files_ordered = self.files[i0:] + self.files[:i0]
        self._file_iter = itertools.cycle(self._files_ordered)
        self._shard: _Shard | None = None
        self._order: torch.Tensor | None = None
        self._start_pos = self.rank
        self._pos = self._start_pos
        self._pad_id = None
        self._debug_logging = logger.isDebugEnabled()
        self._use_non_blocking = str(self.device).startswith("cuda") and torch.cuda.is_available()

    def _load_next(self):
        d = next(self._file_iter)
        self._shard = _Shard(d)
        n = len(self._shard)
        self._pad_id = self._shard.pad_id

        shard_salt = _stable_int_from_name(d.name) & 0xFFFFFFFF
        gen = torch.Generator(device="cpu").manual_seed(self.seed ^ shard_salt)

        self._order = torch.randperm(n, generator=gen)
        self._pos = self._start_pos
        assert (self._pos % self.world_size == self.rank)

    def reset(self):
        self._file_iter = itertools.cycle(self._files_ordered)
        self._shard = None
        self._order = None
        self._pos = self._start_pos
        assert(self._pos % self.world_size == self.rank)
        self._pad_id = None

    def __iter__(self):
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._shard is None: self._load_next()
        b = []
        need = self.sequence_length
        MAX_RETRIES = 1000
        retries = 0
        while need > 0 and retries < MAX_RETRIES:
            if self._pos >= self._order.size(0):
                self._load_next()
            idx = self._order[self._pos]
            self._pos += self.world_size
            assert(self._pos % self.world_size == self.rank)
            x, y, T = self._shard.next_sample(idx)
            if T > need:
                if not b:
                    retries += 1
                    continue
                else: break

            b.append((x,y))
            need -= T
        if not b:
            logger.warning(f"Stopping generation: unable to find example with sequence_length < {self.sequence_length} after {MAX_RETRIES} retries.")
            raise StopIteration
        x, y = _batch_to_tensors(batch=b, T=self.sequence_length, pad_id=self._pad_id, ignore_index=-100, debug_logging=self._debug_logging)

        x = x.to(self.device, non_blocking=self._use_non_blocking)
        y = y.to(self.device, non_blocking=self._use_non_blocking)
        return x, y
