import hashlib
import os, math, numpy as np, torch, itertools, json
from pathlib import Path

class _Shard:
    def __init__(self, d: Path):
        self.dir = Path(d)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        assert self.meta["magic"] == 20240520
        self.tokens = np.load(self.dir / "tokens.npy", mmap_mode="r")
        self.labels = np.load(self.dir / "labels.npy", mmap_mode="r")
        self.offsets = np.load(self.dir / "offsets.npy", mmap_mode="r")
        assert self.tokens.shape[0] == self.labels.shape[0] == int(self.offsets[-1])
        assert self.meta["version"] >= 2
        self.pad_id = int(self.meta["eos_id"])  # pad with EOS

    def __len__(self): return len(self.offsets) - 1

    def get(self, i: int):
        s, e = int(self.offsets[i]), int(self.offsets[i+1])
        return self.tokens[s:e], self.labels[s:e]

def next_multiple_of_n(v: int, *, n: int):
    return next(x for x in range(n, v + 1 + n, n) if x >= v)

# TODO super slow - rework like data_gen_stream with pinning
def _pad(batch, pad_id: int, length: int = 1):
    L = [len(x[0]) for x in batch]
    T = next_multiple_of_n(max(L), n=length)
    B = len(batch)
    x = torch.full((B, T), pad_id, dtype=torch.int32)
    y = torch.full((B, T), -100, dtype=torch.long)
    for i, (inp, lab) in enumerate(batch):
        t = len(inp)
        x[i, :t] = torch.from_numpy(inp.astype(np.int64, copy=False))
        y[i, :t] = torch.from_numpy(lab.astype(np.int64, copy=False))
    return x, y

def _stable_int_from_name(name: str) -> int:
    h = hashlib.sha256(name.encode("utf-8")).digest()
    # take first 8 bytes as an unsigned 64-bit int
    return int.from_bytes(h[:8], "big")

class TaskDataGenerator:
    def __init__(self, root: str,
                 split: str,
                 global_batch_size: int,
                 world_size: int = 1,
                 rank: int = 0,
                 seed: int = 1337,
                 device: str = "cpu",
                 start_shard: int | None = None,
                 drop_remainder: bool = False,
                 infinite: bool = True,
                 squeeze_singleton_batch: bool = True,
                 pad_to_multiple: int = 1
                 ):
        p = Path(root) / split
        self.files = sorted([d for d in p.iterdir() if d.is_dir() and (d / "meta.json").exists()])
        if not self.files: raise FileNotFoundError(f"no shards in {p}")
        assert global_batch_size % world_size == 0
        self.batch_size = int(global_batch_size)
        self.world_size = int(world_size)
        self.local_bsz = global_batch_size // world_size
        self.rank = int(rank)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.drop_remainder = drop_remainder
        self.infinite = infinite
        self.squeeze_single = bool(squeeze_singleton_batch)
        i0 = (start_shard or 0) % len(self.files)
        # Preserve the chosen file ordering so we can reset back to it.
        self._files_ordered = self.files[i0:] + self.files[:i0]
        self._file_iter = itertools.cycle(self._files_ordered)
        self._rng = np.random.default_rng(self.seed)
        self._shard = None
        self._order = None
        self._pos = 0
        self._pad_id = None
        self._pad_to_multiple = pad_to_multiple

    def _load_next(self):
        d = next(self._file_iter)
        self._shard = _Shard(d)
        n = len(self._shard)
        self._pad_id = self._shard.pad_id

        shard_salt = _stable_int_from_name(d.name) & 0xFFFFFFFF
        self._rng = np.random.default_rng(self.seed ^ shard_salt)

        self._order = self._rng.permutation(n).tolist()
        self._pos = 0

    def reset(self):
        self._file_iter = itertools.cycle(self._files_ordered)
        self._rng = np.random.default_rng(self.seed)
        self._shard = None
        self._order = None
        self._pos = 0
        self._pad_id = None

    def __iter__(self): return self

    def __next__(self):
        if self._shard is None: self._load_next()
        b = []
        need = self.local_bsz
        while need > 0:
            if self._pos >= len(self._order):
                if self.drop_remainder and b: break
                self._load_next()
                if not self.infinite and not b: raise StopIteration
            if self._pos >= len(self._order): continue
            idx = self._order[self._pos]; self._pos += 1
            if (self._pos - 1) % self.world_size != self.rank: continue
            b.append(self._shard.get(idx))
            need -= 1
        if len(b) < self.local_bsz and self.drop_remainder: raise StopIteration
        x, y = _pad(b, self._pad_id, self._pad_to_multiple)
        non_blocking = self.device.type == "cuda" and torch.cuda.is_available()
        x = x.to(self.device, non_blocking=non_blocking)
        y = y.to(self.device, non_blocking=non_blocking)
        if self.squeeze_single and x.size(0) == 1:
            return x[0], y[0]
        return x, y