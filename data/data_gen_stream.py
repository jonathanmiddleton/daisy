import itertools

import torch
from pathlib import Path
from tools.master_logger import MasterLogger

logger = MasterLogger

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32, device='cpu') # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        use_pin = bool(torch.cuda.is_available())
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=use_pin)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

class DistributedDataGenerator:
    def __init__(self, filename_pattern: str, sequence_length: int, rank: int, world_size: int, start_shard: int | None = None, device: str = "cuda"):
        # Resolve files matching the pattern; support absolute and relative paths
        if filename_pattern.startswith("/"):
            p = Path(filename_pattern)
            files = sorted(list(p.parent.glob(p.name)))
        else:
            files = sorted(Path.cwd().glob(filename_pattern))
        if not files:
            raise FileNotFoundError(f"No files matched pattern: {filename_pattern}")

        self.files: list[Path] = files
        self.world_size = int(world_size)
        self.sequence_length = int(sequence_length)
        self.global_batch_size = self.world_size * self.sequence_length
        self.rank = int(rank)
        self.device = device
        self._use_non_blocking = str(self.device).startswith("cuda") and torch.cuda.is_available()
        self.debug_logging = logger.isDebugEnabled()

        # Determine starting file index using a 1-based shard number with wrap-around safety
        if start_shard is not None:
            # Convert to zero-based index; allow any integer and wrap within [0, len(files))
            start_idx = (int(start_shard) - 1) % len(self.files)
        else:
            start_idx = 0

        # Create a cycling iterator starting from the chosen file
        files_ordered = self.files[start_idx:] + self.files[:start_idx]
        self._files_ordered = files_ordered
        self._file_iter = itertools.cycle(files_ordered)
        self._current_file: Path = next(self._file_iter)
        self._tokens: torch.Tensor = _load_data_shard(self._current_file)
        self._pos: int = 0

    def get_sequence_length(self):
        return self.sequence_length

    def reset(self):
        """Reset the generator to the beginning of the current file ordering."""
        self._file_iter = itertools.cycle(self._files_ordered)
        self._current_file = next(self._file_iter)
        self._tokens = _load_data_shard(self._current_file)
        self._pos = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._pos + self.global_batch_size + 1 >= len(self._tokens):
            if self.debug_logging: logger.debug(f"Current file: {self._current_file}")
            self._current_file = next(self._file_iter)
            self._tokens = _load_data_shard(self._current_file)
            self._pos = 0
        start = self._pos + self.rank * self.sequence_length
        buf = self._tokens[start:][: self.sequence_length + 1]
        inputs = buf[:-1].to(device=self.device, dtype=torch.int32, non_blocking=self._use_non_blocking)
        targets = buf[1:].to(device=self.device, dtype=torch.int64, non_blocking=self._use_non_blocking)
        self._pos += self.global_batch_size
        return inputs, targets





