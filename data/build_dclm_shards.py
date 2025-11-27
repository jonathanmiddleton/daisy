#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset, disable_caching
from tqdm import tqdm

MAGIC = 20240520
VERSION = 1


class DocTokenStream:
    def __init__(self, dataset_iter, enc):
        self._iter = iter(dataset_iter)
        self.enc = enc
        self.eot = enc.eot_token
        self._current: np.ndarray | None = None  # np.ndarray[uint16]
        self._pos = 0

    def _load_next_doc(self):
        while True:
            example = next(self._iter)  # may raise StopIteration
            text = example.get("text")
            if not isinstance(text, str) or not text:
                continue
            ids = [self.eot]
            ids.extend(self.enc.encode_ordinary(text))
            arr = np.asarray(ids, dtype=np.int32)
            if arr.size == 0:
                continue
            if not ((arr >= 0).all() and (arr < 2**16).all()):
                raise ValueError("Token ID out of uint16 range")
            self._current = arr.astype(np.uint16)
            self._pos = 0
            return

    def read_into(self, out: np.ndarray, offset: int, n: int) -> int:
        if n <= 0:
            return 0
        if offset < 0 or offset + n > out.shape[0]:
            raise ValueError("read_into bounds are invalid")
        written = 0
        while written < n:
            if self._current is None or self._pos >= len(self._current):
                try:
                    self._load_next_doc()
                except StopIteration:
                    break
            avail = len(self._current) - self._pos
            if avail <= 0:
                self._current = None
                self._pos = 0
                continue
            take = min(avail, n - written)
            out[offset + written : offset + written + take] = \
                self._current[self._pos : self._pos + take]
            self._pos += take
            written += take
        return written


def write_shard(path: Path, tokens: np.ndarray):
    num_tokens = int(tokens.size)
    if num_tokens == 0:
        return
    if num_tokens >= 2**31:
        raise ValueError("num_tokens too large for int32 header")
    header = np.zeros(256, dtype=np.int32)
    header[0] = MAGIC
    header[1] = VERSION
    header[2] = num_tokens
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header.tobytes())
        f.write(tokens.tobytes())


def build_dclm_shards(
    out_dir: Path,
    dataset_name: str,
    split: str,
    total_train_tokens: int,
    tokens_per_shard: int,
    val_tokens: int,
    encoding_name: str = "gpt2",
):
    if total_train_tokens <= 0:
        raise ValueError("total_train_tokens must be positive")
    if tokens_per_shard <= 0:
        raise ValueError("tokens_per_shard must be positive")
    if val_tokens < 0:
        raise ValueError("val_tokens must be nonnegative")

    disable_caching()  # avoid Arrow cache on disk

    ds = load_dataset(
        dataset_name,
        split=split,
        streaming=True,
    )

    enc = tiktoken.get_encoding(encoding_name)
    stream = DocTokenStream(ds, enc)

    shard_buf = np.empty(tokens_per_shard, dtype=np.uint16)
    print(f"Starting dataset_name={dataset_name}")
    # Validation shards
    remaining_val = val_tokens
    val_idx = 0
    while remaining_val > 0:
        target = min(tokens_per_shard, remaining_val)
        filled = stream.read_into(shard_buf, 0, target)
        if filled == 0:
            raise RuntimeError(
                "DCLM stream exhausted before finishing validation tokens"
            )
        fname = out_dir / f"dclm_baseline_val_{val_idx:06d}.bin"
        write_shard(fname, shard_buf[:filled])
        remaining_val -= filled
        val_idx += 1

    # Training shards
    remaining_train = total_train_tokens
    train_idx = 1
    pbar = tqdm(
        total=remaining_train,
        unit="tokens",
        desc="DCLM train tokens",
        smoothing=0.01,
    )

    while remaining_train > 0:
        target = min(tokens_per_shard, remaining_train)
        filled = stream.read_into(shard_buf, 0, target)
        if filled == 0:
            raise RuntimeError(
                "DCLM stream exhausted before reaching requested train tokens"
            )
        fname = out_dir / f"dclm_baseline_train_{train_idx:06d}.bin"
        write_shard(fname, shard_buf[:filled])
        remaining_train -= filled
        train_idx += 1
        pbar.update(filled)

    pbar.close()


def parse_args():
    p = argparse.ArgumentParser(
        description="Stream DCLM-baseline and build NanoGPT-style uint16 token shards."
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="dclm_baseline_gpt2_token_shards",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="mlfoundations/dclm-baseline-1.0-parquet",
    )
    p.add_argument(
        "--split",
        type=str,
        default="train",
    )
    p.add_argument(
        "--total-tokens",
        type=int,
        default=int(20e9),
        help="Number of training tokens to materialize.",
    )
    p.add_argument(
        "--tokens-per-shard",
        type=int,
        default=int(1e8),
        help="Max tokens per shard.",
    )
    p.add_argument(
        "--val-tokens",
        type=int,
        default=None,
        help="Number of validation tokens (default = tokens_per_shard). "
             "Set 0 to skip validation shard(s).",
    )
    p.add_argument(
        "--encoding",
        type=str,
        default="gpt2",
        help="tiktoken encoding name.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    val_tokens = args.val_tokens
    if val_tokens is None:
        val_tokens = args.tokens_per_shard

    build_dclm_shards(
        out_dir=out_dir,
        dataset_name=args.dataset,
        split=args.split,
        total_train_tokens=args.total_tokens,
        tokens_per_shard=args.tokens_per_shard,
        val_tokens=val_tokens,
        encoding_name=args.encoding,
    )


if __name__ == "__main__":
    main()
