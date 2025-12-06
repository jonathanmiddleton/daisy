
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


@dataclass
class Prompt:
    id: str
    text: Optional[str]
    tokens: List[int]


def _hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_parquet_or_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path))
    if pd is not None:
        try:
            df = pd.DataFrame(rows)
            if path.endswith(".parquet"):
                try:
                    df.to_parquet(path, index=False)
                    return
                except Exception:
                    pass
            # Fallback: CSV
            csv_path = path[:-8] + ".csv" if path.endswith(".parquet") else path
            df.to_csv(csv_path, index=False)
            return
        except Exception:
            pass
    # Minimal fallback without pandas
    # Write CSV
    csv_path = path[:-8] + ".csv" if path.endswith(".parquet") else path
    if not rows:
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("")
        return
    keys = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_prompts_jsonl(path: str, tokenizer=None, max_prompts: Optional[int] = None) -> List[Prompt]:
    out: List[Prompt] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_prompts is not None and i >= max_prompts:
                break
            d = json.loads(line)
            pid = d.get("id") or d.get("prompt_id") or f"p{i:05d}"
            text = d.get("text") or d.get("prompt")
            toks = d.get("tokens")
            if toks is None:
                if tokenizer is None:
                    raise ValueError("Tokenizer required to tokenize prompt text")
                toks = tokenizer(text)
            out.append(Prompt(id=str(pid), text=text, tokens=[int(t) for t in toks]))
    return out


def prompts_from_token_shards(pattern: str, device: str, count: int, bucket_lens: List[int]) -> List[Prompt]:
    # Sample simple prompts from the token shards via DistributedDataGenerator
    from data.data_gen_stream import DistributedDataGenerator
    import torch

    gen = DistributedDataGenerator(filename_pattern=pattern, sequence_length=128, rank=0, world_size=1, device=device)
    out: List[Prompt] = []
    for i in range(count):
        inputs, _ = next(gen)
        ids = inputs.to("cpu", dtype=torch.int64)
        # pick a random length bucket
        L = bucket_lens[min(len(bucket_lens) - 1, i % len(bucket_lens))]
        pid = f"shard_{i:06d}"
        out.append(Prompt(id=pid, text=None, tokens=inputs[:L].tolist()))
    return out


def write_prompts_manifest(path: str, prompts: List[Prompt]) -> None:
    rows = [{"id": p.id, "hash": _hash_text(p.text or ""), "len": len(p.tokens)} for p in prompts]
    write_jsonl(path, rows)
