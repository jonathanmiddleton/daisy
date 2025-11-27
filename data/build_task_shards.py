import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json, math, random, numpy as np, re, time
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

MAGIC = 20240520
VERSION = 2

def _final_from_gsm8k(ans: str) -> str:
    m = re.findall(r"####\s*([^\n]+)", ans)
    return (m[-1] if m else ans).strip()

def _fmt(instr: str, resp: str):
    instr, resp = (instr or "").strip(), (resp or "").strip()
    if not instr or not resp:
        return None
    return instr, resp

def _iter_arc(subset: str, split: str):
    ds = load_dataset("allenai/ai2_arc", subset, split=split).shuffle(seed=1337)
    for r in ds:
        q = r["question"]
        labels = r["choices"]["label"]
        texts = r["choices"]["text"]
        key = r["answerKey"]
        instr = (
            q
            + "\nOptions:\n"
            + "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
            + "\nAnswer with the correct option letter."
        )
        resp = key
        y = _fmt(instr, resp)
        if y:
            yield y

def _iter_gsm8k(subset: str, split: str):
    ds = load_dataset("openai/gsm8k", subset, split=split).shuffle(seed=1337)
    for r in ds:
        instr = r["question"]
        resp = _final_from_gsm8k(r["answer"])
        y = _fmt(instr, resp)
        if y:
            yield y

def _iter_smoltalk(split: str, stop: int | None = None):
    ds = load_dataset("HuggingFaceTB/smol-smoltalk", split=split).shuffle(seed=1337)
    n = 0
    for r in ds:
        msgs = r.get("messages") or []
        u, a = None, None
        for m in msgs:
            if m.get("role") == "user" and u is None:
                u = m.get("content")
            elif m.get("role") == "assistant" and u is not None:
                a = m.get("content")
                break
        y = _fmt(u, a)
        if y:
            yield y
            n += 1
            if stop and n >= stop:
                break

def _iter_oasst1(split: str, stop: int | None = None):
    ds = load_dataset("OpenAssistant/oasst1", split=split)
    id2row = {r["message_id"]: r for r in ds}
    n = 0
    for r in ds:
        if r.get("role") == "assistant":
            p = id2row.get(r.get("parent_id"))
            if p and p.get("role") == "prompter" and (r.get("lang") or "en").startswith("en"):
                y = _fmt(p.get("text"), r.get("text"))
                if y:
                    yield y
                    n += 1
                    if stop and n >= stop:
                        break

def _iter_dolly15k(split: str, stop: int | None = None):
    ds = load_dataset("databricks/databricks-dolly-15k", split=split)
    n = 0
    for r in ds:
        ctx = r.get("context") or ""
        instr = (r.get("instruction") or "") + (f"\n\n{ctx}" if ctx else "")
        y = _fmt(instr, r.get("response"))
        if y:
            yield y
            n += 1
            if stop and n >= stop:
                break

def _iter_openhermes(split: str, stop: int | None = None):
    ds = load_dataset("HuggingFaceTB/OpenHermes-2.5-H4", split=split)
    n = 0
    for r in ds:
        msgs = r.get("messages") or r.get("conversations")
        if not msgs:
            continue
        u, a = None, None
        for m in msgs:
            role = m.get("role")
            if role in ("user", "human") and u is None:
                u = m.get("content")
            elif role in ("assistant", "gpt") and u is not None:
                a = m.get("content")
                break
        y = _fmt(u, a)
        if y:
            yield y
            n += 1
            if stop and n >= stop:
                break

def _iter_openorca(split: str, stop: int | None = None):
    ds = load_dataset("Open-Orca/OpenOrca", split=split)
    n = 0
    for r in ds:
        instr = (r.get("system_prompt") or "").strip()
        q = (r.get("question") or r.get("input") or "").strip()
        resp = r.get("response") or r.get("output")
        if q:
            instr = (instr + ("\n\n" if instr else "") + q)
        y = _fmt(instr, resp)
        if y:
            yield y
            n += 1
            if stop and n >= stop:
                break

def _iter_no_robots(split: str, stop: int | None = None):
    ds = load_dataset("HuggingFaceH4/no_robots", split=split)
    n = 0
    for r in ds:
        msgs = r.get("messages")
        if msgs:
            u = next((m.get("content") for m in msgs if m.get("role") == "user"), None)
            a = next((m.get("content") for m in msgs if m.get("role") == "assistant"), None)
            y = _fmt(u, a)
        else:
            y = _fmt(r.get("prompt"), r.get("response") or r.get("output"))
        if y:
            yield y
            n += 1
            if stop and n >= stop:
                break

def _iter_codealpaca(split: str, stop: int | None = None):
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split=split)
    n = 0
    for r in ds:
        instr = r.get("instruction") or ""
        inp = r.get("input") or ""
        if inp:
            instr = instr + "\n\n" + inp
        y = _fmt(instr, r.get("output"))
        if y:
            yield y
            n += 1
            if stop and n >= stop:
                break

def _iter_metamath(split: str, stop: int | None = None, max_rows: int = 30000):
    ds = load_dataset("meta-math/MetaMathQA", split=split)
    n = 0
    limit = min(len(ds), max_rows)
    for i in range(limit):
        r = ds[i]
        q = r.get("query") or r.get("problem") or r.get("instruction")
        a = r.get("response") or r.get("answer") or r.get("output")
        y = _fmt(q, a)
        if y:
            yield y
            n += 1
            if stop and n >= stop:
                break

def _iter_dclm_it(split: str, stop: int | None = None):
    ds = load_dataset("mlfoundations-dev/DCLM-IT-Dataset", split=split, streaming=True)
    n = 0
    for r in ds:
        conv = r.get("conversation") or []
        u, a = None, None
        for m in conv:
            frm = m.get("from")
            if frm == "human" and u is None:
                u = m.get("value")
            elif frm == "gpt" and u is not None:
                a = m.get("value")
                break
        y = _fmt(u, a)
        if y:
            yield y
            n += 1
            if stop and n >= stop:
                break

def _tok_pair(tok, instr: str, resp: str, eos_id: int):
    prompt = f"### Instruction:\n{instr}\n\n### Response:\n"
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    r_ids = tok(resp, add_special_tokens=False).input_ids + [eos_id]
    x = p_ids + r_ids

    y = [-100] * len(x)
    for t in range(len(x) - 1):
        nxt = x[t + 1]
        if t + 1 < len(p_ids):
            y[t] = -100
        else:
            y[t] = nxt

    x = np.array(x, dtype=np.uint32)
    y = np.array(y, dtype=np.int32)
    return x, y

def _source_target_count(name: str, kw: dict) -> int:
    stop = kw.get("stop")
    split = kw.get("split", "train")
    if name == "ARC-Easy":
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split)
        n = len(ds)
    elif name == "ARC-Challenge":
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
        n = len(ds)
    elif name == "GSM8K":
        subset = kw.get("subset", "main")
        ds = load_dataset("openai/gsm8k", subset, split=split)
        n = len(ds)
    elif name == "smol-smoltalk":
        ds = load_dataset("HuggingFaceTB/smol-smoltalk", split=split)
        n = len(ds)
    elif name == "oasst1":
        ds = load_dataset("OpenAssistant/oasst1", split=split)
        n = len(ds)
    elif name == "dolly15k":
        ds = load_dataset("databricks/databricks-dolly-15k", split=split)
        n = len(ds)
    elif name == "openhermes":
        ds = load_dataset("HuggingFaceTB/OpenHermes-2.5-H4", split=kw.get("split", "train_sft"))
        n = len(ds)
    elif name == "openorca":
        ds = load_dataset("Open-Orca/OpenOrca", split=split)
        n = len(ds)
    elif name == "no_robots":
        ds = load_dataset("HuggingFaceH4/no_robots", split=split)
        n = len(ds)
    elif name == "codealpaca":
        ds = load_dataset("sahil2801/CodeAlpaca-20k", split=split)
        n = len(ds)
    elif name == "metamath":
        ds = load_dataset("meta-math/MetaMathQA", split=split)
        max_rows = kw.get("max_rows", 30000)
        n = min(len(ds), max_rows)
    elif name == "DCLM-IT":
        n = kw.get("stop")
        if n is None:
            raise ValueError("DCLM-IT requires 'stop' to be set for mixture.")
    else:
        raise ValueError(f"unknown source {name}")
    if stop is not None and name != "DCLM-IT":
        n = min(n, stop)
    return int(n)

def _mixture(train: list[tuple[str, dict]]):
    gens = []
    remaining = []
    for name, kw in train:
        if name == "ARC-Easy":
            gens.append(_iter_arc("ARC-Easy", kw.get("split", "train")))
        elif name == "ARC-Challenge":
            gens.append(_iter_arc("ARC-Challenge", kw.get("split", "train")))
        elif name == "GSM8K":
            gens.append(_iter_gsm8k(kw.get("subset", "main"), kw.get("split", "train")))
        elif name == "smol-smoltalk":
            gens.append(_iter_smoltalk(kw.get("split", "train"), kw.get("stop")))
        elif name == "oasst1":
            gens.append(_iter_oasst1(kw.get("split", "train"), kw.get("stop")))
        elif name == "dolly15k":
            gens.append(_iter_dolly15k(kw.get("split", "train"), kw.get("stop")))
        elif name == "openhermes":
            gens.append(_iter_openhermes(kw.get("split", "train_sft"), kw.get("stop")))
        elif name == "openorca":
            gens.append(_iter_openorca(kw.get("split", "train"), kw.get("stop")))
        elif name == "no_robots":
            gens.append(_iter_no_robots(kw.get("split", "train"), kw.get("stop")))
        elif name == "codealpaca":
            gens.append(_iter_codealpaca(kw.get("split", "train"), kw.get("stop")))
        elif name == "metamath":
            gens.append(_iter_metamath(kw.get("split", "train"), kw.get("stop"), kw.get("max_rows", 30000)))
        elif name == "DCLM-IT":
            gens.append(_iter_dclm_it(kw.get("split", "train"), kw.get("stop")))
        else:
            raise ValueError(f"unknown source {name}")
        remaining.append(_source_target_count(name, kw))
    indices = [i for i, r in enumerate(remaining) if r > 0]
    while indices:
        weights = [remaining[i] for i in indices]
        idx = random.choices(indices, weights=weights, k=1)[0]
        try:
            y = next(gens[idx])
        except StopIteration:
            remaining[idx] = 0
            indices = [i for i, r in enumerate(remaining) if r > 0]
            continue
        remaining[idx] -= 1
        if remaining[idx] < 0:
            remaining[idx] = 0
        yield y
        indices = [i for i, r in enumerate(remaining) if r > 0]

def _write_shard(out_dir: Path, shard_id: int, X: list[np.ndarray], Y: list[np.ndarray], tok_name: str, eos_id: int):
    L = [len(x) for x in X]
    offsets = np.zeros(len(L) + 1, dtype=np.int64)
    np.cumsum(np.array(L, dtype=np.int64), out=offsets[1:])
    flat_x = np.concatenate(X).astype(np.uint16, copy=False)
    flat_y = np.concatenate(Y).astype(np.int32, copy=False)
    d = out_dir / f"shard_{shard_id:05d}"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "tokens.npy", flat_x)
    np.save(d / "labels.npy", flat_y)
    np.save(d / "offsets.npy", offsets)
    meta = {
        "magic": MAGIC,
        "version": VERSION,
        "num_examples": len(L),
        "num_tokens": int(offsets[-1]),
        "tokenizer": tok_name,
        "eos_id": int(eos_id),
        "format": "tasks-v1",
        "labels": "next-token-shifted",
    }
    (d / "meta.json").write_text(json.dumps(meta))

def _build_task_shards(
    out_root: Path,
    split: str,
    tok,
    tokenizer_name: str,
    max_examples_per_shard: int,
    sources: list[tuple[str, dict]],
    seed: int,
):
    random.seed(seed)
    out = out_root / split
    out.mkdir(parents=True, exist_ok=True)
    eos = int(tok.eos_token_id)
    shard, buf_x, buf_y, n = 0, [], [], 0
    for instr, resp in _mixture(sources):
        x, y = _tok_pair(tok, instr, resp, eos)
        buf_x.append(x)
        buf_y.append(y)
        n += 1
        if n >= max_examples_per_shard:
            idx = list(range(len(buf_x)))
            random.shuffle(idx)
            buf_x = [buf_x[i] for i in idx]
            buf_y = [buf_y[i] for i in idx]
            _write_shard(out, shard, buf_x, buf_y, tokenizer_name, eos)
            shard += 1
            buf_x, buf_y, n = [], [], 0
    if n:
        idx = list(range(len(buf_x)))
        random.shuffle(idx)
        buf_x = [buf_x[i] for i in idx]
        buf_y = [buf_y[i] for i in idx]
        _write_shard(out, shard, buf_x, buf_y, tokenizer_name, eos)

def build_task_shards(
    out_dir: str,
    split: str,
    tokenizer_name: str,
    max_examples_per_shard: int,
    sources: list[tuple[str, dict]],
    seed: int = 1337,
):
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, model_max_length=1_000_000)
    assert tok.eos_token_id is not None, "Tokenizer has no EOS. Configure EOS in the tokenizer/model before building shards."
    _build_task_shards(out_root, split, tok, tokenizer_name, max_examples_per_shard, sources, seed)

def _build_staged_task_shards(
    out_root: Path,
    split: str,
    tok,
    tokenizer_name: str,
    max_examples_per_shard: int,
    stages: list[list[tuple[str, dict]]],
    seed: int,
):
    assert stages
    for stage_id, sources in enumerate(stages):
        stage_split = f"{split}_stage_{stage_id:02d}"
        stage_seed = seed + stage_id
        _build_task_shards(out_root, stage_split, tok, tokenizer_name, max_examples_per_shard, sources, stage_seed)

def build_staged_task_shards(
    out_dir: str,
    split: str,
    tokenizer_name: str,
    max_examples_per_shard: int,
    stages: list[list[tuple[str, dict]]],
    seed: int = 1337,
):
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, model_max_length=1_000_000)
    assert tok.eos_token_id is not None, "Tokenizer has no EOS. Configure EOS in the tokenizer/model before building shards."
    _build_staged_task_shards(out_root, split, tok, tokenizer_name, max_examples_per_shard, stages, seed)

if __name__ == "__main__":
    train_sources = [
        # stage 1 ~300k samples
        ("DCLM-IT",   {"split": "train",     "stop": 160_000}),
        ("oasst1",    {"split": "train",     "stop": 25_000}),
        ("dolly15k",  {"split": "train"}),   # full (≈15k)
        ("no_robots", {"split": "train"}),   # full (9.5k)
        ("openhermes",{"split": "train_sft", "stop": 40_000}),
        ("openorca",  {"split": "train",     "stop": 25_000}),
        ("codealpaca",{"split": "train"}),   # full (≈20k)

        # stage 2 ~26k samples
        ("ARC-Easy",      {"split": "train"}),                      # 2,251 train
        ("ARC-Challenge", {"split": "train"}),                      # 1,119 train
        ("GSM8K",         {"subset": "main", "split": "train"}),    # 7,473 train
        ("metamath",      {"split": "train", "stop": 15_000, "max_rows": 30_000}),
    ]

    stage_splits = [
        (0, 7),
        (7, len(train_sources)),
    ]
    train_stages = [train_sources[i:j] for (i, j) in stage_splits]

    # Stage 1 val mixture (~5% of each stage-1 train source)
    # Held-out splits where available:
    #   - oasst1:      validation split (true hold-out)
    #   - no_robots:   test split (500 examples total)
    #   - openhermes:  test_sft split
    # Non-held-out: DCLM-IT, dolly15k, openorca, codealpaca.
    val_stage_0_sources = [
        ("DCLM-IT",   {"split": "train",      "stop": 8_000}),
        ("oasst1",    {"split": "validation", "stop": 1_250}),
        ("dolly15k",  {"split": "train",      "stop": 750}),
        ("no_robots", {"split": "test",       "stop": 480}),
        ("openhermes",{"split": "test_sft",   "stop": 2_000}),
        ("openorca",  {"split": "train",      "stop": 1_250}),
        ("codealpaca",{"split": "train",      "stop": 1_000}),
    ]

    # Stage 2 validation mixture (~5% of each stage-2 train source)
    # Held-out:
    #   - ARC-Easy / ARC-Challenge: validation split
    #   - GSM8K: subset "main", test split
    # Non-held-out: MetaMathQA
    val_stage_1_sources = [
        ("ARC-Easy",      {"split": "validation",               "stop": 112}),
        ("ARC-Challenge", {"split": "validation",               "stop": 56}),
        ("GSM8K",         {"subset": "main", "split": "test",   "stop": 374}),
        ("metamath",      {"split": "train",  "stop": 750, "max_rows": 30_000}),
    ]

    val_stages = [val_stage_0_sources, val_stage_1_sources]
    val_sources = val_stage_0_sources + val_stage_1_sources

    out_dir = "data/instruct_tasks"
    tokenizer_name = "gpt2"
    max_examples_per_shard = 200_000

    build_staged_task_shards(
        out_dir,
        "train",
        tokenizer_name,
        max_examples_per_shard,
        train_stages,
    )
    build_staged_task_shards(
        out_dir,
        "val",
        tokenizer_name,
        max_examples_per_shard,
        val_stages,
    )
    build_task_shards(
        out_dir,
        "val",
        tokenizer_name,
        max_examples_per_shard,
        val_sources,
    )
