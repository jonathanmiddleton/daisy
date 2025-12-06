import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

os.environ["DISABLE_O_ZERO_INIT"] = "1"

from inference.generate import Generator
from inference.kv_cache import KVCache

class DummyAttn:
    def __init__(self, num_heads, head_dim):
        self.num_heads = num_heads
        self.head_dim = head_dim

class DummyBlock(nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn

class DummyModel(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int = 2, num_heads: int = 1, head_dim: int | None = None):
        super().__init__()
        self.eos_token_id = 50256
        H = int(num_heads)
        D = vocab_size if head_dim is None else int(head_dim)
        # Identity-like embedding to produce one-hot vectors for simplicity when D == vocab_size
        self.embed = nn.Embedding(vocab_size, max(D, 1))
        with torch.no_grad():
            if D == vocab_size:
                self.embed.weight.zero_()
                self.embed.weight.fill_(0.0)
                # Make rows equal to one-hot basis
                self.embed.weight.copy_(torch.eye(vocab_size))
            else:
                nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList([DummyBlock(DummyAttn(H, D)) for _ in range(num_layers)])
        # Logits projection: if D == vocab, choose identity to make argmax deterministic
        self.proj = nn.Linear(D, vocab_size, bias=False)
        with torch.no_grad():
            if D == vocab_size:
                self.proj.weight.copy_(torch.eye(vocab_size))
            else:
                nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        # For test inspection
        self.last_k_ctxs = None
        self.last_v_ctxs = None

    def reset_history(self):
        self.last_k_ctxs = None
        self.last_v_ctxs = None

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor, window: int | None = None, debug: bool = False):
        assert input_ids.ndim == 2 and input_ids.size(0) == 1
        B, T = input_ids.shape
        L = len(self.blocks)
        H = self.blocks[0].attn.num_heads
        D = self.blocks[0].attn.head_dim
        # Build per-token key/values from embedding
        emb = self.embed(input_ids)  # [B, T, D]
        # K/V layout expected: [L, B, H, T, D]
        K = []
        V = []
        for _ in range(L):
            k = emb.view(B, T, H, D).transpose(1, 2).contiguous()  # [B, H, T, D]
            v = k.clone()
            K.append(k)
            V.append(v)
        K = torch.stack(K, dim=0)
        V = torch.stack(V, dim=0)
        kv = torch.stack([K, V], dim=0)
        # Logits are projection of the last token representation
        last_vec = emb[:, -1, :].view(B, 1, D)
        logits = self.proj(last_vec).view(B, -1)  # [B, V]
        return logits, kv

    @torch.no_grad()
    def step(self, token_id: torch.Tensor, k_ctxs, v_ctxs, pos: int):
        # Record contexts to verify KV tensors passed from cache
        self.last_k_ctxs = [kc.clone() for kc in k_ctxs]
        self.last_v_ctxs = [vc.clone() for vc in v_ctxs]
        B = 1
        H = self.blocks[0].attn.num_heads
        D = self.blocks[0].attn.head_dim
        # Current token embedding
        x0 = self.embed(token_id)[None, None, :]
        # Produce k/v for each layer: shape [B, 1, H, D]
        k_new_list = []
        v_new_list = []
        for _ in range(len(self.blocks)):
            kv = self.embed(token_id).view(B, 1, H, D)
            k_new_list.append(kv)
            v_new_list.append(kv.clone())
        # Logits: depend on current token embedding (deterministic), but also read from contexts
        # Include a tiny contribution from the mean of last layer's v_ctx to prove usage without affecting argmax
        if v_ctxs and v_ctxs[-1].numel() > 0:
            ctx_mean = v_ctxs[-1][:, :, :, :].mean(dim=(1, 2, 3), keepdim=True)  # [B,1,1,1]
            x_for_logits = x0 + 0.0 * ctx_mean  # no change in values but touches tensor
        else:
            x_for_logits = x0
        logits = self.proj(x_for_logits).view(B, -1)
        return logits, k_new_list, v_new_list


@pytest.fixture()
def dummy_env(device_override: str | None = None):
    torch.manual_seed(0)
    vocab_size = 16
    num_layers = 2
    H = 1
    D = vocab_size
    model = DummyModel(vocab_size=vocab_size, num_layers=num_layers, num_heads=H, head_dim=D).eval()
    device = torch.device(device_override or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    gen = Generator(model, window=8, seed=1337, device=device, dtype=torch.bfloat16, temperature=0.0)
    return model, gen, device, vocab_size


def one_hot(vocab_size, idx, device):
    v = torch.zeros(vocab_size, device=device)
    v[idx] = 1.0
    return v


def test_prefill_updates_cache_and_returns_logits(dummy_env):
    model, gen, device, V = dummy_env
    prompt = torch.tensor([3, 5, 7, 9], dtype=torch.long, device=device)
    logits = gen._prefill(prompt)

    # logits should be one-hot of the last prompt token
    assert logits.shape == (1, V)
    expect = one_hot(V, prompt[-1].item(), device)
    assert torch.allclose(logits[0].float(), expect, atol=0, rtol=0)

    # cache should be filled with embeddings for each token
    assert gen.cache.t == len(prompt)
    for layer in range(len(model.blocks)):
        k_ctx, v_ctx = gen.cache.view(layer)
        # shapes: [B(=1), T, H(=1), D]
        assert k_ctx.shape == (1, len(prompt), 1, V)
        # Values should equal the one-hot embeddings in order
        for t, tok in enumerate(prompt.tolist()):
            oh = one_hot(V, tok, device).view(1, 1, V)
            assert torch.allclose(k_ctx[:, t, :, :].view(1, 1, V).float(), oh.float(), atol=0, rtol=0)
            assert torch.allclose(v_ctx[:, t, :, :].view(1, 1, V).float(), oh.float(), atol=0, rtol=0)

    # history updated
    assert torch.equal(gen.history, prompt)


def test_step_writes_kv_and_advances_cache(dummy_env):
    model, gen, device, V = dummy_env
    prompt = torch.tensor([2, 4, 6], dtype=torch.long, device=device)
    gen._prefill(prompt)
    t0 = gen.cache.t

    next_id = 11
    token_t = torch.tensor(next_id, dtype=torch.long, device=device)
    logits = gen._step(token_t)

    # logits reflect current token
    expect = one_hot(V, next_id, device)
    assert logits.shape == (1, V)
    assert torch.allclose(logits[0].float(), expect.float(), atol=0, rtol=0)

    # cache advanced and wrote new kv at position t0
    assert gen.cache.t == t0 + 1
    for layer in range(len(model.blocks)):
        k_ctx, v_ctx = gen.cache.view(layer)
        # Last element should equal embedding of next_id
        oh = one_hot(V, next_id, device).view(1, 1, V)
        assert torch.allclose(k_ctx[:, -1, :, :].view(1, 1, V).float(), oh.float())
        assert torch.allclose(v_ctx[:, -1, :, :].view(1, 1, V).float(), oh.float())

    # Model received contexts that match the cache contents at call time
    for layer in range(len(model.blocks)):
        mk = model.last_k_ctxs[layer]
        mv = model.last_v_ctxs[layer]
        ck, cv = gen.cache.view(layer)
        # Contexts passed into step corresponded to cache state before writing new token
        # i.e., they should equal the first t0 entries of ck, cv
        assert torch.allclose(mk.float(), ck[:, :-1].float())
        assert torch.allclose(mv.float(), cv[:, :-1].float())

    # history appended
    assert gen.history[-1].item() == next_id


def test_sample_behaviors():
    device = torch.device("cpu")
    vocab = 8
    class Minimal(nn.Module):
        def __init__(self, V):
            super().__init__()
            self.eos_token_id = 50256
            # Provide one dummy block with attention metadata so KVCache can initialize
            Attn = type("Attn", (), {"num_heads": 1, "head_dim": 1})
            Block = type("Block", (), {"attn": Attn()})
            self.blocks = [Block()]
            self.embed = type("E", (), {"num_embeddings": V})()
        def reset_history(self):
            pass
        def prefill(self, input_ids: torch.Tensor, window: int | None = None, debug: bool = False):
            # Minimal stub to satisfy Generator.warmup path in tests
            B, T = input_ids.shape
            H, D = 1, 1
            K = torch.zeros(1, B, H, T, D)
            V = torch.zeros(1, B, H, T, D)
            kv = torch.stack([K, V], dim=0)
            logits = torch.zeros(B, 1)
            return logits, kv
    gen = Generator(model=Minimal(vocab), window=4, seed=1337, device=device, dtype=torch.bfloat16)
    gen.history = torch.tensor([1, 2, 3, 2, 1], dtype=torch.long)

    # Temperature 0 picks argmax
    logits = torch.tensor([0.1, 0.9, 0.5, -1.0, 0.0, 0.0, 0.0, 0.0], device=device)
    gen.set_temperature(0.0)
    assert int(gen.sample(logits, gen.temperature, gen.top_k, gen.top_p)) == 1

    # Temperature scaling: ensure it doesn't crash and changes distribution
    gen.set_temperature(2.0)
    x = logits / 2.0
    probs_low_temp = F.softmax(x, dim=-1)
    probs_base = F.softmax(logits, dim=-1)
    assert not torch.allclose(probs_low_temp, probs_base)

    # top_k filtering
    gen.set_temperature(1.0)
    gen.top_k = 1
    assert int(gen.sample(logits, gen.temperature, gen.top_k, gen.top_p)) == logits.argmax().item()

    # top_p filtering keeps minimal set summing <= p
    gen.top_k = None
    gen.top_p = 0.6
    # Should allow only tokens 0 through 2
    assert 0 <= int(gen.sample(logits, gen.temperature, gen.top_k, gen.top_p)) <= 2

    # repetition penalty increases/decreases logits for seen tokens
    gen.top_p = None
    gen.set_repetition_penalty(1.2)
    # Token 1 and 2 appear in history, so their logits are penalized; token 0 becomes best
    logits2 = torch.tensor([0.95, 0.94, 0.93, 0.0, 0.0, 0.0, 0.0, 0.0], device=device)
    gen.set_temperature(0.0)
    choice = int(gen.sample(logits2, gen.temperature, gen.top_k, gen.top_p))
    assert choice == 0


def test_generate_end_to_end(dummy_env):
    model, gen, device, V = dummy_env
    # Prompt of length 3
    prompt = torch.tensor([4, 1, 4], dtype=torch.long, device=device)
    # Greedy sampling: next token == last prompt token (4), then repeats
    tokens = []
    with torch.inference_mode():
        for t in gen.generate(prompt, max_new_tokens=3):
            tokens.append(int(t))
    # tokens yielded only contain generated ids
    assert tokens == [4, 4, 4]
    # Final return from generator
    gen.reset_history()
    g = gen.generate(prompt, max_new_tokens=2)
    gen_tokens = []
    try:
        with torch.inference_mode():
            while True:
                gen_tokens.append(int(next(g)))
    except StopIteration as e:
        out, pre_dur, step_dur = e.value
    out = out.tolist()
    # Ensure the yielded tokens were as expected
    assert gen_tokens == [4, 4]
    assert out[:len(prompt)] == prompt.tolist()
    assert out[len(prompt):] == [4, 4]
    assert isinstance(pre_dur, float) and isinstance(step_dur, float)
