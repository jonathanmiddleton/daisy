import os
from functools import lru_cache
from math import floor, log2, ceil
from typing import Any, Optional, List, Dict

import torch
from torch import nn, Tensor, SymInt
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask

from models.daisy.block import Block
from models.daisy.functional import norm

WINDOW_BLOCK_SIZE = 128

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

def _pick_value_embedding_layers(attn_layers):
    """Bottom 3 and top 3 attention layers receive value embeddings."""
    K = len(attn_layers)
    if K < 6:
        return []
    attn_layers = sorted(attn_layers)
    return attn_layers[:3] + attn_layers[-3:]

def _build_ve_layer_map(L: int, ve_layers: List[int], _in: int, _out: int) -> Dict[int, Optional[nn.Embedding]]:
    """
    Build a value-embedding layer map with the following pattern:

      - Let K = len(ve_layers) // 2.
      - Create K embeddings.
      - Map the first K layers [0, K-1] to these embeddings (1:1).
      - Map the last K layers [L-K, L-1] to the same embeddings (reused in order).
      - Omit all middle layers
    """
    torch._assert(len(ve_layers) % 2 == 0, f"ve_layers must be an even number of layers, got {len(ve_layers)}")
    if len(ve_layers) == 0:
        return {i: None for i in range(L)}
    K = len(ve_layers) // 2

    embeds = [nn.Embedding(_in, _out) for _ in range(K)]
    ve_map: Dict[int, Optional[nn.Embedding]] = {}

    for i in range(len(ve_layers)):
        ve_map[ve_layers[i]] = embeds[i % K]

    return ve_map

def build_attn_mask(input_seq: Tensor, window_size: int, eos_token_id: int):
    T = input_seq.size(-1)
    q = torch.arange(T, device=input_seq.device)[:, None]  # (T, 1)
    k = torch.arange(T, device=input_seq.device)[None, :]  # (1, T)
    d = q - k  # d[q, k] = q - k

    docs = (input_seq == eos_token_id).cumsum(0)
    docs_q = docs[:, None]
    docs_k = docs[None, :]

    m = torch.zeros(T, T, device=input_seq.device, dtype=torch.float32)
    m[d < 0] = float("-inf")  # forbid future (k > q)
    m[d >= window_size] = float("-inf")  # forbid too-far past
    m[docs_q != docs_k] = float("-inf")
    attn_mask = m[None, None, :, :]
    return attn_mask


def _pick_attention_layers(L, d_model=None, num_heads=None, attn_impl='standard', attn_density=0.75):
    if L <= 0: return []
    if L == 1: return [0]
    if L == 2: return [0, 1]
    if L == 3: return [0, 2]
    if L == 4: return [0, 1, 3]
    if L == 5: return [0, 2, 4]
    if L == 6: return [0, 1, 3, 5]
    if L == 12: return [i for i in range(12) if i != 7]
    if L == 16: return [0, 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 15] #backwards compatibility
    d_head = (d_model // num_heads) if (d_model and num_heads) else 64
    s = max(4, min(12, round(8 * (d_head / 64) ** 0.5)))
    k_log = ceil(2 + log2(L))
    k_ratio = ceil(attn_density * L)
    k_stride = ceil(L / s)
    K = min(L, max(k_log, k_ratio, k_stride))
    idx = [round(i * (L - 1) / (K - 1)) for i in range(K)]
    idx_s = set(map(int, idx))
    if attn_impl == 'kimi_linear':
        idx_s |= set(range(0, L, 4))
    return sorted(idx_s)


class DaisyCore(nn.Module):
    class AttnImplIDs:
        _attn_impl_ids = {"standard":0, "kimi_linear":1}

        STANDARD = _attn_impl_ids["standard"]
        KIMI_LINEAR = _attn_impl_ids["kimi_linear"]

        @classmethod
        def get_attn_impl_id(cls, attn_impl: str):
            if attn_impl in cls._attn_impl_ids:
                return cls._attn_impl_ids[attn_impl]
            else:
                raise ValueError(f"Unknown attn_impl: {attn_impl}")

    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int,
                 head_dim: int, window_size: int = 2048, eos_token_id: int | None = None, desc: dict | None = None,
                 use_value_embeddings: bool = True, use_tied_embeddings: bool = False, attn_all_layers: bool = False,
                 attn_impl: str = 'standard', dynamic_shapes: bool = False):
        super().__init__()
        if eos_token_id is None:
            raise ValueError("eos_token_id is required.")

        def _get_skip_map(L: int):
            """
            Side‑band residual mappings. Places targets just past the midpoint to avoid bypassing too much computation,
            while spacing sources by `s` partitions the first half into `K+1` chunks, giving progressively longer skips
            that cover diverse timescales.
            Parameters:
                L: int
                    Layer count

            Returns:
                dict[int, int]
                    A dictionary mapping target indices to source indices.
            """
            K = max(1, floor(log2(L)) - 1)
            c = L // 2
            s = max(1, c // (K + 1))
            m = {c + t: c - t * s for t in range(1, K + 1)}
            return {i: j for i, j in m.items() if 0 <= j < i < L}

        self.attn_impl_ids = DaisyCore.AttnImplIDs
        self.dynamic_shapes = dynamic_shapes

        self.skip_map = _get_skip_map(num_layers)
        self.eos_token_id = int(eos_token_id)
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.attn_layers = [i for i in range(num_layers)] if attn_all_layers else _pick_attention_layers(num_layers, attn_impl=attn_impl)
        self.use_value_embeddings = use_value_embeddings

        self.ve_layers = []
        self.ve_modules: nn.ModuleList[nn.Embedding] = nn.ModuleList([])
        self.ve_module_map = {}
        if self.use_value_embeddings:
            self.ve_layers = _pick_value_embedding_layers(self.attn_layers)
            self.ve_modules: nn.ModuleList[nn.Embedding] = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(len(self.ve_layers)//2)])
            self.ve_module_map: Dict[int, Optional[nn.Embedding]] = _build_ve_layer_map(num_layers, self.ve_layers, vocab_size, model_dim)


        self.attn_impl_id = self.attn_impl_ids.get_attn_impl_id(attn_impl)
        self.blocks = nn.ModuleList(
            [Block(model_dim, num_heads, max_seq_len, i, head_dim, i in self.attn_layers, attn_impl, dynamic_shapes=dynamic_shapes, receives_ve=(i in self.ve_layers)) for i in range(num_layers)])
        if use_tied_embeddings:
            nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
            self.lm_head_w = self.embed.weight
        else:
            if os.getenv("DISABLE_O_ZERO_INIT", "") != "1":
                # != 1 training
                self.lm_head_w = nn.Parameter(torch.zeros(next_multiple_of_n(vocab_size, n=128), model_dim))
            else:
                # == 1 to allow backpropagation for lr_sweep or cases where the LM head is frozen for testing
                self.lm_head_w = nn.Parameter(torch.empty(next_multiple_of_n(vocab_size, n=128), model_dim))
                nn.init.normal_(self.lm_head_w, mean=0.0, std=0.02)
        self.window_size = window_size
        assert num_layers % 2 == 0
        self.skip_weights = nn.Parameter(torch.ones(num_layers)*2.945) # init 95/5 gate
        self.desc = desc  # non-functional, self-describing metadata

    def reset_history(self):
        for b in self.blocks:
            b.reset_history()

    def create_blockmasks(self, input_seq: Tensor, sliding_window_num_blocks: Tensor, L: int):
        global WINDOW_BLOCK_SIZE
        BLOCK_SIZE = WINDOW_BLOCK_SIZE
        torch._assert(len(input_seq) % BLOCK_SIZE == 0, f"input_seq length {len(input_seq)} must be divisible by BLOCK_SIZE {BLOCK_SIZE}")
        device = input_seq.device
        docs = (input_seq == self.eos_token_id).cumsum(0)

        def document_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask

        def dense_to_ordered(dense_blockmask: Tensor):
            num_blocks = dense_blockmask.sum(dim=-1, dtype=torch.int32)
            indices = dense_blockmask.argsort(dim=-1, descending=False, stable=True).flip(-1).to(torch.int32)
            return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

        assert len(input_seq) % BLOCK_SIZE == 0
        NUM_BLOCKS = len(input_seq) // BLOCK_SIZE
        block_idx = torch.arange(NUM_BLOCKS, dtype=torch.int32, device=device)
        causal_blockmask_any = block_idx[:, None] >= block_idx
        causal_blockmask_all = block_idx[:, None] > block_idx
        docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
        docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()
        document_blockmask_any = (docs_low[:, None] <= docs_high) & (docs_high[:, None] >= docs_low)
        document_blockmask_all = (docs_low[:, None] == docs_high) & (docs_high[:, None] == docs_low)
        blockmask_any = causal_blockmask_any & document_blockmask_any
        blockmask_all = causal_blockmask_all & document_blockmask_all
        partial_kv_num_blocks, partial_kv_indices = dense_to_ordered(blockmask_any & ~blockmask_all)
        full_kv_num_blocks, full_kv_indices = dense_to_ordered(blockmask_all)

        def build_bm(window_size_blocks: Tensor) -> BlockMask:
            # print(f"partial_kv_num_blocks.device={partial_kv_num_blocks.device.type} window_size_blocks.device={window_size_blocks.device} full_kv_num_blocks.device={full_kv_num_blocks.device.type}")
            return BlockMask.from_kv_blocks(
                torch.clamp_max(partial_kv_num_blocks, torch.clamp_min(window_size_blocks - full_kv_num_blocks, 1)),
                partial_kv_indices,
                torch.clamp_max(full_kv_num_blocks, window_size_blocks - 1),
                full_kv_indices,
                BLOCK_SIZE=BLOCK_SIZE,
                mask_mod=document_causal,
            )

        # Long-short SWA block masks by @leloykun & @YouJiacheng, adapated from suggestion by @Grad62304977, following Gemma 2 paper
        long_bm, short_bm = build_bm(sliding_window_num_blocks), build_bm(sliding_window_num_blocks // 2)

        cycle = [long_bm] + [short_bm] * 3
        block_masks = (cycle * ((L + 3) // 4))[:L - 1] + [long_bm] #TODO adjust cycle for kimi_linear L,S,L,_
        return block_masks

    def compute_value_embeddings(self, input_seq: Tensor) -> Dict[int, Tensor]:
        ve_map: Dict[int, Tensor] = {}
        for i in self.ve_module_map:
            ve_map[i] = self.ve_module_map[i](input_seq)
        return ve_map

    def forward(self, input_seq: Tensor, sliding_window_num_blocks: Tensor, target_seq: Tensor = None, loss_chunks: int = 4, output_logits: bool = False):
        torch._assert(input_seq.ndim == 1, "input_seq must be 1D")
        L = len(self.blocks)

        ve_map:Dict[int, Tensor] = self.compute_value_embeddings(input_seq)

        x = x0 = norm(self.embed(input_seq)[None])

        skip_map = self.skip_map
        skip_weights = self.skip_weights

        skip_connections = []

        if input_seq.device.type == "cuda" and not self.dynamic_shapes: #  FlexAttention if supported unless dynamic_shape support is required
            block_masks = self.create_blockmasks(input_seq, sliding_window_num_blocks, L=L)
            attn_mask = None
        else:
            block_masks = [None] * L
            # building an attention mask for T>sqrt(2,147,483,647)==sqrt(INT_MAX) will fail
            torch._assert(input_seq.numel() <= 46340, "For attention masks with SDPA, input_seq length must be less than sqrt(2^31) ~= 46340 tokens")
            attn_mask = build_attn_mask(input_seq, self.window_size, self.eos_token_id)

        for i in range(L):
            if i in skip_map:
                gate = torch.sigmoid(skip_weights[skip_map[i]])
                x = x*gate + (1-gate)*skip_connections[skip_map[i]]
            if i in ve_map:
                x = self.blocks[i](x, ve_map[i], x0,  block_mask=block_masks[i], attn_mask=attn_mask)
            else:
                x = self.blocks[i](x, None, x0, block_mask=block_masks[i], attn_mask=attn_mask)
            skip_connections.append(x)

        x = norm(x)
        if self.training:
            logits: Tensor = F.linear(x.flatten(end_dim=1).bfloat16(), self.lm_head_w.bfloat16()).float()
            loss = F.cross_entropy(15 * logits * torch.rsqrt(logits.square() + 225), target_seq)
            return loss

        if output_logits:
            logits: Tensor = F.linear(x.flatten(end_dim=1).bfloat16(), self.lm_head_w.bfloat16()).float()
            return logits
        else:
            loss = 0
            for i in range(loss_chunks):
                torch._assert(input_seq.numel() % loss_chunks == 0,
                              f"input_seq must be divisible by {loss_chunks} when not in training")
                logits: Tensor = F.linear(x.flatten(end_dim=1).chunk(loss_chunks)[i].bfloat16(),
                                          self.lm_head_w.bfloat16()).float()
                logits = 15 * logits * torch.rsqrt(logits.square() + 225)
                chunk = target_seq.chunk(loss_chunks)[i]
                loss += F.cross_entropy(logits, chunk) / loss_chunks
            return loss

    def step(self, token_id: Tensor, k_ctxs, v_ctxs, pos: int, window: int):
        assert token_id.ndim == 0
        B = T = 1
        token_id = token_id.view(B, T)
        x0 = norm(self.embed(token_id))
        L = len(self.blocks)

        ve = self.compute_value_embeddings(token_id)

        skip_map = self.skip_map
        skip_weights = self.skip_weights

        x = x0
        k_new_list = []
        v_new_list = []
        skip_connections = []
        ve_map: Dict[int, Tensor] = self.compute_value_embeddings(token_id)
        for i in range(L):
            if i in skip_map:
                gate = torch.sigmoid(skip_weights[skip_map[i]])
                x = x*gate + (1-gate)*skip_connections[skip_map[i]]
            if i in ve_map:
                y, k_new, v_new = self.blocks[i].step(x, ve_map[i], x0, k_ctxs[i], v_ctxs[i], pos, window)
            else:
                y, k_new, v_new = self.blocks[i].step(x, None, x0, k_ctxs[i], v_ctxs[i], pos, window)
            x = y
            skip_connections.append(x)
            k_new_list.append(k_new)
            v_new_list.append(v_new)
        x = norm(x)
        logits = F.linear(x.flatten(end_dim=1).bfloat16(), self.lm_head_w.bfloat16()).float()
        return logits, k_new_list, v_new_list

# TODO restore windowing
    def prefill(self, input_seq: Tensor, window: Optional[int] = None, debug: bool = False): #TODO   merge prefill/forward
        assert input_seq.ndim == 2
        B, T = input_seq.shape
        h = None
        d = None
        for b in self.blocks:
            if getattr(b, "attn", None) is not None:
                h = b.attn.num_heads
                d = b.attn.head_dim
                break
        L = len(self.blocks)

        x = norm(self.embed(input_seq))
        x0 = x

        ve = self.compute_value_embeddings(input_seq)

        skip_map = self.skip_map
        skip_weights = self.skip_weights

        k_list, v_list, skip_connections = [], [], []
        ve_map: Dict[int, Tensor] = self.compute_value_embeddings(input_seq)
        for i in range(L):
            if i in skip_map:
                gate = torch.sigmoid(skip_weights[skip_map[i]])
                x = x * gate + (1 - gate) * skip_connections[skip_map[i]]
            if i in ve_map:
                x, k, v = self.blocks[i].prefill(x, ve_map[i], x0, debug=debug)
            else:
                x, k, v  = self.blocks[i].prefill(x, None, x0, debug=debug)
            skip_connections.append(x)
            k_list.append(k)
            v_list.append(v)

        x = norm(x)
        logits = torch.nn.functional.linear(x[:, -1].bfloat16(), self.lm_head_w.bfloat16()).float()

        attn = next(b.attn for b in self.blocks if b.attn is not None)
        H, D = attn.num_heads, attn.head_dim
        device = x.device
        dtype = x.dtype

        K = []
        V = []
        for k, v in zip(k_list, v_list):
            if k is None:
                K.append(torch.zeros(B, H, T, D, device=device, dtype=dtype))
                V.append(torch.zeros(B, H, T, D, device=device, dtype=dtype))
            else:
                K.append(k)
                V.append(v)
        K = torch.stack(K, dim=0)
        V = torch.stack(V, dim=0)
        kv = torch.stack([K, V], dim=0)
        return logits, kv


