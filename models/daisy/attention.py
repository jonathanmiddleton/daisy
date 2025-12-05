import os
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from torch.nn import functional as F
from models.daisy.functional import norm, init_linear
from tools.master_logger import MasterLogger
logger = MasterLogger

WINDOW_BLOCK_SIZE = 128

def is_flex_available(enable_for_cpu: bool = False, dynamic_shapes: bool = False):
    # FlexAttention is supported only on cuda (limited on CPU)
    return (not dynamic_shapes) and (torch.cuda.is_available() or enable_for_cpu)


def _apply_rope(x_BTHD, cos, sin):
    x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), dim=-1).type_as(x_BTHD)


def _flex_call(q, k, v, block_mask, scale):
    return flex_attention(q, k, v, block_mask=block_mask, scale=scale)

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        # For small head_dim (< 4), do NOT truncate; otherwise keep half-truncation.
        half = dim // 2
        keep = dim // 4  # number of active frequencies when truncating by half
        base = 1024
        if keep == 0:
            # head_dim < 4 -> no truncation; use standard RoPE with 'half' frequencies
            angular_freq = (1 / base) ** torch.linspace(0, 1, steps=half, dtype=torch.float32)
        else:
            active = (1 / base) ** torch.linspace(0, 1, steps=keep, dtype=torch.float32)
            # pad with zeros to keep total = half (dim//2)
            angular_freq = torch.cat([active, active.new_zeros(half - keep)])
        # Register buffers so they move with module.to(device) exactly once
        self.inv_freq = nn.Buffer(angular_freq, persistent=False)
        # Preallocate cos/sin tables up to max_seq_len on construction; slice at runtime
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, self.inv_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)
        # Track capacity for assertions
        self._max_seq_len = int(max_seq_len)

    def _get_cos_sin(self, L):
        torch._assert(L <= self.cos.shape[0], f"Rotary buffers too small {self.cos.shape[0]} < {L}")
        cos = self.cos.narrow(0, 0, L)
        sin = self.sin.narrow(0, 0, L)
        return cos, sin

    def forward(self, x_BTHD: torch.Tensor):
        L = x_BTHD.size(-3)
        cos, sin = self._get_cos_sin(L)
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        return _apply_rope(x_BTHD, cos, sin)

    def step(self, x_BTHD: Tensor, pos: int):
        # Use preallocated tables and select a single position
        torch._assert(x_BTHD.size(1) == 1, "step() requires single position only")
        cos, sin = self._get_cos_sin(pos + 1)
        cos = cos.unsqueeze(0).narrow(1, pos, 1).unsqueeze(2)
        sin = sin.unsqueeze(0).narrow(1, pos, 1).unsqueeze(2)
        return _apply_rope(x_BTHD, cos, sin)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim, dynamic_shapes: bool = False, receives_ve: bool = False):
        super().__init__()
        torch._assert(dim % num_heads == 0, "dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim: int = head_dim
        self.m_dim = dim
        self.dynamic_shapes = dynamic_shapes
        # merged QKV weights: suggested by many, implemented by @fernbear.bsky.social, and further improved by @YouJiacheng
        # https://x.com/hi_tysam/status/1879699187107033311
        self.qkvo_w = nn.Parameter(init_linear(torch.empty(4, self.m_dim, self.m_dim)).bfloat16())
        if os.getenv("DISABLE_O_ZERO_INIT", "") != "1":  # 1 for unittests
            self.qkvo_w.detach()[3].zero_()
        self.rotary = Rotary(head_dim, 131072) # magic = max val length TODO fix this
        # scale the attention logits by given constant, instead of the default head_dim**-0.5, by @leloykun
        # inspired by learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        self.attn_scale = 0.12
        self.last_q = None
        self.last_k = None
        self.receives_ve = receives_ve
        self.g_ve = nn.Parameter(torch.tensor(0.0)) if self.receives_ve else None # init 50/50 gate

        self.debug_logging = logger.isDebugEnabled() # quasi-static compile-friendly boolean for logging

# TODO can't create a new Module after compile
    # def maybeResizeRotary(self, x: Tensor):
    #     T = x.size(-2)
    #     if self.debug_logging: logger.debug(f"maybeResizeRotary newSize={T} Rotary capacity= {self.rotary._max_seq_len}")
    #     if T <= self.rotary._max_seq_len: return
    #
    #     if self.debug_logging: logger.debug(f"Resizing Rotary to {T} positions.")
    #     from math import pow
    #     def next_power_of_2(s: int):
    #         pows = [int(pow(2, n)) for n in range(19)]
    #         return next(n for n in pows if n > s)
    #     self.rotary = Rotary(self.head_dim, next_power_of_2(T))
    #     self.rotary.to(x.device)

    def reset_history(self):
        self.last_q = None
        self.last_k = None

    def _calc_qkv(self, x: Tensor, pos: int = None):
        B, T = x.size(0), x.size(1)
        x = x.to(self.qkvo_w.dtype)
        qkv = self.qkvo_w[:3].flatten(end_dim=1)
        q, k, v = F.linear(x, qkv).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k, v = norm(q), norm(k), norm(v)
        if pos is not None:
            q, k = self.rotary.step(q, pos), self.rotary.step(k, pos)
        else:
            q, k = self.rotary(q), self.rotary(k)

        return q, k, v

    def _qkv_common(self, x: torch.Tensor, ve: Optional[torch.Tensor]):
        q, k, v = self._calc_qkv(x)
        dtype = q.dtype
        if self.receives_ve:
            g_x = torch.sigmoid(self.g_ve).to(dtype)
            torch._assert(ve is not None, "ve must be provided when receives_ve=True")
            v = g_x * v + (1.0 - g_x) * ve.view_as(v).to(dtype)
        q_ = q.transpose(1, 2)
        k_ = k.transpose(1, 2)
        v_ = v.transpose(1, 2)
        return q_, k_, v_

    def _sdpa_common(self, x: torch.Tensor, ve: torch.Tensor,  attn_mask: Optional[Tensor] = None):
        B, T = x.size(0), x.size(1)
        q_, k_, v_ = self._qkv_common(x, ve)
        if attn_mask is not None:
            y = torch.nn.functional.scaled_dot_product_attention(q_, k_, v_, is_causal=False, attn_mask=attn_mask, scale=self.attn_scale)
        else:
            y = torch.nn.functional.scaled_dot_product_attention(q_, k_, v_, is_causal=True, scale=self.attn_scale)
        y = y.transpose(1, 2).contiguous().view(B, T, self.m_dim)
        y = torch.nn.functional.linear(y, self.qkvo_w[3])
        return y, k_.transpose(1, 2), v_.transpose(1, 2), q_.transpose(1, 2)

    def forward_sdpa(self, x: torch.Tensor,ve: Optional[torch.Tensor],  attn_mask: Optional[Tensor] = None):
        y, _, _, _ = self._sdpa_common(x, ve,  attn_mask=attn_mask)
        return y

    def forward_flex(self, x: torch.Tensor, ve: Optional[torch.Tensor],  block_mask: BlockMask):
        B, T = x.size(0), x.size(1)
        q_, k_, v_ = self._qkv_common(x, ve)
        y = _flex_call(q_, k_, v_, block_mask=block_mask, scale=self.attn_scale)
        y = y.transpose(1, 2).contiguous().view(B, T, self.m_dim)
        y = torch.nn.functional.linear(y, self.qkvo_w[3])
        return y

    def forward(self, x: torch.Tensor, ve: Optional[torch.Tensor],  block_mask: Optional[BlockMask] = None, attn_mask: Optional[Tensor] = None):
        if self.debug_logging: logger.debug(f"forward(x.shape={x.shape}), self.dynamic_shapes={self.dynamic_shapes}, block_mask={block_mask is not None}, attn_mask={attn_mask is not None}")
        if is_flex_available(dynamic_shapes=self.dynamic_shapes) and block_mask is not None:
            if self.debug_logging: logger.debug(f"Using FlexAttention with block_mask.")
            return self.forward_flex(x, ve, block_mask=block_mask)
        else:
            if self.debug_logging: logger.debug(f"Using SDPA with attn_mask = {attn_mask is not None}.")
            return self.forward_sdpa(x, ve, attn_mask=attn_mask)

    def prefill(self, x: torch.Tensor, ve: Optional[torch.Tensor],  attn_mask: Tensor = None, debug: bool = False):
        y, k, v, q = self._sdpa_common(x, ve, attn_mask=attn_mask)

        if debug:
            self.last_q = q
            self.last_k = k
        return y, k.transpose(1, 2), v.transpose(1, 2)

    def step(self, x, k_ctx: Tensor, v_ctx: Tensor, pos: int, ve: Optional[torch.Tensor], window: int):
        B, T = x.size(0), 1
        q, k, v = self._calc_qkv(x, pos)
        dtype = q.dtype

        if self.receives_ve:
            ve = ve.to(dtype)
            g_x = torch.sigmoid(self.g_ve).to(dtype)
            torch._assert(ve is not None, "ve must be provided when receives_ve=True")
            v = g_x * v + (1.0 - g_x) * ve.view_as(v).to(dtype)

        n = k_ctx.size(1)
        r = n if window is None else min(n, max(window - 1, 0))
        k_ = torch.cat([k_ctx[:, n - r:n], k], 1).transpose(1, 2).to(dtype)
        v_ = torch.cat([v_ctx[:, n - r:n], v], 1).transpose(1, 2).to(dtype)
        q_ = q.transpose(1, 2)

        # since q holds a single position `is_causal` must be False or indices after 0 will be masked out
        y = F.scaled_dot_product_attention(q_, k_, v_, scale=self.attn_scale, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, self.m_dim)
        y = F.linear(y, self.qkvo_w[3]).narrow(1, -1, 1)
        return y, k, v
