from typing import Optional

import torch
from models.daisy.fla_custom_ops import fused_kda_gate
from torch import nn, Tensor

#noinspection PyBroadException
try:
    from fla.ops.kda import chunk_kda
    from fla.modules import ShortConvolution
    from fla.modules.fused_norm_gate import rms_norm_gated
except Exception:
    pass

from einops import rearrange

#noinspection PyBroadException
try:
    from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
except Exception:
    def get_unpad_data(mask):
        torch._assert(mask is None or bool(mask.dtype == torch.long or mask.dtype == torch.int or mask.dtype == torch.bool), "mask dtype unsupported")
        torch._assert(mask is None or bool(mask.min().item() >= 1), "varlen requires flash-linear-attention install")
        B, S = mask.shape if mask is not None else (0, 0)
        idx = torch.arange(B*S, device=mask.device) if mask is not None else None
        return idx, None, S
    # noinspection PyUnusedLocal
    def index_first_axis(x, idx):
        return x
    # noinspection PyUnusedLocal
    def pad_input(x, indices, batch_size, q_len):
        return x.squeeze(0)

#noinspection PyBroadException
try:
    from fla.models.utils import Cache
except Exception:
    Cache = None


class CompilableFusedRMSNormGated(nn.Module):
    """Replacement for FusedRMSNormGated that works with torch.compile"""

    def __init__(self, hidden_size: int, activation: str = "sigmoid", dtype=torch.bfloat16):
        assert activation == "sigmoid", "Only sigmoid activation is supported"
        super().__init__()
        self.hidden_size = hidden_size
        self.activation = activation

        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.bias = None

    def forward(self, x: Tensor, gate: Tensor) -> Tensor:
        # x: (..., hidden_size)
        # gate: (..., hidden_size) - will be passed through sigmoid
        normed = rms_norm_gated(x, gate, self.weight, self.bias)

        # Ensure gate is on the same device as normed (handles fake tensor case)
        gate = gate.to(device=normed.device, dtype=normed.dtype)
        gate_activated = torch.sigmoid(gate)

        return normed * gate_activated

# TODO refactor for ve gates
class KimiLinearSelfAttention(nn.Module):
    # noinspection PyUnusedLocal
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim: int, expand_v: float = 1.0, mode: str = 'chunk',
                 use_short_conv: bool = True, allow_neg_eigval: bool = False, conv_size: int = 4, conv_bias: bool = False, layer_idx: int = 0, receives_ve: bool = False):
        assert allow_neg_eigval == False, "allow_neg_eigval is not yet supported"
        assert mode == 'chunk', "mode must be 'chunk'; fused_recurrent unsupported for training, future support for inference"
        super().__init__()
        self.mode = mode
        self.allow_neg_eigval = allow_neg_eigval
        self.hidden_size = dim
        self.num_heads = num_heads
        self.num_v_heads = num_heads
        self.head_dim = head_dim
        self.head_k_dim = head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.layer_idx = layer_idx
        self.q_proj = nn.Linear(dim, self.key_dim, bias=False, dtype=torch.bfloat16)
        self.k_proj = nn.Linear(dim, self.key_dim, bias=False, dtype=torch.bfloat16)
        self.v_proj = nn.Linear(dim, self.value_dim, bias=False, dtype=torch.bfloat16)
        if use_short_conv:
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation="silu", bias=conv_bias, dtype=torch.bfloat16)
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation="silu", bias=conv_bias, dtype=torch.bfloat16)
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation="silu", bias=conv_bias, dtype=torch.bfloat16)
        self.f_proj = nn.Sequential(nn.Linear(dim, self.head_v_dim, bias=False, dtype=torch.bfloat16), nn.Linear(self.head_v_dim, self.key_dim, bias=False, dtype=torch.bfloat16))
        self.b_proj = nn.Linear(dim, self.num_heads, bias=False, dtype=torch.bfloat16)
        self.A_log = nn.Parameter(
            torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16)).view(1, 1, -1, 1)
        )
        self.A_log._no_weight_decay = True
        self.dt_bias = nn.Parameter(torch.zeros(self.key_dim, dtype=torch.float32))
        self.dt_bias._no_weight_decay = True
        self.g_proj = nn.Sequential(nn.Linear(dim, self.head_v_dim, bias=False, dtype=torch.bfloat16), nn.Linear(self.head_v_dim, self.value_dim, bias=True, dtype=torch.bfloat16) )
        # self.o_norm = FusedRMSNormGated(self.head_v_dim, activation="sigmoid", eps=1e-5, dtype=torch.bfloat16)
        self.o_norm = CompilableFusedRMSNormGated(self.head_v_dim, activation="sigmoid", dtype=torch.bfloat16)
        self.o_proj = nn.Linear(self.value_dim, dim, bias=False, dtype=torch.bfloat16)
        self._recurrent_state = None
        self._conv_state = None
        self._seen = 0
        self._cache = Cache() if Cache is not None else None
        # TODO init weights

    def reset_history(self):
        self._recurrent_state = None
        self._conv_state = None
        self._seen = 0
        if self._cache is not None:
            self._cache = Cache()

    def _mix_v_with_ve(self, v, ve, sa_lambdas):
        if ve is None:
            return v
        lam0 = sa_lambdas[0].to(dtype=v.dtype)
        lam1 = sa_lambdas[1].to(dtype=v.dtype)
        ve_ = ve.to(dtype=v.dtype, device=v.device).view_as(v)
        # print(f"lam0.device={lam0.device}, lam1.device={lam1.device}, v.device={v.device}, ve_.device={ve_.device}")
        return lam0 * v + lam1 * ve_

    # @torch.compiler.disable()
    def _kda_eager(self, x, q, k, v, g, beta, rec, cu_seqlens, use_cache: bool):
        A_log = self.A_log.to(device=g.device)
        dt_bias = self.dt_bias.to(device=g.device)
        # g = rearrange(g, "... (h d) -> ... h d", h=self.num_heads, d=self.head_k_dim) # expects g.shape(-1) == H * head_dim
        g = fused_kda_gate(g, A_log, int(self.head_k_dim), dt_bias, None, 1.0, 20.0)
        o, rec = chunk_kda(q=q, k=k, v=v, g=g, beta=beta, initial_state=rec, output_final_state=use_cache,
                           use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens)
        gp_raw = self.g_proj(x)
        gp = rearrange(gp_raw.to(device=o.device, dtype=o.dtype), "... (h d) -> ... h d", d=self.head_v_dim)
        o = self.o_norm(o, gp)
        return o, rec


    def _forward_core(self, x: Tensor, ve: Optional[Tensor],  attn_mask: Optional[Tensor], use_cache: bool):
        torch._assert(x.shape[0] == 1, "batch size must be 1") # TODO remove
        b, s, _ = x.shape
        # mode = "fused_recurrent" if s <= 64 and not self.training else self.mode
        last_state = None
        if self._cache is not None and use_cache and len(self._cache) > self.layer_idx:
            last_state = self._cache[self.layer_idx]
        elif self._recurrent_state is not None and use_cache:
            last_state = {"recurrent_state": self._recurrent_state, "conv_state": self._conv_state}
        indices, cu_seqlens, _ = get_unpad_data(attn_mask[:, -s:]) if attn_mask is not None else (None, None, None)
        if cu_seqlens is not None:
            cu_seqlens = cu_seqlens.detach()
        if indices is not None:
            x = index_first_axis(rearrange(x, "b t d -> (b t) d"), indices).unsqueeze(0)
        if self.use_short_conv:
            cs_q, cs_k, cs_v = (last_state.get("conv_state") if isinstance(last_state, dict) else (None, None, None)) if last_state is not None else (None, None, None)
            q, cs_q = self.q_conv1d(x=self.q_proj(x), cache=cs_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            k, cs_k = self.k_conv1d(x=self.k_proj(x), cache=cs_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            v, cs_v = self.v_conv1d(x=self.v_proj(x), cache=cs_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            conv_state = (cs_q, cs_k, cs_v)
        else:
            q = torch.nn.functional.silu(self.q_proj(x))
            k = torch.nn.functional.silu(self.k_proj(x))
            v = torch.nn.functional.silu(self.v_proj(x))
            conv_state = None
        g = self.f_proj(x)
        # g = fused_kda_gate(g, self.A_log, self.head_k_dim, g_bias=self.dt_bias)
        q, k = (rearrange(t, "... (h d) -> ... h d", d=self.head_k_dim) for t in (q, k))
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)
        if sa_lambdas is not None and ve is not None:
            v = self._mix_v_with_ve(v, ve, sa_lambdas)
        rec = last_state["recurrent_state"] if isinstance(last_state, dict) else None
        beta = self.b_proj(x).sigmoid().to(dtype=torch.float32, device=x.device)
        o, rec = self._kda_eager(x, q, k, v, g, beta, rec, cu_seqlens, use_cache=use_cache)
        if use_cache:
            self._recurrent_state = rec
            self._conv_state = conv_state
            self._seen += s
            if self._cache is not None:
                self._cache.update(recurrent_state=rec, conv_state=conv_state, layer_idx=self.layer_idx, offset=s)
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o.to(dtype=self.o_proj.weight.dtype, device=self.o_proj.weight.device))
        if indices is not None:
            o = pad_input(o.squeeze(0), indices, b, s)
        return o

    # noinspection PyUnusedLocal
    def forward(self, x: Tensor, ve: Tensor, sa_lambdas: Tensor, attn_mask: Tensor = None, sliding_window_num_blocks: Tensor = None, block_mask: Optional[Tensor] = None):
        assert block_mask is None, "block_mask not supported for KimiLinearSelfAttention"
        return self._forward_core(x, ve, sa_lambdas, attn_mask, use_cache=False)

    # noinspection PyUnusedLocal
    def prefill(self, x: Tensor, ve: Tensor, sa_lambdas: Tensor, attn_mask: Tensor = None, debug: bool = False):
        y = self._forward_core(x, ve, sa_lambdas, attn_mask, use_cache=True)
        return y, None, None

    # noinspection PyUnusedLocal
    def step(self, x: Tensor, k_ctx: Optional[Tensor], v_ctx: Optional[Tensor], pos: int, ve: Tensor, sa_lambdas: Tensor, window: int | None):
        y = self._forward_core(x, ve, sa_lambdas, attn_mask=None, use_cache=True)
        return y, None, None
