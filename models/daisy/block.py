from typing import Optional
import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import BlockMask

from models.daisy.attention_protocol import AttentionProtocol
from models.daisy.attention import CausalSelfAttention
from models.daisy.mlp import MLP
from models.daisy.functional import norm
from torch import Tensor

# class NoOpAttention(nn.Module):
#     """Attention stub that returns zeros, so we can avoid Python branches in Block.forward."""
#
#     def forward(
#         self,
#         x: Tensor,
#         ve: Optional[Tensor],
#         block_mask: Optional[BlockMask] = None,
#     ) -> Tensor:
#         # preserves shape & device, pure tensor op
#         return  zeros_like(x)
#
#     def step(self, x, k_ctx, v_ctx, pos, ve,  window):
#         # return zero output and no new KV state
#         return  zeros_like(x), None, None
#
#     def prefill(self, x, ve: Optional[Tensor],  attn_mask, debug=False):
#         # zero output, no KV state
#         return  zeros_like(x), None, None

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int, head_dim: int, has_attn: bool, attn_impl: str = 'standard', receives_ve: bool = False):
        super().__init__()
        self.g_x = nn.Parameter(torch.tensor(10.0)) # side-band residual scaling
        self.attn: AttentionProtocol | None = None
        if has_attn:
            if attn_impl == 'kimi_linear':
                from models.daisy.attention_kimi import KimiLinearSelfAttention
                if layer_idx % 4 == 0: self.attn = KimiLinearSelfAttention(dim, num_heads, max_seq_len, head_dim,receives_ve)
                else: self.attn = CausalSelfAttention(dim, num_heads, head_dim, receives_ve)
            elif attn_impl == 'standard':
                self.attn = CausalSelfAttention(dim, num_heads, head_dim, receives_ve)
            else:
                raise ValueError(f'Unknown attn_impl: {attn_impl}')
        self.mlp = MLP(dim)
        self.layer_idx = layer_idx

    def reset_history(self):
        if self.attn is not None:
            self.attn.reset_history()

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, block_mask: Optional[BlockMask] = None, attn_mask: Optional[Tensor] = None):
        g_x = torch.sigmoid(self.g_x)
        x = g_x * x + (1.0 - g_x) * x0
        if self.attn is not None:
            x = x + self.attn(x, ve, block_mask=block_mask, attn_mask=attn_mask)
        x = x + self.mlp(norm(x))
        return x

    def step(self, x, ve, x0, k_ctx, v_ctx, pos,  window):
        g_x = torch.sigmoid(self.g_x)
        x = g_x * x + (1.0 - g_x) * x0
        if self.attn is not None:
            y_att, k_new, v_new = self.attn.step(x, k_ctx, v_ctx, pos, ve, window=window)
            x = x + y_att
        else:
            k_new = v_new = None
        x = x + self.mlp(norm(x))
        return x, k_new, v_new

    def prefill(self, x, ve: Optional[Tensor], x0,  attn_mask=None, debug=False):
        g_x = torch.sigmoid(self.g_x)
        x = g_x * x + (1.0 - g_x) * x0
        if self.attn is not None:
            y, k, v = self.attn.prefill(x, ve,  attn_mask, debug=debug)
            x = x + y
        else:
            k = v = None
        x = x + self.mlp(norm(x))
        return x, k, v
