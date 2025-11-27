from typing import Optional

import torch.nn as nn
from torch.nn.attention.flex_attention import BlockMask

from models.daisy.attention_protocol import AttentionProtocol
from models.daisy.attention import CausalSelfAttention
from models.daisy.mlp import MLP
from models.daisy.functional import norm
from torch import Tensor, zeros_like

# class NoOpAttention(nn.Module):
#     """Attention stub that returns zeros, so we can avoid Python branches in Block.forward."""
#
#     def forward(
#         self,
#         x: Tensor,
#         ve: Optional[Tensor],
#         sa_lambdas: Optional[Tensor],
#         block_mask: Optional[BlockMask] = None,
#     ) -> Tensor:
#         # preserves shape & device, pure tensor op
#         return  zeros_like(x)
#
#     def step(self, x, k_ctx, v_ctx, pos, ve, sa_lambdas, window):
#         # return zero output and no new KV state
#         return  zeros_like(x), None, None
#
#     def prefill(self, x, ve: Optional[Tensor], sa_lambdas, attn_mask, debug=False):
#         # zero output, no KV state
#         return  zeros_like(x), None, None

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int, head_dim: int, has_attn: bool, attn_impl: str = 'standard', dynamic_shapes: bool = False):
        super().__init__()

        self.attn: AttentionProtocol | None = None
        if has_attn:
            if attn_impl == 'kimi_linear':
                from models.daisy.attention_kimi import KimiLinearSelfAttention
                if layer_idx % 4 == 0: self.attn = KimiLinearSelfAttention(dim, num_heads, max_seq_len, head_dim)
                else: self.attn = CausalSelfAttention(dim, num_heads, max_seq_len, head_dim)
            elif attn_impl == 'standard':
                self.attn = CausalSelfAttention(dim, num_heads, max_seq_len, head_dim, dynamic_shapes)
            else:
                raise ValueError(f'Unknown attn_impl: {attn_impl}')
        self.mlp = MLP(dim)

    def reset_history(self):
        if self.attn is not None:
            self.attn.reset_history()

    def forward(self, x: Tensor, ve: Tensor, x0: Tensor, lambdas: Tensor, sa_lambdas: Tensor, block_mask: Optional[BlockMask] = None, attn_mask: Optional[Tensor] = None):
        x = lambdas[0] * x + lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(x, ve, sa_lambdas, block_mask=block_mask, attn_mask=attn_mask)
        x = x + self.mlp(norm(x))
        return x

    def step(self, x, ve, x0, k_ctx, v_ctx, pos, lambdas, sa_lambdas, window):
        x = lambdas[0] * x + lambdas[1] * x0
        if self.attn is not None:
            y_att, k_new, v_new = self.attn.step(x, k_ctx, v_ctx, pos, ve, sa_lambdas, window=window)
            x = x + y_att
        else:
            k_new = v_new = None
        x = x + self.mlp(norm(x))
        return x, k_new, v_new

    def prefill(self, x, ve: Optional[Tensor], x0, lambdas, sa_lambdas, attn_mask=None, debug=False):
        x = lambdas[0] * x + lambdas[1] * x0
        if self.attn is not None:
            y, k, v = self.attn.prefill(x, ve, sa_lambdas, attn_mask, debug=debug)
            x = x + y
        else:
            k = v = None
        x = x + self.mlp(norm(x))
        return x, k, v
