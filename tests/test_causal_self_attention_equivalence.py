
import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask, BlockMask

from models.daisy.attention import CausalSelfAttention, is_flex_available


def _make_block_mask(T: int, H: int, W: int, device: torch.device):
    dev_str = "cuda" if device.type == "cuda" else "cpu"
    def mask_mod(b, h, q_idx, kv_idx):
        return (kv_idx <= q_idx) & (kv_idx >= q_idx - (W - 1))
    return create_block_mask(mask_mod, 1, H, T, T, device=dev_str)


@pytest.mark.parametrize("T,W", [(128,128), (131,128)])
def test_causal_self_attention_forward_equals_step(T, W,  monkeypatch):
    monkeypatch.setenv("DISABLE_O_ZERO_INIT", "1")
    with torch.inference_mode():
        torch.manual_seed(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        H, D = 1, 2
        dim, max_seq_len = H * D, 256

        dtype = torch.bfloat16
        x_t = torch.ones(T, dtype=torch.int, device=device) #pseudo-tokens
        x = torch.randn(1, T, dim, device=device, dtype=dtype) #pseudo-embedding
        lambdas = torch.tensor([1.0, 0.5], device=device, dtype=dtype)
        ve_full = torch.randn(1, T, H, D, device=device, dtype=dtype)

        if torch.cuda.is_available():
            m = CausalSelfAttention(dim=dim, num_heads=H, max_seq_len=max_seq_len, head_dim=D).to(device).eval()
            mask = _make_block_mask(T, H, W, device)
            y_full = m.forward(x, ve=ve_full, block_mask=mask, sa_lambdas=lambdas)
        else:
            m = CausalSelfAttention(dim=dim, num_heads=H, max_seq_len=max_seq_len, head_dim=D).to(device).eval()
            y_full = m.forward(x, ve=ve_full, sa_lambdas=lambdas)
        # Initialize empty KV caches and append per token to match causal semantics
        K = torch.empty(1, 0, H, D, device=device, dtype=dtype)
        V = torch.empty(1, 0, H, D, device=device, dtype=dtype)
        ys = []
        for pos in range(T):
            x_pos = x[:, pos:pos+1]
            ve_pos = None if ve_full is None else ve_full[:, pos:pos+1]
            y_pos, k_new, v_new = m.step(x_pos, k_ctx=K, v_ctx=V, pos=pos, ve=ve_pos, sa_lambdas=lambdas, window=W)
            K = torch.cat([K, k_new], dim=1)
            V = torch.cat([V, v_new], dim=1)
            ys.append(y_pos)
        y_step = torch.cat(ys, dim=1)

        assert y_full.shape == y_step.shape
        max_err = (y_full.float() - y_step.float()).abs().max().item()
        assert torch.allclose(y_full, y_step, atol=2e-2, rtol=2e-2), f"max_err={max_err}"
