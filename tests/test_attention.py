import torch
import pytest
from models.daisy.attention import CausalSelfAttention


def make_attn(dim=16, heads=2, head_dim=8, max_seq_len=64):
    attn = CausalSelfAttention(dim=dim, num_heads=heads, max_seq_len=max_seq_len, head_dim=head_dim)
    # For numerical stability and CPU support, run in float32 for tests
    attn.qkvo_w = torch.nn.Parameter(attn.qkvo_w.float())
    # Reinitialize output projection to non-zero so outputs aren't trivially zero
    with torch.no_grad():
        attn.qkvo_w[3].normal_(mean=0.0, std=0.02)
    return attn


def batched_sdpa_reference(attn: CausalSelfAttention, x: torch.Tensor, lambdas: torch.Tensor, ve: torch.Tensor):
    # Compute reference outputs by reproducing the internal computations in batched form using SDPA
    B, T, D = x.shape
    H, Hd = attn.num_heads, attn.head_dim
    qkv = torch.nn.functional.linear(x, attn.qkvo_w[:3].flatten(end_dim=1))
    q, k, v = qkv.view(B, T, 3 * H, Hd).chunk(3, dim=-2)
    from models.daisy.functional import norm
    q, k = norm(q), norm(k)
    q, k = attn.rotary(q), attn.rotary(k)
    v = norm(v)
    # Match the model's behavior: if the attention module mixes external values (receives_ve=True),
    # use its learnable gate; otherwise, the step() path ignores ve entirely.
    if ve is not None:
        if getattr(attn, "receives_ve", False):
            g_x = torch.sigmoid(attn.g_ve).to(v.dtype)
            v = g_x * v + (1.0 - g_x) * ve.view_as(v)
        else:
            # receives_ve=False -> model ignores ve in step() so do the same here
            v = v
    q_ = q.transpose(1, 2)  # (B, H, T, Hd)
    k_ = k.transpose(1, 2)  # (B, H, T, Hd)
    v_ = v.transpose(1, 2)  # (B, H, T, Hd)
    y = torch.nn.functional.scaled_dot_product_attention(
        q_, k_, v_, is_causal=True, scale=attn.attn_scale
    ).transpose(1, 2)
    y = y.contiguous().view(B, T, H * Hd)
    y = torch.nn.functional.linear(y, attn.qkvo_w[3])
    return y


def per_token_windowed_reference(attn: CausalSelfAttention, x: torch.Tensor, lambdas: torch.Tensor, ve: torch.Tensor, window: int):
    assert ve is not None, "Ve must be provided for windowed reference"
    # Compute q/k/v for the whole sequence, then for each token t apply SDPA over the last `window` keys/values
    B, T, D = x.shape
    H, Hd = attn.num_heads, attn.head_dim
    qkv = torch.nn.functional.linear(x, attn.qkvo_w[:3].flatten(end_dim=1))
    q, k, v = qkv.view(B, T, 3 * H, Hd).chunk(3, dim=-2)
    from models.daisy.functional import norm
    q, k = norm(q), norm(k)
    # Use rotary on full sequences for equivalence to step
    q, k = attn.rotary(q), attn.rotary(k)
    v = norm(v)

    v = lambdas[0] * v + lambdas[1] * ve.view_as(v)

    q_all = q.transpose(1, 2)  # (B, H, T, Hd)
    k_all = k.transpose(1, 2)
    v_all = v.transpose(1, 2)

    outs = []
    for t in range(T):
        start = max(0, t + 1 - window)
        q_t = q_all[:, :, t : t + 1, :]  # (B, H, 1, Hd)
        k_t = k_all[:, :, start : t + 1, :]  # (B, H, S, Hd)
        v_t = v_all[:, :, start : t + 1, :]  # (B, H, S, Hd)
        y_t = torch.nn.functional.scaled_dot_product_attention(
            q_t, k_t, v_t, is_causal=False, scale=attn.attn_scale
        )  # (B, H, 1, Hd)
        y_t = y_t.transpose(1, 2).reshape(B, 1, H * Hd)
        y_t = torch.nn.functional.linear(y_t, attn.qkvo_w[3])
        outs.append(y_t)
    return torch.cat(outs, dim=1)


@pytest.mark.parametrize("T,window", [(12, 12), (8, 16)])
def test_step_matches_batched_sdpa_full_context(T, window):
    with torch.no_grad():
        torch.manual_seed(0)
        dim, heads, head_dim = 16, 2, 8

        attn = make_attn(dim=dim, heads=heads, head_dim=head_dim, max_seq_len=64)

        B = 1
        x = torch.randn(B, T, dim, dtype=torch.float32)
        lambdas = torch.tensor([0.7, 0.3], dtype=torch.float32)
        ve = torch.randn(B, T, heads * head_dim, dtype=torch.float32)

        # Reference with full causal context in batch
        y_ref = batched_sdpa_reference(attn, x, lambdas, ve)

        # Streaming with step (window >= T ensures full context equivalence)
        k_ctx = torch.empty(B, 0, heads, head_dim, dtype=torch.float32)
        v_ctx = torch.empty(B, 0, heads, head_dim, dtype=torch.float32)
        ys = []
        for t in range(T):
            x_t = x[:, t : t + 1, :]
            ve_t = None if ve is None else ve[:, t : t + 1, :]
            y_t, k_t, v_t = attn.step(x_t, k_ctx, v_ctx, pos=t, ve=ve_t, window=window)
            ys.append(y_t)
            # update caches
            k_ctx = torch.cat([k_ctx, k_t], dim=1)
            v_ctx = torch.cat([v_ctx, v_t], dim=1)

        y_step = torch.cat(ys, dim=1)

        assert y_ref.shape == y_step.shape == (B, T, heads * head_dim)
        assert torch.allclose(y_ref, y_step, rtol=1e-4, atol=1e-4)


def test_step_matches_windowed_reference():
    with torch.no_grad():
        torch.manual_seed(123)
        dim, heads, head_dim = 16, 2, 8

        attn = make_attn(dim=dim, heads=heads, head_dim=head_dim, max_seq_len=64)

        # B, T = 1, 20
        B, T = 1, 3
        window = 2
        x = torch.randn(B, T, dim, dtype=torch.float32)
        lambdas = torch.tensor([1.0, 0.0], dtype=torch.float32)
        ve_full = torch.randn(B, T, heads * head_dim, dtype=torch.float32)
        # Reference computed per token using only the last `window` keys/values
        y_ref = per_token_windowed_reference(attn, x, lambdas, ve_full, window=window)

        # Streaming step with caches
        k_ctx = torch.empty(B, 0, heads, head_dim, dtype=torch.float32)
        v_ctx = torch.empty(B, 0, heads, head_dim, dtype=torch.float32)
        ys = []
        for t in range(T):
            ve_step = ve_full.narrow(1, t, 1)
            x_t = x[:, t : t + 1, :]
            y_t, k_t, v_t = attn.step(x_t, k_ctx, v_ctx, pos=t, ve=ve_step, window=window)
            ys.append(y_t)
            k_ctx = torch.cat([k_ctx, k_t], dim=1)
            v_ctx = torch.cat([v_ctx, v_t], dim=1)

        y_step = torch.cat(ys, dim=1)

        assert y_ref.shape == y_step.shape == (B, T, heads * head_dim)
        assert torch.allclose(y_ref, y_step, rtol=1e-4, atol=1e-4)
