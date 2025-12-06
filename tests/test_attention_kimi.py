import os
import sys
import importlib.util

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

def _import_from_path(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for these tests")
    return torch.device("cuda")

@pytest.fixture(scope="module")
def kimi_impl(device):
    here = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/daisy/"))
    mod = _import_from_path(os.path.join(here, "attention_kimi.py"), "attention_kimi_impl")
    return mod

@pytest.fixture(scope="module")
def kimi_ref():
    try:
        from fla.layers.kda import KimiDeltaAttention as RefKimi
    except Exception as e:
        pytest.skip(f"Reference KimiDeltaAttention unavailable: {e}")
    return RefKimi

def _make_modules(kimi_impl, kimi_ref_cls, *, dim=64, num_heads=4, head_dim=16, expand_v=1.0, use_short_conv=True):
    impl = kimi_impl.KimiLinearSelfAttention(
        dim=dim,
        num_heads=num_heads,
        max_seq_len=1024,
        head_dim=head_dim,
        expand_v=expand_v,
        mode="chunk",
        use_short_conv=use_short_conv,
        allow_neg_eigval=False,
        conv_size=4,
        conv_bias=False,
        layer_idx=0,
    )
    ref = kimi_ref_cls(
        mode="chunk",
        hidden_size=dim,
        expand_v=expand_v,
        head_dim=head_dim,
        num_heads=num_heads,
        num_v_heads=num_heads,
        use_short_conv=use_short_conv,
        allow_neg_eigval=False,
        conv_size=4,
        conv_bias=False,
        layer_idx=0,
    )
    impl.eval()
    ref.eval()
    return impl, ref

def _make_inputs(model, T, device, dtype=torch.bfloat16):
    x = torch.randn(1, T, model.hidden_size, device=device, dtype=dtype)
    ve = torch.randn(1, T, model.num_v_heads, model.head_v_dim, device=device, dtype=dtype)
    sa_lambdas = torch.tensor([0.7, 0.3], device=device, dtype=dtype)
    return x, ve, sa_lambdas

def _copy_weights(dst_module, src_module):
    dst_sd = dst_module.state_dict()
    src_sd = src_module.state_dict()
    copied, missing = [], []

    def try_set(k_dst, k_src):
        try:
            if dst_sd[k_dst].shape == src_sd[k_src].shape:
                dst_sd[k_dst].copy_(src_sd[k_src])
                copied.append((k_dst, k_src))
                return True
            return False
        except KeyError:
            return False

    for k in list(dst_sd.keys()):
        if k in src_sd and dst_sd[k].shape == src_sd[k].shape:
            dst_sd[k].copy_(src_sd[k])
            copied.append((k, k))

    name_map = [
        ("q_proj", ["attn_q", "q_proj", "q"]),
        ("k_proj", ["attn_k", "k_proj", "k"]),
        ("v_proj", ["attn_v", "v_proj", "v"]),
        ("o_proj", ["attn_out", "o_proj", "out_proj", "o"]),
        ("q_norm", ["q_norm"]),
        ("k_norm", ["k_norm"]),
        ("v_norm", ["v_norm"]),
        ("g_proj", ["g_proj", "gate_proj"]),
        ("b_proj", ["b_proj"]),
        ("f_proj", ["f_proj"]),
        ("o_norm", ["o_norm", "out_norm"]),
        ("A_log", ["A_log", "A"]),
        ("dt_bias", ["dt_bias", "dt"]),
        ("q_conv1d", ["q_conv1d", "q_conv"]),
        ("k_conv1d", ["k_conv1d", "k_conv"]),
        ("v_conv1d", ["v_conv1d", "v_conv"]),
    ]

    for k_dst in list(dst_sd.keys()):
        if any(k_dst == d for d, _ in copied):
            continue
        base = k_dst.split(".")[0]
        for dst_token, src_tokens in name_map:
            if base == dst_token:
                suffix = k_dst[len(base):]
                for st in src_tokens:
                    cand = st + suffix
                    if try_set(k_dst, cand):
                        break

    dst_module.load_state_dict(dst_sd, strict=False)

    for k in dst_sd.keys():
        if not any(k == d for d, _ in copied):
            missing.append(k)
    return copied, missing

@pytest.mark.cuda
def test_forward_shapes_and_dtypes(device, kimi_impl):
    torch.manual_seed(0)
    m = kimi_impl.KimiLinearSelfAttention(
        dim=96, num_heads=6, max_seq_len=256, head_dim=16, expand_v=1.0,
        mode="chunk", use_short_conv=True, allow_neg_eigval=False, conv_size=4, conv_bias=False, layer_idx=0
    ).to(device)
    m.eval()

    x, ve, sa_l = _make_inputs(m, T=33, device=device)
    with torch.no_grad():
        y = m.forward(x, ve, sa_l, attn_mask=None, sliding_window_num_blocks=None, block_mask=None)
    assert y.shape == (1, 33, m.hidden_size)
    assert y.dtype == m.o_proj.weight.dtype
    assert y.is_cuda

@pytest.mark.cuda
def test_masking_unpad_matches_trimmed_forward(device, kimi_impl):
    torch.manual_seed(1)
    m = kimi_impl.KimiLinearSelfAttention(
        dim=64, num_heads=4, max_seq_len=512, head_dim=16, expand_v=1.0,
        mode="chunk", use_short_conv=True, allow_neg_eigval=False, conv_size=4, conv_bias=False, layer_idx=0
    ).to(device)
    m.eval()

    T = 40
    valid = 29
    x, ve, sa_l = _make_inputs(m, T=T, device=device)
    attn_mask = torch.zeros(1, T, device=device, dtype=torch.bool)
    attn_mask[:, :valid] = True

    with torch.no_grad():
        y_mask = m.forward(x, ve, sa_l, attn_mask=attn_mask, sliding_window_num_blocks=None, block_mask=None)
        y_trim = m.forward(x[:, :valid], ve[:, :valid], sa_l, attn_mask=None, sliding_window_num_blocks=None, block_mask=None)

    assert torch.allclose(y_mask[:, :valid], y_trim, rtol=1e-2, atol=1e-2)

@pytest.mark.cuda
def test_streaming_step_matches_full_forward(device, kimi_impl):
    torch.manual_seed(2)
    m = kimi_impl.KimiLinearSelfAttention(
        dim=64, num_heads=4, max_seq_len=1024, head_dim=16, expand_v=1.0,
        mode="chunk", use_short_conv=True, allow_neg_eigval=False, conv_size=4, conv_bias=False, layer_idx=0
    ).to(device)
    m.eval()

    T = 48
    x, ve, sa_l = _make_inputs(m, T=T, device=device)

    with torch.no_grad():
        y_full = m.forward(x, ve, sa_l, attn_mask=None, sliding_window_num_blocks=None, block_mask=None)

    m.reset_history()
    outs = []
    with torch.no_grad():
        for i in range(T):
            yi, _, _ = m.step(x[:, i:i+1], k_ctx=None, v_ctx=None, pos=i, ve=ve[:, i:i+1], sa_lambdas=sa_l, window=None)
            outs.append(yi)
    y_stream = torch.cat(outs, dim=1)

    assert torch.allclose(y_full, y_stream, rtol=2e-2, atol=2e-2)

@pytest.mark.cuda
def test_prefill_then_step_matches_full_forward(device, kimi_impl):
    torch.manual_seed(3)
    m = kimi_impl.KimiLinearSelfAttention(
        dim=80, num_heads=5, max_seq_len=1024, head_dim=16, expand_v=1.0,
        mode="chunk", use_short_conv=True, allow_neg_eigval=False, conv_size=4, conv_bias=False, layer_idx=0
    ).to(device)
    m.eval()

    T = 64
    split = 17
    x, ve, sa_l = _make_inputs(m, T=T, device=device)

    with torch.no_grad():
        y_full = m.forward(x, ve, sa_l, attn_mask=None, sliding_window_num_blocks=None, block_mask=None)

    m.reset_history()
    with torch.no_grad():
        y0, _, _ = m.prefill(x[:, :split], ve[:, :split], sa_l, attn_mask=None, debug=False)
        outs = [y0]
        for i in range(split, T):
            yi, _, _ = m.step(x[:, i:i+1], k_ctx=None, v_ctx=None, pos=i, ve=ve[:, i:i+1], sa_lambdas=sa_l, window=None)
            outs.append(yi)
    y_inc = torch.cat(outs, dim=1)

    assert torch.allclose(y_full, y_inc, rtol=2e-2, atol=2e-2)

@pytest.mark.cuda
def test_backward_grads_exist_and_finite(device, kimi_impl):
    torch.manual_seed(4)
    m = kimi_impl.KimiLinearSelfAttention(
        dim=48, num_heads=3, max_seq_len=256, head_dim=16, expand_v=1.0,
        mode="chunk", use_short_conv=True, allow_neg_eigval=False, conv_size=4, conv_bias=False, layer_idx=0
    ).to(device)
    m.train()

    x, ve, sa_l = _make_inputs(m, T=24, device=device)
    x.requires_grad_(True)
    out = m.forward(x, ve, sa_l, attn_mask=None, sliding_window_num_blocks=None, block_mask=None)
    loss = out.float().pow(2).mean()
    loss.backward()

    n_grads = 0
    for n, p in m.named_parameters():
        if p.grad is None:
            continue
        assert torch.isfinite(p.grad).all(), f"Non-finite grad in {n}"
        n_grads += 1
    assert n_grads > 0

@pytest.mark.cuda
def test_equivalence_with_reference_if_weights_aligned(device, kimi_impl, kimi_ref):
    torch.manual_seed(5)
    impl, ref = _make_modules(kimi_impl, kimi_ref, dim=64, num_heads=4, head_dim=16, expand_v=1.0, use_short_conv=True)
    impl = impl.to(device)
    ref = ref.to(device)

    copied, missing = _copy_weights(impl, ref)
    copied_ratio = len(copied) / max(1, len(list(impl.state_dict().keys())))
    if copied_ratio < 0.6:
        pytest.skip(f"Could not align enough weights with reference (copied {len(copied)}; missing {len(missing)})")

    x, ve, sa_l = _make_inputs(impl, T=32, device=device)
    with torch.no_grad():
        y_impl = impl.forward(x, ve, sa_l, attn_mask=None, sliding_window_num_blocks=None, block_mask=None)
        y_ref = ref.forward(x, ve, sa_l, attn_mask=None)

    assert torch.allclose(y_impl, y_ref, rtol=3e-2, atol=3e-2)
