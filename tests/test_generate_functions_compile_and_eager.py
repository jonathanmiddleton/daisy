import math
import pytest
import torch
import torch.nn.functional as F

from inference.generate import (
    _topk_filter,
    _topp_filter,
    _gumbel,
    _sample,
    _repetition_penalty,
)


def has_cuda():
    return torch.cuda.is_available()


def has_mps():
    return hasattr(torch, "mps") and torch.mps.is_available()


def accel_available():
    return has_cuda() or has_mps()


def pick_device():
    if has_cuda():
        return torch.device("cuda")
    if has_mps():
        return torch.device("mps")
    return torch.device("cpu")


def compile_backend_for_device():
    # Work around known inductor issues on MPS by using eager backend for torch.compile
    if has_mps() and not has_cuda():
        return "eager"
    return None


def manual_seed_all(seed: int):
    torch.manual_seed(seed)
    if has_cuda():
        torch.cuda.manual_seed_all(seed)


def maybe_compile(fn):
    backend = compile_backend_for_device()
    return (lambda f: torch.compile(f, backend=backend) if backend else torch.compile(f))(fn)


@pytest.mark.parametrize("compiled", [False, True] if accel_available() else [False])
def test_topk_filter_correct_and_compiled_equivalence(compiled):
    device = pick_device()
    dtype = torch.float32
    x = torch.tensor([[0.1, -1.0, 2.5, 0.0], [5.0, 4.0, 3.0, 2.0]], device=device, dtype=dtype)

    fn = _topk_filter
    if compiled:
        fn = maybe_compile(fn)

    # Edge cases: k None, <=0, >= last dim -> unchanged
    for k in [None, 0, -3, x.size(-1), x.size(-1) + 5]:
        y = fn(x, k)
        assert torch.allclose(y, x), f"Expected unchanged for k={k}"
        assert y.device == x.device and y.dtype == x.dtype

    # Proper top-k filtering
    k = 2
    y = fn(x, k)
    # Build expected manually
    v, i = torch.topk(x, k)
    expected = torch.full_like(x, float("-inf"))
    expected.scatter_(dim=-1, index=i, src=v)
    assert torch.allclose(y, expected)


@pytest.mark.parametrize("compiled", [False, True] if accel_available() else [False])
def test_topp_filter_correct_and_compiled_equivalence(compiled):
    device = pick_device()
    dtype = torch.float32
    # Chosen logits so that probabilities are easy to reason about
    x = torch.tensor([[0.0, 2.0, 1.0, -1.0], [3.0, 3.0, 1.0, -2.0]], device=device, dtype=dtype)

    fn = _topp_filter
    if compiled:
        fn = maybe_compile(fn)

    # Edge cases: p None, <=0, >=1 -> unchanged
    for p in [None, 0.0, -0.1, 1.0, 1.5]:
        y = fn(x, p)
        assert torch.allclose(y, x), f"Expected unchanged for p={p}"
        assert y.device == x.device and y.dtype == x.dtype

    # Proper top-p filtering
    p = 0.70
    y = fn(x, p)
    # Manual reference implementation (mirrors the algorithm)
    probs = F.softmax(x, dim=-1)
    s, i = torch.sort(probs, dim=-1, descending=True)
    c = torch.cumsum(s, dim=-1)
    ms = c <= p
    ms[..., 0] = True
    m = torch.zeros_like(ms, dtype=torch.bool).scatter(-1, i, ms)
    expected = x.masked_fill(~m, float("-inf"))
    assert torch.allclose(y, expected)


@pytest.mark.parametrize("compiled", [False, True] if accel_available() else [False])
def test_gumbel_correct_and_compiled_equivalence(compiled):
    device = pick_device()
    dtype = torch.float32
    shape = (4, 5)

    fn = _gumbel
    if compiled:
        fn = maybe_compile(fn)

    seed = 1234
    manual_seed_all(seed)
    out = fn(shape, device, dtype)

    manual_seed_all(seed)
    u = torch.rand(shape, device=device, dtype=dtype)
    expected = -torch.log(-torch.log(u.clamp_min_(1e-6)))

    assert out.shape == shape
    assert out.device.type == device.type
    assert out.dtype == dtype
    assert torch.allclose(out, expected, atol=0, rtol=0), "_gumbel must match analytic transform"


@pytest.mark.parametrize("compiled", [False, True] if accel_available() else [False])
def test_sample_device_correct_and_compiled_equivalence(compiled):
    device = pick_device()
    dtype = torch.float32

    fn = _sample
    if compiled:
        fn = maybe_compile(fn)

    # Deterministic path: temperature == 0.0 -> pure argmax, ignores top_k/top_p
    logits = torch.tensor([[0.1, -0.2, 0.3], [1.0, 5.0, 5.0], [-1.0, -2.0, -3.0]], device=device, dtype=dtype)
    y = fn(logits, temperature=0.0, top_k=1, top_p=0.1)
    expected_argmax = torch.argmax(logits, dim=-1)
    assert torch.equal(y, expected_argmax)

    # Stochastic path: reproduce exact algorithm with a fixed seed
    B, V = 8, 17
    seed = 2025
    # Build deterministic logits so RNG stream is only used for Gumbel samples
    base = torch.arange(B * V, device=device, dtype=dtype).view(B, V)
    logits = torch.sin(base / 7.0) * 2.0 + torch.cos(base / 5.0)
    temperature = 1.3
    top_k = 7
    top_p = 0.8

    # Reference implementation (do not call _sample_device to avoid circularity)
    manual_seed_all(seed)
    x = logits / max(temperature, 1e-6)
    # top-k
    if top_k is not None and 0 < top_k < V:
        v, i = torch.topk(x, top_k)
        x_k = torch.full_like(x, float("-inf"))
        x_k.scatter_(dim=-1, index=i, src=v)
    else:
        x_k = x
    # top-p
    if top_p is not None and 0.0 < top_p < 1.0:
        probs = F.softmax(x_k, dim=-1)
        s, i = torch.sort(probs, dim=-1, descending=True)
        c = torch.cumsum(s, dim=-1)
        ms = c <= top_p
        ms[..., 0] = True
        m = torch.zeros_like(ms, dtype=torch.bool).scatter(-1, i, ms)
        x_filt = x_k.masked_fill(~m, float("-inf"))
    else:
        x_filt = x_k
    g = _gumbel(x_filt.shape, device=device, dtype=dtype)
    expected = torch.argmax(x_filt + g, dim=-1)

    # Now compute with function under test from the same RNG seed
    manual_seed_all(seed)
    got = fn(logits, temperature=temperature, top_k=top_k, top_p=top_p)

    assert torch.equal(got, expected), "_sample_device must match the Gumbel-max reference exactly"


@pytest.mark.parametrize("compiled", [False, True] if accel_available() else [False])
def test_repetition_penalty_device_correct_and_compiled_equivalence(compiled):
    device = pick_device()
    dtype = torch.float32

    fn = _repetition_penalty
    if compiled:
        fn = maybe_compile(fn)

    # Unchanged when rep_p == 1 or no history
    logits = torch.tensor([2.0, -1.0, 0.5, 0.0], device=device, dtype=dtype)
    prev_ids_empty = torch.empty(0, dtype=torch.long, device=device)
    one = torch.tensor(1.0, device=device, dtype=dtype)
    rep_p = torch.tensor(1.0, device=device, dtype=dtype)

    y1 = fn(logits, prev_ids_empty, rep_p, one)
    assert torch.allclose(y1, logits)

    prev_ids_some = torch.tensor([0, 1, 2, 1, 0, 0], dtype=torch.long, device=device)
    y2 = fn(logits, prev_ids_some, rep_p, one)
    assert torch.allclose(y2, logits)

    # Actual effect with rep_p > 1
    rep_p = torch.tensor(1.2, device=device, dtype=dtype)
    # Use small rep_w/rep_h/cap to make numbers noticeable and deterministic
    rep_w = 8
    rep_h = 10.0
    cap = 2.0

    prev_slice = prev_ids_some[-rep_w:]
    y = fn(logits, prev_slice, rep_p, one, rep_h=rep_h, cap=cap)

    # Manual computation
    prev = prev_slice
    d = torch.arange(prev.numel(), 0, -1, device=device, dtype=dtype)
    w = (0.5 ** (d / rep_h))
    h = torch.zeros_like(logits)
    h.scatter_add_(0, prev, w)
    h.clamp_max_(cap)
    scale = torch.pow(rep_p, h)
    factor = torch.where(logits > 0, 1.0 / scale, scale)
    expected = logits * factor

    assert torch.allclose(y, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not accel_available(), reason="Requires CUDA or MPS for torch.compile")
def test_topk_filter_eager_vs_compiled_same():
    device = pick_device()
    dtype = torch.float32
    x = torch.randn(3, 11, device=device, dtype=dtype)
    k = 5
    eager = _topk_filter(x, k)
    compiled = maybe_compile(_topk_filter)(x, k)
    assert torch.allclose(eager, compiled)


@pytest.mark.skipif(not accel_available(), reason="Requires CUDA or MPS for torch.compile")
def test_topp_filter_eager_vs_compiled_same():
    device = pick_device()
    dtype = torch.float32
    x = torch.randn(4, 13, device=device, dtype=dtype)
    p = 0.9
    eager = _topp_filter(x, p)
    compiled = maybe_compile(_topp_filter)(x, p)
    assert torch.allclose(eager, compiled)


@pytest.mark.skipif(not accel_available(), reason="Requires CUDA or MPS for torch.compile")
def test_gumbel_eager_vs_compiled_same():
    device = pick_device()
    dtype = torch.float32
    shape = (7, 5)
    seed = 2468
    manual_seed_all(seed)
    eager = _gumbel(shape, device, dtype)
    manual_seed_all(seed)
    compiled = maybe_compile(_gumbel)(shape, device, dtype)
    assert torch.allclose(eager, compiled)


@pytest.mark.skipif(not accel_available(), reason="Requires CUDA or MPS for torch.compile")
def test_sample_device_eager_vs_compiled_same():
    device = pick_device()
    dtype = torch.float32
    B, V = 6, 23
    base = torch.arange(B * V, device=device, dtype=dtype).view(B, V)
    logits = torch.cos(base / 11.0) * 1.7 + torch.sin(base / 3.0)
    temperature = 0.8
    top_k = 9
    top_p = 0.85
    seed = 777
    manual_seed_all(seed)
    r1 = torch.randint(0, V - 1, (B,), device=device, dtype=torch.long)
    eager = _sample(logits, temperature=temperature, top_k=top_k, top_p=top_p)
    manual_seed_all(seed)
    r2 = torch.randint(0, V - 1, (B,), device=device, dtype=torch.long)
    assert torch.equal(r1, r2)
    compiled = maybe_compile(_sample)(logits, temperature=temperature, top_k=top_k, top_p=top_p)
    assert torch.equal(eager, compiled)


@pytest.mark.skipif(not accel_available(), reason="Requires CUDA or MPS for torch.compile")
def test_repetition_penalty_device_eager_vs_compiled_same():
    device = pick_device()
    dtype = torch.float32
    logits = torch.tensor([2.0, -1.0, 0.5, 0.0, -3.1, 4.2], device=device, dtype=dtype)
    prev_ids = torch.tensor([0, 5, 2, 3, 2, 1, 0, 5, 0, 0, 3, 3, 5, 5, 5], dtype=torch.long, device=device)
    rep_p = torch.tensor(1.3, device=device, dtype=dtype)
    one = torch.tensor(1.0, device=device, dtype=dtype)
    prev_slice = prev_ids[-10:]
    kwargs = dict(rep_h=20.0, cap=2.5)
    eager = _repetition_penalty(logits, prev_slice, rep_p, one, **kwargs)
    compiled = torch.compile(_repetition_penalty)(logits, prev_slice, rep_p, one, **kwargs)
    assert torch.allclose(eager, compiled, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(accel_available(), reason="Only runs when neither CUDA nor MPS is available")
def test_compiled_skipped_when_no_accel():
    # The rest of the tests parametrize compiled=[False] automatically when no accelerator
    # This test simply asserts our helper detects the environment as intended.
    assert not accel_available()
