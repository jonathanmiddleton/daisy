import time
import logging
from typing import Generator, Callable
import torch
import torch.nn.functional as F

from tools.compile_helpers import torch_compiled_callable_debug_wrapper, torch_get_guards_from_callable
from inference.kv_cache import KVCache
import os

logger = logging.getLogger(__name__)

def _topk_filter(x, k):
    if k is None or k <= 0 or k >= x.size(-1):
        return x
    v, i = torch.topk(x, k)
    y = torch.full_like(x, float("-inf"))
    y.scatter_(dim=-1, index=i, src=v)
    return y

def _topp_filter(x, p):
    if p is None or p <= 0.0 or p >= 1.0:
        return x
    probs = F.softmax(x, dim=-1)
    s, i = torch.sort(probs, dim=-1, descending=True)
    c = torch.cumsum(s, dim=-1)
    ms = c <= p
    ms[..., 0] = True
    m = torch.zeros_like(ms, dtype=torch.bool).scatter(-1, i, ms)
    return x.masked_fill(~m, float("-inf"))

def _gumbel(shape, device, dtype):
    u = torch.rand(shape, device=device, dtype=dtype)
    return -torch.log(-torch.log(u.clamp_min_(1e-6)))

def _sample(logits, temperature=1.0, top_k=None, top_p=None):
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1)
    x = logits / max(temperature, 1e-6)
    x = _topk_filter(x, top_k)
    x = _topp_filter(x, top_p)
    # Gumbel-Softmax trick
    g = _gumbel(x.shape, x.device, x.dtype)
    y = torch.argmax(x + g, dim=-1)
    return y

def _repetition_penalty(logits, prev: torch.Tensor, rep_p: torch.Tensor, _one: torch.Tensor, rep_h=140.0, cap=3.0):
    # if torch.equal(rep_p, _one) or prev_ids is None or prev_ids.numel() == 0:
    #     return logits
    # prev = prev_ids[-rep_w:]
    d = torch.arange(prev.numel(), 0, -1, device=logits.device, dtype=logits.dtype)
    w = (0.5 ** (d / rep_h))
    h = torch.zeros_like(logits)
    h.scatter_add_(0, prev, w)
    h.clamp_max_(cap)
    scale = torch.pow(rep_p, h)
    factor = torch.where(logits > 0, 1.0 / scale, scale)
    return logits * factor


class Generator:
    def __init__(self, model, window, seed, device=None, dtype=torch.float16, max_seq_len: int = 65536, temperature=1.0, top_k=None, top_p=None, repetition_penalty=1.0, eos_token_id=None):
        if eos_token_id is None:
            eos_token_id = getattr(model, "eos_token_id", None)
        assert eos_token_id is not None
        self.model = model.eval()
        self.device = next(model.parameters()).device if device is None else device
        self.dtype = dtype
        assert self.device is not None
        self.devtype = "cuda" if str(self.device).startswith("cuda") else ("mps" if str(self.device).startswith("mps") else "cpu")
        h = None; d = None
        for b in self.model.blocks:
            a = getattr(b, "attn", None)
            if a is not None:
                h = a.num_heads; d = a.head_dim; break
        self.cache = KVCache(L=len(self.model.blocks), B=1, H=h, W=window, D=d, device=self.device, dtype=dtype)
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.rep_p_t = torch.tensor(repetition_penalty, device=self.device, dtype=dtype)
        self.eos_token_id = torch.tensor(eos_token_id, device=self.device)
        self.max_seq_len = max_seq_len
        self.window = window
        self.vocab_size = self.model.embed.num_embeddings
        self._one = torch.tensor(1, device=self.device, dtype=dtype) # cached value for passing to compiled function
        # set seed across all devices
        self.seed = seed
        torch.manual_seed(self.seed)
        self.history = torch.empty(0, dtype=torch.long, device=self.device)
        self.mps_torch_compile = os.environ.get("DAISY_MPS_TORCH_COMPILE", "0") == "1"
        self.torch_backend_aot = os.environ.get("DAISY_DEBUG_COMPILES", "0") == "1"
        self.debug_mode = os.environ.get("DAISY_DEBUG_MODE", "0") == "1"
        if self.torch_backend_aot:
            logger.info("Enabling debug compiles")
        # maybe compile
        COMPILE_SAMPLE = False
        self.sample = torch.compile(_sample, dynamic=False) if COMPILE_SAMPLE else _sample
        self.apply_repetition_penalty = self._maybe_compile(_repetition_penalty)
        self.model_prefill_not_compiled = self.model.prefill
        self.use_mark_dynamic = True
        self.use_dynamic_compile = False

    def _maybe_compile(self, func):
        """
        Compile if a device is available, with the appropriate context and options. Otherwise return the original function.
        If self.torch_backend_aot is True, perform a test compile with backend="aot_eager".
        """
        if self.devtype == 'mps' and not self.mps_torch_compile:
            return func

        backend = 'inductor' if not self.torch_backend_aot else "aot_eager"

        f_c = None
        with torch.inference_mode():
            if self.devtype == 'mps':
                try:
                    logger.debug(f"Compiling function '{func.__name__}' with backend '{backend}'")
                    f_c = torch.compile(func, dynamic=self.use_dynamic_compile, backend=backend)
                except Exception as e:
                    logger.debug(f"Failed to compile function '{func.__name__}' with backend '{backend}': {e}")
                    f_c = func
            elif self.devtype == 'cuda':
                try:
                    logger.debug(f"Compiling function '{func.__name__}' with backend '{backend}'")
                    f_c = torch.compile(func, dynamic=self.use_dynamic_compile, backend=backend)
                except Exception as e:
                    logger.debug(f"Failed to compile function '{func.__name__}' with backend '{backend}': {e}")
                    f_c = func
            else:
                logger.debug(f"Not compiling function '{func.__name__}'. Unsupported device: '{self.devtype}'")
                f_c = func

        assert f_c is not None
        if self.debug_mode:
            f_c = torch_compiled_callable_debug_wrapper(fn=f_c,post_exec_hook=torch_get_guards_from_callable)

        return f_c

    def _prefill_func_bounds(self) -> list[tuple[Callable[[int], bool], tuple[str, tuple[int,int]]]]:
        """
            Dynamo computed static boundaries for guards. For MPS and torch 2.8.0-2.10.0-pre. Determined empirically.
            For dynamic shapes, Dynamo infers the following constraints:
            1) that sequence length is bounded from above by DaisyCore's window_size-1
            2) that sequence length is bounded from below by 9
            3) that sequence length _may_ be bounded from above by 1023 (MPS SDPA kernel fast path), or
            4) that sequence length _may_ be bounded from below by 2048
         """
        min_sdpa = 9 # magic
        max_for_fast_sdpa = 1023 # inferred from torch 2.9.0 MPS SDPA kernel fast path
        two_zero_four_eight = 2048
        max_seq_len_sdpa = self.window-1
        bounds: list[tuple[Callable[[int], bool], tuple[str, tuple[int,int]]]] = [
            # (Predicate, (Function name, (min,max))
            (lambda n: n <= min_sdpa, ('_model_prefill_not_compiled', (1, min_sdpa)),),
            (lambda n: min_sdpa < n <= max_for_fast_sdpa, (f'_model_prefill_{min_sdpa+1}', (min_sdpa+1, max_for_fast_sdpa))),
            (lambda n: max_for_fast_sdpa < n <= two_zero_four_eight, (f'_model_prefill_{max_for_fast_sdpa + 1}', (max_for_fast_sdpa + 1, two_zero_four_eight))),
            (lambda n: two_zero_four_eight < n, (f'_model_prefill_{two_zero_four_eight + 1}', (two_zero_four_eight+1, max_seq_len_sdpa))),
        ]
        return bounds

    def _prefill_fn_for_input(self, x: torch.Tensor):
        """
        This is an hack to overcome lack of dynamic shape support in torch.compile for MPS.

        Lazily compile model_prefill for different sequence lengths. Mark the input tensor as dynamic as appropriate.
        This is a workaround for MPS related defects in torch 2.8.0, 2.9.0, and 2.10.0-pre.
        (1) dynamic=True is unsupported on MPS [2.8.0, 2.9.0, 2.10.0-pre]
        (2) compilations fail within a torch.no_grad/inference_mode context [2.9.0, 2.10.0-pre] (a defect causing a
            comparison between a SymInt and an Int)
        (3) torch.fx.experimental.symbolic_shapes.ConstraintViolationError resulting from specialized guards for
            SDPA fast and 2-pass kernels [2.9.0, 2.10.0-pre]

        Note that while we cache per-bounds compilations of prefill, the torch compiler uses the same code object to
        cache frames and - presumably - guards. Effectively, this means that each subsequent compilation is a recompilation
        and subject to the torch._dynamo.config.recompile_limit. This also means that you may see one recompilation
        for prefill (two total compilations).
        """
        assert x.ndim == 2
        assert x.size(1) > 0
        bounds = self._prefill_func_bounds()
        fn = None
        for pred, (fn_name,(min_,max_)) in bounds:
            if pred(x.size(1)):
                fn = getattr(self, fn_name, None)
                if fn is None:
                    # lazy maybe-compile
                    compiled = self._maybe_compile(self.model.prefill)
                    setattr(self, fn_name, compiled)
                    fn = compiled
                    if self.use_mark_dynamic:
                        torch._dynamo.mark_dynamic(x, 1, min=min_, max=max_)
                    logger.debug(f"Compiled function '{fn_name}' for input shape {x.shape} and bounds [{min_},{max_}].")
                    break

        assert fn is not None
        return fn


    def _sync(self):
        d = str(self.device)
        if d.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif d.startswith("mps") and hasattr(torch, "mps") and torch.mps.is_available():
            torch.mps.synchronize()

    def set_temperature(self, temperature: float):
        self.temperature = float(temperature)
        return self

    def set_repetition_penalty(self, repetition_penalty: float):
        self.rep_p_t = torch.tensor(repetition_penalty, device=self.rep_p_t.device, dtype=self.rep_p_t.dtype)
        return self

    def reset_history(self):
        self.model.reset_history()
        self.cache.reset_history()
        # dynamic-sized history to match tests' expectations
        self.history = torch.empty(0, dtype=torch.long, device=self.device)

    def _prefill(self, prompt_ids: torch.Tensor):
        prompt_ids = prompt_ids[None,:]
        fn = self._prefill_fn_for_input(prompt_ids)
        logits, kv = fn(prompt_ids, window=self.window)
        logits, kv = logits.to(self.dtype), kv.to(self.dtype)
        logits = logits[..., :self.vocab_size]
        prompt_ids = prompt_ids.squeeze(0)
        self.cache.bulk_write_packed(kv, len(prompt_ids), window=self.window)
        self.history = torch.cat([self.history, prompt_ids.to(dtype=torch.long, device=self.device)], dim=0)
        return logits

    def _step(self, token: torch.Tensor):
        k_ctxs, v_ctxs = [], []
        for i in range(len(self.model.blocks)):
            kc, vc = self.cache.view(i)
            assert(kc.dtype == self.dtype); assert(vc.dtype == self.dtype)
            k_ctxs.append(kc); v_ctxs.append(vc)
        logits, k_new, v_new = self.model.step(token, k_ctxs, v_ctxs, self.cache.t, self.window)
        logits = logits.to(self.dtype)
        # some layers may not have attention but we can assume the first layer does
        k_new = [x.to(self.dtype) if x is not None else torch.zeros_like(k_new[0]) for x in k_new]
        v_new = [x.to(self.dtype) if x is not None else torch.zeros_like(v_new[0]) for x in v_new]
        logits = logits[..., :self.vocab_size]
        for i in range(len(self.model.blocks)):
            assert(k_new[i] is not None); assert(v_new[i] is not None)
            self.cache.write(i, k_new[i], v_new[i])
        self.cache.advance()
        self.history = torch.cat([self.history, token.view(1)], dim=0)
        return logits

    @torch.inference_mode()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens) -> Generator[torch.Tensor, None, tuple[torch.Tensor, float, float]]:
        assert prompt_ids.ndim == 1
        assert prompt_ids.size(0) > 0 # must have at least one token
        assert prompt_ids.size(0) <= self.window, f"prompt length {prompt_ids.size(0)} must be <= attention window {self.window}"
        assert prompt_ids.size(0)
        assert max_new_tokens > 0
        self.reset_history()
        self._sync(); t0 = time.perf_counter()
        logits = self._prefill(prompt_ids)
        self._sync(); t1 = time.perf_counter()
        for i in range(max_new_tokens):
            # prepare tensors for passing to potentially compiled functions, avoiding recompiles for shape changes
            logits = logits[-1]
            if self.use_mark_dynamic:
                torch._dynamo.mark_dynamic(logits, 0, min=1, max=self.max_seq_len)
            rep_pen_w = 128
            prev_ids = self.history[-rep_pen_w:]
            if self.use_mark_dynamic:
                torch._dynamo.mark_dynamic(prev_ids, 0, min=2, max=self.max_seq_len)
            logits = self.apply_repetition_penalty(logits, prev_ids, self.rep_p_t, self._one)
            tok = self.sample(logits, self.temperature, self.top_k, self.top_p)
            if tok == self.eos_token_id:
                break
            yield tok.view(1)
            if i < max_new_tokens - 1:
                logits = self._step(tok)
            else:
                # Final token: append to history to reflect all generated tokens in the return value
                t = tok.to(device=self.device, dtype=torch.long).reshape(())
                self.history = torch.cat([self.history, t.view(1)], dim=0)
                logits = None
        self._sync(); t2 = time.perf_counter()
        prefill_duration = t1 - t0
        step_duration = t2 - t1
        out = self.history.detach().clone()
        return out, prefill_duration, step_duration
