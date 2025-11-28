import torch
from torch import Tensor
from functools import lru_cache
import torch.distributed as dist
from typing import Any
from torch import nn
import math
from tools.master_logger import MasterLogger

logger = MasterLogger

# from tools.compile_helpers import dynamo_force_static_param_shapes


def derive_named_param_groups(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    """Derive canonical named parameter groups from the model.

    Returns a mapping with keys:
      - hidden_matrix_params: all module parameters with ndim >= 2 inside model.blocks (sorted by size desc)
      - embed_params: parameters of model.embed and model.value_embeds
      - scalar_params: [model.scalars]
      - head_params: [model.lm_head_w]
    """
    # Hidden matrices (e.g., linear/attention weight matrices) from transformer blocks
    hidden_matrix_params = []
    hidden_scalar_params = []
    attn_time_constant_params = []
    for name, p in model.blocks.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith("A_log") or name.endswith("dt_bias"): # kimi linear attention time constant
            attn_time_constant_params.append(p)
        elif p.ndim == 1:
            hidden_scalar_params.append(p)
        elif p.ndim >= 2:
            # if p.dtype != torch.bfloat16:
            #     print(name, p.dtype)
            hidden_matrix_params.append(p)
    hidden_matrix_params.sort(key=lambda x: x.size(), reverse=True)
    hidden_scalar_params.sort(key=lambda x: x.size(), reverse=True)
    # Embedding parameters
    embed_params = [*model.embed.parameters(),
                    *model.value_embeds.parameters()] if model.value_embeds is not None else [*model.embed.parameters()]

    if getattr(model, "scalars", None) is not None:
        # v1 model
        scalar_params = [*model.scalars.parameters()]
    else:
        # v2 model
        scalar_params = [*model.skip_weights]
        for b in model.blocks:
            if getattr(b, "g_x", None) is not None:
                scalar_params.append(b.g_x)
            attn = getattr(b, "attn", None)
            if attn is not None and getattr(attn, "g_ve", None) is not None:
                scalar_params.append(attn.g_ve)

    # Output head weights
    a = model.embed.weight
    b = model.lm_head_w
    if model.embed.weight is model.lm_head_w:
        # tied embeddings
        head_params = None
    else:
        head_params: list[nn.Parameter] = [model.lm_head_w]

    # Sanity: ensure exact partitioning of all model parameters
    params_collections = [hidden_matrix_params, embed_params, scalar_params, head_params, hidden_scalar_params, attn_time_constant_params]
    params_collections = [p for p in params_collections if p is not None]
    optimized_parameters_set = {p for params in params_collections if params for p in params}
    all_params = set(model.parameters())
    missing = all_params - optimized_parameters_set  # in all_params but not in optimized_parameters_set
    extra = optimized_parameters_set - all_params  # in optimized_parameters_set but not in all_params

    if missing or extra:
        logger.error(f"Missing from optimized_parameters_set: {missing}")
        logger.error(f"Extra in optimized_parameters_set: {extra}")

    assert optimized_parameters_set == all_params
    assert len(optimized_parameters_set) == sum(len(lst) for lst in params_collections)

    p_dict = {
        "hidden_matrix_params": hidden_matrix_params,
        "embed_params": embed_params,
        "scalar_params": scalar_params,
    }

    if head_params:
        p_dict["head_params"] = head_params

    if hidden_scalar_params:
        p_dict["hidden_scalar_params"] = hidden_scalar_params

    if attn_time_constant_params:
        p_dict["attn_time_constant_params"] = attn_time_constant_params

    return p_dict


# -----------------------------------------------------------------------------
# Muon optimizer

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ∈ [1 - l, 1 + r], which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for a, b, c in [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]:
        A = X @ X.mT
        B = b * A + c * A @ A  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

@torch.compile
def update_faster(acc_bf16_view_u16: Tensor, mantissa: Tensor, momentum_buffer: Tensor, grad: Tensor, momentum: Tensor,
                  eff_lr: Tensor, eff_weight_decay: Tensor):
    """
    Mixed precision with full precision math. Support for uint* ops requires torch compile
    """
    assert acc_bf16_view_u16.dtype == mantissa.dtype == torch.uint16
    grad = grad.float()
    momentum_buffer.copy_(momentum * momentum_buffer + (1 - momentum) * grad)
    v = zeropower_via_newtonschulz5(momentum * momentum_buffer + (1 - momentum) * grad)

    acc_m_u32 = (acc_bf16_view_u16.to(torch.uint32) << 16) | mantissa.to(torch.uint32)
    acc_m_u32.view(torch.float32).mul_(1 - eff_weight_decay)
    acc_m_u32.view(torch.float32).add_(other=v, alpha=-eff_lr)
    acc_bf16_view_u16.copy_((acc_m_u32 >> 16).to(torch.uint16))
    mantissa.copy_(acc_m_u32.to(torch.uint16))

# with dynamo_force_static_param_shapes(False):
@torch.compile
def update_slower(acc: Tensor, momentum_buffer: Tensor, grad: Tensor, momentum: Tensor, eff_lr: Tensor,
                  eff_weight_decay: Tensor):
    """
    Full precision alternative to update_fast for platforms that don't support uint* ops or if unable to compile
    """
    grad = grad.float()
    momentum_buffer.copy_(momentum * momentum_buffer + (1 - momentum) * grad)
    v = zeropower_via_newtonschulz5(momentum * momentum_buffer + (1 - momentum) * grad)
    acc.mul_(1 - eff_weight_decay).add_(other=v, alpha=-eff_lr)


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Warning: This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    """

    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        self.uint16_ops = torch.cuda.is_available()

        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)
        # flse = [p for group in self.param_groups for p in group["params"] if p.dtype != torch.bfloat16 ]
        assert all(p.dtype == torch.bfloat16 for group in self.param_groups for p in group["params"])

    @torch.no_grad()
    def step(self):
        futures: list[torch.Future] = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * self.world_size
            momentum = torch._as_tensor_fullprec(group["momentum"])
            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    # Skip if no gradient for this parameter on this step/rank
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if self.uint16_ops:
                        if len(state) == 0:
                            state["mantissa"] = torch.zeros_like(p, dtype=torch.uint16)
                            state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
                        update_faster(
                            p.view(torch.uint16), state["mantissa"], state["momentum_buffer"],
                            p.grad, momentum,
                            eff_lr=torch._as_tensor_fullprec(group["lr"] * max(1, p.size(-2) / p.size(-1)) ** 0.5),
                            eff_weight_decay=torch._as_tensor_fullprec(
                                group["lr"] * group["weight_decay"] * getattr(p, "wd_mul", 1.0)),
                        )
                    else:
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
                        update_slower(
                            p, state["momentum_buffer"],
                            p.grad, momentum,
                            eff_lr=torch._as_tensor_fullprec(group["lr"] * max(1, p.size(-2) / p.size(-1)) ** 0.5),
                            eff_weight_decay=torch._as_tensor_fullprec(
                                group["lr"] * group["weight_decay"] * getattr(p, "wd_mul", 1.0)),
                        )
                if self.world_size > 1 and dist.is_available() and dist.is_initialized():
                    futures.append(
                        dist.all_gather(
                            params_pad[base_i:base_i + self.world_size],
                            params_pad[base_i + self.rank],
                            async_op=True
                        ).get_future()
                    )
                if futures:
                    torch.futures.collect_all(futures).wait()


class AdaptiveLR:
    def __init__(self,
                 H: int = 50,
                 H_tail: int = 200,
                 H_eval: int = 20,
                 H_guard: int = 50,
                 H_stable: int = 150,
                 tau_hi: float = 0.03,
                 tau_lo: float = 0.01,
                 rho1_lo: float = -0.30,
                 rho1_hi: float = -0.10,
                 dnr_hi: float = 3.0,
                 dnr_lo: float = 1.6,
                 m_min: float = 0.1,
                 cosine_frac: float = 1.0,
                 eps: float = 1e-12):
        self.beta = 2 ** (-1.0 / H)
        self.beta_tail = 2 ** (-1.0 / H_tail)
        self.H_eval = H_eval
        self.H_guard = H_guard
        self.H_stable = H_stable
        self.tau_hi, self.tau_lo = tau_hi, tau_lo
        self.rho1_lo, self.rho1_hi = rho1_lo, rho1_hi
        self.dnr_hi, self.dnr_lo = dnr_hi, dnr_lo
        self.m_min = m_min
        self.cosine_frac = cosine_frac
        self.eps = eps

        # Optional learning rate floor applied to the returned scale (fraction of initial LR)
        self.lr_floor: float = 0.0

        self.mu = None
        self.v = 0.0
        self.tau = 0.0
        self.c1 = 0.0
        self.s = 0.0
        self.r_prev = 0.0
        self.mu_prev = None
        self.d = 0.0

        self.m = 1.0
        self.prev_step = None
        self.since_change = 0
        self.eval_accum = 0
        self.stable_accum = 0

    def _update_stats(self, loss: float, delta_steps: int):
        b = self.beta ** delta_steps
        bt = self.beta_tail ** delta_steps

        if self.mu is None:
            self.mu = loss
            self.mu_prev = loss
            self.prev_step = 0

        self.mu = (1.0 - b) * loss + b * self.mu
        r = loss - self.mu
        self.v = (1.0 - b) * (r * r) + b * self.v
        z = r / math.sqrt(self.v + self.eps)
        self.tau = (1.0 - bt) * (abs(z) > 3.0) + bt * self.tau
        self.c1 = (1.0 - b) * (r * self.r_prev) + b * self.c1
        self.s = (1.0 - b) * (r * r) + b * self.s
        rho1 = self.c1 / (self.s + self.eps)
        dmu = abs(self.mu - self.mu_prev)
        self.mu_prev = self.mu
        self.d = (1.0 - b) * dmu + b * self.d
        dnr = math.sqrt(self.v + self.eps) / (self.d + self.eps)
        self.r_prev = r
        return self.tau, rho1, dnr

    def _maybe_adapt(self, tau: float, rho1: float, dnr: float, delta_steps: int):
        self.eval_accum += delta_steps
        self.since_change += delta_steps

        if self.eval_accum < self.H_eval or self.since_change < self.H_guard:
            return

        triggers = (tau > self.tau_hi) + (rho1 < self.rho1_lo) + (dnr > self.dnr_hi)
        if triggers >= 2:
            self.m = max(self.m * 0.70, self.m_min)
            self.since_change = 0
            self.stable_accum = 0
            self.eval_accum = 0
            return
        if triggers == 1 and (tau > 0.045 or dnr > 4.0):
            self.m = max(self.m * 0.85, self.m_min)
            self.since_change = 0
            self.stable_accum = 0
            self.eval_accum = 0
            return

        if tau < self.tau_lo and rho1 > self.rho1_hi and dnr < self.dnr_lo:
            self.stable_accum += self.eval_accum
            if self.stable_accum >= self.H_stable:
                self.m = min(1.0, self.m * 1.01)
                self.since_change = 0
                self.stable_accum = 0

        self.eval_accum = 0

    @staticmethod
    def _lr_scale_base(s: float, cooldown_frac: float, cosine_frac: float = 1.0) -> float:
        x = 0.0 if s < 0.0 else (1.0 if s > 1.0 else s)
        if cooldown_frac <= 0.0 or x < 1.0 - cooldown_frac:
            return 1.0
        t = (x - (1.0 - cooldown_frac)) / max(cooldown_frac, 1e-8)
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        k = 0.0 if cosine_frac < 0.0 else (1.0 if cosine_frac > 1.0 else cosine_frac)
        if t <= k:
            return 0.5 * (1.0 + math.cos(math.pi * t))
        yk = 0.5 * (1.0 + math.cos(math.pi * k))
        return yk * (1.0 - (t - k) / max(1.0 - k, 1e-8))

    def get_lr(self, step: int, loss: float, s: float, cooldown_frac: float) -> float:
        if self.prev_step is None:
            delta = 1
        else:
            delta = step - self.prev_step
            if delta < 1:
                delta = 1
        self.prev_step = step

        tau, rho1, dnr = self._update_stats(loss, delta)
        self._maybe_adapt(tau, rho1, dnr, delta)
        base = self._lr_scale_base(s, cooldown_frac, self.cosine_frac)
        scale = base * self.m
        # Apply optional floor to the scale (fraction of initial LR)
        if scale < self.lr_floor:
            scale = self.lr_floor
        return scale


def get_linear_decay_lr_s(s: float, cooldown_frac: float) -> float:
    """Return LR scale factor given normalized progress s in [0,1].
    1.0 during main phase, linear decay during cooldown.
    """
    x = 0.0 if s < 0 else (1.0 if s > 1 else s)
    return 1.0 if x < 1 - cooldown_frac else max((1 - x) / max(cooldown_frac, 1e-8), 0.0)


def get_lincos_lr_s(s: float, cooldown_frac: float) -> float:
    """Linear warmup to 1.0, then cosine decay to 0.0 during cooldown.

    Args:
        s: normalized training progress in [0, 1].
        cooldown_frac: fraction of total progress reserved for cosine cooldown (in [0,1]).

    Behavior:
        - Warmup phase (0 <= s < 1 - cooldown_frac): linear ramp from 0 -> 1 over the warmup span.
        - Cooldown phase (1 - cooldown_frac <= s <= 1): cosine decay from 1 -> 0 over the cooldown span.
        - Inputs are clamped to valid ranges; cooldown_frac <= 0 disables cooldown (pure warmup).
    """
    # Clamp inputs
    x = 0.0 if s < 0.0 else (1.0 if s > 1.0 else s)
    c = 0.0 if cooldown_frac < 0.0 else (1.0 if cooldown_frac > 1.0 else cooldown_frac)

    # No cooldown: pure linear warmup over the whole schedule
    if c == 0.0:
        return x  # ramps 0->1 as s goes 0->1

    # Warmup: length is (1 - c), scale up to exactly 1 at the boundary
    warmup_len = max(1.0 - c, 1e-8)
    if x < 1.0 - c:
        return x / warmup_len

    # Cooldown: cosine decay over the final c fraction
    t = (x - (1.0 - c)) / max(c, 1e-8)  # t in [0,1]
    return 0.5 * (1.0 + math.cos(math.pi * t))


def get_cosine_decay_lr_s(s: float, cooldown_frac: float) -> float:
    """Constant 1.0 until cooldown, then cosine decay to 0.0 over cooldown span."""
    x = 0.0 if s < 0.0 else (1.0 if s > 1.0 else s)
    c = 0.0 if cooldown_frac < 0.0 else (1.0 if cooldown_frac > 1.0 else cooldown_frac)
    if c <= 0.0:
        return 1.0
    if x < 1.0 - c:
        return 1.0
    t = (x - (1.0 - c)) / max(c, 1e-8)  # normalized to [0,1]
    return 0.5 * (1.0 + math.cos(math.pi * t))


# Dispatch table for LR schedules
LEARNING_RATE_SCHEDULES: dict[str, callable] = {
    "linear_decay": get_linear_decay_lr_s,
    "linear_warmup_cosine_decay": get_lincos_lr_s,
    "cosine_decay": get_cosine_decay_lr_s,
}


def get_lr_scale(schedule_name: str, s: float, cooldown_frac: float) -> float:
    """Resolve a schedule by name and compute the LR scale for progress s.

    Raises ValueError for unknown schedule names.
    """
    fn = LEARNING_RATE_SCHEDULES.get(schedule_name)
    if fn is None:
        raise ValueError(
            f"Unknown learning_rate_schedule '{schedule_name}'. Valid options: {sorted(LEARNING_RATE_SCHEDULES.keys())}")
    return fn(s, cooldown_frac)


# Global flag to force full-sized attention windows regardless of training progress
_force_full_windows: bool = False


def set_full_windows(flag: bool):
    global _force_full_windows
    _force_full_windows = bool(flag)


def next_multiple_of_n(v: float | int, *, n: int):
    """Return the smallest multiple of n greater than or equal to v."""
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)


@lru_cache(1)
def get_window_size_blocks_helper(window_size_tokens: int, window_block_size: int):
    """Return number of window blocks as a small CPU tensor.
    Avoids unconditional CUDA allocation to prevent crashes on CPU-only setups/tests.
    """
    if window_block_size <= 0:
        raise ValueError("window_block_size must be > 0")
    blocks = int(window_size_tokens) // int(window_block_size)
    return torch.tensor(blocks, dtype=torch.int32)

# move to train.py
def get_num_window_blocks(schedule: float, *, attention_window_len: int, window_block_size: int) -> torch.Tensor:
    """Attention window schedule driven by normalized progress schedule s∈[0,1].
    Returns the number of blocks for the sliding window, parameterized by attention_window_len and window_block_size.
    """
    if _force_full_windows:
        x = 1.0
    else:
        x = 0.0 if schedule < 0 else (1.0 if schedule > 1 else schedule)
    # Cubic increase (by @jadenj3o)
    factor = 4 * x ** 3 - 6 * x ** 2 + 3 * x
    #scaled attn window
    window_tokens = next_multiple_of_n(attention_window_len * factor, n=window_block_size)
    return get_window_size_blocks_helper(window_tokens, window_block_size)


def resolve_optimizer_class(opt_type: str):
    """Resolve optimizer type name to a class.
    Supports torch.optim.* and custom Muon optimizer.

    Important: prefer the local Muon implementation (supports batched >2D params)
    over torch.optim.Muon, which only supports 2D parameters.
    """
    if opt_type == "Muon":
        # Always use the local Muon defined in this module
        return Muon
    if hasattr(torch.optim, opt_type):
        return getattr(torch.optim, opt_type)
    raise ValueError(f"Unknown optimizer type: {opt_type}")


def get_referenced_groups(cfg_list: list[dict[str, Any]]) -> list[str]:
    referenced_group_names: set[str] = set()
    for idx, opt_cfg in enumerate(cfg_list):
        pg_cfgs = opt_cfg.get("params")
        for pg in pg_cfgs:
            referenced_group_names.add(pg.get("group"))
    return list(referenced_group_names)


def build_optimizers_from_cfg(
        *,
        cfg_list: list[dict[str, Any]],
        model: nn.Module,
        rank: int,
        world_size: int,
        frozen_groups: list[str] | None = None,
) -> list[torch.optim.Optimizer]:
    """Build optimizers from configuration list and sets no_grad on frozen groups.

    Each item in cfg_list is a dict with keys:
      - type: optimizer class name (e.g., "AdamW" or "Muon")
      - params: list of param-group dicts containing at least {group: <name>} and any group overrides like lr
      - other keys are passed as kwargs to the optimizer constructor (betas, eps, weight_decay, fused, momentum, ...)
      - cfg_list should include frozen groups as well, if any

    This function also enforces that the union of all param groups referenced in the config
    exactly matches the required named groups for the model (no missing or extra groups).
    """
    if not isinstance(cfg_list, list) or not cfg_list:
        raise ValueError("optimizers config must be a non-empty list")

    # Derive canonical named parameter groups from the model
    param_groups_by_name = derive_named_param_groups(model)
    required_group_names = set(param_groups_by_name.keys())
    frozen_groups = frozen_groups or []
    for name in frozen_groups:
        if name not in required_group_names:
            raise ValueError(f"Frozen group '{name}' is not present in the model")
        for p in param_groups_by_name[name]:
            p.requires_grad_(False)

    # Collect all group names referenced across all optimizers in config
    referenced_group_names: set[str] = set()

    optimizers: list[torch.optim.Optimizer] = []

    for idx, opt_cfg in enumerate(cfg_list):
        if not isinstance(opt_cfg, dict):
            raise ValueError(f"Optimizer #{idx} must be a mapping, got {type(opt_cfg).__name__}")
        opt_type = opt_cfg.get("type")
        if not opt_type:
            raise ValueError(f"Optimizer #{idx} is missing 'type'")

        pg_cfgs = opt_cfg.get("params")
        if not isinstance(pg_cfgs, list) or not pg_cfgs:
            raise ValueError(f"Optimizer '{opt_type}' requires a non-empty 'params' list")

        param_groups: list[dict[str, Any]] = []
        for pg in pg_cfgs:
            if not isinstance(pg, dict):
                raise ValueError(f"param group for optimizer '{opt_type}' must be a mapping, got {type(pg).__name__}")
            name = pg.get("group")
            if name not in param_groups_by_name:
                raise ValueError(
                    f"Unknown param group '{name}' for optimizer '{opt_type}'. "
                    f"Known groups: {sorted(param_groups_by_name.keys())}"
                )
            if frozen_groups and pg.get("group") in frozen_groups:
                logger.info(f"Freezing group '{name}' for optimizer '{opt_type}'")
                continue
            referenced_group_names.add(name)
            group_opts = {k: v for k, v in pg.items() if k != "group"}
            # Preserve the canonical group name for downstream tooling (e.g., LR sweeps)
            group_opts["name"] = name
            group_opts["params"] = param_groups_by_name[name]
            param_groups.append(group_opts)

        if not param_groups:
            continue
        OptClass = resolve_optimizer_class(opt_type)
        opt_kwargs = {k: v for k, v in opt_cfg.items() if k not in ("type", "params")}

        if OptClass is Muon:
            # Enforce explicit weight_decay in config for Muon (breaking change)
            if "weight_decay" not in opt_kwargs:
                raise ValueError(
                    "Muon optimizer requires 'weight_decay' to be set explicitly in the training config")
            opt_kwargs.setdefault("rank", rank)
            opt_kwargs.setdefault("world_size", world_size)

        optimizer = OptClass(param_groups, **opt_kwargs)
        optimizers.append(optimizer)

    # Validate exhaustive coverage: all required groups must be present exactly once across optimizers
    missing = required_group_names - referenced_group_names - set(frozen_groups)
    extra = referenced_group_names - required_group_names
    if missing or extra:
        msgs = []
        if missing:
            msgs.append(f"missing groups: {sorted(missing)}")
        if extra:
            msgs.append(f"unknown groups: {sorted(extra)}")
        raise ValueError(
            "Optimizer config param groups must be exhaustive and valid; " + ", ".join(msgs) +
            f". Required groups: {sorted(required_group_names)}"
        )

    # Additionally ensure that the parameters are fully covered without overlap
    # Only consider trainable parameters (requires_grad=True) for coverage; frozen groups are excluded
    covered_params = {p for name in referenced_group_names for p in param_groups_by_name[name]}
    all_trainable_params = {p for p in model.parameters() if p.requires_grad}
    if covered_params != all_trainable_params:
        raise ValueError(
            "Param groups in optimizer config do not cover trainable model parameters exactly. "
            f"covered={len(covered_params)} total_trainable={len(all_trainable_params)}"
        )

    return optimizers
