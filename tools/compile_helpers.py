from typing import Callable, Tuple, Any, Dict
from torch._dynamo.eval_frame import innermost_fn
import functools
from tools.master_logger import MasterLogger
from tools.helpers import _default_sanitizer
import contextlib
import torch

logger = MasterLogger

@contextlib.contextmanager
def dynamo_force_static_param_shapes(value: bool, enabled: bool = True):
    if not enabled:
        yield
        return
    old = torch._dynamo.config.force_parameter_static_shapes
    torch._dynamo.config.force_parameter_static_shapes = value
    try:
        yield
    finally:
        torch._dynamo.config.force_parameter_static_shapes = old

def torch_get_guards_from_callable(fn: Callable, *args, **kwargs):
    """
    Hook for debug level logging of guards associated with a compiled and executed function.
    """
    try:
        import torch
        import dis

        # Resolve to innermost compiled function (avoids empty cache hits)
        target = innermost_fn(fn)
        code = target.__code__ if hasattr(target, "__code__") else (
            target.forward.__code__ if hasattr(getattr(target, "forward", None), "__code__") else None
        )
        if code is None:
            logger.debug(f"No code object found for {fn}")
            return

        # cache entries per compiled variant
        entries = torch._C._dynamo.eval_frame._debug_get_cache_entry_list(code)

        out = []
        for e in entries:
            # e.guard_manager.check_fn
            guards = getattr(getattr(e, "guard_manager", None), "check_fn", None)
            if guards is None:
                continue
            for guard in guards:
                for part in guard.code_parts:
                    logger.debug(f"Function '{fn}' guard clause: {part}")

                #  disassemble guard / code
                # dis.dis(guard)
                # dis.dis(code)

                #  evaluate guard manually on a locals() dict L for the frame
                # (keys are the original parameter names; run once to see names)
                # L = {"a": a_tensor, "b": b_tensor}
                # log.debug(guard(L))

        logger.debug(f"Guards for {fn}: {out}")
    except Exception:
        pass


def torch_compiled_callable_debug_wrapper(
    *,
    fn: Callable,
    post_exec_hook: Callable,
    expose_args: bool = False,
    sanitizer: Callable[[Tuple[Any, ...], Dict[str, Any]], Tuple[Tuple[Any, ...], Dict[str, Any]]] = _default_sanitizer,
) -> Callable:
    """
    Hook must have signature ~ (fn: Callable, *args, **kwargs)
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        y = fn(*args, **kwargs)
        try:
            if expose_args:
                safe_args, safe_kwargs = sanitizer(args, kwargs)
                post_exec_hook(fn, *safe_args, **safe_kwargs)
            else:
                post_exec_hook(fn)
        except Exception as e:
            logger.debug("Post-exec hook failed: %s", e, exc_info=False)
        return y
    return wrapper