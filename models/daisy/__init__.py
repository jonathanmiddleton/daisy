# daisy package exposing Daisy model modules under models.daisy
from __future__ import annotations

# Re-export key classes/functions for convenience
from .daisy_core import DaisyCore  # noqa: F401
from .attention import CausalSelfAttention, Rotary  # noqa: F401
from .block import Block  # noqa: F401
from .functional import norm, init_linear  # noqa: F401
from .mlp import MLP  # noqa: F401
from .attention_protocol import AttentionProtocol # noqa: F401
try:
    from .attention_kimi import KimiLinearSelfAttention # noqa: F401
except ImportError:
    pass

__all__ = [
    "DaisyCore",
    "CausalSelfAttention",
    "KimiLinearSelfAttention",
    "Rotary",
    "Block",
    "norm",
    "init_linear",
    "MLP",
    "AttentionProtocol",
]
