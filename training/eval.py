import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.distributed as dist
from torch import nn

from training.optim import get_num_window_blocks

WINDOW_BLOCK_SIZE = 128


@dataclass
class EvalResult:
    val_loss: float
    val_acc: Optional[float]
    epoch: Optional[int]
    ema_dloss_per_token: float


class Evaluator:
    """
    Generic evaluator that works with any data generator yielding (inputs, targets).

    - For pretraining: typically used with DistributedDataGenerator (1D token streams).
    - For SFT: typically used with TaskDataGenerator (instruction/response sequences).

    The 'total_tokens' argument to eval() is interpreted as a *global* token
    budget for this evaluation call. The evaluator will consume enough batches
    so that steps * world_batch_tokens ~= total_tokens (integer division).
    """

    def __init__(
        self,
        data_generator: Any,
        distributed_enabled: bool,
        rank: int,
        train_attention_window_len: int,
    ):
        self._ddg = data_generator
        self._distributed_enabled = bool(distributed_enabled)
        self._rank = int(rank or 0)
        self._train_attention_window_len = int(train_attention_window_len)

        self._last_val_loss: Optional[float] = None
        self._ema_dloss_per_token: Optional[float] = None
        self._last_tokens_seen: int = 0

        # Approximate global tokens processed per eval step
        self._world_batch_tokens: Optional[int] = None

        if self._distributed_enabled and not dist.is_initialized():
            raise RuntimeError(
                "Evaluator: distributed_enabled=True but dist process group is not initialized"
            )

    @property
    def world_batch_tokens(self) -> Optional[int]:
        """
        Approximate number of global tokens processed per eval step from the
        most recent eval() call.
        """
        return self._world_batch_tokens

    def reset_generator(self) -> None:
        """
        Reset the underlying data generator, if it exposes a 'reset' method.
        """
        reset = getattr(self._ddg, "reset", None)
        if callable(reset):
            reset()

    def _compute_world_batch_tokens(self, inputs: torch.Tensor) -> int:
        """
        Compute approximate global tokens per eval step from the local batch.
        """
        local_tokens = int(inputs.numel())
        if self._distributed_enabled:
            world_size = dist.get_world_size()
        else:
            world_size = 1
        return local_tokens * world_size

    def eval(self, model: nn.Module, total_tokens: int, is_sft: bool = False) -> Dict[str, float]:
        """
        Run evaluation on approximately 'total_tokens' global tokens.

        The underlying generator is assumed to yield (inputs, targets) pairs
        that are directly consumable by the model, as in training.

        Returns a dict with:
            - 'val_loss': average loss over eval steps
            - 'val_acc': always None (placeholder for compatibility)
            - 'epoch': always None (no epoch tracking)
            - 'ema_dloss_per_token': exponential moving average of d(loss)/d(token)
        """
        if total_tokens <= 0:
            raise ValueError("Evaluator.eval: total_tokens must be > 0")

        device = next(model.parameters()).device
        model_was_training = model.training
        model.eval()

        # First batch defines the approximate world-batch token span
        inputs, targets = next(self._ddg)
        self._world_batch_tokens = self._compute_world_batch_tokens(inputs)
        world_batch_tokens = self._world_batch_tokens

        # Number of eval steps based on target global tokens
        # (integer division: we may use slightly fewer tokens than requested)
        steps = max(1, total_tokens // world_batch_tokens)

        loss_acc = torch.zeros((), device=device, dtype=torch.float32)

        def run_step(x: torch.Tensor, y: torch.Tensor) -> None:
            nonlocal loss_acc
            # Use final training schedule (s=1.0) for eval; match train.py signature
            n_blocks = get_num_window_blocks(
                1.0,
                attention_window_len=self._train_attention_window_len,
                window_block_size=WINDOW_BLOCK_SIZE,
            ).to(device.type)

            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    # model's chunking optimization for linear/cross_entropy will fail for SFT due to masking, producing NaN
                    loss = model(x, n_blocks, y) if not is_sft else model(x, n_blocks, y, loss_chunks=1)

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                raise RuntimeError(
                    f"NaN/Inf loss during eval on rank {self._rank}: "
                    f"inputs.shape={tuple(x.shape)}, targets.shape={tuple(y.shape)}, "
                    f"n_blocks={n_blocks}"
                )

            loss_acc += loss.detach()

        # Consume the first batch
        run_step(inputs, targets)

        # Remaining steps
        for _ in range(steps - 1):
            inputs, targets = next(self._ddg)
            run_step(inputs, targets)

        # Average across ranks
        if self._distributed_enabled:
            dist.all_reduce(loss_acc, op=dist.ReduceOp.AVG)

        cur_val = float(loss_acc.item() / steps)

        # Update EMA of d(loss)/d(token) based on requested token budget
        tokens_since_last = total_tokens
        if self._last_val_loss is not None and tokens_since_last > 0:
            dpt = (cur_val - self._last_val_loss) / tokens_since_last
            if self._ema_dloss_per_token is None:
                self._ema_dloss_per_token = dpt
            else:
                self._ema_dloss_per_token = 0.7 * self._ema_dloss_per_token + 0.3 * dpt

        self._last_val_loss = cur_val
        self._last_tokens_seen += tokens_since_last

        if model_was_training:
            model.train()

        return {
            "val_loss": cur_val,
            "val_acc": None,
            "epoch": None,
            "ema_dloss_per_token": self._ema_dloss_per_token
            if self._ema_dloss_per_token is not None
            else float("nan"),
        }
