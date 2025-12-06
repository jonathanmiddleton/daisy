import os
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.distributed as dist
from torch import nn

from data import DataGeneratorProtocol
from training.optim import get_num_window_blocks
from tools.master_logger import MasterLogger

WINDOW_BLOCK_SIZE = 128
logger = MasterLogger


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
    - For Task SFT: typically used with TaskDataGenerator (instruction/response sequences).

    The 'total_tokens' argument to eval() is interpreted as a *global* token
    budget for this evaluation call. The evaluator will consume enough batches
    so that steps * world_batch_tokens ~= total_tokens (integer division).
    """

    def __init__(
        self,
        data_generator: DataGeneratorProtocol,
        distributed_enabled: bool,
        rank: int,
        world_size: int,
        attn_window_len: int,
        global_val_tokens: int,
        val_type: str = "pretraining",  # "pretraining" or "task"
        log_samples: bool = False,
        sample_log_path: Optional[str] = None,
        tokenizer_name: str = "gpt2",
    ):
        self._ddg = data_generator
        self._distributed_enabled = bool(distributed_enabled)
        self._rank = int(rank or 0)
        self._val_type = val_type
        self._attn_window_len = attn_window_len
        self._global_val_tokens = global_val_tokens
        self._world_size = world_size
        self._sequence_length = data_generator.get_sequence_length()

        # Per-run bookkeeping
        self._last_val_loss: Optional[float] = None
        self._ema_dloss_per_token: Optional[float] = None
        self._last_tokens_seen: int = 0

        # Optional per-sample logging of eval losses + text
        env_log = os.environ.get("EVAL_LOG_SAMPLES", "0").strip()
        self._log_samples = bool(log_samples) or env_log in ("1", "true", "True")

        # Determine sample log path (per rank)
        env_path = os.environ.get("EVAL_SAMPLE_LOG_PATH", "").strip()
        if sample_log_path:
            base_path = sample_log_path
        elif env_path:
            base_path = env_path
        else:
            base_path = f"eval_samples_rank{self._rank}.jsonl"

        # Allow {rank} placeholder in the path
        self._sample_log_path = base_path.format(rank=self._rank)

        # Optional tokenizer for decoding tokens -> text
        self._tokenizer = None
        tok_name = tokenizer_name
        if tok_name:
            try:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
                logger.debug(f"[eval] Loaded tokenizer '{tok_name}' for sample logging.")
            except Exception as e:
                logger.error(f"[eval] Failed to load tokenizer '{tok_name}' for sample logging: {e}")
                self._tokenizer = None

        if self._distributed_enabled and not dist.is_initialized():
            raise RuntimeError(
                "Evaluator: distributed_enabled=True but dist process group is not initialized"
            )

    def reset_generator(self) -> None:
        """
        Reset the underlying data generator, if it exposes a 'reset' method.
        """
        reset = getattr(self._ddg, "reset", None)
        if callable(reset):
            reset()
        else:
            logger.warning(f"[eval] Generator '{self._ddg}' does not expose a 'reset' method.")

    def _log_sample(
        self,
        step_idx: int,
        loss_tensor: torch.Tensor,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """
        Write a single eval sample record to the sample log file as JSONL.
        Includes loss, shapes, and decoded text (if tokenizer set).
        """
        if not self._log_samples:
            return
        if logger.isDebugEnabled(): logger.debug(f"[eval] _log_sample(step_idx={step_idx})")
        loss_scalar = float(loss_tensor.detach().item())
        tokens_cpu = inputs.detach().to("cpu")

        # Handle 1D vs 2D shaped inputs (batch_size may be 1)
        if tokens_cpu.dim() == 2:
            # assume shape (B, T), log the first row
            tok_ids = tokens_cpu[0].tolist()
        else:
            tok_ids = tokens_cpu.tolist()

        text = None
        if self._tokenizer is not None:
            try:
                text = self._tokenizer.decode(tok_ids, skip_special_tokens=False)
            except Exception as e:
                logger.error(f"[eval] Token decode failed on rank={self._rank}, step={step_idx}: {e}")
                text = None

        record = {
            "rank": self._rank,
            "step": step_idx,
            "loss": loss_scalar,
            "inputs_shape_padded": list(inputs.shape),
            "targets_shape_padded": list(targets.shape),
            "text": text,
        }

        try:
            with open(self._sample_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(
                f"[eval] Failed to write sample log on rank={self._rank}, step={step_idx} "
                f"to '{self._sample_log_path}': {e}"
            )

    def eval(self, model: nn.Module, schedule: float) -> Dict[str, float]:
        """
        The underlying generator is assumed to yield (inputs, targets) pairs
        that are directly consumable by the model, as in training.
        
        Returns a dict with:
            - 'val_loss': average loss over eval steps
            - 'val_acc': always None (placeholder for compatibility)
            - 'epoch': always None (no epoch tracking)
            - 'ema_dloss_per_token': exponential moving average of d(loss)/d(token)
        """
        device = next(model.parameters()).device
        model_was_training = model.training
        model.eval()

        # First batch defines the approximate world-batch token span
        inputs, targets = next(self._ddg)

        steps = max(1, self._global_val_tokens // self._sequence_length // self._world_size)

        loss_acc = torch.zeros((), device=device, dtype=torch.float32)

        step_idx = 0  # local step index within this eval() call

        def run_step(x: torch.Tensor, y: torch.Tensor) -> None:
            nonlocal loss_acc, step_idx
            if logger.isDebugEnabled():
                logger.debug(f"[eval] run_step(x.shape={x.shape}, y.shape={y.shape})")
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    n_blocks = get_num_window_blocks(schedule=schedule,
                                                     attention_window_len=self._attn_window_len,
                                                     window_block_size=WINDOW_BLOCK_SIZE).to(device)
                    if logger.isDebugEnabled(): logger.debug(f"[eval] n_blocks={n_blocks}")
                    loss = model(x, n_blocks, y) if self._val_type == 'pretraining' else model(x, n_blocks, y, loss_chunks=1)

            # Optional per-sample debug logging
            self._log_sample(step_idx, loss, x, y)

            loss_acc += loss.detach()
            step_idx += 1

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
        tokens_since_last = self._global_val_tokens
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
