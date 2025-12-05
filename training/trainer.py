import copy
import itertools
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional, Iterable, Tuple, Dict, List

import torch
import torch.distributed as dist
from torch import nn

from data.data_gen_stream import DistributedDataGenerator
from data.data_gen_task import TaskDataGenerator
from models import model_from_spec
from tools.checkpoint import model_from_checkpoint, save_checkpoint
from tools.model_report import build_report, format_report_text
from tools.master_logger import MasterLogger
from training.eval import Evaluator
from training.hparams import Hyperparameters
from training.optim import Muon, get_lr_scale, build_optimizers_from_cfg, get_num_window_blocks, set_full_windows
from training.progress import ProgressMeter


WINDOW_BLOCK_SIZE = 128
logger = MasterLogger


@dataclass(frozen=True)
class CompileKey:
    model_spec: str
    model_class: str
    vocab_size: int
    eos_token_id: int
    num_layers: int
    num_heads: int
    model_dim: int
    head_dim: int
    attention_window_len: int
    max_seq_len: int
    torch_coordinate_descent_tuning: bool
    device_type: str
    world_size: int
    compiled_backend: str
    dynamic: bool


def derive_compile_key(args: Hyperparameters, *, device_type: str, world_size: int, dynamic: bool) -> CompileKey:
    backend = 'inductor' if device_type == 'cuda' else 'aot_eager'
    return CompileKey(
        model_spec=args.model_spec,
        model_class=args.model_class,
        vocab_size=int(args.vocab_size),
        eos_token_id=int(args.eos_token_id),
        num_layers=int(args.num_layers),
        num_heads=int(args.num_heads),
        model_dim=int(args.model_dim),
        head_dim=int(args.head_dim),
        attention_window_len=int(args.attention_window_len),
        max_seq_len=int(getattr(args, "max_seq_len", args.training_sequence_length)),
        torch_coordinate_descent_tuning=bool(getattr(args, "torch_coordinate_descent_tuning", False)),
        device_type=device_type,
        world_size=world_size,
        compiled_backend=backend,
        dynamic=bool(dynamic),
    )


def _maybe_compile(model: nn.Module, *, device_type: str, dynamic: bool, is_task: bool) -> nn.Module:
    disable_compile = os.environ.get("TORCH_DISABLE_MODEL_COMPILE", "0") == "1"
    if disable_compile:
        logger.info(f"Model compilation disabled: TORCH_DISABLE_MODEL_COMPILE={disable_compile}")
        return model
    backend = 'inductor' if device_type == 'cuda' else 'aot_eager'
    use_dynamic = bool(dynamic or is_task)
    if is_task:
        torch._dynamo.config.force_parameter_static_shapes = False
    torch._inductor.config.coordinate_descent_tuning = bool(
        os.environ.get("TORCH_COORDINATE_DESCENT_TUNING", "0") == "1" or False
    )
    torch._dynamo.config.compiled_autograd = True
    torch._dynamo.config.error_on_nested_fx_trace = False
    logger.info(f"Compiling model (dynamic={use_dynamic}) (backend={backend}). This may take several minutes and occurs in phases during the initial eval and forward.")
    return torch.compile(model, dynamic=use_dynamic, backend=backend)


def _maybe_reset_peak_memory_stats(device_type: str):
    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def _get_max_memory_allocated(device_type: str) -> Optional[int]:
    if device_type == "cuda":
        return torch.cuda.max_memory_allocated()
    return None


class CompiledRuntime:
    """Persistent per-compile-key runtime holding device/DDP, compiled model and initial weights."""

    def __init__(self, args_for_group: Hyperparameters, *, dynamic: bool):
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        # Apply configurable RNG seed
        try:
            seed = int(getattr(args_for_group, "seed", 1337))
        except Exception:
            seed = 1337
        try:
            import random
            random.seed(seed)
        except Exception:
            logger.error(f"Failed to seed random.seed({seed})")
        try:
            import numpy as np
            np.random.seed(seed)
        except Exception:
            logger.error(f"Failed to seed numpy.random.seed({seed})")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            try:
                torch.cuda.manual_seed_all(seed)
            except Exception:
                logger.error(f"Failed to seed torch.cuda.manual_seed_all({seed})")

        self.DEBUG_LOG_ENABLED = logger.isDebugEnabled() # quasi-static compiler-friendly bool

        # World setup from torchrun
        try:
            self.world_size = int(os.environ.get("WORLD_SIZE"))
            self.rank = int(os.environ.get("RANK"))
            self.local_rank = int(os.environ.get("LOCAL_RANK"))
        except Exception as e:
            # explicitly log as opposed to get(,default)
            logger.info(f"Torchrun vars not set. Assuming standalone.")
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0

        self.use_distributed = self.world_size > 1
        logger.info(f"CompiledRuntime initialized on rank={self.rank} world_size={self.world_size} local_rank={self.local_rank} use_distributed={self.use_distributed}")

        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.local_rank)
        elif torch.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        logger.debug(f"CompiledRuntime initialized on device={self.device}")
        if self.DEBUG_LOG_ENABLED: logger.debug(f"CompiledRuntime initialized on device={self.device}")
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            if self.use_distributed:
                dist.init_process_group(backend="nccl", device_id=self.device, world_size=self.world_size)
                dist.barrier()
            self.is_master = self.rank == 0
        elif self.device.type == "mps":
            if self.use_distributed:
                raise ValueError("Distributed training is not supported on macOS/MPS")
            self.is_master = True
        else:
            self.is_master = True

        # Configure attention windows mode globally
        set_full_windows(args_for_group.full_windows)

        is_task = (args_for_group.train_mode == "task")

        # Initialize model
        if args_for_group.init_checkpoint:
            model, _ = model_from_checkpoint(args_for_group.init_checkpoint, device=self.device, dynamic_shapes=is_task)
            logger.info("Rehydrated model from checkpoint.")
        else:
            # Make sure max_seq_len is large enough (group-level)
            msl = int(getattr(args_for_group, "max_seq_len", args_for_group.training_sequence_length))
            setattr(args_for_group, "max_seq_len", msl)
            model = model_from_spec(args_for_group.model_spec, device=self.device.type, overrides=asdict(args_for_group))

        # dtype tweaks and broadcast
        for m in model.modules():
            if isinstance(m, nn.Embedding):
                m.bfloat16()
        if self.use_distributed:
            for param in model.parameters():
                dist.broadcast(param.detach(), 0)

        self.model = model
        # Capture initial weights after dtype/distribution
        _sd = copy.deepcopy(self.model.state_dict())
        from tools.checkpoint import restore_prefix
        self._initial_state = restore_prefix(_sd)

        # Compile once
        self.model = _maybe_compile(self.model, device_type=self.device.type, dynamic=dynamic, is_task=is_task)

        report = build_report(self.model)
        logger.info(f"Initial model report:\n{format_report_text(report)}")

    def reset_model_to_initial(self):
        #noinspection PyBroadException
        try:
            self.model.load_state_dict(self._initial_state, strict=True)
        except Exception as e:
            # uncompiled models expect weight names without a prefix
            from tools.checkpoint import remove_prefix
            _sd = remove_prefix(self._initial_state)
            self.model.load_state_dict(_sd, strict=True)
        self.model.zero_grad(set_to_none=True)

    def destroy(self):
        if self.use_distributed and dist.is_initialized():
            dist.destroy_process_group()


class TrainingSession:
    """Ephemeral per-run session using a shared CompiledRuntime."""

    def __init__(self, runtime: CompiledRuntime, args: Hyperparameters, run_id: int):
        self.rt = runtime
        self.args = args
        self.run_id = run_id
        self._last_run_ckpt_path: Optional[str] = None
        self._wandb = None
        self._wandb_enabled = False

    # ------- W&B helpers -------
    def _wandb_init(self):
        if self.rt.is_master and self.args.wandb_log:
            try:
                import wandb

                project = self.args.wandb_project
                # If an lr_scale suffix was applied to the run name, do NOT append a run-id suffix.
                # Otherwise, keep previous behavior and append "-r{run_id}" for disambiguation.
                base_name = getattr(self.args, "wandb_run_name", "") or ""
                if getattr(self, "_lr_suffix_applied", False):
                    name = base_name
                else:
                    name = f"{base_name}-r{self.run_id}"
                self._wandb = wandb
                group = getattr(self.args, "wandb_group", None)
                self._wandb.init(
                    project=project,
                    name=name,
                    config=asdict(self.args),
                    group=group,  # None is fine if no group is set
                )
                self._wandb_enabled = True
                logger.info(f"wandb logging enabled: project={project} name={name} group={group}")
            except Exception as e:
                logger.error(f"[warn] Failed to initialize wandb logging: {e}")
                self._wandb = None
                self._wandb_enabled = False

    def _log_wandb(self, d: dict):
        if self._wandb_enabled:
            try:
                self._wandb.log(d)
            except Exception as e:
                logger.error(f"[warn] wandb.log failed: {e}")

    def _update_wandb_config(self, d: dict):
        if self._wandb_enabled:
            try:
                self._wandb.config.update(d, allow_val_change=True)
            except Exception as e:
                logger.error(f"[warn] Failed to update wandb config: {e}")

    # ------- Checkpoint helpers -------
    @staticmethod
    def _ckpt_filename(*, val_value: Optional[float], step: int, tokens: int, run_start_minute: str, run_id: int, suffix: Optional[str] = None) -> str:
        os.makedirs("checkpoints", exist_ok=True)
        _val_trunc = math.trunc(val_value * 100) / 100 if val_value is not None else float("nan")
        return (
            f"checkpoints/{run_start_minute}-val{_val_trunc:.3f}-step{step:06d}-tokens{tokens}-run{run_id}"
            + (f"-{suffix}" if suffix else "")
            + ".pt"
        )

    def _save_checkpoint(self, *, val_value: Optional[float], step: int, run_start_minute: str, model: nn.Module, best_val: float, tokens: int, progress: ProgressMeter, overwrite: bool = False, suffix: Optional[str] = None) -> str:
        fname = self._ckpt_filename(val_value=val_value, step=step, tokens=tokens, run_start_minute=run_start_minute, run_id=self.run_id, suffix=suffix)
        if overwrite and self._last_run_ckpt_path and os.path.exists(self._last_run_ckpt_path) and self._last_run_ckpt_path != fname:
            try:
                os.remove(self._last_run_ckpt_path)
            except OSError:
                logger.warning(f"Failed to remove previous run checkpoint: {self._last_run_ckpt_path}")

        save_checkpoint(
            fname,
            model=model,
            hparams=asdict(self.args),
            step=step,
            best_val=best_val,
            progress_state=progress.state_dict(),
        )
        self._last_run_ckpt_path = fname
        return fname

    # ------- Build per-run state -------
    def _apply_lr_scale_inplace(self):
        # Be tolerant to various representations of lr_scale coming from CLI overrides.
        # Accept: float/int, numeric strings, and 1-element list/tuple.
        raw = getattr(self.args, "lr_scale", 1.0)
        def _parse_lr_scalar(x) -> float:
            # Fast-path numbers
            if isinstance(x, (int, float)):
                return float(x)
            # Single-element collections
            if isinstance(x, (list, tuple)):
                if len(x) == 1:
                    return _parse_lr_scalar(x[0])
                raise ValueError("lr_scale list/tuple must contain exactly one value")
            # Strings like "1.3" or "[1.3]" or quoted
            if isinstance(x, str):
                s = x.strip()
                # Strip surrounding brackets if present (common when users pass [1.3])
                if len(s) >= 2 and s[0] == "[" and s[-1] == "]":
                    s = s[1:-1].strip()
                # Strip surrounding quotes
                if len(s) >= 2 and ((s[0] == s[-1]) and s[0] in "\"'"):
                    s = s[1:-1]
                return float(s)
            # Last resort: attempt direct float conversion
            return float(x)

        try:
            lr_scale = _parse_lr_scalar(raw)
        except Exception:
            logger.warning(f"Failed to parse lr_scale={raw} as float.")
            lr_scale = 1.0
        # Normalize the stored value to the parsed float so downstream logs/configs are consistent
        try:
            setattr(self.args, "lr_scale", float(lr_scale))
        except Exception:
            pass
        try:
            for opt in (self.args.optimizers or []):
                if isinstance(opt, dict) and "lr" in opt and isinstance(opt["lr"], (int, float)):
                    opt["lr"] = float(opt["lr"]) * lr_scale
                pgs = opt.get("params") if isinstance(opt, dict) else None
                if isinstance(pgs, list):
                    for pg in pgs:
                        if isinstance(pg, dict) and "lr" in pg and isinstance(pg["lr"], (int, float)):
                            pg["lr"] = float(pg["lr"]) * lr_scale
            logger.info(f"Applied lr_scale={lr_scale} to configured learning rates.")
        except Exception:
            logger.error(f"Failed to apply lr_scale={lr_scale} to learning rates.")
            sys.exit(1)

        # If an lr_scale was provided (and is not effectively 1.0), append it as a suffix to the wandb run name
        try:
            if abs(float(lr_scale) - 1.0) > 1e-12 and not getattr(self, "_lr_suffix_applied", False):
                base = getattr(self.args, "wandb_run_name", "") or ""
                # compact human-friendly formatting for floats, avoids long reprs
                fmt = f"{float(lr_scale):.6g}"
                suffix = f"-lr{fmt}"
                new_name = (base + suffix) if base else f"lr{fmt}"
                setattr(self.args, "wandb_run_name", new_name)
                setattr(self, "_lr_suffix_applied", True)
                logger.info(f"wandb_run_name updated with lr_scale suffix: {new_name}")
        except Exception:
            logger.warning(f"Failed to add lr_scale suffix to wandb_run_name: {self.args.wandb_run_name}")

    def _build_data_and_evals(self) -> Tuple[Iterable, List[Tuple[str, Evaluator, int]], Optional[int]]:
        args = self.args
        is_task = (args.train_mode == "task")
        world_size = self.rt.world_size
        rank = self.rt.rank
        device_type = self.rt.device.type

        begin_shard_env = os.environ.get("BEGIN_SHARD")
        begin_shard = int(begin_shard_env) if begin_shard_env not in (None, "") else None

        val_evals: list[tuple[str, Evaluator, int]] = []

        if not is_task:
            train_ddg = DistributedDataGenerator(
                filename_pattern=args.train_shards,
                batch_size=world_size * args.training_sequence_length,
                rank=rank,
                world_size=world_size,
                start_shard=begin_shard,
                device=device_type,
            )
            tokens_per_step = world_size * args.training_sequence_length
        else:
            pad_to_multiple = WINDOW_BLOCK_SIZE if device_type == "cuda" else 1
            train_ddg = TaskDataGenerator(
                root=args.task_train_root,
                split=getattr(args, "task_train_split", "train"),
                batch_size=world_size,
                world_size=world_size,
                rank=rank,
                seed=int(getattr(args, "task_seed", 1337)),
                device=device_type,
                start_shard=begin_shard,
                drop_remainder=False,
                infinite=True,
                squeeze_singleton_batch=True,
                pad_to_multiple=pad_to_multiple,
            )
            # For Task SFT we use dynamic token counting
            tokens_per_step = None

            task_val_shards = getattr(args, "task_val_shards", []) or []
            for v in task_val_shards:
                label = v.get("type", "task")
                path = v.get("path")
                split = v.get("split", "val")
                t_tokens = int(v.get("target_tokens"))
                ddg = TaskDataGenerator(
                    root=path,
                    split=split,
                    batch_size=world_size,
                    world_size=world_size,
                    rank=rank,
                    seed=int(getattr(args, "task_seed", 1337)),
                    device=device_type,
                    start_shard=None,
                    drop_remainder=False,
                    infinite=True,
                    squeeze_singleton_batch=True,
                    pad_to_multiple=pad_to_multiple,
                )
                ev = Evaluator(
                    data_generator=ddg,
                    distributed_enabled=self.rt.use_distributed,
                    rank=rank,
                    attn_window_len=args.train_attention_window_len,
                    val_type='task',
                    log_samples=getattr(args, "task_val_debug_log_samples", False),
                )
                val_evals.append((label, ev, t_tokens))

        # Common eval setup (pretraining-style)
        vcfgs = getattr(args, "val_shards", []) or []
        if len(vcfgs) != 0:
            for v in vcfgs:
                label = v.get("type")
                path = v.get("path")
                seq_len = int(v.get("sequence_length"))
                t_tokens = int(v.get("target_tokens"))
                val_batch = self.rt.world_size * seq_len
                if t_tokens % val_batch != 0:
                    raise ValueError(
                        f"val shard '{label}': target_tokens ({t_tokens}) must be divisible by val_batch_size ({val_batch})"
                    )
                ddg = DistributedDataGenerator(path, val_batch, self.rt.rank, self.rt.world_size, device=device_type)
                ev = Evaluator(
                    data_generator=ddg,
                    distributed_enabled=self.rt.use_distributed,
                    rank=self.rt.rank,
                    attn_window_len=args.train_attention_window_len,
                    val_type='pretraining',
                )
                val_evals.append((label, ev, t_tokens))

        return train_ddg, val_evals, tokens_per_step

    def _perform_eval(self, training_time_ms: float, step: int, t0: float, progress: ProgressMeter, ema_dloss_per_token: float, best_val: float, run_start_minute: str,  val_evals: List[Tuple[str, Evaluator, int]]) -> Tuple[float, float, float, float, float]:
        logger.info("[eval] starting evaluations...")
        if self.rt.use_distributed:
            if logger.isDebugEnabled(): logger.debug(f"_perform_eval: beginning dist.barrier()")
            dist.barrier()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        self.rt.model.eval()

        per_ds_results: list[tuple[str, dict]] = []
        for label, ev, target_tokens in val_evals:
            logger.info(f"[eval] start dataset={label} target_tokens={target_tokens})")
            out = ev.eval(model=self.rt.model, total_tokens=target_tokens, schedule=progress.s)
            if logger.isDebugEnabled(): logger.debug(f"[eval] finished dataset={label} out={out}")
            per_ds_results.append((label, out))

        primary_label, primary_out = per_ds_results[0]
        cur_val = float(primary_out.get("val_loss"))
        last_val_loss = cur_val
        ema_dloss_per_token = primary_out.get("ema_dloss_per_token", ema_dloss_per_token)

        parts = [f"{lbl}:{float(out.get('val_loss')):.6f}" for lbl, out in per_ds_results]
        logger.info(
            f"step:{step} tokens:{progress.tokens_processed:,}/{progress.target_tokens:,} (s={progress.s:.4f}) "
            + " ".join(parts)
            + f" train_time:{training_time_ms:,.0f}ms ema_dloss_per_1e6_tokens:{ema_dloss_per_token * 1e6:.6f}"
        )

        wb = {
            "val/loss": cur_val,
            "val/ppl": math.exp(cur_val) if cur_val < 20 else float("inf"),
            "val/ema_dloss_per_token": ema_dloss_per_token,
            "tokens": progress.tokens_processed,
            "s": progress.s,
            "train/time_ms": training_time_ms + 1000 * (time.perf_counter() - t0),
            "step": step,
        }
        for lbl, out in per_ds_results:
            _loss = out.get("val_loss")
            wb[f"val/{lbl}/loss"] = _loss
            wb[f"val/{lbl}/ppl"] = math.exp(_loss) if _loss < 20 else float("inf")
        self._log_wandb(wb)

        if self.rt.is_master and self.args.save_checkpoint and progress.should_checkpoint():
            if cur_val < best_val:
                best_val = cur_val
                fname = self._save_checkpoint(
                    val_value=cur_val,
                    step=step,
                    run_start_minute=run_start_minute,
                    model=self.rt.model,
                    best_val=best_val,
                    tokens=progress.tokens_processed,
                    progress=progress,
                    overwrite=False,
                    suffix="best",
                )
                logger.info(f"Saved checkpoint to {fname} with val loss {float(cur_val):.6f}")
            else:
                logger.info(f"No improvement in val loss: best={best_val:.6f}, current={cur_val:.6f}. Skipping checkpoint.")

            progress.mark_checkpoint_done()

        self.rt.model.train()
        if self.rt.use_distributed:
            if logger.isDebugEnabled(): logger.debug(f"_perform_eval: final dist.barrier()")
            dist.barrier()
        t0 = time.perf_counter()
        return training_time_ms, t0, ema_dloss_per_token, best_val, last_val_loss

    def run(self) -> None:
        if logger.isDebugEnabled():
            import socket
            hostname = socket.gethostname()
            logger.debug(f"Trainer.run() starting: world_size={self.rt.world_size}, rank={self.rt.rank}, device={self.rt.device.type} hostname={hostname}")

        # Reset model and per-run state
        self.rt.reset_model_to_initial()
        _maybe_reset_peak_memory_stats(self.rt.device.type)
        self._apply_lr_scale_inplace()
        self._wandb_init()

        args = self.args
        if args.train_mode == "task":
            if not getattr(args, "task_train_root", None):
                raise ValueError("Task mode requires 'task_train_root'.")
            if int(args.target_tokens) <= 0:
                raise ValueError("Task mode requires 'target_tokens' > 0.")

        if not args.optimizers:
            raise ValueError("Training config must provide 'optimizers' list")

        # Build optimizers, freeze groups
        frozen_groups = [
            p_cfg["group"]
            for opt_cfg in args.optimizers
            for p_cfg in (opt_cfg.get("params") or [])
            if p_cfg.get("frozen")
        ]
        optimizers: list[torch.optim.Optimizer] = build_optimizers_from_cfg(
            cfg_list=args.optimizers,
            model=self.rt.model,
            rank=self.rt.rank,
            world_size=self.rt.world_size,
            frozen_groups=frozen_groups,
        )
        def _opt_params(opt: torch.optim.Optimizer) -> list[nn.Parameter]:
            return [p for g in opt.param_groups for p in g["params"]]
        opt2params = {opt: _opt_params(opt) for opt in optimizers}
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]

        # Build data and evals
        train_ddg, val_evals, tokens_per_step = self._build_data_and_evals()
        eval_every_tokens = None
        vlet = int(args.val_loss_every_tokens)
        if val_evals and vlet > 0:
            eval_every_tokens = vlet

        progress = ProgressMeter(
            target_tokens=int(args.target_tokens),
            eval_every_tokens=eval_every_tokens,
            checkpoint_per_n_tokens=int(args.checkpoint_per_n_tokens),
            checkpoint_warmup_tokens=int(args.checkpoint_warmup_tokens),
            log_every_tokens=int(self.args.log_interval),
        )

        # Tracking for eval stats and ETA
        last_val_loss = None
        ema_dloss_per_token = math.inf
        training_time_ms = 0.0
        best_val = float("inf")

        if self.rt.use_distributed:
            dist.barrier()
        step = 0
        t0 = time.perf_counter()
        warmup_end = 0.0

        # Timestamp for filenames
        run_start_dt = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        run_start_minute = run_start_dt.strftime("%Y%m%dT%H%M")

        # Training loop
        while progress.tokens_processed < progress.target_tokens:
            # Eval
            if progress.should_eval():
                training_time_ms, t0, ema_dloss_per_token, best_val, last_val_loss = self._perform_eval(
                    training_time_ms, step, t0, progress, ema_dloss_per_token, best_val, run_start_minute, val_evals
                )
                progress.mark_eval_done()

            # TRAIN
            ga_steps = max(1, int(args.grad_acc_steps))
            total_train_loss = 0.0
            tokens_this_step = 0
            for micro_step in range(ga_steps):
                if logger.isDebugEnabled(): logger.debug(f"micro_step={micro_step} in ga_steps={ga_steps}")

                inputs, targets = next(train_ddg)
                if logger.isDebugEnabled(): logger.debug(f"inputs.shape={inputs.shape}, targets.shape={targets.shape}")

                seq_len = inputs.size(-1)
                if args.train_mode == "task":
                    skip_count = 0
                    while seq_len > int(args.training_sequence_length):
                        # attempt to recover by getting a new batch
                        skip_count += 1
                        if skip_count > 100:
                            # significantly outside expectations - either the data must be cleansed or the model configuration is improper
                            logger.error(f"Fatal: failed to find inputs with seq_len <= {int(args.training_sequence_length)} after 100 attempts.")
                            sys.exit(1)
                        if logger.isDebugEnabled(): logger.debug(f"seq_len={seq_len} > int(args.training_sequence_length)={int(args.training_sequence_length)}")
                        inputs, targets = next(train_ddg)
                        seq_len = inputs.size(-1)

                tokens_this_step += int(seq_len) # TODO consider returning count from data generator to avoid sync
                assert tokens_this_step != 0
                if logger.isDebugEnabled(): logger.debug(f"step={step} cumulative tokens_microstep={tokens_this_step}")

                n_blocks = get_num_window_blocks(
                    progress.s,
                    attention_window_len=args.train_attention_window_len,
                    window_block_size=WINDOW_BLOCK_SIZE,
                ).to(self.rt.device.type)
                if logger.isDebugEnabled(): logger.debug(f"n_blocks={n_blocks} device={n_blocks.device}")
                with torch.autocast(self.rt.device.type, dtype=torch.bfloat16):
                    loss = self.rt.model(inputs, n_blocks, targets)

                loss_to_backward = loss / ga_steps
                if self.rt.use_distributed:
                    self.rt.model.require_backward_grad_sync = micro_step == ga_steps - 1
                loss_to_backward.backward()
                total_train_loss += float(loss.item()) # TODO reduce syncs by reporting only every N steps

            if logger.isDebugEnabled(): logger.debug(f"step={step} tokens_this_step={tokens_this_step} total_train_loss={total_train_loss}")
            if logger.isDebugEnabled() and self.rt.use_distributed: logger.debug(f"start opt2futures: dist.all_reduce()")
            opt2futures = {
                opt: (
                    [
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                        for p in params
                        if p.grad is not None
                    ]
                    if self.rt.use_distributed
                    else []
                )
                for opt, params in opt2params.items()
            }
            if logger.isDebugEnabled() and self.rt.use_distributed: logger.debug(f"len(opt2futures)={len([g for g in opt2futures.values() if g is not None])}")

            s = progress.s
            lr_scale_base = get_lr_scale(args.learning_rate_schedule, s, args.cooldown_frac)
            lr_scale = max(lr_scale_base, float(getattr(args, "learning_rate_floor", 0.0)))

            for opt in optimizers:
                if isinstance(opt, Muon):
                    continue
                for group in opt.param_groups:
                    group["lr"] = group["initial_lr"] * lr_scale
                    # if logger.isDebugEnabled(): logger.debug(f"set lr={group['lr']} for opt={opt}")

            for opt in optimizers:
                if isinstance(opt, Muon):
                    for group in opt.param_groups:
                        frac = s
                        group["momentum"] = (1 - frac) * 0.85 + frac * 0.95

            for opt in optimizers:
                if self.rt.use_distributed:
                    if logger.isDebugEnabled(): logger.debug(f"wait for opt2futures...")
                    torch.futures.collect_all(opt2futures[opt]).wait()
                    if logger.isDebugEnabled(): logger.debug(f"done waiting for opt2futures")
                opt.step()
            self.rt.model.zero_grad(set_to_none=True)

            if args.train_mode == "task":
                # task token counts vary across each batch
                if self.rt.use_distributed:
                    t = torch.tensor(tokens_this_step, device=self.rt.device, dtype=torch.int32)
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    global_step_tokens = int(t.item())
                else:
                    global_step_tokens = tokens_this_step
            else:
                # pretraining token counts are constant across each batch so we can infer global_step_tokens
                global_step_tokens = tokens_this_step * self.rt.world_size

            progress.update(global_step_tokens)
            if logger.isDebugEnabled(): logger.debug(f"update {global_step_tokens} -> progress.tokens_processed={progress.tokens_processed}")

            step += 1

            train_loss_est = total_train_loss / ga_steps

            if self.rt.use_distributed:
                if logger.isDebugEnabled(): logger.debug(f"Starting dist.all_reduce() -> train_loss_est={train_loss_est}")
                loss_tensor = torch.tensor(train_loss_est, device=self.rt.device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                train_loss_est = loss_tensor.item()
                if logger.isDebugEnabled(): logger.debug(f"Finished dist.all_reduce() -> train_loss_est={train_loss_est}")

            approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
            if step == 9:
                warmup_end = approx_training_time_ms
            avg_step = (
                f"avg_step:{(approx_training_time_ms - warmup_end) / max(step - 9, 1):.2f}ms"
                if step >= 10
                else "avg_step: (warmup to step 10)"
            )

            if progress.should_log():
                logger.info(
                    f"step:{step} train_loss:{train_loss_est:.4f} tokens:{progress.tokens_processed:,}/{progress.target_tokens:,} "
                    f"(s={progress.s:.4f}) train_time:{approx_training_time_ms:,.0f}ms {avg_step} "
                    f"lr_scale:{lr_scale:.4f} (base:{lr_scale_base:.4f} floor:{float(getattr(args, 'learning_rate_floor', 0.0)):.4f})"
                )
                self._log_wandb(
                    {
                        "train/loss": train_loss_est,
                        "train/ppl": math.exp(train_loss_est) if train_loss_est < 20 else float("inf"),
                        "tokens": progress.tokens_processed,
                        "s": progress.s,
                        "lr_scale": lr_scale,
                        "lr_scale_base": lr_scale_base,
                        "learning_rate_floor": float(getattr(args, "learning_rate_floor", 0.0)),
                        "train/time_ms": approx_training_time_ms,
                        "avg_step_ms": avg_step,
                    }
                )
                progress.mark_log_done()

        # End of training
        training_time_ms, t0, ema_dloss_per_token, best_val, last_val_loss = self._perform_eval(
            training_time_ms, step, t0, progress, ema_dloss_per_token, best_val, run_start_minute, val_evals
        )
        if self.rt.is_master and self.args.save_checkpoint:
            _ = self._save_checkpoint(
                val_value=last_val_loss,
                step=step,
                run_start_minute=run_start_minute,
                model=self.rt.model,
                best_val=best_val,
                tokens=progress.tokens_processed,
                progress=progress,
                overwrite=False,
                suffix="final",
            )

        peak_mem = _get_max_memory_allocated(self.rt.device.type)
        if peak_mem is not None:
            logger.info(f"peak memory allocated: {peak_mem // 1024 // 1024} MiB")
        if self._wandb_enabled:
            try:
                self._wandb.finish()
            except Exception as e:
                logger.warning("wandb.finish(): " + str(e))

        report = build_report(self.rt.model)
        logger.info(f"Final model report:\n{format_report_text(report)}")


def compute_group_max_seq_len(arg_list: List[Hyperparameters]) -> int:
    max_train = max(int(a.training_sequence_length) for a in arg_list)
    max_val = 0
    for a in arg_list:
        try:
            lens = [int(v.get("sequence_length")) for v in (getattr(a, "val_shards", []) or [])]
        except Exception:
            logger.error(f"Failed to parse val_shards for {a}")
            lens = []
        if lens:
            max_val = max(max_val, max(lens))
    return max(max_train, max_val)


def partition_runs_by_compile_key(run_args: List[Tuple[int, Hyperparameters]], *, device_type: str, world_size: int) -> Dict[CompileKey, List[Tuple[int, Hyperparameters]]]:
    groups: Dict[CompileKey, List[Tuple[int, Hyperparameters]]] = {}
    for run_id, a in run_args:
        dynamic = (a.train_mode == "task")
        ck = derive_compile_key(a, device_type=device_type, world_size=world_size, dynamic=dynamic)
        groups.setdefault(ck, []).append((run_id, a))
    return groups


__all__ = [
    "CompileKey",
    "derive_compile_key",
    "CompiledRuntime",
    "TrainingSession",
    "compute_group_max_seq_len",
    "partition_runs_by_compile_key",
]
