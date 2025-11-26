import json
import math
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional

from data.data_gen_stream import DistributedDataGenerator
from data.data_gen_task import TaskDataGenerator
from models import model_from_spec
from tools.checkpoint import model_from_checkpoint
from tools.checkpoint import save_checkpoint
from tools.model_report import build_report, format_report_text
from training.eval import Evaluator
from training.hparams import Hyperparameters, load_hparams_from_yaml, apply_cli_overrides
from training.optim import Muon, get_lr_scale
from training.optim import build_optimizers_from_cfg
from training.optim import get_num_window_blocks, set_full_windows
from training.progress import ProgressMeter

WINDOW_BLOCK_SIZE = 128

########################################
#        App Config & Setup            #
########################################
# Load config from YAML (first CLI arg if provided)
_config_path = sys.argv[1] if len(sys.argv) > 1 else None
args = load_hparams_from_yaml(_config_path)
# Apply overrides from the remaining CLI args: support --key=value or key=value
args = apply_cli_overrides(args, sys.argv[2:])
# Apply optional scheduling toggle for attention windows
set_full_windows(args.full_windows)
run_id = int(os.environ.get("RUN_ID", 0))
### End App Config ###

train_mode = getattr(args, "train_mode", "pretrain")
if train_mode not in ("pretrain", "task"):
    raise ValueError(f"Unsupported train_mode={train_mode!r}; expected 'pretrain' or 'task'.")

is_task = (train_mode == "task")

########################################
#        Torch Setup                   #
########################################
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import torch
from torch import nn
import torch.distributed as dist

torch.manual_seed(1337)

TORCH_DISABLE_MODEL_COMPILE = os.environ.get("TORCH_DISABLE_MODEL_COMPILE", "0") == "1"
if not TORCH_DISABLE_MODEL_COMPILE:
    # Configure inductor/dynamo compile/tuning
    torch._inductor.config.coordinate_descent_tuning = bool(getattr(args, "torch_coordinate_descent_tuning", False))
    torch._dynamo.config.compiled_autograd = True
    torch._dynamo.config.error_on_nested_fx_trace = False  # workaround/diagnostic for dynamo error related to FlexAttention
# torchrun sets these env variables
rank = int(os.environ.get("RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))
local_rank = int(os.environ.get("LOCAL_RANK", "0"))

use_distributed = world_size > 1

device = (
    torch.device("cuda", local_rank)
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.mps.is_available()
    else torch.device("cpu")
)
if device.type == "cuda":
    torch.cuda.set_device(device)
    if use_distributed:
        dist.init_process_group(backend="nccl", device_id=device, world_size=world_size)
        dist.barrier()
    is_master = rank == 0
elif device.type == "mps":
    if use_distributed:
        raise ValueError("Distributed training is not supported on macOS/MPS")
    is_master = True
else:
    is_master = True


def maybe_compile(model: nn.Module, dynamic: bool = False) -> nn.Module:
    if TORCH_DISABLE_MODEL_COMPILE:
        logger.info(f"Model compilation disabled: TORCH_DISABLE_MODEL_COMPILE={TORCH_DISABLE_MODEL_COMPILE}")
        return model
    else:
        backend = 'inductor' if device.type == 'cuda' else 'aot_eager'
        use_dynamic = dynamic or is_task
        if is_task:
            torch._dynamo.config.force_parameter_static_shapes = False
        logger.info(f"Compiling model (dynamic={use_dynamic}) (backend={backend}). This may take several minutes and occur during the initial evals.")
        model = torch.compile(model, dynamic=use_dynamic, backend=backend)
        return model


def maybe_reset_peak_memory_stats():
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    else:
        logger.debug(f"reset_memory_stats() unsupported on device.type={device.type}")


def get_max_memory_allocated() -> Optional[int]:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated()
    else:
        logger.debug(f"max_memory_allocated() unsupported on device.type={device.type}")
        return None


########################################
#        Logging                       #
########################################
from tools.master_logger import MasterLogger

logger = MasterLogger
### End Logging ###

# Run start timestamp truncated to the minute (UTC)
_run_start_dt = datetime.now(timezone.utc).replace(second=0, microsecond=0)
run_start_minute = _run_start_dt.strftime("%Y%m%dT%H%M")
# Track last checkpoint file saved for this run so we can replace it
_last_run_ckpt_path: Optional[str] = None

########################################
#       Weights & Biases (Optional)    #
########################################
_wandb = None
_wandb_enabled = False
if is_master and args.wandb_log:
    try:
        import wandb as _wandb

        _project = args.wandb_project or "daisy"
        _name = args.wandb_run_name+"-TAWL"+str(args.train_attention_window_len)+"-TSL"+str(args.training_sequence_length)
        _wandb.init(project=_project, name=_name, config=asdict(args))
        _wandb_enabled = True
        logger.info(f"wandb logging enabled: project={_project} name={_name}")
        art = _wandb.Artifact(name=f"models-src-{run_id}", type="code")
        import glob

        files = [p for p in glob.glob("models/**/*.py", recursive=True) if os.path.basename(p) != "__init__.py"]
        for p in files:
            art.add_file(p, name=os.path.relpath(p, start="models"))
        _wandb.log_artifact(art)
    except Exception as _e:
        logger.error(f"[warn] Failed to initialize wandb logging: {_e}")
        _wandb = None
        _wandb_enabled = False


def log_wandb(d: dict):
    if _wandb_enabled:
        try:
            _wandb.log(d)
        except Exception as _e:
            logger.error(f"[warn] wandb.log failed: {_e}")


def update_wandb_config(d: dict):
    if _wandb_enabled:
        try:
            _wandb.config.update(d, allow_val_change=True)
        except Exception as _e:
            logger.error(f"[warn] Failed to update wandb config: {_e}")


########################################
#               Helpers                #
########################################
def _build_hparams_from_args(args: Hyperparameters) -> dict:
    """Build a checkpoint hparams dict from training args."""
    return asdict(args)


def _get_ckpt_filename(
    *,
    val_value: Optional[float],
    step: int,
    tokens: int,
    run_start_minute: str,
    run_id: int,
    suffix: Optional[str] = None,
) -> str:
    os.makedirs("checkpoints", exist_ok=True)
    _val_trunc = math.trunc(val_value * 100) / 100 if val_value is not None else float("nan")
    return (
        f"checkpoints/{run_start_minute}-val{_val_trunc:.3f}-step{step:06d}-tokens{tokens}-run{run_id}"
        + (f"-{suffix}" if suffix else "")
        + ".pt"
    )


def _save_checkpoint(
    *,
    val_value: Optional[float],
    step: int,
    run_start_minute: str,
    run_id: int,
    model: nn.Module,
    best_val: float,
    args: Hyperparameters,
    tokens_per_step: int,
    tokens: int,
    progress,
    overwrite: bool = False,
    suffix: Optional[str] = None,
) -> str:
    """Create a run-scoped checkpoint filename, remove the previous run checkpoint if different,
    save the new checkpoint, and remember its path.
    """
    global _last_run_ckpt_path
    fname = _get_ckpt_filename(
        val_value=val_value,
        step=step,
        tokens=tokens,
        run_start_minute=run_start_minute,
        run_id=run_id,
        suffix=suffix,
    )
    if overwrite and _last_run_ckpt_path and os.path.exists(_last_run_ckpt_path) and _last_run_ckpt_path != fname:
        try:
            os.remove(_last_run_ckpt_path)
        except OSError:
            logger.warning(f"Failed to remove previous run checkpoint: {_last_run_ckpt_path}")

    save_checkpoint(
        fname,
        model=model,
        hparams=_build_hparams_from_args(args),
        step=step,
        best_val=best_val,
        tokens_per_step=tokens_per_step,
        progress_state=progress.state_dict(),
    )
    _last_run_ckpt_path = fname
    return fname


########################################
#        Model & Optimizers            #
########################################

# Rehydrate critical hyperparameters from checkpoint if available
best_val_from_ckpt = None
resume_from_step = None
_resume_tokens_per_step: Optional[int] = None
_ckpt_obj = None
if args.init_checkpoint:
    # TODO diff args/hparams/modelspec from checkpoint
    model, hparams = model_from_checkpoint(args.init_checkpoint, device=device, dynamic_shapes=is_task) #dynamic_shapes required for variable task shapes
    logger.info("Rehydrated model from checkpoint.")
else:
    # Initialize rotary buffers with the maximum sequence length that the model might see
    _msl = int(getattr(args, "max_seq_len", args.training_sequence_length))
    _val_lens = []
    try:
        _val_lens = [int(v.get("sequence_length")) for v in (getattr(args, "val_shards", []) or [])]
    except Exception:
        _val_lens = []
    _max_val_len = max(_val_lens) if _val_lens else int(args.training_sequence_length)
    max_len = max(_msl, int(args.training_sequence_length), _max_val_len)
    setattr(args, "max_seq_len", max_len)
    model = model_from_spec(args.model_spec, device=device.type, overrides=asdict(args))
    hparams = _build_hparams_from_args(args)

logger.info("Hyperparameters:\n" + json.dumps(asdict(args), indent=2, sort_keys=True))
# Ensure wandb sees the final effective hyperparameters (after overrides and any rehydration)
update_wandb_config(asdict(args))  # TODO without diff of checkpoint/model_spec_from_args the reported model args may be incorrect

if is_task:
    if not getattr(args, "task_train_root", None):
        raise ValueError("Task mode requires 'task_train_root'.")
    if int(args.target_tokens) <= 0:
        raise ValueError("Task mode requires 'target_tokens' > 0.")

for m in model.modules():
    if isinstance(m, nn.Embedding):
        m.bfloat16()
for param in model.parameters():
    if use_distributed:
        dist.broadcast(param.detach(), 0)

if not args.optimizers:
    raise ValueError("Training config must provide 'optimizers' list")

frozen_groups = [
    p_cfg["group"]
    for opt_cfg in args.optimizers
    for p_cfg in (opt_cfg.get("params") or [])
    if p_cfg.get("frozen")
]
optimizers: list[torch.optim.Optimizer] = build_optimizers_from_cfg(
    cfg_list=args.optimizers,
    model=model,
    rank=rank,
    world_size=world_size,
    frozen_groups=frozen_groups
)


def opt_params(opt: torch.optim.Optimizer) -> list[nn.Parameter]:
    return [p for g in opt.param_groups for p in g["params"]]


opt2params = {opt: opt_params(opt) for opt in optimizers}
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

report = build_report(model)
logger.info(f"Model report:\n{format_report_text(report)}")
model = maybe_compile(model, dynamic=is_task) # need dynamic compile for varible task shapes


########################################
#    Training and Validation Setup     #
########################################

maybe_reset_peak_memory_stats()

_begin_shard_env = os.environ.get("BEGIN_SHARD")
_begin_shard = int(_begin_shard_env) if _begin_shard_env not in (None, "",) else None

_val_evals: list[tuple[str, Evaluator, int]] = []

if train_mode == "pretrain":
    _train_ddg = DistributedDataGenerator(
        filename_pattern=args.train_shards,
        batch_size=world_size * args.training_sequence_length,
        rank=rank,
        world_size=world_size,
        start_shard=_begin_shard,
        device=device.type,
    )

    tokens_per_step = world_size * args.training_sequence_length

elif train_mode == "task":
    pad_to_multiple = WINDOW_BLOCK_SIZE if device.type == "cuda" else 1
    _train_ddg = TaskDataGenerator(
        root=args.task_train_root,
        split=getattr(args, "task_train_split", "train"),
        batch_size=world_size,  # one example per rank
        world_size=world_size,
        rank=rank,
        seed=int(getattr(args, "task_seed", 1337)),
        device=device.type,
        start_shard=_begin_shard,
        drop_remainder=False,
        infinite=True,
        squeeze_singleton_batch=True,
        pad_to_multiple=pad_to_multiple
    )
    _task_val_shards = getattr(args, "task_val_shards", []) or []
    for _v in _task_val_shards:
        _label = _v.get("type", "task")
        _path = _v.get("path")
        _split = _v.get("split", "val")
        _tokens = int(_v.get("target_tokens"))
        _ddg = TaskDataGenerator(
            root=_path,
            split=_split,
            batch_size=world_size,  # one example per rank
            world_size=world_size,
            rank=rank,
            seed=int(getattr(args, "task_seed", 1337)),
            device=device.type,
            start_shard=None,
            drop_remainder=False,
            infinite=True,
            squeeze_singleton_batch=True,
            pad_to_multiple=pad_to_multiple
        )
        _eval = Evaluator(
            data_generator=_ddg,
            distributed_enabled=use_distributed,
            rank=rank,
            training_sequence_length=args.training_sequence_length,
            val_type='task',
            log_samples=getattr(args, "task_val_debug_log_samples", False),
            mark_dynamic_dim = 0,
            mark_dynamic_min = 1,
            mark_dynamic_max = int(args.training_sequence_length)
        )
        _val_evals.append((_label, _eval, _tokens))

    # For Task SFT we use dynamic token counting; tokens_per_step is not meaningful.
    tokens_per_step = None
else:
    raise ValueError(f"Unknown train_mode: {train_mode}")


#  Common Eval Setup
_val_cfgs = getattr(args, "val_shards", []) or []
if len(_val_cfgs) != 0:
    for _v in _val_cfgs:
        _label = _v.get("type")
        _path = _v.get("path")
        _seq_len = int(_v.get("sequence_length"))
        _tokens = int(_v.get("target_tokens"))
        val_batch_size = world_size * _seq_len
        if _tokens % val_batch_size != 0:
            raise ValueError(
                f"val shard '{_label}': target_tokens ({_tokens}) must be divisible by val_batch_size ({val_batch_size})"
            )
        _ddg = DistributedDataGenerator(
            _path,
            val_batch_size,
            rank,
            world_size,
            device=device.type,
        )
        _eval = Evaluator(
            data_generator=_ddg,
            distributed_enabled=use_distributed,
            rank=rank,
            training_sequence_length=args.training_sequence_length,
            val_type='pretraining'
        )
        _val_evals.append((_label, _eval, _tokens))

_ga_steps_cfg = max(1, int(args.grad_acc_steps))
if tokens_per_step is not None:
    _tokens_per_optim_step = tokens_per_step * _ga_steps_cfg
else:
    # TODO: for Task SFT we temporarily use an upper bound based on max sequence length for checkpoint metadata, should be better estimated but probably inconsequential
    _tokens_per_optim_step = int(args.training_sequence_length) * _ga_steps_cfg

_eval_every_tokens = None
if _val_evals and int(args.val_loss_every_tokens) > 0:
    _eval_every_tokens = int(args.val_loss_every_tokens)

progress = ProgressMeter(
    target_tokens=int(args.target_tokens),
    eval_every_tokens=_eval_every_tokens,
    checkpoint_per_n_tokens=int(args.checkpoint_per_n_tokens),
    checkpoint_warmup_tokens=int(args.checkpoint_warmup_tokens),
)

# Tracking for eval stats and ETA
last_val_loss = None
ema_dloss_per_token = math.inf
training_time_ms = 0
best_val = float("inf") if best_val_from_ckpt is None else best_val_from_ckpt

if use_distributed:
    dist.barrier()
step = 0
t0 = time.perf_counter()
warmup_end = 0.0
max_seq_len = args.training_sequence_length
while progress.tokens_processed < progress.target_tokens:
    # --------------- Evaluation -----------------
    if progress.should_eval():
        logger.info("[eval] starting evaluations...")
        if use_distributed:
            dist.barrier()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()

        per_ds_results: list[tuple[str, dict]] = []
        for _label, _ev, _target_tokens in _val_evals:
            _ev.reset_generator()
            logger.info(
                f"[eval] start dataset={_label} target_tokens={_target_tokens})"
            )
            _world_batch = _ev.world_batch_tokens or 0
            _steps = _target_tokens // _world_batch if _world_batch > 0 else 0
            _out = _ev.eval(model=model, total_tokens=_target_tokens)
            per_ds_results.append((_label, _out))

        # Canonical/primary val metrics use the first dataset
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
        log_wandb(wb)

        if is_master and args.save_checkpoint and progress.should_checkpoint():
            if cur_val < best_val:
                best_val = cur_val
                fname = _save_checkpoint(
                    val_value=cur_val,
                    step=step,
                    run_start_minute=run_start_minute,
                    run_id=run_id,
                    model=model,
                    best_val=best_val,
                    args=args,
                    tokens_per_step=_tokens_per_optim_step,
                    tokens=progress.tokens_processed,
                    progress=progress,
                    overwrite=False,
                    suffix="best",
                )
                logger.info(f"Saved checkpoint to {fname} with val loss {float(cur_val):.6f}")
            else:
                logger.info(
                    f"No improvement in val loss: best={best_val:.6f}, current={cur_val:.6f}. Skipping checkpoint."
                )

            progress.mark_checkpoint_done()

        model.train()
        if use_distributed:
            dist.barrier()
        t0 = time.perf_counter()
        progress.mark_eval_done()

    # --------------- TRAINING SECTION -----------------
    ga_steps = max(1, int(args.grad_acc_steps))
    total_train_loss = 0.0
    tokens_this_step = 0
    skipped = False
    for micro_step in range(ga_steps):
        inputs, targets = next(_train_ddg)

        # TODO hack remove
        min_seq_len = 4096

        def seq_len_from_progress(s: float,
                                  min_len: int,
                                  max_len: int,
                                  block_size: int) -> int:
            n_blocks = (max_len - min_len) // block_size
            # s=0  -> 0, s=1 -> n_blocks
            block_idx = int(n_blocks * s)
            return min_len + block_size * block_idx
        target_seq_len = seq_len_from_progress(progress.s, min_seq_len, max_seq_len, WINDOW_BLOCK_SIZE)
        inputs = inputs[:target_seq_len]
        tokens_per_step = inputs.size(0)

        ####################

        if is_task:
            seq_len = inputs.size(-1)
            if seq_len > int(args.training_sequence_length):
                logger.debug(
                    f"Skipping example: Task example length {seq_len} exceeds training_sequence_length "
                    f"{int(args.training_sequence_length)}"
                )
                skipped = True
                continue

            tokens_this_step += int(seq_len)

        logger.debug(
            f"Pre-autocast: inputs.shape={inputs.shape} inputs.device.type={inputs.device.type} "
            f"targets.shape={targets.shape} targets.device.type={targets.device.type}"
        )

        n_blocks = get_num_window_blocks(progress.s,
                                         attention_window_len=args.train_attention_window_len,
                                         window_block_size=WINDOW_BLOCK_SIZE,
                                         ).to(device.type)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            loss = model(inputs, n_blocks, targets)

        loss_to_backward = loss / ga_steps
        if use_distributed:
            model.require_backward_grad_sync = micro_step == ga_steps - 1
        loss_to_backward.backward()
        total_train_loss += float(loss.item())

    if not skipped:
        opt2futures = {
            opt: (
                [
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                    for p in params
                    if p.grad is not None
                ]
                if use_distributed
                else []
            )
            for opt, params in opt2params.items()
        }

        s = progress.s
        lr_scale_base = get_lr_scale(args.learning_rate_schedule, s, args.cooldown_frac)
        lr_scale = max(lr_scale_base, float(getattr(args, "learning_rate_floor", 0.0)))

        for opt in optimizers:
            if isinstance(opt, Muon):
                continue
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * lr_scale

        for opt in optimizers:
            if isinstance(opt, Muon):
                for group in opt.param_groups:
                    frac = s
                    group["momentum"] = (1 - frac) * 0.85 + frac * 0.95

        for opt in optimizers:
            if use_distributed:
                torch.futures.collect_all(opt2futures[opt]).wait()
            opt.step()
        model.zero_grad(set_to_none=True)

        if not is_task:
            progress.update(tokens_per_step * ga_steps)
        else:
            progress.update(tokens_this_step * world_size)

        step += 1

        train_loss_est = total_train_loss / ga_steps

        if use_distributed:
            loss_tensor = torch.tensor(train_loss_est, device=device)
            torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.AVG)
            train_loss_est = loss_tensor.item()

        approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
        if step == 9:
            warmup_end = approx_training_time_ms
        avg_step = (
            f"avg_step:{(approx_training_time_ms - warmup_end) / max(step - 9, 1):.2f}ms"
            if step >= 10
            else "avg_step: (warmup to step 10)"
        )
        logger.info(
            f"step:{step} train_loss:{train_loss_est:.4f} tokens:{progress.tokens_processed:,}/{progress.target_tokens:,} "
            f"(s={progress.s:.4f}) train_time:{approx_training_time_ms:,.0f}ms {avg_step} "
            f"lr_scale:{lr_scale:.4f} (base:{lr_scale_base:.4f} floor:{float(getattr(args, 'learning_rate_floor', 0.0)):.4f})"
        )
        log_wandb(
            {
                "train/loss": train_loss_est,
                "train/ppl": math.exp(train_loss_est) if train_loss_est < 20 else float("inf"),
                "tokens": progress.tokens_processed,
                "s": progress.s,
                "lr_scale": lr_scale,
                "lr_scale_base": lr_scale_base,
                "learning_rate_floor": float(getattr(args, "learning_rate_floor", 0.0)),
                "train/time_ms": approx_training_time_ms,
                "target_seq_len":target_seq_len
            }
        )

# End of training: save final checkpoint
if is_master and args.save_checkpoint:
    _ = _save_checkpoint(
        val_value=last_val_loss,
        step=step,
        run_start_minute=run_start_minute,
        run_id=run_id,
        model=model,
        best_val=best_val,
        args=args,
        tokens_per_step=_tokens_per_optim_step,
        tokens=progress.tokens_processed,
        progress=progress,
        overwrite=False,
        suffix="final",
    )

_peak_mem = get_max_memory_allocated()
if _peak_mem is not None:
    logger.info(f"peak memory allocated: {_peak_mem // 1024 // 1024} MiB")
if _wandb_enabled:
    try:
        _wandb.finish()
    except Exception as _e:
        logger.warning("wandb.finish(): " + str(_e))
if use_distributed and dist.is_initialized():
    dist.destroy_process_group()
