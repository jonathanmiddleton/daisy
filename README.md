# Daisy

**A compact, high-efficiency GPT training stack for small-to-mid-scale language models**

Daisy is designed for engineers who want to train decoder-only transformer models (150M–1.6B parameters) with limited computational resources while maintaining production-grade performance. The framework emphasizes clarity, efficiency, and pragmatic design choices that make training accessible without sacrificing quality.

Whether you're pretraining on billions of tokens, fine-tuning for domain adaptation, or building instruction-following models, Daisy provides the tools and optimizations you need to get results quickly.

<table border="0" style="border: 0 !important; border-collapse: collapse;">
  <tr style="border: 0 !important;">
    <td style="border: 0 !important;">
        <ul>
        <li><b>Scale:</b> 150M–1.6B parameter models</li>
        <li><b>Context:</b> Long-context via sliding windows (up to 65K tokens)</li>
        <li><b>Precision:</b> BFloat16-first with selective FP32 upcasting</li>
        <li><b>Attention:</b> Merged projections, windowed attention, RoPE</li>
        <li><b>Training:</b> DDP via torchrun, token-based scheduling</li>
        <li><b>Configs:</b> YAML-based with CLI overrides</li>
        </ul>
    </td>
    <td style="border: 0 !important;">
      <img src="assets/daisy.png" alt="Daisy" width="160">
    </td>
  </tr>
</table>



## Quick Start

### Installation

```bash
conda env create -f environment.yml
conda activate daisy
# or: pip install -r requirements.txt
```

### Prepare Data (Optional)

```bash
python data/cached_fineweb100B.py
# or the EDU variant:
# python data/cached_fineweb_edu_100B.py
```

### Train a Model
Model training requires a CUDA device. (8xA100/H100 with BFloat16 support is recommended.)
```bash
./run.sh config/pretrain_420m.yml -n 8
```

The `-n` flag specifies the number of GPUs. Any configuration parameter can be overridden on the command line:

```bash
./run.sh config/pretrain_420m.yml -n 8 target_tokens=3000000000 learning_rate_schedule=linear_decay
```

### Generate Text
CPU, CUDA, and MPS are supported for inference.
```bash
python sample.py checkpoints/20251018T0111-val1.750-step001300-run1-best.pt \
    --device cuda \
    --max_tokens 256 \
    --temperature 0.7 \
    --top_k 50 \
    --repetition_penalty 1.15
```

For interactive chat:

```bash[env.example](../../Downloads/env.example)
python sample.py checkpoints/instruct_model.pt --chat
```

### (minimal) OpenAI Responses-style REST API

```bash
pip install -r app.requirements.txt
# set CHECKPOINT_PATH and optional device/auth in env
export CHECKPOINT_PATH=/abs/path/to/checkpoints/model.pt
uvicorn app:app --host 0.0.0.0 --port 8000
```

```bash
# Example env exports. Checkpoint path is required.
CHECKPOINT_PATH=/abs/path/to/checkpoints/model.pt

# Optional
DEVICE=
BASIC_AUTH_USER=user
BASIC_AUTH_PASS=pass
SEED=1337
```

#### Usage
An example client is available in `examples/app_client.py`. 

```bash
curl -u user:pass \
  -X POST http://localhost:8000/v1/responses \
  -H 'content-type: application/json' \
  -d '{
    "model":"daisy",
    "input":[
      {"role":"system","content":[{"type":"input_text","text":"You are concise."}]},
      {"role":"user","content":[{"type":"input_text","text":"Name two prime numbers > 10."}]}
    ],
    "max_output_tokens":64
  }'
```

#### Example Request
```json
{
  "model": "daisy",
  "input": [
    {"role":"system","content":[{"type":"input_text","text":"..."}]},
    {"role":"user","content":[{"type":"input_text","text":"..."}]},
    {"role":"assistant","content":[{"type":"input_text","text":"..."}]}
  ],
  "session_id": "optional",
  "max_output_tokens": 256,
  "temperature": 0.7,
  "top_k": 50,
  "top_p": 0.95,
  "repetition_penalty": 1.15,
  "stream": false
}
```
#### Example Response
```json
{
  "id": "resp_...",
  "object": "response",
  "created": 1730...,
  "model": "daisy",
  "output": [{"type":"message","role":"assistant","content":[{"type":"output_text","text":"..."}]}],
  "usage": {"prompt_tokens":N,"completion_tokens":M,"total_tokens":N+M},
  "meta": {"prefill_ms":..., "gen_ms":...},
  "session_id": "sess_..."
}

```
---

## Model Architecture

Daisy implements a decoder-only transformer architecture with several key optimizations for efficiency and performance. The design philosophy prioritizes memory efficiency, training stability, and long-context capability while maintaining simplicity.

### Core Components

#### DaisyCore (models/daisy/daisy_core.py)

The main model class orchestrates the forward pass through embedding layers, transformer blocks, and the language modeling head. Key architectural features include:

**Token Embeddings with Value Augmentation**: The model maintains a primary token embedding (`self.embed`) plus three auxiliary value embeddings (`self.value_embeds`). These value embeddings are strategically injected into the first three and last three transformer layers, providing additional representational capacity at critical points in the network. Mid-layers operate without value embeddings for efficiency.

**Skip Connections**: Non-adjacent layer connections are defined by `_get_skip_map(L)`, which creates skip paths from layers around the midpoint to earlier layers. For a network with L layers, K = max(1, L // 8) skip connections are created, centered at layer c = L // 2, with spacing s = max(1, L // (2 * (K + 1))). Each skip connection is weighted by a learnable scalar in `self.scalars[:L]`.

**Merged Language Model Head**: The output projection (`self.lm_head_w`) is initialized to zero (unless `DISABLE_O_ZERO_INIT=1`), following the principle that early training should rely on residual paths. The head dimension is rounded up to the next multiple of 128 for memory alignment. During training, logits are computed in BF16 and scaled by `15 * logits * rsqrt(logits^2 + 225)` before cross-entropy loss, providing adaptive gradient scaling.

**Learnable Scalar Gates**: The model maintains three sets of learnable scalars in a single parameter tensor:
- Skip weights (num_layers scalars): Scale skip connections
- Residual mixing weights (num_layers × 2): Blend current activations with initial embeddings
- Value embedding mixing weights (num_layers × 2): Blend attention values with token value embeddings

**Block Mask Generation**: The `create_blockmasks()` method constructs document-aware attention masks that respect EOS token boundaries. It generates two mask types: a "long" mask using the full sliding window and a "short" mask using half the window size. These are cycled in a long-short-short-short pattern across layers, inspired by Gemma 2. The masks use block-sparse representation with `window_block_size` (typically 128) as the fundamental granularity.

#### CausalSelfAttention (models/daisy/attention.py)

The attention mechanism implements several optimizations for efficiency and stability:

**Merged QKVO Projections**: All four projection matrices (Query, Key, Value, Output) are fused into a single weight tensor `self.qkvo_w` with shape `(4, num_heads * head_dim, model_dim)`. This reduces memory reads from four separate operations to one, improving cache efficiency. The output projection is zero-initialized unless disabled for testing.

**Rotary Position Embeddings (RoPE)**: Position information is encoded via rotary embeddings applied to Q and K tensors after projection. The implementation uses half-truncated RoPE with base frequency tuning (base=1024). For head dimensions < 4, standard RoPE is used; otherwise, only the first quarter of frequencies are active, with the remainder zero-padded. This reduces computational cost while maintaining positional awareness.

**QK Normalization**: Both query and key tensors undergo RMS normalization before the attention operation. This stabilizes training by preventing attention logit magnitudes from growing unbounded, particularly important for long sequences.

**Fixed Attention Scale**: Rather than using the standard `head_dim^(-0.5)` scaling, attention logits are scaled by a fixed constant `self.attn_scale = 0.12`. This value was tuned empirically for the model scale and provides more stable gradients than adaptive scaling.

**Dual Attention Paths**: The forward pass uses PyTorch's `flex_attention` with block masks for efficient training on long sequences. The `step()` method (for autoregressive generation) uses `scaled_dot_product_attention` with a sliding window over the KV cache. The `prefill()` method handles batch prefilling during inference with optional attention masks.

**Value Embedding Integration**: When value embeddings are provided (in the first/last three layers), they are blended with the projected values using learnable mixing weights: `v = lambda[0] * v + lambda[1] * ve`. Mid-layers use only `lambda[0] * v` for efficiency.

#### Rotary (models/daisy/attention.py)

The rotary position embedding implementation preallocates cosine and sine tables up to `max_seq_len` during initialization, avoiding repeated computation. Angular frequencies are computed as `(1/base)^linspace(0,1,steps)` where base=1024. For half-truncation, only the first quarter of frequencies are active. The `_apply_rope()` function splits the input into two halves, applies the rotation `[x1*cos + x2*sin, -x1*sin + x2*cos]`, and concatenates the result.

#### Block (models/daisy/block.py)

Each transformer block combines attention and feedforward layers with residual connections:

**Pre-Normalization Architecture**: RMS normalization is applied before each sub-layer (attention and MLP), following the pre-norm transformer design that improves training stability.

**Residual Mixing**: Before processing, the input is blended with the initial embedding via `x = lambda[0] * x + lambda[1] * x0`, allowing the network to directly access token embeddings at any depth.

**Attention Integration**: If the block has an attention layer (all layers by default), the normalized input is passed through attention and added to the residual stream.

**MLP Integration**: The MLP operates on normalized activations and its output is added to the residual stream.

#### MLP (models/daisy/mlp.py)

The feedforward network uses a gated activation design:

**Expansion Ratio**: Hidden dimension is 4× the model dimension (`hdim = 4 * dim`).

**Squared ReLU Activation**: The activation function is `ReLU(x)^2`, which provides approximately 1-2% better performance than GELU while being computationally simpler. This activation was suggested by the ReLA paper (arXiv:2109.08668v2).

**Zero-Initialized Output**: The output projection `self.proj_w` is initialized to zero, ensuring early training relies on residual paths.

**Weight Decay Multipliers**: Both weight matrices have `wd_mul = 2.0` attributes for use with the Muon optimizer, which applies differential weight decay.

**BFloat16 Computation**: All operations are performed in BF16 for efficiency, with inputs explicitly cast to match weight dtype.

### Precision Strategy

Daisy follows a "BFloat16-first" design philosophy:

- **Embeddings and Projections**: Stored and computed in BF16
- **Normalization**: RMS norm computed in the input dtype (typically BF16)
- **Attention**: QK normalization and attention operations in BF16, with selective FP32 for numerically sensitive operations
- **Residuals**: Accumulated in BF16
- **Logits**: Computed in BF16, upcast to FP32 for loss computation
- **Gradients**: Backpropagation in mixed precision with automatic gradient scaling

This approach balances speed (BF16 is 2× faster on modern hardware) with stability (BF16's larger dynamic range handles the wide range of values in transformer training).

### Sliding Window Attention

Long-context capability is achieved through sliding window attention with document awareness:

**Window Scheduling**: The attention window size is progressively expanded during training (unless `full_windows=true`). The window size is specified in blocks, where each block contains `window_block_size` tokens (typically 128).

**Block-Sparse Masks**: Attention masks are constructed at block granularity, reducing memory from O(T²) to O((T/B)²) where B is the block size. Within each block, fine-grained causal masking is applied via the `mask_mod` function.

**Document Boundaries**: EOS tokens define document boundaries. Attention is prevented from crossing these boundaries, ensuring the model doesn't mix context from different documents.

**Long-Short Pattern**: Layers alternate between long windows (full size) and short windows (half size) in a 1-3-3-3 pattern, providing a mix of local and global context throughout the network.

### Training Optimizations

**Compiled Execution**: The model is compiled with `torch.compile(dynamic=False)` for optimized execution graphs.

**Gradient Accumulation**: Supports arbitrary accumulation steps to simulate larger batch sizes.

**Distributed Data Parallel**: Uses PyTorch DDP with NCCL backend for multi-GPU training.

**Efficient Checkpointing**: Only saves checkpoints when validation loss improves, avoiding unnecessary I/O.

---

## Repository Layout

```
daisy/
├── config/                      # Training configurations (YAML)
│   ├── pretrain_450m.yml       # 450M pretraining config
│   ├── pretrain_1.6B.yml       # 1.6B pretraining config
│   ├── pretrain_150m.yml       # 150M pretraining config
│   ├── fine_tune_450m.yml      # Fine-tuning configs
│   └── instruct_task_450m.yml   # Instruction tuning configs
│
├── model_specs/                 # Model architecture definitions (YAML)
│   ├── daisy_450m.yml          # 450M architecture spec
│   ├── daisy_1p6b.yml          # 1.6B architecture spec
│   └── daisy_150m.yml          # 150M architecture spec
│
├── models/                      # Model implementations
│   └── daisy/
│       ├── daisy_core.py       # Main model class (DaisyCore)
│       ├── attention.py        # Attention + RoPE implementations
│       ├── block.py            # Transformer block
│       ├── mlp.py              # Feedforward network
│       └── functional.py       # Utility functions (norm, init)
│
├── training/                    # Training infrastructure
│   ├── hparams.py              # Hyperparameter management
│   ├── optim.py                # Optimizer builders (AdamW, Muon)
│   ├── eval.py                 # Evaluation logic
│   ├── progress.py             # Token-based progress tracking
│   └── data_gen_stream.py      # Distributed data loading
│
├── inference/                   # Generation system
│   ├── generate.py             # Generator class with sampling
│   └── kv_cache.py             # KV cache for autoregressive generation
│
├── tools/                       # Utilities
│   ├── checkpoint.py           # Checkpoint load/save
│   ├── runner.py               # Multi-run orchestration
│   ├── lr_sweep.py             # Learning rate search
│   └── inspect_checkpoint.py   # Checkpoint inspection
│
├── data/                        # Data preparation scripts
│   ├── cached_fineweb100B.py
│   └── cached_fineweb_edu_100B.py
│
├── train.py                     # Main training script
├── sample.py                    # Inference CLI
├── run.sh                       # Training wrapper script
└── nohup_run.sh                # Background execution helper
```

### Key Files

**train.py**: Main training loop with gradient accumulation, distributed training, evaluation, and checkpointing. Supports warm-starting from checkpoints and token-based scheduling.

**sample.py**: Command-line interface for text generation with support for one-shot prompts and interactive chat mode. Includes streaming output and adjustable sampling parameters.

**run.sh**: Wrapper script that launches training via `torchrun` with proper distributed configuration. Handles command-line overrides and environment setup.

**config/*.yml**: Training configurations specifying datasets, optimization settings, evaluation cadence, and checkpoint frequency. Inherits from model specs and allows CLI overrides.

**model_specs/*.yml**: Architecture definitions specifying layer counts, dimensions, attention parameters, and vocabulary size. Separate from training configs for reusability.

---

## Training Usage

### Basic Training

The simplest way to start training is with the `run.sh` wrapper:

```bash
./run.sh config/pretrain_420m.yml -n 8
```

This launches training on 8 GPUs using the 450M model configuration. The script automatically:
- Sets up distributed training via `torchrun`
- Loads the configuration and model spec
- Initializes data loaders
- Runs training with periodic evaluation
- Saves checkpoints when validation improves

### Configuration System

Daisy uses a two-layer YAML configuration system:

**Model Specs** (`model_specs/*.yml`) define the architecture:
```yaml
model_class: models.daisy.daisy_core.DaisyCore
vocab_size: 50257
num_layers: 16
num_heads: 8
head_dim: 128
model_dim: 1024
attention_window_len: 3456
max_seq_len: 65536
window_block_size: 128
eos_token_id: 50256
```

**Training Configs** (`config/*.yml`) define the training process:
```yaml
model_spec: daisy_450m
train_shards: "data/fineweb-edu/edu_fineweb_train_*.bin"
target_tokens: 8000000000
cooldown_frac: 0.5
learning_rate_schedule: linear_decay
training_sequence_length: 65536
train_attention_window_len: 3456
grad_acc_steps: 1

optimizers:
  - type: AdamW
    betas: [0.9, 0.95]
    eps: 1.0e-10
    weight_decay: 0.0
    fused: true
    params:
      - group: head_params
        lr: 0.003464
      - group: embed_params
        lr: 0.1732
      - group: scalar_params
        lr: 0.01299
  
  - type: Muon
    momentum: 0.95
    weight_decay: 0.0
    params:
      - group: hidden_matrix_params
        lr: 0.02

val_shards:
  - type: "fineweb-edu"
    path: "data/fineweb-edu/edu_fineweb_val_*.bin"
val_loss_every_tokens: 262144000
checkpoint_per_n_tokens: 1
save_checkpoint: true
```

### Common Training Scenarios

#### Pretraining from Scratch

```bash
./run.sh config/pretrain_420m.yml -n 8 \
    target_tokens=8000000000 \
    wandb_log=true \
    wandb_project=my-project \
    wandb_run_name=pretrain-450m-v1
```

#### Fine-Tuning from a Checkpoint

```bash
./run.sh config/fine_tune_450m.yml -n 8 \
    -p checkpoints/20251018T0111-val1.750-step001300-run1-best.pt \
    target_tokens=1500000000
```

The `-p` flag loads model weights from the checkpoint but resets the optimizer and learning rate schedule. This is a "warm start" rather than a full resume.

#### Single-GPU Debugging

```bash
./run.sh config/pretrain_280m.yml -n 1 \
    target_tokens=100000000 \
    val_loss_every_tokens=10000000
```

#### Adjusting Learning Rates

```bash
./run.sh config/pretrain_420m.yml -n 8 \
    optimizers[0].params[0].lr=0.005 \
    optimizers[1].params[0].lr=0.03
```

#### Forcing Full Attention Windows

By default, attention windows expand progressively during training. To use full windows throughout:

```bash
./run.sh config/pretrain_420m.yml -n 8 --full_windows
```

### Direct Training (Without Wrapper)

For more control, invoke `train.py` directly:

```bash
torchrun --nproc_per_node=8 train.py config/pretrain_420m.yml target_tokens=8000000000
```

### Multi-Node Training

```bash
torchrun --nproc_per_node=8 --nnodes=4 --node_rank=0 \
    --master_addr=192.168.1.1 --master_port=29500 \
    train.py config/pretrain_1.6B.yml
```

### Token-Based Scheduling

All training progress is measured in tokens processed, not steps. This ensures consistent comparisons across different batch sizes and gradient accumulation settings.

**Key Parameters:**
- `target_tokens`: Total tokens to process (e.g., 8B for pretraining)
- `val_loss_every_tokens`: Evaluation frequency (e.g., 262M tokens)
- `checkpoint_per_n_tokens`: Checkpoint frequency after warmup
- `checkpoint_warmup_tokens`: Tokens before first checkpoint

### Learning Rate Schedules

Three schedules are available via `learning_rate_schedule`:

**linear_decay**: Constant LR until `1 - cooldown_frac`, then linear decay to 0
```bash
learning_rate_schedule=linear_decay cooldown_frac=0.5
```

**linear_warmup_cosine_decay**: Linear warmup until `1 - cooldown_frac`, then cosine decay
```bash
learning_rate_schedule=linear_warmup_cosine_decay cooldown_frac=0.7
```

**cosine_decay**: Constant LR until `1 - cooldown_frac`, then cosine decay
```bash
learning_rate_schedule=cosine_decay cooldown_frac=0.9
```

Note: Schedules apply to AdamW optimizers. Muon optimizers use linear momentum warmup from 0.85 to 0.95 based on progress.

### Per-Group Learning Rates

Daisy uses different learning rates for different parameter groups:

- **embed_params**: Token embeddings (typically 50-100× higher than head)
- **head_params**: Output projection (base LR)
- **scalar_params**: Learnable scalars (typically 3-5× higher than head)
- **hidden_matrix_params**: Attention and MLP weights (Muon optimizer)

This differential learning rate strategy is crucial for training stability and convergence speed.

### Weights & Biases Integration

Enable experiment tracking with W&B:

```bash
./run.sh config/pretrain_420m.yml -n 8 \
    wandb_log=true \
    wandb_project=daisy-experiments \
    wandb_run_name=450m-baseline
```

Logged metrics include:
- Training loss (per step)
- Validation loss and perplexity (per evaluation)
- Learning rates (per optimizer group)
- Tokens processed and training time
- EMA of loss delta (for convergence monitoring)

### Checkpointing

Checkpoints are saved only when validation loss improves. Each checkpoint contains:
- Model weights
- Hyperparameters (merged from model spec and training config)
- Training metadata (step, best validation loss, tokens per step)
- Progress state (for tracking evaluation/checkpoint cadence)

Checkpoint filenames encode key information:
```
20251018T0111-val1.750-step001300-run1-best.pt
│           │         │         │    └─ suffix
│           │         │         └─ run ID
│           │         └─ training step
│           └─ validation loss
└─ timestamp (YYYYMMDDTHHMM)
```

### Inspecting Checkpoints

```bash
python -m tools.inspect_checkpoint checkpoints/model.pt
python -m tools.inspect_checkpoint checkpoints/model.pt --json
```

### Learning Rate Sweeps

Find optimal learning rates for a parameter group:

```bash
python -m tools.lr_sweep \
    --config config/pretrain_420m.yml \
    --group head_params \
    --num_scales 200 \
    --scale_min 0.01 \
    --scale_max 100
```

This runs short training trials across a geometric scale of learning rates and reports the optimal value based on EMA of loss delta.

---

## Inference Usage

### One-Shot Generation

Generate text from a prompt:

```bash
python sample.py checkpoints/model.pt \
    --prompt "Once upon a time" \
    --max_tokens 256 \
    --temperature 0.7 \
    --top_k 50 \
    --top_p 0.95 \
    --repetition_penalty 1.25 \
    --seed 1337 \
    --device cuda
```

**Sampling Parameters:**
- `--temperature` / `-t`: Controls randomness (0.0 = greedy, higher = more random)
- `--top_k`: Keep only top-k tokens by probability
- `--top_p`: Nucleus sampling threshold
- `--repetition_penalty` / `-rp`: Penalize recently generated tokens (1.0 = no penalty)
- `--seed` / `-s`: Random seed for reproducibility
- `--max_tokens`: Maximum tokens to generate

### Interactive Chat Mode

Start a conversational interface:

```bash
python sample.py checkpoints/instruct_model.pt --chat
```

**Chat Commands:**
- Type normally to send messages
- `/t=0.4` - Change temperature
- `/rp=1.2` - Change repetition penalty
- `/new` - Start a new conversation (clear history)
- `exit`, `quit`, or Ctrl-D - Exit chat

**Example Session:**
```
You: /t=0.3 What is the capital of France?
Assistant: The capital of France is Paris.

You: /rp=1.5 Tell me more about it.
Assistant: Paris is located in northern France...

You: /new
[Screen clears, conversation resets]
```

### Instruction-Tuned Models

For instruction-tuned checkpoints, prompts are automatically formatted:

```
### Instruction:
{your prompt}

### Response:
```

To use base model formatting (no template), add `--base`:

```bash
python sample.py checkpoints/model.pt --base --prompt "Once upon a time"
```

### Programmatic Generation

```python
import torch
from tools.checkpoint import model_from_checkpoint
from inference.generate import Generator

# Load model
model, hparams = model_from_checkpoint('checkpoints/model.pt', device='cuda')
model.eval()

# Create generator
gen = Generator(
    model=model,
    window=int(hparams['train_attention_window_len']),
    seed=1337,
    eos_token_id=hparams['eos_token_id'],
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.25,
    device='cuda',
)

# Encode prompt
import tiktoken
enc = tiktoken.get_encoding("gpt2")
prompt_ids = torch.tensor(enc.encode("Once upon a time"), device='cuda')

# Generate
with torch.inference_mode():
    gen_iter = gen.generate(prompt_ids, max_new_tokens=256)
    for token in gen_iter:
        print(enc.decode([int(token)]), end="", flush=True)
```

### Performance Optimization

The generator automatically:
- Compiles sampling and repetition penalty functions (on CUDA)
- Warms up with representative sequence lengths
- Uses KV caching for efficient autoregressive generation
- Manages sliding window context automatically

Typical performance on A100:
- Prefill: 10,000-50,000 tokens/sec (depending on sequence length)
- Generation: 50-150 tokens/sec (450M model)

### Repetition Penalty

The repetition penalty uses exponential decay over the last 128 tokens:

```python
weight = 0.5^(distance / 140)
scale = repetition_penalty^(accumulated_weight)
logits = logits * (1/scale if logits > 0 else scale)
```

This penalizes recent tokens more strongly than distant ones, with a half-life of 140 tokens. The penalty is capped at 3.0× to prevent extreme suppression.

---

## Design Philosophy

Daisy embodies several key design principles:

**BFloat16-First**: Default to BF16 for speed, selectively upcast for stability. Modern accelerators (A100, H100) provide 2× speedup for BF16 over FP32, and BF16's larger dynamic range handles transformer training better than FP16.

**Token-Based Scheduling**: Measure progress in tokens processed, not training steps. This enables fair comparisons across different batch sizes, gradient accumulation settings, and hardware configurations.

**Sliding Window Attention**: Achieve long-context capability (65K tokens) with O(T×W) complexity instead of O(T²). Block-sparse masks reduce memory further while maintaining quality.

**Disciplined Defaults**: Provide sensible baseline configurations that work out of the box. Users can override any parameter, but defaults should produce good results without tuning.

**Pragmatic Tooling**: Include utilities for common workflows (LR sweeps, checkpoint inspection, multi-run orchestration) rather than requiring external tools.

**Merged Operations**: Fuse related operations (QKVO projections, value embedding mixing) to reduce memory traffic and improve cache efficiency.

**Zero-Initialized Outputs**: Initialize output projections to zero so early training relies on residual paths. This improves stability and convergence speed.

**Differential Learning Rates**: Use different learning rates for embeddings, hidden layers, and output heads. This reflects the different optimization landscapes of these parameter groups.

---

## Acknowledgements

Originally launched as a fork of **modded-nanogpt** (Keller Jordan & contributors) and inspired by **nanoGPT** (Andrej Karpathy), this project reproduced a substantial portion of optimized training code.

---

## License

See `LICENSE`.
