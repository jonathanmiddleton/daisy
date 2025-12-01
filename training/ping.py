#!/usr/bin/env python3
"""
Distributed ping utility for sanity checking NCCL/GPU multi-node connectivity.

Usage: this module is intended to be launched via torchrun or run_ping.sh.
It initializes a process group (default backend: nccl), performs an all_reduce
on a CUDA tensor containing the local rank id, and prints the result.
"""
import os
import sys

import torch
import torch.distributed as dist


def main() -> int:
    backend = os.environ.get("TORCH_DDP_BACKEND", "nccl")
    try:
        dist.init_process_group(backend=backend)
    except Exception as e:
        # Fallback to gloo when NCCL is unavailable (e.g., CPU-only envs)
        if backend != "gloo":
            dist.init_process_group(backend="gloo")
        else:
            raise e

    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        tensor = torch.tensor([rank], device=device, dtype=torch.int64)
        dist.all_reduce(tensor)
        # Ensure synchronization before printing
        if device.type == "cuda":
            torch.cuda.synchronize()
        print(f"Rank {rank}/{world_size}, tensor={tensor.item()}")
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
