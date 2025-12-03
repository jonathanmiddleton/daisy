import os, torch, torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
local_rank = int(os.environ['LOCAL_RANK'])
print("rank", rank)
print("local_rank", local_rank)
x = torch.ones(1, device=f"cuda:{local_rank}")
dist.all_reduce(x)
print("rank", rank, "x", x.item())
dist.barrier(device_ids=[f"cuda:{local_rank}"])
dist.destroy_process_group()