import os, torch, torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
x = torch.ones(1, device=f"cuda:{int(os.environ['LOCAL_RANK'])}")
dist.all_reduce(x)
print("rank", rank, "x", x.item())
dist.barrier()
