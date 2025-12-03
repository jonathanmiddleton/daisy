import os
import torch
import torch.distributed as dist

def main():
    dist.init_process_group("nccl")

    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    assert 0 <= local_rank < torch.cuda.device_count(), "LOCAL_RANK out of range"
    torch.cuda.set_device(local_rank)

    print("rank: ", rank)
    print("local_rank: ", local_rank)

    x = torch.ones(1, device=f"cuda:{local_rank}")
    dist.all_reduce(x)
    print("rank", rank, "x", x.item())
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()