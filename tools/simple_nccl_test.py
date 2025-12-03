import os
import torch
import torch.distributed as dist

def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    assert 0 <= local_rank < torch.cuda.device_count(), "LOCAL_RANK out of range"
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    # Tell c10d which device this process uses
    dist.init_process_group(backend="nccl", device_id=device)

    rank = dist.get_rank()

    print("rank:", rank)
    print("local_rank:", local_rank)

    x = torch.ones(1, device=device)
    dist.all_reduce(x)
    print("rank", rank, "x", x.item())
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()