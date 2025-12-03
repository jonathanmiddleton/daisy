import os
import torch
import torch.distributed as dist

def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    assert 0 <= local_rank < torch.cuda.device_count(), "LOCAL_RANK out of range"
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    dist.init_process_group(backend="nccl", device_id=device)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    x = torch.ones(1, device=device)
    dist.all_reduce(x)  # default is sum

    result = x.item()
    print(f"rank {rank} | world_size={world_size} | result={result}")

    assert result == float(world_size), (
        f"all_reduce sanity check failed: got {result}, "
        f"expected {world_size} (nnodes * nproc_per_node)"
    )
    print(f"Success!")
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()