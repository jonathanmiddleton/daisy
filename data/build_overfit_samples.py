from concurrent.futures import ThreadPoolExecutor

import torch
from pathlib import Path
import os, sys
import shutil

from huggingface_hub import hf_hub_download


def create_overfit_sample(input_file_name: str, output_file: str, num_tokens: int = 1_000_000):
    local_dir = os.path.join(os.path.dirname(__file__), 'overfit', '.cache')
    os.makedirs(local_dir, exist_ok=True)

    def get(fname):
        if not os.path.exists(os.path.join(local_dir, fname)):
            if fname.startswith("dclm_baseline"):
                hf_hub_download(
                    repo_id="JonathanMiddleton/dclm-baseline",
                    filename=fname,
                    repo_type="dataset",
                    local_dir=local_dir,
                )
            elif fname.startswith("edu_fineweb"):
                hf_hub_download(
                    repo_id="karpathy/fineweb-edu-100B-gpt2-token-shards",
                    filename=fname,
                    repo_type="dataset",
                    local_dir=local_dir,
                )
            else:
                raise ValueError(f"Unknown dataset: {fname}")
    get(input_file_name)

    input_path = Path(local_dir) / input_file_name
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read the header from input file
    header = torch.from_file(str(input_path), False, 256, dtype=torch.int32, device='cpu')
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    original_num_tokens = int(header[2])

    print(f"Original file has {original_num_tokens:,} tokens")
    tokens_to_read = min(num_tokens, original_num_tokens)
    print(f"Extracting {tokens_to_read:,} tokens...")

    with input_path.open("rb", buffering=0) as f:
        tokens = torch.empty(tokens_to_read, dtype=torch.uint16)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * tokens_to_read, "number of tokens read does not match expected"

    new_header = torch.zeros(256, dtype=torch.int32)
    new_header[0] = 20240520  # magic number
    new_header[1] = 1  # version
    new_header[2] = tokens_to_read

    print(f"Writing to {output_path}...")
    with output_path.open("wb") as f:
        f.write(new_header.numpy().tobytes())
        f.write(tokens.numpy().tobytes())

    print(f"Created {output_path} with {tokens_to_read:,} tokens")
    print(f"File size: {output_path.stat().st_size:,} bytes")


if __name__ == "__main__":
    print("Building overfit samples...")
    tasks = [
        dict(
            input_file_name="dclm_baseline_train_000001.bin",
            output_file="data/overfit/dclm_baseline_overfit_1M.bin",
            num_tokens=1_000_000,
        ),
        dict(
            input_file_name="edu_fineweb_train_000001.bin",
            output_file="data/overfit/edu_fineweb_overfit_1M.bin",
            num_tokens=1_000_000,
        ),
    ]

    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        futures = [
            executor.submit(create_overfit_sample, **task)
            for task in tasks
        ]
        for f in futures:
            f.result()

    cache_dir = os.path.join(os.path.dirname(__file__), 'overfit', '.cache')
    try:
        if os.path.isdir(cache_dir):
            print(f"Removing cache directory {cache_dir}...")
            shutil.rmtree(cache_dir)
    except Exception as e:
        print(f"Warning: failed to remove cache directory {cache_dir}: {e}")