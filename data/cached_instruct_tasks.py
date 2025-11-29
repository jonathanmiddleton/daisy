import os
from huggingface_hub import snapshot_download

repo_root = os.path.dirname(os.path.dirname(__file__))
local_dir = os.path.join(repo_root, "data", "instruct_mix")

snapshot_download(
    repo_id="JonathanMiddleton/instruct-task-mix",
    repo_type="dataset",
    local_dir=local_dir,
)