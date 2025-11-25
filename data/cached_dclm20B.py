import os
import sys
from huggingface_hub import hf_hub_download
def get(fname):
    local_dir = os.path.join(os.path.dirname(__file__), 'dclm_baseline')
    if not os.path.exists(os.path.join(local_dir, fname)):
        hf_hub_download(repo_id="JonathanMiddleton/dclm-baseline", filename=fname,
                        repo_type="dataset", local_dir=local_dir)
get("dclm_baseline_val_%06d.bin" % 0)
num_chunks = 200 # full dclm_baseline is 200 chunks. Each chunk is 100M tokens
if len(sys.argv) >= 2:
    num_chunks = int(sys.argv[1])
for i in range(1, num_chunks+1):
    get("dclm_baseline_train_%06d.bin" % i)
