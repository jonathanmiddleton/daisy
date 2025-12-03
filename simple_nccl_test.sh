MASTER_ADDR=$(ip -4 -o addr show scope global | awk '$4 ~ /^10\./ {print $4; exit}' | cut -d/ -f1)
MASTER_PORT=29500
MASTER_HOSTNAME=$HOSTNAME
echo "Assuming hosts are on the same 10.0.0.0/24 network..."
echo "Open ports 1 - 65535 for hostmask 10.0.0.0/24"
echo "Execute on workers:"
echo "export MASTER_ADDR=$MASTER_ADDR"
echo "export MASTER_PORT=$MASTER_PORT"
echo "echo \"$MASTER_ADDR $MASTER_HOSTNAME\""' | sudo tee -a /etc/hosts'
echo "torchrun \
  --nproc_per_node=1 \
  --nnodes=2 \
  --rdzv_backend=c10d \
  --rdzv_endpoint="'$MASTER_ADDR'":"'$MASTER_PORT' \
  "--node_rank=1 \
  tools/simple_nccl_test.py"

torchrun \
  --nproc_per_node=1 \
  --nnodes=2 \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR":$MASTER_PORT \
  --node_rank=0 \
  tools/simple_nccl_test.py


