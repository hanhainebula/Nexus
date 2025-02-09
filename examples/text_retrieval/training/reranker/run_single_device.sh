ACCELERATE_CONFIG=/root/Nexus/examples/text_retrieval/training/single_device.json

# export PET_NNODES=1
# export PET_NODE_RANK=0
# export PET_WORLD_SIZE=1
# export PET_MASTER_PORT=29500
# export PET_MASTER_ADDR=127.0.0.1
# export PET_NPROC_PER_NODE=1

# unset PET_NNODES
# unset PET_NODE_RANK
# unset PET_WORLD_SIZE
# unset PET_MASTER_PORT
# unset PET_MASTER_ADDR
# unset PET_NPROC_PER_NODE

accelerate launch --config_file $ACCELERATE_CONFIG /root/Nexus/examples/text_retrieval/training/reranker/single_device.py

# torchrun --nnodes 1 --nproc-per-node 1 --node-rank 0 --master-addr localhost --master-port 29500 --standalone /root/Nexus/examples/text_retrieval/training/reranker/single_device.py
