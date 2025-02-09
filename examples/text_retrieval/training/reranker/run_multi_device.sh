# torchrun --nproc_per_node=4 --nnodes=1 /root/Nexus/examples/text_retrieval/training/reranker/multi_device.py
ACCELERATE_CONFIG=/root/Nexus/examples/text_retrieval/training/multi_device.json

accelerate launch --config_file $ACCELERATE_CONFIG /root/Nexus/examples/text_retrieval/training/reranker/multi_device.py
