cd /root/Nexus

ACCELERATE_CONFIG=/root/Nexus/examples/text_retrieval/training/multi_device.json

export ACCELERATE_LOG_LEVEL=debug
export NCCL_DEBUG=INFO 
export TORCH_DISTRIBUTED_DEBUG=DETAIL  

accelerate launch --config_file $ACCELERATE_CONFIG Nexus/training/embedder/text_retrieval/__main__.py
