
ACCELERATE_CONFIG=/root/Nexus/examples/text_retrieval/training/single_device.json

accelerate launch --config_file $ACCELERATE_CONFIG /root/Nexus/examples/text_retrieval/training/reranker/single_device.py