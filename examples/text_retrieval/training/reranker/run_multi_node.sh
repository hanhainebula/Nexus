ACCELERATE_CONFIG=/root/Nexus/examples/text_retrieval/training/multi_node.json

accelerate launch --config_file $ACCELERATE_CONFIG /root/Nexus/examples/text_retrieval/training/reranker/multi_device.py