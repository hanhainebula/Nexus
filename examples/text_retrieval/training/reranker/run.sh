ACCELERATE_CONFIG=/data1/home/recstudio/haoran/Nexus/examples/text_retrieval/training/single_node_multi_device.json

cd /data1/home/recstudio/haoran/Nexus

accelerate launch --config_file $ACCELERATE_CONFIG Nexus/training/reranker/text_retrieval/__main__.py