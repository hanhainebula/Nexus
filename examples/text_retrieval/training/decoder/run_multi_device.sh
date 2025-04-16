ACCELERATE_CONFIG=/share/project/aqjiang/Nexus/examples/text_retrieval/training/multi_device.json

accelerate launch --config_file $ACCELERATE_CONFIG /share/project/aqjiang/Nexus/examples/text_retrieval/training/decoder/multi_device.py
