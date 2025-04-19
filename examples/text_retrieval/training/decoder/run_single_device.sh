ACCELERATE_CONFIG=/share/project/aqjiang/Nexus/examples/text_retrieval/training/single_device.json


accelerate launch --config_file $ACCELERATE_CONFIG /share/project/aqjiang/Nexus/examples/text_retrieval/training/decoder/single_device.py
