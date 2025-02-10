import json
from transformers import HfArgumentParser

from Nexus.training.embedder.text_retrieval import *
import time
def main():
    data_config_path='/root/Nexus/examples/text_retrieval/training/embedder/data_config.json'
    train_config_path='/root/Nexus/examples/text_retrieval/training/embedder/training_config.json'
    model_config_path='/root/Nexus/examples/text_retrieval/training/embedder/model_config.json'
    
    model_args = TextEmbedderModelArguments.from_json(model_config_path)
    data_args = TextEmbedderDataArguments.from_json(data_config_path)
    training_args = TextEmbedderTrainingArguments.from_json(train_config_path)
    runner = TextEmbedderRunner(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args
    )
    start = time.time()
    runner.run()
    end = time.time()
    elapsed_time = end-start
    print(f"程序运行耗时: {elapsed_time:.4f} 秒")
    
"""
多机多卡: 14:52
程序运行耗时: 897.2552 秒
程序运行耗时: 897.1733 秒
程序运行耗时: 897.2755 秒
程序运行耗时: 897.6597 秒
程序运行耗时: 897.6720 秒
程序运行耗时: 897.7898 秒
程序运行耗时: 897.6690 秒
程序运行耗时: 897.7205 秒"""

if __name__ == "__main__":
    main()
