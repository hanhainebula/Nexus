from transformers import HfArgumentParser

from Nexus.training.reranker.text_retrieval import *
import time

def main():
    data_config_path='/root/Nexus/examples/text_retrieval/training/reranker/data_config.json'
    model_config_path='/root/Nexus/examples/text_retrieval/training/reranker/model_config.json'
    train_config_path='/root/Nexus/examples/text_retrieval/training/reranker/training_config.json'
    
    model_args = TextRerankerModelArguments.from_json(model_config_path)
    data_args = TextRerankerDataArguments.from_json(data_config_path)
    training_args = TextRerankerTrainingArguments.from_json(train_config_path)
    
    runner = TextRerankerRunner(
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
单机多卡：
{'train_runtime': 3002.3872, 'train_samples_per_second': 161.812, 'train_steps_per_second': 0.809, 'train_loss': 1.2864001208541325, 'epoch': 1.0}  
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 2429/2429 [50:02<00:00,  1.24s/it]
02/09/2025 12:23:28 - INFO - Nexus.training.reranker.text_retrieval.trainer -   Saving model checkpoint to /root/models/reranker_ckpt_multi_device
程序运行耗时: 3008.0696 秒
程序运行耗时: 3009.8632 秒
程序运行耗时: 3008.1315 秒
程序运行耗时: 3010.5069 秒

多机多卡
[15:24<00:00,  
程序运行耗时: 929.1384 秒
程序运行耗时: 929.1168 秒
程序运行耗时: 929.1435 秒
程序运行耗时: 929.1707 秒
程序运行耗时: 929.1756 秒
程序运行耗时: 929.1258 秒
程序运行耗时: 929.1696 秒
程序运行耗时: 929.2332 秒
"""

if __name__ == "__main__":
    main()
