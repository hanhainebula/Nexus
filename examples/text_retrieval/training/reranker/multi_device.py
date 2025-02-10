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

if __name__ == "__main__":
    main()
