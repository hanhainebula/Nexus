import json
from transformers import HfArgumentParser

from Nexus.training.embedder.text_retrieval.decoder_only import *
import time
def main():
    data_config_path='/share/project/aqjiang/Nexus/examples/text_retrieval/training/decoder/data_config.json'
    train_config_path='/share/project/aqjiang/Nexus/examples/text_retrieval/training/decoder/training_config.json'
    model_config_path='/share/project/aqjiang/Nexus/examples/text_retrieval/training/decoder/model_config.json'
    
    model_args = DecoderOnlyEmbedderModelArguments.from_json(model_config_path)
    data_args = DecoderOnlyEmbedderDataArguments.from_json(data_config_path)
    training_args = DecoderOnlyEmbedderTrainingArguments.from_json(train_config_path)
    runner = DecoderOnlyEmbedderRunner(
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
