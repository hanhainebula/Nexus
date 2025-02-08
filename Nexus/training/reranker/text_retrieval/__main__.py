from transformers import HfArgumentParser

from Nexus.training.reranker.text_retrieval import *


# def main():
#     parser = HfArgumentParser((TextRerankerModelArguments, TextRerankerDataArguments, TextRerankerTrainingArguments))
#     model_args, data_args, training_args = parser.parse_args_into_dataclasses()
#     model_args: TextRerankerModelArguments
#     data_args: TextRerankerDataArguments
#     training_args: TextRerankerTrainingArguments

#     runner = TextRerankerRunner(
#         model_args=model_args,
#         data_args=data_args,
#         training_args=training_args
#     )
#     runner.run()

def main():
    data_config_path='/data1/home/recstudio/haoran/Nexus/examples/text_retrieval/training/reranker/data_config.json'
    model_config_path='/data1/home/recstudio/haoran/Nexus/examples/text_retrieval/training/reranker/model_config.json'
    train_config_path='/data1/home/recstudio/haoran/Nexus/examples/text_retrieval/training/reranker/training_config.json'
    
    model_args = TextRerankerModelArguments.from_json(model_config_path)
    data_args = TextRerankerDataArguments.from_json(data_config_path)
    training_args = TextRerankerTrainingArguments.from_json(train_config_path)
    
    runner = TextRerankerRunner(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args
    )
    runner.run()

if __name__ == "__main__":
    main()
