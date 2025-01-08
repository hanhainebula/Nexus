from transformers import HfArgumentParser

from Nexus.training.reranker.text_retrieval import *


def main():
    parser = HfArgumentParser((TextRerankerModelArguments, TextRerankerDataArguments, TextRerankerTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: TextRerankerModelArguments
    data_args: TextRerankerDataArguments
    training_args: TextRerankerTrainingArguments

    runner = TextRerankerRunner(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args
    )
    runner.run()


if __name__ == "__main__":
    main()
