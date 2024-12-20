from transformers import HfArgumentParser

from UniRetrieval.training.reranker.text_retrieval import *
from .runner import EncoderOnlyRerankerRunner


def main():
    parser = HfArgumentParser((AbsTextRerankerModelArguments, AbsTextRerankerDataArguments, AbsTextRerankerTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: AbsTextRerankerModelArguments
    data_args: AbsTextRerankerDataArguments
    training_args: AbsTextRerankerTrainingArguments

    runner = EncoderOnlyRerankerRunner(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args
    )
    runner.run()


if __name__ == "__main__":
    main()
