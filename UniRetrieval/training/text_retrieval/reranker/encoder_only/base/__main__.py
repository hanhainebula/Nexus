from transformers import HfArgumentParser

from UniRetrieval.training.text_retrieval.abc.reranker import (
    AbsRerankerModelArguments,
    AbsRerankerDataArguments,
    AbsRerankerTrainingArguments
)
from UniRetrieval.training.text_retrieval.reranker.encoder_only.base import EncoderOnlyRerankerRunner


def main():
    parser = HfArgumentParser((AbsRerankerModelArguments, AbsRerankerDataArguments, AbsRerankerTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: AbsRerankerModelArguments
    data_args: AbsRerankerDataArguments
    training_args: AbsRerankerTrainingArguments

    runner = EncoderOnlyRerankerRunner(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args
    )
    runner.run()


if __name__ == "__main__":
    main()
