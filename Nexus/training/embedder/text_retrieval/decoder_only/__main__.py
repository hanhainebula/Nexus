from transformers import HfArgumentParser

from . import (
    DecoderOnlyEmbedderModelArguments,
    DecoderOnlyEmbedderRunner,
)
from .arguments import DecoderOnlyEmbedderDataArguments, DecoderOnlyEmbedderTrainingArguments


def main():
    parser = HfArgumentParser(
        (
            DecoderOnlyEmbedderModelArguments,
            DecoderOnlyEmbedderDataArguments,
            DecoderOnlyEmbedderTrainingArguments,
        )
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    runner = DecoderOnlyEmbedderRunner(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args
    )
    runner.run()


if __name__ == "__main__":
    main()
