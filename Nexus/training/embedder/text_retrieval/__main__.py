from transformers import HfArgumentParser

from Nexus.training.embedder.text_retrieval import *

def main():
    parser = HfArgumentParser((
        TextEmbedderModelArguments,
        TextEmbedderDataArguments,
        TextEmbedderTrainingArguments
    ))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: TextEmbedderModelArguments
    data_args: TextEmbedderDataArguments
    training_args: TextEmbedderTrainingArguments

    runner = TextEmbedderRunner(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args
    )
    runner.run()


if __name__ == "__main__":
    main()
