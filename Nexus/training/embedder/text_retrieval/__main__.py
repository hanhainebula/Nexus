from transformers import HfArgumentParser

from Nexus.training.embedder.text_retrieval import (
    TextEmbedderDataArguments,
    TextEmbedderModelArguments,
    TextEmbedderRunner,
    TextEmbedderTrainingArguments,
)

def main():
    parser = HfArgumentParser(
        (
            TextEmbedderModelArguments,
            TextEmbedderDataArguments,
            TextEmbedderTrainingArguments,
        )
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    runner = TextEmbedderRunner(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args
    )
    runner.run()


if __name__ == "__main__":
    main()
