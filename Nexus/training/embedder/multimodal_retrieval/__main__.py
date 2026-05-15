import argparse
import sys

from transformers import HfArgumentParser

from .arguments import (
    MultimodalEmbedderDataArguments,
    MultimodalEmbedderModelArguments,
    MultimodalEmbedderTrainingArguments,
)
from .runner import MultimodalEmbedderRunner


def parse_cli_args(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--model_config")
    config_parser.add_argument("--data_config")
    config_parser.add_argument("--training_config")
    config_args, remaining = config_parser.parse_known_args(argv)

    config_paths = [
        config_args.model_config,
        config_args.data_config,
        config_args.training_config,
    ]
    if any(path is not None for path in config_paths):
        missing = []
        if config_args.model_config is None:
            missing.append("--model_config")
        if config_args.data_config is None:
            missing.append("--data_config")
        if config_args.training_config is None:
            missing.append("--training_config")
        if missing:
            raise ValueError(f"Config-file mode requires the following arguments: {', '.join(missing)}")
        if len(remaining) != 0:
            raise ValueError(f"Unexpected extra arguments in config-file mode: {' '.join(remaining)}")

        return (
            MultimodalEmbedderModelArguments.from_json(config_args.model_config),
            MultimodalEmbedderDataArguments.from_json(config_args.data_config),
            MultimodalEmbedderTrainingArguments.from_json(config_args.training_config),
        )

    parser = HfArgumentParser(
        (
            MultimodalEmbedderModelArguments,
            MultimodalEmbedderDataArguments,
            MultimodalEmbedderTrainingArguments,
        )
    )
    return parser.parse_args_into_dataclasses(args=argv)


def main():
    model_args, data_args, training_args = parse_cli_args()

    runner = MultimodalEmbedderRunner(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )
    runner.run()


if __name__ == "__main__":
    main()
