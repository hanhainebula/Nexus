import argparse
import sys

from transformers import HfArgumentParser

from .arguments import load_config
from .arguments import MultimodalRetrievalEvalArgs, MultimodalRetrievalEvalModelArgs
from .runner import MultimodalRetrievalEvalRunner


def parse_cli_args(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--eval_config")
    config_parser.add_argument("--model_config")
    config_args, remaining = config_parser.parse_known_args(argv)

    if config_args.eval_config is not None or config_args.model_config is not None:
        if config_args.eval_config is None or config_args.model_config is None:
            raise ValueError("Config-file mode requires both --eval_config and --model_config.")
        if len(remaining) != 0:
            raise ValueError(f"Unexpected extra arguments in config-file mode: {' '.join(remaining)}")
        return (
            load_config(config_args.eval_config, MultimodalRetrievalEvalArgs),
            load_config(config_args.model_config, MultimodalRetrievalEvalModelArgs),
        )

    eval_parser = HfArgumentParser(MultimodalRetrievalEvalArgs)
    eval_args, remaining = eval_parser.parse_args_into_dataclasses(
        args=argv,
        return_remaining_strings=True,
    )

    model_parser = HfArgumentParser(MultimodalRetrievalEvalModelArgs)
    model_args, remaining = model_parser.parse_args_into_dataclasses(
        args=remaining,
        return_remaining_strings=True,
    )
    if len(remaining) != 0:
        raise ValueError(f"Unexpected extra arguments: {' '.join(remaining)}")

    if model_args.token is None:
        model_args.token = eval_args.token
    return eval_args, model_args


def main():
    eval_args, model_args = parse_cli_args()

    runner = MultimodalRetrievalEvalRunner(eval_args=eval_args, model_args=model_args)
    runner.run()


if __name__ == "__main__":
    main()
