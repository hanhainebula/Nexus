from transformers import HfArgumentParser

from . import TextRetrievalEvalArgs, TextRetrievalEvalModelArgs, TextRetrievalEvalRunner


def main():
    parser = HfArgumentParser((
        TextRetrievalEvalArgs,
        TextRetrievalEvalModelArgs
    ))

    eval_args, model_args = parser.parse_args_into_dataclasses()
    eval_args: TextRetrievalEvalArgs
    model_args: TextRetrievalEvalModelArgs

    runner = TextRetrievalEvalRunner(
        eval_args=eval_args,
        model_args=model_args
    )

    runner.run()


if __name__ == "__main__":
    main()
