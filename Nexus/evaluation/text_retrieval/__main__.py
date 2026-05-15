from transformers import HfArgumentParser

from Nexus.evaluation.text_retrieval import (
    TextRetrievalEvalArgs,
    TextRetrievalEvalModelArgs,
    TextRetrievalEvalRunner,
)

def main():
    parser = HfArgumentParser((TextRetrievalEvalArgs, TextRetrievalEvalModelArgs))
    eval_args, model_args = parser.parse_args_into_dataclasses()
    runner = TextRetrievalEvalRunner(
        eval_args=eval_args,
        model_args=model_args
    )

    runner.run()


if __name__ == "__main__":
    main()
