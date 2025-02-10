from transformers import HfArgumentParser

from Nexus.evaluation.text_retrieval import TextRetrievalEvalArgs, TextRetrievalEvalModelArgs, TextRetrievalEvalRunner
from Nexus.evaluation.text_retrieval.arguments import load_config

def main():

    eval_config_path='/root/Nexus/examples/text_retrieval/evaluation/config/eval_config_fiqa.json'
    model_config_path='/root/Nexus/examples/text_retrieval/evaluation/config/model_config.json'
    
    eval_args = load_config(eval_config_path, TextRetrievalEvalArgs)
    model_args = load_config(model_config_path, TextRetrievalEvalModelArgs)
    
    eval_args: TextRetrievalEvalArgs
    model_args: TextRetrievalEvalModelArgs

    runner = TextRetrievalEvalRunner(
        eval_args=eval_args,
        model_args=model_args
    )

    runner.run()


if __name__ == "__main__":
    main()
