from transformers import HfArgumentParser

from Nexus.evaluation.text_retrieval.airbench import (
    AIRBenchEvalArgs, AIRBenchEvalModelArgs,
    AIRBenchEvalRunner
)

from Nexus.evaluation.text_retrieval.arguments import load_config

def main():

    eval_config_path='/data1/home/recstudio/haoran/Nexus/examples/text_retrieval/evaluation/eval_config.json'
    model_config_path='/data1/home/recstudio/haoran/Nexus/examples/text_retrieval/evaluation/model_config.json'
    
    eval_args = load_config(eval_config_path, AIRBenchEvalArgs)
    model_args = load_config(model_config_path, AIRBenchEvalModelArgs)
    
    eval_args: AIRBenchEvalArgs
    model_args: AIRBenchEvalModelArgs

    runner = AIRBenchEvalRunner(
        eval_args=eval_args,
        model_args=model_args
    )

    runner.run()


if __name__ == "__main__":
    main()
    print("==============================================")
    print("Search results have been generated.")
    print("For computing metrics, please refer to the official AIR-Bench docs:")
    print("- https://github.com/AIR-Bench/AIR-Bench/blob/main/docs/submit_to_leaderboard.md")