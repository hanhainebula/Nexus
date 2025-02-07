from Nexus.evaluation.recommendation import RecommenderEvalArgs, RecommenderEvalModelArgs, TDERecommenderEvalRunner


def main():
    
    eval_config_path = "./examples/recommendation/eval/eval_config.json"
    model_config_path = "./examples/recommendation/eval/eval_model_config.json"
    
    eval_args = RecommenderEvalArgs.from_json(eval_config_path)
    model_args = RecommenderEvalModelArgs.from_json(model_config_path)
        
    runner = TDERecommenderEvalRunner(
        eval_args=eval_args,
        model_args=model_args
    )

    runner.run()
    if runner.retriever is not None and hasattr(runner.retriever, "_id_transformer_group"):
        runner.retriever._id_transformer_group.__del__()
    if runner.ranker is not None and hasattr(runner.ranker, "_id_transformer_group"):
        runner.ranker._id_transformer_group.__del__()
    


if __name__ == "__main__":
    main()