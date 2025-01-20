from Nexus.training.reranker.recommendation.tde_runner import TDERankerRunner
from Nexus.training.reranker.recommendation.modeling import MLPRanker


def main():
    data_config_path = "./examples/recommendation/config/data/recflow_ranker.json"
    train_config_path = "./examples/recommendation/config/mlp_ranker_tde/train.json"
    model_config_path = "./examples/recommendation/config/mlp_ranker_tde/model.json"
    
    runner = TDERankerRunner(
        model_config_or_path=model_config_path,
        data_config_or_path=data_config_path,
        train_config_or_path=train_config_path,
        model_class=MLPRanker
    )
    runner.run()
    if hasattr(runner.model, "_id_transformer_group"):
        runner.model._id_transformer_group.__del__()

if __name__ == "__main__":
    main()
