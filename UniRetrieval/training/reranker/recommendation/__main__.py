from UniRetrieval.training.reranker.recommendation.runner import RankerRunner
from UniRetrieval.training.reranker.recommendation.modeling import MLPRanker, MyMLPRanker


def main():
    data_config_path = "/data1/home/recstudio/haoran/UniRetrieval/examples/recommendation/config/data/recflow_ranker.json"
    train_config_path = "/data1/home/recstudio/haoran/UniRetrieval/examples/recommendation/config/mlp_ranker/train.json"
    model_config_path = "/data1/home/recstudio/haoran/UniRetrieval/examples/recommendation/config/mlp_ranker/model.json"
    
    runner = RankerRunner(
        model_config_path=model_config_path,
        data_config_path=data_config_path,
        train_config_path=train_config_path,
        model_class=MLPRanker
    )
    runner.run()


if __name__ == "__main__":
    main()
