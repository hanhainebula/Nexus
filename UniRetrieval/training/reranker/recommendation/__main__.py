from UniRetrieval.training.reranker.recommendation.runner import RankerRunner



def main():
    data_config_path = "/data1/home/recstudio/haoran/UniRetrieval/recommender_examples/config/data/recflow_ranker.json"
    train_config_path = "/data1/home/recstudio/haoran/UniRetrieval/recommender_examples/config/mlp_ranker/train.json"
    model_config_path = "/data1/home/recstudio/haoran/UniRetrieval/recommender_examples/config/mlp_ranker/model.json"
    
    runner = RankerRunner(
        model_config_path=model_config_path,
        data_config_path=data_config_path,
        train_config_path=train_config_path
    )
    runner.run()


if __name__ == "__main__":
    main()
